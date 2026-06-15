
import os
import sys

# Resolve third_party relative to this file so the script is location-independent.
sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "third_party")
)

import argparse
from PIL import Image
from einops import rearrange
import numpy as np
import torch
import torchvision.transforms as tf
from tqdm import tqdm

from fastsam import FastSAM, FastSAMPrompt
import tensorflow as tff2
import tensorflow.compat.v1 as tff
import shutil

def rescale(
    image,
    shape,
):
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images,
    shape,
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2
    
    images = images[..., :, row : row + h_out, col : col + w_out]
    
    return images


def process_image(img_path):
    to_tensor = tf.ToTensor()

    img_lst = []
    org_img = Image.open(img_path)
    img = to_tensor(org_img.resize((640, 480)))
    img_lst.append(img[None])

    images = torch.cat(img_lst)

    *_, h_in, w_in = images.shape
    h_out, w_out = [384, 512]

    scale_factor = max(1.015*h_out / h_in, 1.015*w_out / w_in)

    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)

    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])

    images = center_crop(images, [384, 512])
    
    return images


def extract_openseg_img_feature(
        img, 
        openseg_model, 
        text_emb, 
        img_size=None, 
        regional_pool=True
    ):
    '''Extract per-pixel OpenSeg features.'''
        
    def array_to_bytes(array):
        assert array.ndim == 3 and array.shape[2] == 3, "Expected shape (H, W, 3)"
        assert array.dtype in [np.float32, np.float64], "Array must be float"

        uint8_img = tff.image.convert_image_dtype(array, dtype=tff.uint8, saturate=True)
        png_bytes = tff.io.encode_png(uint8_img).numpy()
        return png_bytes

    np_image_string = array_to_bytes(img.permute(1, 2, 0).cpu().numpy())
        
    results = openseg_model.signatures['serving_default'](
        inp_image_bytes=tff.convert_to_tensor(np_image_string),
        inp_text_emb=text_emb)
    img_info = results['image_info']
    crop_sz = [
        int(img_info[0, 0] * img_info[2, 0]),
        int(img_info[0, 1] * img_info[2, 1])
    ]
    if regional_pool:
        image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
    else:
        image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
    if img_size is not None:
        feat_2d = tff.cast(tff.image.resize_nearest_neighbor(
            image_embedding_feat, img_size, align_corners=True)[0], dtype=tff.float16).numpy()
    else:
        feat_2d = tff.cast(image_embedding_feat[[0]], dtype=tff.float16).numpy()

    feat_2d = torch.from_numpy(feat_2d)
        
    return feat_2d


def extract_fastsam_openseg_feats(images, fastsam, fastsam_prompt, openseg, openseg_text_emb):
    B, C, h, w = images.shape
    
    cache = {"clip": [], "fastsam": []}
    with torch.no_grad():
        inputs = (images * 255.).clone()
        everything_results = fastsam(
            inputs,
            device='cuda',
            retina_masks=True,
            imgsz=(w, h),
            conf=0.4,
            iou=0.9
        )
        prompt = fastsam_prompt(inputs, everything_results, device='cuda')

        clip_img_feats = []
        for idx, everything_result in enumerate(everything_results):
            format_results, masks_torch = prompt._format_results(everything_result, 0, sort=True)
            
            image = images[idx]
            clip_feats = extract_openseg_img_feature(
                image, 
                openseg, 
                openseg_text_emb, 
                img_size=[h, w], 
            )
                    
            clip_feats_flat = rearrange(clip_feats, "h w d -> (h w) d").cuda().float()
            masks_flat = rearrange(masks_torch, "n h w -> n (h w)").cuda().float()
            pooled = masks_flat @ clip_feats_flat
                    
            denom = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            mask_feats = pooled / denom
            
            cache["clip"].append(mask_feats)
            cache["fastsam"].append(format_results)

    return cache

def process(args):
    fastsam = FastSAM(args.fastsam_ckpt)
    fastsam_prompt = FastSAMPrompt
    openseg = tff2.saved_model.load(args.openseg_path, tags=[tff.saved_model.tag_constants.SERVING],)
    openseg_text_emb = tff.zeros([1, 1, 768])

    for scene_name in sorted(os.listdir(args.train_path)):
        print("Processing " + scene_name + "...")

        IMG_DIR = os.path.join(args.train_path, scene_name, "color")
        CACHE_DIR = os.path.join(args.train_path, scene_name, "cache")

        os.makedirs(CACHE_DIR, exist_ok=True)

        for i, img_file in enumerate(tqdm(sorted(os.listdir(IMG_DIR)))):
            if os.path.exists(os.path.join(CACHE_DIR, img_file[:-4] + ".pt")):
                continue

            IMG_FILE = os.path.join(IMG_DIR, img_file)
            img = process_image(IMG_FILE)
            cache = extract_fastsam_openseg_feats(img, fastsam, fastsam_prompt, openseg, openseg_text_emb)
            torch.save(cache, os.path.join(CACHE_DIR, img_file[:-4] + ".pt"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract FastSAM + OpenSeg features for ScanNet scenes."
    )
    parser.add_argument(
        "--train-path",
        default="dataset/scannet/train",
        help="Directory containing the per-scene subfolders to process.",
    )
    parser.add_argument(
        "--fastsam-ckpt",
        default="pretrained/FastSAM-x.pt",
        help="Path to the FastSAM checkpoint.",
    )
    parser.add_argument(
        "--openseg-path",
        default="pretrained/saved_openseg/openseg_exported_clip",
        help="Path to the exported OpenSeg saved_model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    process(parse_args())
