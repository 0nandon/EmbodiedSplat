
import argparse
import os
import yaml
from pathlib import Path
from tqdm import tqdm

import clip
import numpy as np
import torch
import torch.nn.functional as F
from local_aggregate import LocalAggregator

from database.class_labels import SCANNET20_CLASS_LABELS
from semseg_metric import evaluate

import sys
sys.path.append("src/third_party")

import torch.nn.functional as F
from maskadapter.clip import CLIP

SCANNET_LABELS_20_FORMAT = [
    "a {} in a photo.".format(cls) for cls in SCANNET20_CLASS_LABELS
]

def load_yaml(path):
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)

def process_gt(raw_gt, labels_info):
    gt = np.ones_like(raw_gt) * -1
    cnt = 0
    for idx, label in labels_info.items():
        if label['name'] in SCANNET20_CLASS_LABELS:
            gt[raw_gt == idx] = cnt
            cnt += 1

    return torch.from_numpy(gt), cnt

def build_text_embedding(text, mode="base"):
    if mode == "base":
        clip.available_models()
        model, preprocess = clip.load("ViT-L/14@336px")
        run_on_gpu = torch.cuda.is_available()

        with torch.no_grad():
            text = clip.tokenize(text)
            if run_on_gpu:
                text = text.cuda()
            
            text_embedding = model.encode_text(text)
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    elif mode == "fast":
        clip_model = CLIP(model_name="convnext_large_d_320", pretrained="laion2b_s29b_b131k_ft_soup").cuda()
        classifier = clip_model.get_text_classifier(text, device='cuda')
        text_embedding = F.normalize(classifier, p=2, dim=-1, eps=1e-5)

    return text_embedding

def compute_costmap_2d(gaussians, text_feat):
    GS = gaussians
    
    feat_dict = torch.cat(GS.clip_features_2d["feat"], dim=0)
    feat_dict = torch.nn.functional.normalize(feat_dict, dim=-1, eps=1e-5)

    costmap = feat_dict.float() @ text_feat.float().t()

    idx = GS.clip_features_2d["idx"][0, GS.valid[0], :]
    idx = idx[:, :5]
    invalid = (idx != -1).sum(dim=-1) == 0.

    non_mask = idx == -1
    idx[non_mask] = 0.

    weights = GS.clip_features_2d["weight"][0, GS.valid[0], :]
    weights = weights[:, :5]
    weights /= weights.sum(dim=-1, keepdim=True)

    costmap_sel = costmap[idx.int()]

    plz = torch.bmm(costmap_sel.permute(0, 2, 1), weights.unsqueeze(-1)).squeeze(-1)

    costmap = (plz * 100).softmax(dim=-1).detach().cpu().numpy()

    invalid = invalid.detach().cpu().numpy()
    costmap_2d = costmap[~invalid, :]
    
    return costmap_2d

def compute_costmap_3d(gaussians, semantics, text_feat):
    GS = gaussians
    feat = semantics[0]
    feat = torch.nn.functional.normalize(feat, dim=-1, eps=1e-5)

    idx = GS.clip_features_2d["idx"][0, GS.valid[0], :]
    invalid = (idx != -1).sum(dim=-1) == 0.
    invalid = invalid

    costmap = ((feat @ text_feat.t().float()) * 100).softmax(dim=-1)
    costmap_3d = costmap[~invalid, :].detach().cpu().numpy()
    
    return costmap_3d

def ensemble(costmap_2d, costmap_3d, exp=1.0):
    ensemble = costmap_2d > costmap_3d

    lambda_balance = np.ones_like(costmap_2d)

    lambda_balance[ensemble == 1] = exp
    lambda_balance[ensemble != 1] = 1 - exp
        
    costmap = (costmap_2d ** lambda_balance) * (costmap_3d ** (1 - lambda_balance))
    
    return costmap

def compute_semseg(gaussians, coords, costmap):
    GS = gaussians
    pts = torch.from_numpy(coords)[None].cuda()
    
    idx = GS.clip_features_2d["idx"][0, GS.valid[0], :]
    invalid = (idx != -1).sum(dim=-1) == 0.

    means3D = GS.means[0, GS.valid[0], :][~invalid, :][None].cuda()
    opas = GS.opacities[0, GS.valid[0]][~invalid][None].cuda()
    semantics = torch.from_numpy(costmap[None]).cuda()
    cov3D = GS.covariances[0, GS.valid[0]][~invalid, :, :][None].cuda()

    eigvals, eigvecs = torch.linalg.eigh(cov3D[0].cpu())
    scales = torch.sqrt(eigvals)[None].cuda()

    agg = LocalAggregator(3, 200, 200, 200, [-20.0, -20.0, -20.0], 0.4).cuda()
    size = semantics.shape[-1]
    outs = []
    for idx in range(np.ceil(size/18).astype(int)):
        if (idx+1)*18 > size:
            s = torch.cat([semantics[:, :, idx*18:], torch.zeros(1, semantics.shape[1], 18 - size % 18).cuda()], dim=-1)
            out = agg(pts, means3D, opas, s, scales, cov3D[0].inverse()[None])
            outs.append(out[:, :size % 18])
        else:
            s = semantics[:, :, idx*18:(idx+1)*18]
            out = agg(pts, means3D, opas, s, scales, cov3D[0].inverse()[None])
            outs.append(out)
    
    pred = torch.cat(outs, dim=-1).argmax(dim=-1)
    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-mode", type=str, default="base")
    parser.add_argument("--labels-yaml", type=str,
                        default="src/evaluation/database/scannet_label_database.yaml")
    parser.add_argument("--pred-path", type=str, default="outputs_semantic/scannet")
    parser.add_argument("--pred-mode", type=str, default="ensemble")
    parser.add_argument("--dataset", type=str, default="scannet20")
    args = parser.parse_args()

    text_feat = build_text_embedding(SCANNET_LABELS_20_FORMAT, mode=args.text_mode)
    LABELS_INFO = load_yaml(args.labels_yaml)
    PRED_PATH = args.pred_path
    PRED_MODE = args.pred_mode
    DATASET = args.dataset

    gts_collate = []
    preds_collate = []

    for scene in tqdm(os.listdir(PRED_PATH)):
        SCENE_PATH = os.path.join(PRED_PATH, scene) 

        gaussians = torch.load(os.path.join(SCENE_PATH, "ckpt_gs.pt"), weights_only=False) 
        
        if PRED_MODE in ["3d", "ensemble"]:
            semantics = torch.load(os.path.join(SCENE_PATH, "ckpt_semantic.pt"), weights_only=False)
       
        costmap_2d = compute_costmap_2d(gaussians, text_feat)
        
        if PRED_MODE in ["3d", "ensemble"]:
            costmap_3d = compute_costmap_3d(gaussians, semantics, text_feat)
            
        if PRED_MODE == "ensemble":
            costmap = ensemble(costmap_2d, costmap_3d, 0.)
        elif PRED_MODE == "2d":
            costmap = costmap_2d
        elif PRED_MODE == "3d":
            costmap = costmap_3d
    
        pcd = np.load("dataset/scannet/test/" + scene + "/" + scene[5:] + ".npy")
        coords, gts = pcd[:, :3], pcd[:, 10]
    
        preds = compute_semseg(gaussians, coords, costmap)
        gts, _ = process_gt(gts, LABELS_INFO)
    
        gts_collate.append(gts)
        preds_collate.append(preds)


    gts_collate = torch.cat(gts_collate)
    preds_collate = torch.cat(preds_collate)
    evaluate(preds_collate.cpu(), gts_collate.cpu(), stdout=True, dataset=DATASET)
