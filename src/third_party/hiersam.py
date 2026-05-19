

import os
import cv2
import numpy as np
from PIL import Image
import torch
import clip
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from typing import Any, Dict, List, Optional, Tuple

class Tools:
    def __init__(
        self,
        model: str = "vit_h",
        points_per_side: Optional[int] = 64,
        pred_iou_thresh: float = 0.88,
        box_nms_thresh: float = 0.7,
        stability_score_thresh: float = 0.95, 
        crop_n_layers: int = 0,  
        crop_n_points_downscale_factor: int = 1,
        min_mask_region_area: int = 0,
        crop_nms_thresh: int = 0.7, 
        crop_overlap_ratio: int = 512 / 1500,
        clip_model: str = "ViT-L/14@336px",
        load_sam: str = "",
        load_tracker: str = "",
    ) -> None:

        self.sam = sam_model_registry[model](checkpoint=load_sam).to('cuda')
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            box_nms_thresh=box_nms_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device)
        self.image = None

    def determine_tracker_model_cfg(self, model_path):
        if "large" in model_path:
            return "configs/samurai/sam2.1_hiera_l.yaml"
        elif "base_plus" in model_path:
            return "configs/samurai/sam2.1_hiera_b+.yaml"
        elif "small" in model_path:
            return "configs/samurai/sam2.1_hiera_s.yaml"
        elif "tiny" in model_path:
            return "configs/samurai/sam2.1_hiera_t.yaml"
        else:
            raise ValueError("Unknown model size in path!")

    def mask_nms(self, masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
        """
        Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
        
        Args:
            masks (torch.Tensor): has shape (num_masks, H, W)
            scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
            iou_thr (float, optional): The threshold for IoU.
            score_thr (float, optional): The threshold for the mask scores.
            inner_thr (float, optional): The threshold for the overlap rate.
            **kwargs: Additional keyword arguments.
        Returns:
            selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
        """

        scores, idx = scores.sort(0, descending=True)
        num_masks = idx.shape[0]
        
        masks_ord = masks[idx.view(-1), :]
        masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

        iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
        inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
        for i in range(num_masks):
            for j in range(i, num_masks):
                intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
                union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
                iou = intersection / union
                iou_matrix[i, j] = iou
                # select mask pairs that may have a severe internal relationship
                if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                    inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                    inner_iou_matrix[i, j] = inner_iou
                if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                    inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                    inner_iou_matrix[j, i] = inner_iou

        iou_matrix.triu_(diagonal=1)
        iou_max, _ = iou_matrix.max(dim=0)
        inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
        inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
        inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
        inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
        
        keep = iou_max <= iou_thr
        keep_conf = scores > score_thr
        keep_inner_u = inner_iou_max_u <= 1 - inner_thr
        keep_inner_l = inner_iou_max_l <= 1 - inner_thr
        
        # If there are no masks with scores above threshold, the top 3 masks are selected
        if keep_conf.sum() == 0:
            index = scores.topk(3).indices
            keep_conf[index, 0] = True
        if keep_inner_u.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_u[index, 0] = True
        if keep_inner_l.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_l[index, 0] = True
        keep *= keep_conf
        keep *= keep_inner_u
        keep *= keep_inner_l

        selected_idx = idx[keep]
        return selected_idx

    def get_seg_img(self, mask, image):
        image = image.copy()
        image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
        x,y,w,h = np.int32(mask['bbox'])
        seg_img = image[y:y+h, x:x+w, ...]
        return seg_img
    
    def get_seg_img_multi(self, mask, image):
        seg_img = []
        seg_img.append(self.get_seg_img(mask, image))
        for i in range(3):
            image_ = image.copy()
            x1, y1, x2, y2 = self.mask2box_multi_level(torch.from_numpy(mask['segmentation']), i)
            seg_img.append(image_[y1:y2, x1:x2])
        return seg_img

    def pad_img(self, img):
        h, w, _ = img.shape
        l = max(w,h)
        pad = np.zeros((l,l,3), dtype=np.uint8)
        if h > w:
            pad[:,(h-w)//2:(h-w)//2 + w, :] = img
        else:
            pad[(w-h)//2:(w-h)//2 + h, :, :] = img
        return pad

    def masks_update(self, *args, **kwargs):
        # remove redundant masks based on the scores and overlap rate between masks
        
        masks_new = ()
        for masks_lvl in (args):
            seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

            scores = stability * iou_pred
            keep_mask_nms = self.mask_nms(seg_pred, scores, **kwargs).tolist()
            
            new_masks_lvl = []
            for i, m in enumerate(masks_lvl):
                if i in keep_mask_nms:
                    new_masks_lvl.append(m)

            masks_new += (new_masks_lvl,)

        return masks_new

    def sam_encoder(self, image):
        # image = cv2.imread(image)
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        # pre-compute masks
        masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(image)
        # pre-compute postprocess
        masks_default, masks_s, masks_m, masks_l = \
            self.masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
        
        def mask2segmap(masks, image):
            seg_img_list = []
            seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        
            for i in range(len(masks)):
                mask = masks[i]
                seg_imgs = self.get_seg_img_multi(mask, image)
                pad_seg_img = [cv2.resize(self.pad_img(seg_img), (224,224)) for seg_img in seg_imgs]
                pad_seg_img = np.stack(pad_seg_img)
                seg_img_list.append(pad_seg_img)
            
                seg_map[masks[i]['segmentation']] = i

            seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3 
            seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,1,4,2,3) / 255.0).to('cuda')

            return seg_imgs, seg_map

        seg_images, seg_maps= {}, {}
        seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
        if len(masks_s) != 0:
            seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
        if len(masks_m) != 0:
            seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
        if len(masks_l) != 0:
            seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
        
        # 0:default 1:s 2:m 3:l
        return seg_images, seg_maps
    
    def register_img(self, img_path):
        self.image = Image.open(img_path).convert("RGB")

    def get_clip_feature(self, mask):
        image_features_final = torch.zeros(1, 768).to(self.device)
        for i in range(3):
            x1, y1, x2, y2 = self.mask2box_multi_level(mask, i)
            cropped_img = self.image.crop((x1, y1, x2, y2))
            image_input = torch.tensor(self.clip_preprocess(cropped_img)).unsqueeze(0)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input.to(self.device)).float()
                image_features /= image_features.norm(dim=-1, keepdim=True) #normalize
            
            image_features_final += image_features
            image_features_final /= 3
            image_features_final /= image_features_final.norm(dim=-1, keepdim=True)

        return image_features_final
        
    def mask2box(self, mask: torch.Tensor):
        row = torch.nonzero(mask.sum(axis=0))[:, 0]
        if len(row) == 0:
            return None
        x1 = row.min().item()
        x2 = row.max().item()
        col = np.nonzero(mask.sum(axis=1))[:, 0]
        y1 = col.min().item()
        y2 = col.max().item()

        return x1, y1, x2 + 1, y2 + 1

    def mask2box_multi_level(self, mask: torch.Tensor, level, expansion_ratio=0.1):
        x1, y1, x2 , y2 = self.mask2box(mask)
        if level == 0:
            return x1, y1, x2, y2
        shape = mask.shape
        x_exp = int(abs(x2- x1)*expansion_ratio) * level
        y_exp = int(abs(y2-y1)*expansion_ratio) * level
        
        return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)


if __name__ == "__main__":
    sam = Tools(
        load_sam="checkpoints/sam_vit_h_4b8939.pth",
        load_tracker="checkpoints/sam2.1_hiera_base_plus.pt"
    )
