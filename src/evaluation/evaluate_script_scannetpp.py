
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

from database.class_labels import SCANNETPP_CLASS_LABELS, SCANNETPP_CLASS_ID, SCANNETPP_IGNORE
from semseg_metric import evaluate

import sys
sys.path.append("src/third_party")

import torch.nn.functional as F
from maskadapter.clip import CLIP

LABELS_FORMAT = [
    "a {} in a photo.".format(cls) for cls in SCANNETPP_CLASS_LABELS if cls not in SCANNETPP_IGNORE
]

def process_gt(raw_gt):
    gt = np.ones_like(raw_gt) * -1
    cnt = 0
    for label_id, idx in SCANNETPP_CLASS_ID.items():
        if SCANNETPP_CLASS_LABELS[idx] in SCANNETPP_IGNORE:
            continue
        gt[raw_gt == label_id] = cnt
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

    agg = LocalAggregator(3, 250, 250, 250, [-50.0, -50.0, -50.0], 0.4).cuda()
    
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
    parser.add_argument("--pred-path", type=str, default="outputs_semantic/scannetpp")
    parser.add_argument("--pred-mode", type=str, default="ensemble")
    parser.add_argument("--dataset", type=str, default="scannetpp")
    args = parser.parse_args()

    R_z_90 = np.array([[0, -1, 0], [1,  0, 0], [0,  0, 1]]).astype(np.float32)

    text_feat = build_text_embedding(LABELS_FORMAT, mode=args.text_mode)
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
            costmap = ensemble(costmap_2d, costmap_3d, 1.)
        elif PRED_MODE == "2d":
            costmap = costmap_2d
        elif PRED_MODE == "3d":
            costmap = costmap_3d
    
        pcd = torch.load("dataset/scannetpp/data/" + scene + "/" + scene + ".pth", weights_only=False)
        coords = pcd['sampled_coords'].astype(np.float32) @ R_z_90

        preds = compute_semseg(gaussians, coords, costmap)
        gts, _ = process_gt(pcd['sampled_labels'])

        gts_collate.append(gts)
        preds_collate.append(preds)
    
    gts_collate = torch.cat(gts_collate)
    preds_collate = torch.cat(preds_collate)
    evaluate(preds_collate.cpu(), gts_collate.cpu(), stdout=True, dataset=DATASET)
