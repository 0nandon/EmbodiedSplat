
### Reconstruct semantic Gaussians in ScanNet

Run the below command with setting the GPU index:
```
bash scripts/scannet/eval_embodiedsplat.sh --gpu 0
```

Resulting semantic Gaussians may be stored in `outputs_semantic` folder.

### Compute metric on ScanNet under 20, 15, 10 classes setting

Run the below command to compute the metric on point clouds:
```
python src/evaluation/evaluate_script_scannet.py \
    --pred-path outputs_semantic/scannet \
    --pred-mode ensemble \
    --text-mode base \
    --dataset scannet20
```

You may want try the other options:
* `--dataset`: Set this argument to `scannet15` or `scannet10` to compute metrics under the 15-class or 10-class setting, respectively (*cf* Tab. 1 of the main paper).
* `--pred-mode`: 
    * `2d`: Use costmaps derived from 2D VLMs only for the final metric computation.
    * `3d`: Use costmaps derived from 3D CLIP features only for the final metric computation.
    * `ensemble`: Use both of 2D and 3D costmaps for the better performance following Eq. 6 of the main paper.

### Compute metric on ScanNet200

Run the below command to compute the metric on point clouds:
```
python src/evaluation/evaluate_script_scannet200.py \
    --pred-path outputs_semantic/scannet \
    --pred-mode ensemble \
    --text-mode base \
    --dataset scannet200
```
