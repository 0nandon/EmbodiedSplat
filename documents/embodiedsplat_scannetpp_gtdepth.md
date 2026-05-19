
### Reconstruct semantic Gaussians in ScanNet++

Run the below command with setting the GPU index:
```
bash scripts/scannetpp/eval_embodiedsplat_gtdepth.sh --gpu 0
```

Resulting semantic Gaussians may be stored in `outputs_semantic` folder.

### Compute metric on ScanNet++

Run the below command to compute the metric on point clouds:
```
python src/evaluation/evaluate_script_scannetpp.py \
    --pred-path outputs_semantic/scannetpp_gtdepth \
    --pred-mode ensemble \
    --text-mode base \
    --dataset scannetpp
```

You may want try the other options:
* `--pred-mode`: 
    * `2d`: Use costmaps derived from 2D VLMs only for the final metric computation.
    * `3d`: Use costmaps derived from 3D CLIP features only for the final metric computation.
    * `ensemble`: Use both of 2D and 3D costmaps for the better performance following Eq. 6 of the main paper.
