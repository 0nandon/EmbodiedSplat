
### Reconstruct semantic Gaussians in ScanNet++

Run the below command with setting the GPU index:
```
bash scripts/scannetpp/eval_embodiedsplat_fast.sh --gpu 0
```

Resulting semantic Gaussians may be stored in `outputs_semantic` folder.

### Compute metric on ScanNet++

Run the below command to compute the metric on point clouds:
```
python src/evaluation/evaluate_script_scannetpp.py \
    --pred-path outputs_semantic/scannetpp_fast \
    --pred-mode 2d \
    --text-mode fast \
    --dataset scannetpp
```

You may want try the other options:
* `--pred-mode`: 
    * `2d`: *fast* veresion only uses 2D VLMs for the costmap.