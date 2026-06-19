<p align="center">
  <h1 align="center">EmbodiedSplat 🛋️<br>
Online Feed-Forward Semantic 3DGS <br>
for Open-Vocabulary 3D Scene Understanding</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/seungjun-lee-43101a261/">Seungjun Lee</a></span> ·  
    <a href="https://scholar.google.com/citations?user=7rf6Bw4AAAAJ&hl=en">Zihan Wang</a></span> ·
    <a href="https://wangys16.github.io">Yunsong Wang</a></span> ·
    <a href="https://www.comp.nus.edu.sg/~leegh/">Gim Hee Lee</a><sup></sup> <br>
    National University of Singapore<br>
  </p>
  <h2 align="center">CVPR 2026</h2>
  <h3 align="center"><a href="https://github.com/0nandon/EmbodiedSplat">Code</a> | <a href="https://arxiv.org/pdf/2603.04254">Paper</a> | <a href="https://0nandon.github.io/EmbodiedSplat/">Project Page</a> </h3>
  <div align="center">
  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
  </div>
</p>

<p align="center">
  <a href="">
    <img src="https://github.com/0nandon/EmbodiedSplat/blob/main/static/teaser.png" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
<strong>Build and understand at Once!</strong> By taking over 300 streaming images, our <strong>EmbodiedSplat</strong> reconstructs whole-scene open-vocabulary 3DGS in online manner at up to 5-6 FPS per-frame processing time. Reconstructed scene supports diverse perception tasks such as open-vocabulary 3D semantic segmentation, 2D-rendered semantic segmentation and novel-view color synthesis with depth rendering. 
</p>

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#todo">TODO</a>
    </li>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
     <li>
      <a href="#evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## News:

- [2026/02/21] EmbodiedSplat is accepted to CVPR 2026 🔥. The code will be released before June.
- [2026/05/19] The code and pretrained weights are released! 👊🏻
- [2026/06/14] Example training pipeline of EmbodiedSplat is released! 

## TODO
- [x] Release the code of EmbodiedSplat and pretrained weights
- [x] Release the training code of EmbodiedSplat
- [ ] If time permits, we are planning to give some updates (Not for publishing another paper, but just for fun ☺️): 
    * Replacing the reconstruction backbone from FreeSplat++ to the most recent pose-free online 3DGS feed-forward model.
    * Replacing the CLIP(OpenSeg, MaskAdapter) + SAM pipeline into the more stronger 2D VLMs such as SAM3.
    * Attaching LLM to EmbodiedSplat by following the spirit of SplatTalk.
    * Adopting EmbodiedSplat to real robot and release the code.

## Installation
### Dependencies :memo:
The main dependencies of the project are the following:
```yaml
python: 3.10
cuda: 11.8
```
You can set up a conda environment as follows:
```
conda create -n embodiedsplat python=3.10
conda activate embodiedsplat
conda install -c conda-forge libopenblas=0.3.31 openblas-devel=0.3.31

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install "setuptools<81"
pip install -r requirements.txt --no-build-isolation

cd src/third_party/MinkowskiEngine
git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas

pip install --no-build-isolation src/model/encoder/submodules/simple-knn
pip install --no-build-isolation src/ops
pip install --no-build-isolation src/third_party/localagg
pip install --no-build-isolation git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth
pip install --no-build-isolation src/third_party/langsplat-rasterization
pip install git+https://github.com/openai/CLIP.git

# if you face error when you run the evaluation code due to MinkowskiEngine, do:
cd $CONDA_PREFIX/lib
ln -sf libopenblasp-r0.3.31.so libopenblas.so.0
ln -sf libopenblasp-r0.3.31.so libopenblas.so
cd {YOUR_PATH}
```

## Data Preparation

The testing scenes in ScanNet and ScanNet++, and pretrained weights are available <a href="https://huggingface.co/datasets/onandon/EmbodiedSplat">here</a>. You can easily download all the preprocessed data by running:
```
python download_data.py
```

Once you run the above command, two folders must be produced:
* `pretrained`: Including all the pretrained weights of the EmbodiedSplat and auxiliary 2D models.
* `dataset`: Including all the testing scenes and ground-truth annotations from ScanNet and ScanNet++.

## Evaluation

> [!IMPORTANT]
> **NOTE 📌** : We make a minor update to the inference strategy. As mentioned at the end of Sec. 7.2, we apply *floater removal* as a post-refinement step following FreeSplat++. In our original paper, Gaussians identified as floaters are also excluded from semantic prediction on point clouds in Eq. 11. However, we empirically find that this exclusion degrades semantic performance, even though floater removal clearly improves rendered RGB quality. Hence, in the released code, floater Gaussians are excluded only during RGB rendering, while they are still used for semantic prediction. As a result, the evaluation results may be higher than the numbers reported in the paper.
>
> **NOTE 📌** : We support two types of inference strategy:
>* `incremantal`: Among all past frames, we select the N=30 images with the smallest pose differences from the current frame and use them as reference frames.
>* `online`: Simply select the past *N*=30 frames, *i.e.*, [*t*−30,*t*−1], and use them as reference frames for timestep *t*.
> 
>The dafault setting is `incremental`, but it can be changed to `online` by setting `model.encoder.recon_mode=online` in the config files under `config/experiment`. Both settings yield similar performance.

We provide evaluation scripts for diverse settings across ScanNet and ScanNet++, with options to enable or disable GT depth. All the experiments are conducted in single NVIDIA RTX 6000 Ada GPU (48GB).

| Model | EmbodiedSplat | EmbodiedSplat-*fast* |  
| -------- | -------- | -------- |
| ScanNet    |    [Here](documents/embodiedsplat_scannet.md)      |     [Here](documents/embodiedsplat_fast_scannet.md)      |   
| ScanNet, GT Depth    |     [Here](documents/embodiedsplat_scannet_gtdepth.md)     |    [Here](documents/embodiedsplat_fast_scannet_gtdepth.md)      |   
| ScanNet++   |    [Here](documents/embodiedsplat_scannetpp.md)      |     [Here](documents/embodiedsplat_fast_scannetpp.md)     |   
| ScanNet++, GT Depth    |     [Here](documents/embodiedsplat_scannetpp_gtdepth.md)     |    [Here](documents/embodiedsplat_fast_scannetpp_gtdepth.md)      |   

Generated semantic Gaussians are stored in `outputs_semantic` folder and subsequently used for evaluation in point clouds.

## Training

> [!IMPORTANT]
> **NOTE 📌** : You may read the text below to better understand our model.
> 
> * Our CLIP global codebook and Online Sparse Coefficient Field is a **training-free approcah** which stores 2D CLIP features in a memory-efficient manner. Theoretically, they can be adopted by any kinds of feed-forward 3DGS model where we use FreeSplat++ in the main paper.
> 
> * Since EmbodiedSplat-*fast* only uses 2D CLIP features, it doesn't require additional training. 
> 
> * Only 3D CLIP features in EmbodiedSplat require additional training where 2D CLIP is distilled to 3D U-Net with memory adapter. Combining 3D CLIP features show additional performance improvement which is also shown by OpenScene.

If you want to train EmbodiedSplat for 3D CLIP features, please follow the [train.md](documents/train.md).

## Acknowledgement

Our work is inspired a lot from the following works. We sincerely appreciate to their great contributions!

* <a href="https://github.com/wangys16/FreeSplat">FreeSplat</a>
* <a href="https://arxiv.org/abs/2503.22986">FreeSplat++</a>
* <a href="https://largespatialmodel.github.io">LSM</a>
* <a href="https://saimouli.github.io/onlineLang/">Online-LangSplat</a>
* <a href="https://github.com/lhj-git/InstanceGasuusian_code">InstanceGaussian</a>
* <a href="https://drsplat.github.io">Dr.Splat</a>
* <a href="https://arxiv.org/abs/2406.02058">OpenGaussian</a>
* <a href="https://github.com/minghanqin/LangSplat">LangSplat</a>

## Citation
If you find our code or paper useful, please cite
```bibtex
@inproceedings{lee2026embodiedsplat,
          title={EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding},
          author={Lee, Seungjun and Wang, Zihan and Wang, Yunsong and Lee, Gim Hee},
          booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
          pages={23774--23784},
          year={2026}
        }
```
