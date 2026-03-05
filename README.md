<div align="center">

# Latent Policy Steering

</div>

[![Static Badge](https://img.shields.io/badge/arXiv-2600.0000-red)]()
[![Static Badge](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/collections/jellyho/droid-dataset)
[![Static Badge](https://img.shields.io/badge/🌍-Project_Page-blue)](https://jellyho.github.io/LPS/)
![Static Badge](https://img.shields.io/badge/Python-3.10-green)



## 📰 News
- [2026/03/07] We release the code of LPS.
- [2026/03/07] [LPS]() is now on arXiv.

<hr />

<p align="center">
  <a href="https://jellyho.github.io/LPS/">
    <img alt="teaser figure" src="./assets/LPS_DARK.svg">
  </a>
</p>


## Overview
**Latent Policy Steering (LPS)** is a robust offline reinforcement learning framework for robotics that resolves the brittle trade-off between return maximization and behavioral constraints. Instead of relying on lossy proxy latent critics, LPS directly optimizes a latent-action-space actor by backpropagating original-action-space Q-gradients through a differentiable one-step MeanFlow policy. This architecture allows the original critic to guide end-to-end optimization without proxy networks, while the MeanFlow policy serves as a strong generative prior. As a result, LPS works out-of-the-box with minimal tuning, achieving state-of-the-art performance across OGBench and real-world robotic tasks.

## Installation
```bash
conda create -n lps python=3.10
pip install -r requirements.txt
```


## OGBench Simulation

```bash
sh lps_ogbench.sh task_name task_num seed

# Examples
sh lps_ogbench.sh cube-single-play 1 100
sh lps_ogbench.sh cube-double-play 1 100
sh lps_ogbench.sh scene-play 1 100
sh lps_ogbench.sh puzzle-3x3-play 1 100
sh lps_ogbench.sh puzzle-4x4-play 1 100
```

## DROID

### Collect the data (hdf5 format)

Follow the [instruction](https://droid-dataset.github.io/droid/example-workflows/teleoperation.html) of DROID to collect demonstration as hdf5 format.

We assume that the dataset saved as

```text
droid_dataset_dir/
├── task_1/
│   ├── success/
│   │   ├── trajectory_0.hdf5
│   │   ├── trajectory_1.hdf5
|   |   └── ...
│   ├── failure/
│   │   ├── trajectory_7.hdf5
│   │   ├── trajectory_9.hdf5
|   |   └── ...
├── task_2/
│   ├── success/
|   |   └── ...
│   ├── failure/
|   |   └── ...
└── ...
```

### Train LPS using DROID dataset

```bash
sh lps_droid.sh task_name droid_dataset_dir seed

# Example
sh lps_droid.sh task_2 ~/droid_dataset_dir 100
```

### Evaluate LPS on DROID

We again assuming that you installed DROID on the same python environment you installed LPS.

```bash
sh lps_droid_eval.sh checkpoint_dir checkpoint_step

# Example
sh lps_droid_eval.sh /home/rllab2/jellyho/droid_ckpts/LPS/LPS_TEST_1 10000
```


## Acknowledgments
This codebase is built on top of [Reinforcement Learninig with Aciton Chunking](https://github.com/ColinQiyangLi/qc).

## Citation

If you find this work useful, please consider citing:
```bibtex
@misc{

}
```
