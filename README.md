# MetaVQA Training Repository

This repository contains the training and testing scripts for **MetaVQA: A Benchmark for Embodied Scene Understanding of Vision-Language Models**. Please refer to [MetaVQA Main repository](https://github.com/WeizhenWang-1210/MetaVQA) for the dataset, benchmark models, and additional resources,

Our training script is adapted from [Salesforce LAVIS](https://github.com/salesforce/LAVIS) and [ELM](https://github.com/OpenDriveLab/ELM).

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Dadaism6/MetaVQA-Training.git
   cd metavqa-training
   ```
2. **Dependencies**:
    ```bash
    conda create -n metavqa-training python==3.8
    conda activate metavqa-training
    pip install -e .
    ```
3. **Data Preparation**:
    Please refer to the [MetaVQA website](https://metadriverse.github.io/metaVQA/) for data preparation. We will release the data shortly!

## Training
To train a model using the MetaVQA dataset, run the following command:
```bash
python scrips/train.py --cfg-path /path/to/config.yaml
```
You can refer to ```lavis/projects/blip2/train``` for existing training configs.
We have prepared shell scripts for distributed training. Please refer to ```scripts``` folder.

## Evaluation
To evaluate, run the following command:
```bash
python scrips/train.py --cfg-path /path/to/eval/config.yaml
```
You can refer to ```lavis/projects/blip2/eval``` for existing evaluating configs.
We have prepared shell scripts for distributed training. Please refer to ```scripts``` folder.

## Acknowledgements

We acknowledge all the open-source projects that make this work possible:
- [Lavis](https://github.com/salesforce/LAVIS) | [ELM](https://github.com/OpenDriveLab/ELM)
- [MetaDrive](https://github.com/metadriverse/metadrive) | [CAT](https://github.com/metadriverse/cat)
