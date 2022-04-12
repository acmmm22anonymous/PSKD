# Prototype-based Selective knowledge Distillation for Zero-Shot Sketch Based Image Retrieval
This repository is the anonymous Pytorch implementation of the PSKD method.
![Alternative text](./image/overview.png)
## Main Idea
we propose a Prototype-based Selective Knowledge Distillation (PSKD) method for ZS-SBIR. The model would first learn a set of prototypes to represent categories and then utilize an instance-level adaptive learning strategy to strengthen semantic relations between categories. Afterward, a correlation matrix targeted for the downstream task would be established through the prototypes. With the correlation matrix, the teacher signal given by transformers pre-trained on ImageNet and fine-tuned on the downstream dataset, can be reconstructed to weaken the impact of mispredictions and selectively distill knowledge on the student network. We perform the experiment on three widely-used datasets on ZS-SBIR, and the results exhibit that PSKD establishes the new state-of-the-art performance on all datasets.
## The State-of-the-art Performance
![Alternative text](./image/results.png)
![Alternative text](./image/results2.png)
## Visulization
![Alternative text](./image/retrieval.png)

## Installation and Requirements

### Installation

- Python 3.7
- PyTorch 1.8.1
- Numpy 1.22.0

### Prepare datasets and pre-trained models
Download **Sketchy Extended** and **TU-Berlin** dataset by following [SEM-PCYC](https://github.com/AnjanDutta/sem-pcyc).

### Training & Testing
 - >Training 
   - Sketchy 

      `python main_sketch.py `

    - TU-Berlin

      `python main_tuberlin.py `
 - >Testing
   - Sketchy 

      `python main_sketch.py `

    - TU-Berlin

      `python main_tuberlin.py `


