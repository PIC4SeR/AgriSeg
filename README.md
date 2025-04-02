[![CVPR](https://img.shields.io/badge/CVPR-paper-blue)](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Angarano_Domain_Generalization_for_Crop_Segmentation_with_Standardized_Ensemble_Knowledge_Distillation_CVPRW_2024_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-2304.01029-b31b1b.svg)](https://arxiv.org/abs/2304.01029)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

<h1 align="center">  Domain Generalization for Crop Segmentation with Knowledge Distillation
</h1>

<p align="center">
  <img src="https://github.com/PIC4SeR/AgriSeg/blob/46462c38e3e2da198e45696ff02c2c67f5fe5675/assets/scheme.svg" alt="Graphical abstract" width="450"/>
</p>

In recent years, precision agriculture has gradually oriented farming closer to automation processes to support all the activities related to field management. Service robotics plays a predominant role in this evolution by deploying autonomous agents that can navigate fields while performing tasks without human intervention, such as monitoring, spraying, and harvesting. To execute these precise actions, mobile robots need a real-time perception system that understands their surroundings and identifies their targets in the wild. Generalizing to new crops and environmental conditions is critical for practical applications, as labeled samples are rarely available. 
In this paper, we investigate the problem of crop segmentation and propose a novel approach to enhance domain generalization using knowledge distillation. In the proposed framework, we transfer knowledge from an ensemble of models individually trained on source domains to a student model that can adapt to unseen target domains. 
To evaluate the proposed method, we present a synthetic multi-domain dataset for crop segmentation containing plants of variegate shapes and covering different terrain styles, weather conditions, and light scenarios for more than 50,000 samples. We demonstrate significant improvements in performance over state-of-the-art methods. Our approach provides a promising solution for domain generalization in crop segmentation and has the potential to enhance precision agriculture applications.

## Installation
[Explanation on how to install the project goes here]

```
pip install -r requirements.txt
```

The full dataset is available at [this link](https://sites.google.com/view/datasetpic4ser/home-page).

## Usage
The configuration for the experiments can be set via the YAML files in ```cfg/```. Experiments can be run through the ```main.py``` file:
```
python main.py --cfg cfg/cfg_1.yaml --gpu 0
```

# Citations

This repository is intended for scientific research purposes.
If you want to use this code for your research, please cite our work ([Domain Generalization for Crop Segmentation with Knowledge Distillation](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/html/Angarano_Domain_Generalization_for_Crop_Segmentation_with_Standardized_Ensemble_Knowledge_Distillation_CVPRW_2024_paper.html)).

```
@inproceedings{angarano2024domain,
  title={Domain generalization for crop segmentation with standardized ensemble knowledge distillation},
  author={Angarano, Simone and Martini, Mauro and Navone, Alessandro and Chiaberge, Marcello},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5450--5459},
  year={2024}
}
```
