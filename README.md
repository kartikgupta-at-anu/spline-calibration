# Calibration of Neural Networks using Splines

This repository is the official implementation of ICLR 2021 paper: [Calibration of Neural Networks using Splines](https://openreview.net/forum?id=eQe8DEWNN2W).

This code is for research purposes only.

Any questions or discussions are welcomed!


## Installation

Setup python virtual environment.

```
virtualenv -p python3 venv
source venv/bin/activate                                 
pip3 install -r requirements.txt
mkdir saved_logits
```


## Setup

Download the logits for different data and network combinations from [here](https://drive.google.com/drive/folders/1e1ai-bKb7LukKShqpn3S_gYXJGzhUYgm) and put them under `saved_logits` folder. 


## Recalibration

To find a recalibration function and evaluate the calibration:

```bash
python recalibrate.py
```

The results for pre-calibration and post-calibration with various metrics will be saved in csv format under `out/{dataset}/{network}/beforeCALIB_results.csv` and `out/{dataset}/{network}/afterCALIBsplinenatual6_results.csv`. Calibration graphs such as Figure 1 in the main paper will be generated under `out/{dataset}/{network}` folder. 

## Cite

If you make use of this code in your own work, please cite our paper:

```
@inproceedings{
gupta2021calibration,
title={Calibration of Neural Networks using Splines},
author={Kartik Gupta and Amir Rahimi and Thalaiyasingam Ajanthan and Thomas Mensink and Cristian Sminchisescu and Richard Hartley},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=eQe8DEWNN2W}
}
```

#### Contact
Kartik Gupta (kartik.gupta@anu.edu.au).
