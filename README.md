# PoCaPNet: A Novel Approach for Surgical Phase Recognition Using Speech and X-Ray Images

:fire: Accepted to Interspeech 2023

PoCaPNet is a multimodal network based on two-stage TCN architecture and designed for surgical phase recognition using speech and X-Ray data collected during port-catheter placement surgeries.
If you find the project intersting or have an idea for collobration, plese send me an e-mail.

<p align="center">
<img width="960" height="350" src="https://github.com/kubicndmr/PoCaPNet/files/11861012/model.pdf"> 
</p>

## Getting Started

To install dependencies, we recommend creating a virtual environment as following:
```
    - python3 -m venv pocapenv
    - source pocapenv/bin/activate
    - pip install -r requirements.txt
```
Later, you can run following scripts to prepare your data for the training:

    - python preprocessing.py

This will create ```data``` and ```features``` folders containing audio and image data and features respectively. We recommend choosing a large storage via ```utils/paths.py``` for these folders as they can grow very large very quickly. If a large storage is not available, you can manipulate ```data.py``` script, which is responsible for reading data for training epochs, to generate features on the fly instead of reading from the disk.  

```utils/hparams.py``` holds all settings for the training as well as the preprocessing and feature extraction. You can adjust preferred values in this script for these steps.  

## Dataset

The medical dataset PoCap Corpus is unfortunately is not publicly available due to the ethics protocol signed by patients and medical personnel during data collection procedure. However, details of the dataset is explained in the script.

## Training

```main.py``` file encapsulates training and testing modules. You can simply run this script and start the experiment:

    - python main.py

## Notes
If this study is useful for you, please cite as:
```
@inproceedings{demir23_interspeech,
  author={Kubilay Can Demir and Tobias Weise and Matthias May and Axel Schmid and Andreas Maier and Seung Hee Yang},
  title={{PoCaPNet: A Novel Approach for Surgical Phase Recognition Using Speech and X-Ray Images}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2348--2352},
  doi={10.21437/Interspeech.2023-753}
}
```
