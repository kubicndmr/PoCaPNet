# PoCaPNet: A Novel Approach for Surgical Phase Recognition Using Speech and X-Ray Images

PoCaPNet is a multimodal network based on two-stage TCN architecture and designed for surgical phase recognition using speech and X-Ray data collected during port-catheter placement surgeries. 

## Getting Started

To install dependencies, we recommend creating a virtual environment as following:
```
    - python3 -m venv pocapenv
    - source pocapenv/bin/activate
    - pip install -r requirements.txt
```
Later, you can run following scripts to prepare your data for the training:

    - python preprocessing.py
    - python extract_features.py

This will create ```data``` and ```features``` folders containing audio and image data and features respectively. We recommend choosing a large storage via ```utils/paths.py``` for these folders as they can grow very large very quickly. If a large storage is not available, you can manipulate ```data.py``` script, which is responsible for reading data for training epochs, to generate features on the fly instead of reading from the disk.  

```utils/hparams.py``` holds all settings for the training as well as the preprocessing and feature extraction. You can adjust preferred values in this script for these steps.  

## Prerequisites

The main medical dataset, PoCap Corpus [1], is unfortunately is not publicly available in order to ensure data security of patients and medical personal.

## Training

```main.py``` file encapsulates training and testing modules. You can simply run this script and start the experiment:

    - python main.py

## Contributing

Instructions for how others can contribute to the project. This might include guidelines for submitting pull requests, reporting issues, and coding standards.

## License

Information on the license under which the project is released. This might include a link to the full license text.

## References
```
[1] Demir, Kubilay Can, et al. "PoCaP Corpus: A Multimodal Dataset for Smart Operating Room Speech Assistant Using Interventional Radiology Workflow Analysis." Text, Speech, and Dialogue: 25th International Conference, TSD 2022, Brno, Czech Republic, September 6–9, 2022, Proceedings. Cham: Springer International Publishing, 2022.
```
