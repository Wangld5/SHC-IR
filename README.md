# SHC-IR
The code of paper "Image Retrieval with Well-Separated Semantic Hash Centers" accepted by ACCV 2022
## Environment

- PyTorch 1.7.1 + cuda 11.2

## Get Started
1. three dataset ImageNet, Stanford Cars, NABirds should be download on their official website
2. run `main_cls.py` to generate and save semantic category for three datasets.  
3. run `optimizeAccel.py` to generate and save hash centers for each datasets.  
4. run `main.py` to get mAP results for three datasets.

## Note
the parameter setting can be modified in `scripts\utils.py`   
some setting should be modified in `scripts\head.py` to fit for generate semantic category or retrieval  



