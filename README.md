# StackMix and Blot Augmentations for Handwritten Recognition using CTCLoss

[TODO] arxiv 

## Table with results

[TODO] 

## Demo Neptune with all experiments

[TODO]

## Pretrained Models

[TODO] 

## Recommended structure of experiments:

```
--- StackMix-OCR/
--- StackMix-OCR-DATA/
--- StackMix-OCR-SAVED_MODELS/
--- StackMix-OCR-MWE_TOKENS/
```


## Datasets

### There are two ways to get a dataset:
#### The first way:
1. Download selected dataset and annotations from original site (for example Bentham: http://www.transcriptorium.eu/~tsdata/BenthamR0/)  
2. Prepare dataset using jupyter notebook in jupyters folder  
3. Put dataset in StackMix-OCR-DATA folder  

####  The second way:
Downlad prepared dataset using our script ```download_dataset.py``` (for example Bentham: ```python scripts/download_dataset.py --dataset_name=bentham```) 
And now you can use train script.  

You can change out folder by key --data_dir='your path', by default --data_dir=../StackMix-OCR-DATA.
All dataset names: bentham, peter, hkr, iam.


How to get char masks:
```
sh runners/bentham/train_blots.sh
sh runners/bentham/prepare_char_masks.sh 
```

## Run in docker:

[TODO] 

## Run locally:

install requirements:
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Run experiments from root project (StackMix-OCR):

1. Train "base" experiment:
```
sh runners/bentham/train.sh
```

2. Train "base + blots" experiment:
```
sh runners/bentham/train_blots.sh
```

3. Train "base + stackmix" experiment:
```
sh runners/bentham/prepare_stackmix.sh
sh runners/bentham/train_stackmix.sh
```

4. Train "base + blots + stackmix" experiment:
```
sh runners/bentham/prepare_stackmix.sh
sh runners/bentham/train_blots_stackmix.sh
```


## Supported by:

- Sber
- OCRV
- Sirius University
- RZHD


## Citation

[TODO]


## Contacts

- [A. Shonenkov](https://www.kaggle.com/shonenkov) shonenkov@phystech.edu
- [D. Karachev](https://github.com/thedenk/)
- [M. Novopoltsev](https://github.com/maximazzik)
- [D. Dimitrov]
- [M. Potanin]
