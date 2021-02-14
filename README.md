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

[TODO] 

How to get char masks:
```
sh runners/train_bentham_blots.sh
sh runners/prepare_bentham_char_masks.sh 
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
sh runners/train_bentham.sh
```

2. Train "base + blots" experiment:
```
sh runners/train_bentham_blots.sh
```

3. Train "base + stackmix" experiment:
```
sh runners/prepare_bentham_stackmix.sh
sh runners/train_bentham_stackmix.sh
```

4. Train "base + blots + stackmix" experiment:
```
sh runners/prepare_bentham_stackmix.sh
sh runners/train_bentham_blots_stackmix.sh
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
- [M. Novopoltsev](https://github.com/maximazzik))
- [D. Dimitrov]
- [M. Potanin]
