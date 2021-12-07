# StackMix and Blot Augmentations for Handwritten Recognition using CTCLoss

[This paper](https://arxiv.org/abs/2108.11667) presents a new text generation method StackMix. StackMix can be
applied to the standalone task of generating handwritten text based on printed text.

## Config file
Create a new config file need  in ```configs/__init__.py```.
An individual config file is required for each dataset.

## Datasets

### There are two ways to get a dataset:
#### The first way:
Download a dataset and annotations (for example Bentham: http://www.transcriptorium.eu/~tsdata/BenthamR0/). Prepare dataset and create marking.csv file

####  The second way:
Downlad prepared dataset using the script ```download_dataset.py``` (for example Bentham: ```python scripts/download_dataset.py --dataset_name=bentham```)
And now you can use train script.

You can change out folder by key --data_dir='your path', by default --data_dir=../StackMix-OCR-DATA.
All dataset names: bentham, peter, hkr, iam.

### Dataset format

The dataset should contain a directory with images and a csv file `marking.csv` with annotations. The csv file must contain the "stage" field with information about which sample the image belongs to `(train / valid / test)`. An example of the structure and content of a csv file is given below
```
sample_id,path,stage,text
270-01,washington/images/270-01.png,train,"270. Letters, Orders and Instructions. October 1755."
270-03,washington/images/270-03.png,train,"only for the publick use, unless by particu-"
270-04,washington/images/270-04.png,train,lar Orders from me. You are to send
270-05,washington/images/270-05.png,train,"down a Barrel of Flints with the Arms, to"
```

## Setup

install requirements:
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Example run train
```
python scripts/run_train.py \
    --checkpoint_path "" \
    --experiment_name "HKR_exp" \
    --dataset_name "hkr" \
    --data_dir "data/HKR/" \
    --output_dir "exp/" \
    --experiment_description \
    "[Base] Training Base OCR on HKR dataset" \
    --image_w 512  \
    --image_h 64 \
    --num_epochs 200 \
    --bs 256 \
    --num_workers 8 \
    --use_blot 0 \
    --use_augs 1 \
    --use_progress_bar 0 \
    --seed 6955
```

## Example run evaluation
```
python scripts/run_evaluation.py \
    --experiment_folder "exp/" \
    --dataset_name "hkr" \
    --data_dir "data/HKR/" \
    --image_w 512  \
    --image_h 64 \
    --bs 128
```

## Generating all_char_masks.json for stackmix
```
python scripts/prepare_char_masks.py \
    --checkpoint_path "exp/HKR_exp/best_cer.pt" \
    --data_dir "data/HKR/" \
    --dataset_name "hkr" \
    --image_w 512  \
    --image_h 64 \
    --bs 128 \
    --num_workers 8
```

`dataset_name` must be taken from the config

## Image generation

The code for generation can be found [here](/jupyters/usage_stackmix.ipynb)

Example of generating images with stackmix

![Example of generating images](https://sun9-64.userapi.com/impg/xAFmDnVuuTmc4FM_FKhLPnq-KvrppD4x-DvUKg/hy1qKbRbS58.jpg?size=402x305&quality=96&sign=5bdfa7702f2e655cc991e274d4bb7b3f&type=album)


## Supported by:

- Sber
- OCRV
- Sirius University
- RZHD


## Citation

Please cite the related works in your publications if it helps your research:
```
@inproceedings{10.1145/3476887.3476892,
author = {Mark, Potanin and Denis, Dimitrov and Alex, Shonenkov and Vladimir, Bataev and Denis, Karachev and Maxim, Novopoltsev and Andrey, Chertok},
title = {Digital Peter: New Dataset, Competition and Handwriting Recognition Methods},
year = {2021},
isbn = {9781450386906},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3476887.3476892},
doi = {10.1145/3476887.3476892},
abstract = { This paper presents a new dataset of Peter the Great’s manuscripts and describes
a segmentation procedure that converts initial images of documents into lines. This
new dataset may be useful for researchers to train handwriting text recognition models
as a benchmark when comparing different models. It consists of 9694 images and text
files corresponding to different lines in historical documents. The open machine learning
competition ”Digital Peter” was held based on the considered dataset. The baseline
solution for this competition and advanced methods on handwritten text recognition
are described in the article. The full dataset and all codes are publicly available.},
booktitle = {The 6th International Workshop on Historical Document Imaging and Processing},
pages = {43–48},
numpages = {6},
keywords = {handwritten text recognition, Digital Peter, historical dataset, Russian},
location = {Lausanne, Switzerland},
series = {HIP '21}
}
```


## Contacts

- [A. Shonenkov](https://www.kaggle.com/shonenkov) shonenkov@phystech.edu
- [D. Karachev](https://github.com/thedenk/)
- [M. Novopoltsev](https://github.com/maximazzik)
- [D. Dimitrov](https://github.com/denndimitrov)
- [M. Potanin](https://github.com/MarkPotanin)
