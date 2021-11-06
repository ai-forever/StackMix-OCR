# StackMix and Blot Augmentations for Handwritten Recognition using CTCLoss

[This paper](https://arxiv.org/abs/2108.11667) presents a new text generation method StackMix. StackMix can be
applied to the standalone task of generating handwritten text based on printed text.

## Config file
Create a new config file need  in ```configs/__init__.py```.
An individual config file is required for each dataset

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

### Dataset format

The dataset should contain a directory with images, a csv file `marking.csv` and a json file with markup information. An example of the file location can be seen below

![Dataset format](https://sun9-7.userapi.com/impg/GpIzvjYF9AbpGOQbamvCcgwRA9fVfHo2SaPOcg/Ox847-h0m8o.jpg?size=174x106&quality=96&sign=fb2ce9af30b54f09cfc8542ee8f84fad&type=album)

The csv file must contain the "stage" field with information about which sample the image belongs to `(training / valid / test)`.
An example of the structure and content of a csv file is given below
```
sample_id,path,stage,text
270-01,washington/images/270-01.png,train,"270. Letters, Orders and Instructions. October 1755."
270-03,washington/images/270-03.png,train,"only for the publick use, unless by particu-"
270-04,washington/images/270-04.png,train,lar Orders from me. You are to send
270-05,washington/images/270-05.png,train,"down a Barrel of Flints with the Arms, to"
```

## Run:

install requirements:
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Example run train:
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

## Example run evaluation:
```
python scripts/run_evaluation.py \
    --experiment_folder "exp/" \
    --dataset_name "hkr" \
    --data_dir "data/HKR/" \
    --image_w 512  \
    --image_h 64 \
    --bs 128
```

## Generating char_mask for stackmix:
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

### Note

 - `dataset_name` must be taken from the config
 - all file paths must be relative

Example of generating images with stackmix

![Example of generating images](https://sun9-64.userapi.com/impg/xAFmDnVuuTmc4FM_FKhLPnq-KvrppD4x-DvUKg/hy1qKbRbS58.jpg?size=402x305&quality=96&sign=5bdfa7702f2e655cc991e274d4bb7b3f&type=album)

The code for generation can be found [here](/jupyters/usage_stackmix.ipynb)

## Supported by:

- Sber
- OCRV
- Sirius University
- RZHD


## Citation

Please cite the related works in your publications if it helps your research:

[TODO]

## Contacts

- [A. Shonenkov](https://www.kaggle.com/shonenkov) shonenkov@phystech.edu
- [D. Karachev](https://github.com/thedenk/)
- [M. Novopoltsev](https://github.com/maximazzik)
- [D. Dimitrov]
- [M. Potanin](https://github.com/MarkPotanin)
