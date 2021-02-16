# if you need resume experiment, should set checkpoint_path
python scripts/prepare_stackmix.py \
  --dataset_name "saintgall" \
  --data_dir "../StackMix-OCR-DATA" \
  --mwe_tokens_dir "../StackMix-OCR-MWE_TOKENS" \
  --image_w 2048 \
  --image_h 128
