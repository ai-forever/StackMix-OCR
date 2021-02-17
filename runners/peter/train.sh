# if you need resume experiment, should set checkpoint_path
python scripts/run_train.py \
  --checkpoint_path "" \
  --experiment_name "peter_base" \
  --dataset_name "peter" \
  --data_dir "../StackMix-OCR-DATA" \
  --output_dir "../StackMix-OCR-SAVED_MODELS" \
  --mwe_tokens_dir "../StackMix-OCR-MWE_TOKENS" \
  --experiment_description "[Base] Training OCR on Digital Peter dataset" \
  --image_w 2048 \
  --image_h 128 \
  --num_epochs 300 \
  --bs 16 \
  --num_workers 4 \
  --use_blot 0 \
  --use_stackmix 0 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --seed 6955
