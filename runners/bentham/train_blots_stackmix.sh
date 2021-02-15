# if you need resume experiment, should set checkpoint_path
python scripts/run_train.py \
  --checkpoint_path "" \
  --experiment_name "bentham_blots_stackmix" \
  --dataset_name "bentham" \
  --data_dir "../StackMix-OCR-DATA" \
  --output_dir "../StackMix-OCR-SAVED_MODELS" \
  --mwe_tokens_dir "../StackMix-OCR-MWE_TOKENS" \
  --experiment_description "[StackMix+Blots] Training StackMix with Blots OCR on Bentham dataset" \
  --image_w 2048 \
  --image_h 128 \
  --num_epochs 1000 \
  --bs 16 \
  --num_workers 4 \
  --use_blot 1 \
  --use_stackmix 1 \
  --use_progress_bar 0 \
  --neptune_project "" \
  --neptune_token "" \
  --seed 6955
