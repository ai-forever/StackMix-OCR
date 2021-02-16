# if you need resume experiment, should set checkpoint_path
python scripts/prepare_char_masks.py \
  --checkpoint_path "../StackMix-OCR-SAVED_MODELS/hkr_blots/best_cer.pt" \
  --dataset_name "hkr" \
  --data_dir "../StackMix-OCR-DATA" \
  --experiment_description "[CHAR MASKS] Prepare char masks for HKR dataset" \
  --image_w 1024 \
  --image_h 128 \
  --bs 16 \
  --num_workers 4
