# if you need resume experiment, should set checkpoint_path
python scripts/prepare_char_masks.py \
  --checkpoint_path "../StackMix-OCR-SAVED_MODELS/saintgall_base/last.pt" \
  --dataset_name "saintgall" \
  --data_dir "../StackMix-OCR-DATA" \
  --experiment_description "[CHAR MASKS] Prepare char masks for Saint Gall dataset" \
  --image_w 2048 \
  --image_h 128 \
  --bs 16 \
  --num_workers 4
