ALPHAPOSE_DIR="../AlphaPose"  
MOTIONBERT_DIR="../MotionBERT"

VIDEO_DIR=$1
VIDEO_NAME=$2

cd "$ALPHAPOSE_DIR"

python scripts/demo_inference.py --cfg configs/coco/hrnet/256x192_w32_lr1e-3.yaml --checkpoint pretrained_models/hrnet_w32_256x192.pth --video "$MOTIONBERT_DIR/$VIDEO_DIR/$VIDEO_NAME.mp4" --outdir "$MOTIONBERT_DIR/$VIDEO_DIR"

cd "$MOTIONBERT_DIR"

mv "$MOTIONBERT_DIR/$VIDEO_DIR/alphapose-results.json" "$MOTIONBERT_DIR/$VIDEO_DIR/$VIDEO_NAME.json"

python train_action_infer.py --config configs/action/MB_ft_aihub_xsub_custom.yaml --pretrained checkpoint/action/FT_aihub --video $VIDEO_DIR/$VIDEO_NAME.mp4