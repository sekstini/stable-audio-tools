python train.py \
    --model-config "stable_audio_tools/configs/model_configs/txt2audio/transformer.json" \
    --dataset-config "stable_audio_tools/configs/dataset_configs/local_libritts.json" \
    --save-dir "checkpoints/transformer/encodec_32khz" \
    --batch-size 16 \
    --gradient-clip-val 0.2 \