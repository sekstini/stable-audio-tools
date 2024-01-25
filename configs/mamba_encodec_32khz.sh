python train.py \
    --model_config "stable_audio_tools/configs/model_configs/txt2audio/mamba.json" \
    --dataset_config "stable_audio_tools/configs/dataset_configs/local_libritts.json" \
    --save_dir "checkpoints/mamba/encodec_32khz" \
    --batch_size 16 \
    --gradient_clip_val 0.2 \