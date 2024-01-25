python train.py \
    --model_config "stable_audio_tools/configs/model_configs/txt2audio/mamba_dac_44khz.json" \
    --dataset_config "stable_audio_tools/configs/dataset_configs/local_libritts.json" \
    --save_dir "checkpoints/mamba/dac_44khz" \
    --batch_size 8 \
    --gradient_clip_val 0.2 \