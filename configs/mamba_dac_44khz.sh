python train.py \
    --model-config "stable_audio_tools/configs/model_configs/txt2audio/mamba_dac_44khz.json" \
    --dataset-config "stable_audio_tools/configs/dataset_configs/local_libritts.json" \
    --save-dir "checkpoints/mamba/dac_44khz" \
    --batch-size 8 \
    --gradient-clip-val 0.2 \