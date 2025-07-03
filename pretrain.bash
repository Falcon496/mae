export IMAGENET_DIR=/path/to/your/imagenet
export OUTPUT_DIR=/path/to/your/output/dir
uv python main_pretrain.py \ 
    # environment variables
    --data_path ${IMAGENET_DIR} \
    --job_dir ${OUTPUT_DIR} \
    --nodes 1 \
    --ngpus 1 \
    --use_volta32 \

    --model mae_deit_small_patch16_dec512d8b \
    --batch_size 64 \
    --accum_iter 4 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 