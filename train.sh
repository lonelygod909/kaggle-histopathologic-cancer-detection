#!/bin/bash

DEFAULT_MODEL_NAME="SEresNeXt50_32x4d"
BATCH_SIZE=64
NUM_EPOCHS=24
K_FOLDS=10
SEED=25

MODEL_NAME=${1:-$DEFAULT_MODEL_NAME}

LOG_DIR="logs/${MODEL_NAME}_epochs${NUM_EPOCHS}_frFalse_prTrue"
MODEL_PATH="${LOG_DIR}/model_fold_1.pth"
OUTPUT_FILE="submission_${MODEL_NAME}_${NUM_EPOCHS}epochs_final.csv"

echo "使用模型: $MODEL_NAME"
echo "日志目录: $LOG_DIR"
echo "模型路径: $MODEL_PATH"
echo "输出文件: $OUTPUT_FILE"

python train.py --image_dir '../autodl-tmp/data/train' \
                --model_name $MODEL_NAME \
                --pretrained \
                --batch_size $BATCH_SIZE \
                --num_epochs $NUM_EPOCHS \
                --k_folds $K_FOLDS \
                --seed $SEED \
                --resume None


python test.py --model_path $MODEL_PATH \
               --test_dir ../autodl-tmp/data/test \
               --output_file $OUTPUT_FILE \
               --model_name $MODEL_NAME \
               --batch_size $BATCH_SIZE \
               --use_tta

echo "训练和测试完成!"
echo "提交文件: $OUTPUT_FILE"