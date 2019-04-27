EXPERIMENT="new"
DATA="data/GLUE/MRPC"
TASK="MRPC"
CONFIG="config/bert-base-uncased.json"
WEIGHTS="pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt"
VOCAB="config/bert-uncased-vocab.txt"

# Prepare experiment
OUTPUT="experiments/${TASK,,}/$EXPERIMENT"
echo "Removing $OUTPUT if it exists"
if [ -d "$OUTPUT" ]; then rm -r $OUTPUT; fi
mkdir -p $OUTPUT

# Copy this script and the model config to the experiment directory
cp $0 $OUTPUT
cp $CONFIG $OUTPUT

CUDA_VISIBLE_DEVICES=0 python classify.py \
  --model bert \
  --cfg $CONFIG \
  --load_weights $WEIGHTS \
  --exp_name $EXPERIMENT \
  --task_name $TASK \
  --data_dir $DATA \
  --vocab $VOCAB \
  --do_lower_case \
  --seed 42 \
  --val_every 1 \
  --max_seq_length 128 \
  --train_batch_size 40 \
  --num_train_epochs 4.0 \
  --learning_rate 2e-5
