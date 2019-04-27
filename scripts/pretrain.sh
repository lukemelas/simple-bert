EXPERIMENT="test-finetuning-from-pretrained-weights"
DATA="data/Wiki/wiki.train.processed"
CONFIG="config/bert-base-uncased.json"
TASK="pretrain"
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

CUDA_VISIBLE_DEVICES=0,1 python pretrain.py \
  --model bert \
  --text_file $DATA \
  --cfg $CONFIG \
  --exp_name $EXPERIMENT \
  --vocab $VOCAB \
  --do_lower_case \
  --seed 42 \
  --val_every 1 \
  --max_seq_length 256 \
  --train_batch_size 32 \
  --total_iterations 100000 \
  --learning_rate 5e-6 \
  --load_weights $WEIGHTS 
