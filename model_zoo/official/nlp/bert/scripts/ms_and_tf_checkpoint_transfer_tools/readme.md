# Mindspore and Tensorflow checkpoint transfer tools

# How to use
## 1. For Mindspore to Tensorflow
```
python ms_and_tf_checkpoint_transfer_for_bert_large.py \
--transfer_option='ms2tf' \
--ms_ckpt_path='/data/ms-bert/checkpoint_bert-1500_100.ckpt' \
--tf_ckpt_path='/data/tf-bert/bs64k_32k_ckpt_model.ckpt' \
--new_ckpt_path='/data/ms2tf/tf_bert_large_1500-100.ckpt'
```
## 2. For Tensorflow to Mindspore
```
python ms_and_tf_checkpoint_transfer_for_bert_large.py \
--transfer_option='tf2ms' \
--tf_ckpt_path='/data/tf-bert/tf_bert_large_1500-100.ckpt' \
--ms_ckpt_path='/data/ms-bert/checkpoint_bert-1500_100.ckpt' \
--new_ckpt_path='/data/tf2ms/ms_bert_large_1500-100.ckpt'
```

# Note
Please note that both tf2ms and ms2tf require two inputs, one output, one of the two inputs is the checkpoint to be converted, and the other is the target checkpoint to be referred to. Because there are many types of bert models, the meaning of the target checkpoint is to prevent you from using different checkpoints for conversion errors.