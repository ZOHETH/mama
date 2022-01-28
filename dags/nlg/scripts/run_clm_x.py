import logging
import math
import os
import random
import sys
import logging
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional, List
from types import MethodType
sys.path.append('dags')

from datasets import load_dataset
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, XLNetConfig
from nlg.scripts.transformers.models.xlnet.modeling_tf_xlnet import TFXLNetLMHeadModel
from nlg.scripts.transformers.trainer_tf import TFTrainer, TFTrainingArguments
from transformers import create_optimizer
from sklearn.model_selection import train_test_split


output_dir = '/datafile/kaixuan/nlg/mymodel6s'
max_seq_length = 256
block_size = 256
batch_size = 4
num_train_epochs = 4
learning_rate = 5e-5
preprocessing_num_workers = 1
overwrite_cache = False
validation_split_percentage = 1

logger = logging.Logger(__name__)


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)




def data_from_tfrecord():
    dataset=tf.data.TFRecordDataset(['/datafile/kaixuan/nlg/s0125.tfrecords'])
    seq_len=256
    m=128
    record_spec = {
            "input_ids": tf.io.FixedLenFeature([seq_len], tf.int64),
            "labels": tf.io.FixedLenFeature([m], tf.int64),
            "token_type_ids": tf.io.FixedLenFeature([seq_len], tf.int64),
            "target_mapping": tf.io.FixedLenFeature([m,seq_len], tf.float32),
            # "perm_mask": tf.io.FixedLenFeature([m,seq_len], tf.float32),
        }

    def _parse_function(example_proto):
        return tf.io.parse_single_example(example_proto, record_spec)

    parsed_dataset = dataset.map(_parse_function)
    train_dataset=parsed_dataset.batch(batch_size=batch_size, drop_remainder=True).repeat(int(num_train_epochs))
    return train_dataset, sum(1 for _ in dataset)


def create_mask(self, qlen, mlen):
    """
    Creates causal attention mask. Float mask where 1.0 indicates masked, 0.0 indicates not-masked.
    """
    attn_mask = tf.ones([qlen, qlen])
    mask_u = tf.linalg.band_part(attn_mask, 0, -1)
    mask_dia = tf.linalg.band_part(attn_mask, 0, 0)
    attn_mask_pad = tf.zeros([qlen, mlen])
    ret = tf.concat([attn_mask_pad, mask_u], 1)
    if self.same_length:
        mask_l = tf.linalg.band_part(attn_mask, -1, 0)
        ret = tf.concat([ret[:, :qlen] + mask_l - mask_dia, ret[:, qlen:]], 1)
    return ret

def main():
    model = TFXLNetLMHeadModel.from_pretrained("hfl/chinese-xlnet-base")
    model.transformer.attn_type = 'uni'
    model.transformer.create_mask = MethodType(create_mask, model.transformer)
    tf_train_dataset, ds_size=data_from_tfrecord()

    batches_per_epoch = ds_size // batch_size
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=int(num_train_epochs * batches_per_epoch),
        num_warmup_steps=2
    )
    
    def dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
    # tf.config.experimental_run_functions_eagerly(True)
    history = model.fit(
        tf_train_dataset,
        epochs=int(num_train_epochs),
        steps_per_epoch=batches_per_epoch,
        callbacks=[SavePretrainedCallback(output_dir=output_dir)],
    )
    model.save_pretrained(output_dir)
    try:
        train_perplexity = math.exp(history.history["loss"][-1])
    except OverflowError:
        train_perplexity = math.inf
    print(f"  Final train loss: {history.history['loss'][-1]:.3f}")
    print(f"  Final train perplexity: {train_perplexity:.3f}")



if __name__ == "__main__":
    main()
