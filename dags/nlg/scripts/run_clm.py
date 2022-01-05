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


output_dir = 'mymodel4'
train_file = 'data.csv'
max_seq_length = 256
block_size = 256
batch_size =4
num_train_epochs = 3
learning_rate = 5e-5
preprocessing_num_workers = 1
overwrite_cache = False
validation_split_percentage = 30

logger = logging.Logger(__name__)


class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


def sample_generator(dataset, tokenizer):
    # Trim off the last partial batch if present
    sample_ordering = np.random.permutation(len(dataset))
    for sample_idx in sample_ordering:
        example = dataset[int(sample_idx)]
        # Handle dicts with proper padding and conversion to tensor.
        example = {key: tf.convert_to_tensor(arr, dtype_hint=tf.int64) for key, arr in example.items()}
        yield example, example["labels"]  # TF needs some kind of labels, even if we don't use them
    return

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


def data_from_tfrecord():
    dataset=tf.data.TFRecordDataset(['/home/yangkaixuan/repo/models/official/nlp/xlnet/long_pretrain/tfrecords/train-0-0.bsz-4.seqlen-256.reuse-128.uni.alpha-6.beta-1.fnp-256.tfrecords'])
    # dataset=tf.data.TFRecordDataset(['/home/yangkaixuan/project/mama/1.tfrecord'])
    seq_len=256
    record_spec = {
            "input": tf.io.FixedLenFeature([seq_len], tf.int64),
            "target": tf.io.FixedLenFeature([seq_len], tf.int64),
            "seg_id": tf.io.FixedLenFeature([seq_len], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.int64),
            "is_masked": tf.io.FixedLenFeature([seq_len], tf.int64),
        }
    def _parse_function(example_proto):
        # Parse the input `tf.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, record_spec)

    parsed_dataset = dataset.map(_parse_function)
    parsed_dataset=parsed_dataset.map(lambda x : {'labels':x['target'], 'input_ids':x['input'],
                                                  'token_type_ids':x['seg_id'],'attention_mask':x['is_masked']})
    train_dataset=parsed_dataset.batch(batch_size=batch_size, drop_remainder=True).repeat(int(num_train_epochs))
    return train_dataset, sum(1 for _ in dataset)

def data_from_hf():
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
    data_files = {}
    dataset_args = {}
    data_files["train"] = train_file
    # dataset_args["keep_linebreaks"] = False
    raw_datasets = load_dataset('csv', data_files=data_files, **dataset_args)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))), test_size=validation_split_percentage / 100
    )
    eval_dataset = train_dataset.select(val_indices)
    train_dataset = train_dataset.select(train_indices)

    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_generator = partial(sample_generator, train_dataset, tokenizer)

    train_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in train_dataset.features
        if feature != "special_tokens_mask"
    }
    train_sig = (train_signature, train_signature["labels"])
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    tf_train_dataset = (
        tf.data.Dataset.from_generator(train_generator, output_signature=train_sig)
            .with_options(options)
            .batch(batch_size=batch_size, drop_remainder=True)
            .repeat(int(num_train_epochs))
    )
    eval_generator = partial(sample_generator, eval_dataset, tokenizer)
    eval_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in eval_dataset.features
        if feature != "special_tokens_mask"
    }
    eval_sig = (eval_signature, eval_signature["labels"])
    tf_eval_dataset = (
        tf.data.Dataset.from_generator(eval_generator, output_signature=eval_sig)
            .with_options(options)
            .batch(batch_size=batch_size, drop_remainder=True)
            .repeat(int(num_train_epochs))
    )
    batches_per_epoch = len(train_dataset) // batch_size
    return tf_train_dataset,tf_eval_dataset,len(train_dataset)

def main():
    
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-xlnet-base")
    model = TFXLNetLMHeadModel.from_pretrained("hfl/chinese-xlnet-base")
    model.transformer.attn_type = 'uni'
    model.transformer.reuse_len=128
    model.transformer.mem_len=192
    model.transformer.use_mems_train=True
    model.transformer.create_mask = MethodType(create_mask, model.transformer)

    tf_train_dataset, ds_size=data_from_tfrecord()
    # tf_train_dataset,_, ds_size=data_from_hf()

    
    batches_per_epoch = ds_size // batch_size
    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=int(num_train_epochs * batches_per_epoch),
        num_warmup_steps=2
    )

    def shape_list(tensor: tf.Tensor) -> List[int]:
        dynamic = tf.shape(tensor)

        if tensor.shape == tf.TensorShape(None):
            return dynamic

        static = tensor.shape.as_list()

        return [dynamic[i] if s is None else s for i, s in enumerate(static)]

    def loss(labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to -100 affect the loss
        active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
        reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        return loss_fn(labels, reduced_logits)

    def dummy_loss(y_true, y_pred):
        return tf.reduce_mean(y_pred)

    model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
    tf.config.experimental_run_functions_eagerly(True)
    history = model.fit(
        tf_train_dataset,
        # validation_data=tf_eval_dataset,
        epochs=int(num_train_epochs),
        steps_per_epoch=batches_per_epoch,
        callbacks=[SavePretrainedCallback(output_dir=output_dir)],
    )
    model.save_pretrained(output_dir)
    try:
        train_perplexity = math.exp(history.history["loss"][-1])
    except OverflowError:
        train_perplexity = math.inf
    # try:
    #     validation_perplexity = math.exp(history.history["val_loss"][-1])
    # except OverflowError:
    #     validation_perplexity = math.inf
    print(f"  Final train loss: {history.history['loss'][-1]:.3f}")
    print(f"  Final train perplexity: {train_perplexity:.3f}")
    # print(f"  Final validation loss: {history.history['val_loss'][-1]:.3f}")
    # print(f"  Final validation perplexity: {validation_perplexity:.3f}")
    # endregion


if __name__ == "__main__":
    main()
