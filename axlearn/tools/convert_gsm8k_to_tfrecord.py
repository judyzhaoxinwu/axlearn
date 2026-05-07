# Copyright © 2026 Apple Inc.

"""Utility tool serializing GSM8K datasets into versioned structures natively."""


import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags, logging

from axlearn.common import input_tf_data

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir",
    None,
    "Base storage directory or project bucket (e.g., data/tensorflow_datasets/)",
    required=True,
)
flags.DEFINE_integer(
    "verify_records",
    3,
    "Number of records to read back and verify after conversion",
    required=False,
)
flags.DEFINE_integer(
    "num_shards",
    1,
    "Number of file shards to split the dataset into for multi-host TPU compatibility",
    required=False,
)


def serialize_sample(question: str, answer: str, annotation: str, short_answer: str) -> bytes:
    """Packs problem strings variables into serializable Protobuf features."""
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "question": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[question.encode("utf-8")])
                ),
                "answer": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[answer.encode("utf-8")])
                ),
                "annotation": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[annotation.encode("utf-8")])
                ),
                "short_answer": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[short_answer.encode("utf-8")])
                ),
            }
        )
    )
    return example.SerializeToString()


def convert_dataset(split: str, version_dir: str, num_shards: int) -> int:
    """Downloads original split and writes sharded TFRecords, returning total row count."""
    logging.info("Fetching split data %s via TFDS...", split)
    data_stream = tfds.load("gsm8k", split=split)

    shard_writers = []
    for s in range(num_shards):
        target_filename = f"gsm8k-{split}.tfrecord-{s:05d}-of-{num_shards:05d}"
        target_path = os.path.join(version_dir, target_filename)
        shard_writers.append(tf.io.TFRecordWriter(target_path))

    logging.info("Writing %d shards for split %s in directory %s", num_shards, split, version_dir)

    count = 0
    try:
        for item in data_stream.as_numpy_iterator():
            binary_record = serialize_sample(
                question=item["question"].decode("utf-8"),
                answer=item["answer"].decode("utf-8"),
                annotation=item["annotation"].decode("utf-8"),
                short_answer=item["short_answer"].decode("utf-8"),
            )

            assigned_shard = count % num_shards
            shard_writers[assigned_shard].write(binary_record)
            count += 1
    finally:
        for writer in shard_writers:
            writer.close()

    logging.info("Wrote out %d examples across %d shards for subset %s.", count, num_shards, split)
    return count


def create_metadata_files(version_dir: str, num_shards: int, counts: dict):
    """Writes the dataset_info.json and features.json metadata dynamically using compiled counts."""
    logging.info("Writing metadata JSON structures inside: %s", version_dir)

    # Construct shards distribution profiles dynamically
    def shard_distribution(total_count: int, shards: int) -> list:
        base = total_count // shards
        rem = total_count % shards
        return [str(base + 1 if i < rem else base) for i in range(shards)]

    dataset_info = {
        "name": "gsm8k",
        "version": "1.0.0",
        "description": "Grade School Math 8k reasoning dataset, sharded for multi-host TPUs.",
        "splits": [
            {"name": "train", "shardLengths": shard_distribution(counts["train"], num_shards)},
            {"name": "test", "shardLengths": shard_distribution(counts["test"], num_shards)},
            {
                "name": "train_socratic",
                "shardLengths": shard_distribution(counts["train_socratic"], num_shards),
            },
            {
                "name": "test_socratic",
                "shardLengths": shard_distribution(counts["test_socratic"], num_shards),
            },
        ],
    }

    with tf.io.gfile.GFile(os.path.join(version_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)

    features = {
        "feature": {
            "type": "STRUCT",
            "struct": {
                "features": {
                    "question": {"type": "TENSOR", "tensor": {"shape": [], "dtype": "STRING"}},
                    "answer": {"type": "TENSOR", "tensor": {"shape": [], "dtype": "STRING"}},
                    "annotation": {"type": "TENSOR", "tensor": {"shape": [], "dtype": "STRING"}},
                    "short_answer": {"type": "TENSOR", "tensor": {"shape": [], "dtype": "STRING"}},
                }
            },
        }
    }

    with tf.io.gfile.GFile(os.path.join(version_dir, "features.json"), "w") as f:
        json.dump(features, f, indent=2)


def verify_and_print_records(tfrecord_path: str, num_records: int):
    """Reads back compiled tfrecord binary payloads natively to verify data integrity."""
    feature_description = {
        "question": tf.io.FixedLenFeature([], tf.string),
        "answer": tf.io.FixedLenFeature([], tf.string),
        "annotation": tf.io.FixedLenFeature([], tf.string),
        "short_answer": tf.io.FixedLenFeature([], tf.string),
    }

    build_dataset_fn = input_tf_data.tfrecord_dataset(
        glob_path=tfrecord_path,
        is_training=False,
        shuffle_buffer_size=0,
        features=feature_description,
    )

    parsed_dataset = build_dataset_fn()

    count = 0
    for record in parsed_dataset.take(num_records):
        count += 1
        q_text = record["question"].numpy().decode("utf-8")
        a_text = record["answer"].numpy().decode("utf-8")
        ann_text = record["annotation"].numpy().decode("utf-8")
        sa_text = record["short_answer"].numpy().decode("utf-8")

        logging.info("[eshenlog] [Record #%d]", count)
        logging.info("[eshenlog] Question: %s", q_text)
        logging.info("[eshenlog] Answer: %s", a_text)
        logging.info("[eshenlog] Annotation: %s", ann_text)
        logging.info("[eshenlog] Short Answer: %s", sa_text)
        logging.info("[eshenlog] %s", "-" * 40)


def main(_):
    version_dir = os.path.join(FLAGS.output_dir, "gsm8k", "1.0.0")
    tf.io.gfile.makedirs(version_dir)

    num_shards = FLAGS.num_shards

    # Track exact counts dynamically during compilation loop
    counts = {}
    counts["train"] = convert_dataset("train", version_dir, num_shards)
    counts["test"] = convert_dataset("test", version_dir, num_shards)
    counts["train_socratic"] = convert_dataset("train_socratic", version_dir, num_shards)
    counts["test_socratic"] = convert_dataset("test_socratic", version_dir, num_shards)

    create_metadata_files(version_dir, num_shards, counts)

    if FLAGS.verify_records > 0:
        sample_shard = os.path.join(version_dir, "gsm8k-train.tfrecord-00000-of-*")
        verify_and_print_records(sample_shard, FLAGS.verify_records)


if __name__ == "__main__":
    app.run(main)
