import os
import config.global_var as gl
import tensorflow as tf
from data_utils.logger_config import get_logger
logger = get_logger(gl.LOG_DIR + '/train.log')

def parse_single_example(x, params):
    expected_features = {'ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                         'masks': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                         'segment_ids': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                         'labels': tf.io.FixedLenFeature(shape=[], dtype=tf.string)}
    parsed_features = tf.io.parse_single_example(x, features=expected_features)
    ids = tf.io.decode_raw(parsed_features['ids'], out_type=tf.float32)
    ids = tf.reshape(ids, shape=[gl.MAX_SEQ_LENGTH])

    masks = tf.io.decode_raw(parsed_features['masks'], out_type=tf.float32)
    masks = tf.reshape(masks, shape=[gl.MAX_SEQ_LENGTH])

    segment_ids = tf.io.decode_raw(parsed_features['segment_ids'], out_type=tf.float32)
    segment_ids = tf.reshape(segment_ids, shape=[gl.MAX_SEQ_LENGTH])

    features = {'ids': ids, 'masks': masks, 'segment_ids': segment_ids}
    if params.mode == 'predict':
        return features
    else:
        labels = tf.io.decode_raw(parsed_features['labels'], out_type=tf.float32)
        labels = tf.reshape(labels, shape=[gl.MAX_SEQ_LENGTH])
        return features, labels


def input_fn(file_dir_list, params):
    files = tf.data.Dataset.list_files(file_dir_list)
    if params.mode == 'train':
        data_set = tf.data.TFRecordDataset(files, buffer_size=gl.batch_size * gl.batch_size) \
            .map(lambda x: parse_single_example(x, params), num_parallel_calls=4).shuffle(buffer_size=10 * gl.batch_size).batch(
            batch_size=gl.batch_size, drop_remainder=True).prefetch(1)
        iterator = data_set.make_one_shot_iterator()
        features, labels = iterator.get_next()
        logger.debug('input features/labels are generated!')
        #logger.debug(labels)
        return features, labels

    else:
        data_set = tf.data.TFRecordDataset(files, buffer_size=gl.batch_size * gl.batch_size) \
            .map(lambda x: parse_single_example(x, params), num_parallel_calls=4).batch(batch_size=gl.batch_size,
                                                                  drop_remainder=False).prefetch(1)
        iterator = data_set.make_one_shot_iterator()
        features, labels = iterator.get_next()
        return features


def get_file_list(dir):
    file_list = tf.gfile.ListDirectory(dir)
    file_dir_list = [dir + '/' + i for i in file_list]
    logger.debug(file_dir_list)
    return file_dir_list
