import tensorflow as tf
tf.disable_eager_execution()
import config.global_var as gl

def parse_example(x):
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

    labels = tf.io.decode_raw(parsed_features['labels'], out_type=tf.float32)
    labels = tf.reshape(labels, shape=[gl.MAX_SEQ_LENGTH])
    return features, labels

reader = tf.TFRecordReader()
filename = gl.TFRECORDS_ROOT+'/train'+'/2021_11_01_13_27_37.tfrecords'

ds = tf.data.TFRecordDataset(filename)
ds = ds.map(lambda x: parse_example(x)).prefetch(buffer_size=10).batch(10)
itr = ds.make_one_shot_iterator()
batch_data = itr.get_next()
res = tf.Session().run(batch_data)

print(res)