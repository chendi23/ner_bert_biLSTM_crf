import tensorflow as tf
import argparse
import config.global_var as gl
from model import bert_bilstm_crf_estimator, data_loader, model_ops
tf.enable_eager_execution()
params = argparse.ArgumentParser()
params.add_argument('--mode', default='train')
params = params.parse_args()
print(params.mode)


def main():

    model = bert_bilstm_crf_estimator.model_estimator(params=params)
    train_file = data_loader.get_file_list(gl.TFRECORDS_ROOT+'/train')
    valid_file = data_loader.get_file_list(gl.TFRECORDS_ROOT+'/valid')
    model_ops.model_fit(model=model, params=params, train_file=train_file, valid_file=valid_file)
    return

main()