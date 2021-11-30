import os
import time
import tensorflow as tf
from model import data_loader
import config.global_var as gl
from data_utils.logger_config import get_logger

logger = get_logger(gl.LOG_DIR + '/train.log')


def model_fit(model, params, train_file, valid_file):
    valid_metric_list = []

    for ep in range(int(gl.EPOCH)):
        begin_time = time.time()
        logger.debug('ep:{} is about to train'.format(ep))
        model.train(input_fn=lambda: data_loader.input_fn(train_file, params))
        results = model.evaluate(input_fn=lambda: data_loader.input_fn(valid_file, params))
        end_time = time.time()
        logger.debug(
            'Epoch:{}\t loss={:.5f}\t recall={:.5f} \t precision={:.5f}\t f1={:.5f}\train plus eval time={:.5f}'.format(ep,
                                                                                                             results[
                                                                                                                 'loss'],
                                                                                                             results[
                                                                                                                 'recall'],
                                                                                                             results[
                                                                                                                 'precision'],
                                                                                                                  results['f1'],
                                                                                                             end_time - begin_time))
        # logger.debug('Epoch:{}\t loss={:.5f}\t  batch_acc={:.5f}\t train plus eval time={:.5f}'.format(ep, results['loss'], results['batch_acc'], end_time - begin_time))

    valid_metric_list.append(results['recall_metric'])
    if early_stop(valid_metric_list, backstep_num=10):
        logger.debug('Training early stops!!!')
        trained_model_path = model_save_pb(params, model)
        return trained_model_path, results
    logger.debug('model_pb saved!!!')
    trained_model_path = model_save_pb(params, model)
    return trained_model_path, results


def early_stop(valid_metric_list, backstep_num):
    length = len(valid_metric_list)
    best_metric_score = max(valid_metric_list)
    if length > 15:
        backstep_count = 0
        for i in range(backstep_num):
            if valid_metric_list[-1 * (i + 1)] < best_metric_score:
                backstep_count += 1
                if backstep_count == backstep_num:
                    return 1
        return 0


def model_predict():
    return


def model_save_pb(params, model):
    tf.disable_eager_execution()
    input_spec = {'ids': tf.placeholder(shape=[None, gl.MAX_SEQ_LENGTH], dtype=tf.int32, name='Xi'),
                  'masks': tf.placeholder(shape=[None, gl.MAX_SEQ_LENGTH], dtype=tf.int32, name='Xv')}

    model_input_receiving_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=input_spec)
    if not os.path.exists(os.path.join(params.model_pb)):
        os.mkdir(os.path.join(params.model_pb))
    return model.export_savedmodel(params.model_pb, model_input_receiving_fn)
