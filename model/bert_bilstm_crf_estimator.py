import tensorflow as tf
import numpy as np
from data_utils.logger_config import get_logger
import config.global_var as gl
from bert.modeling import BertConfig, BertModel, get_assignment_map_from_checkpoint
from data_utils import eval

logger = get_logger(gl.LOG_DIR + '/train.log')




def model_fn(features, labels, params, mode):
    ids = tf.to_int32(features['ids'])
    masks = tf.to_int32(features['masks'])
    segment_ids = tf.to_int32(features['segment_ids'])
    labels = tf.to_int32(labels)


    # 得到句子长度。只有【pad】标记为0，其他非0项的数量即为句子实际长度
    used = tf.sign(tf.abs(ids))
    length = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(length, tf.int32)

    bert_config = BertConfig.from_json_file(gl.ROOT_PATH + '/' + "chinese_L-12_H-768_A-12/bert_config.json")
    bert_base = BertModel(config=bert_config, input_ids=ids, input_mask=masks, token_type_ids=segment_ids,
                          is_training=False)
    bert_out = bert_base.get_sequence_output()

    logger.debug('bert_fp_finished')

    with tf.variable_scope('biLSTM'):
        fw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=100, forget_bias=1.0, state_is_tuple=True)
        bw_lstm = tf.nn.rnn_cell.LSTMCell(num_units=100, forget_bias=1.0, state_is_tuple=True)
        (fw_output, bw_output), status = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, bert_out, dtype=tf.float32,
                                                                         time_major=False, scope=None)
        bilstm_out = tf.concat([fw_output, bw_output], axis=2)
        bilstm_dropout = tf.nn.dropout(bilstm_out, 0.5, name='bilstm_dropout')

    logger.debug('biLSTM_fp_finished')

    with tf.variable_scope('biLSTM_projection'):
        projection_out = tf.layers.dense(inputs=bilstm_dropout, units=gl.ann_num_tags,
                                         kernel_initializer=tf.keras.initializers.glorot_normal,
                                         bias_initializer=tf.keras.initializers.glorot_normal,
                                         )

    logger.debug('biLSTM_projection_fp_finished')

    with tf.variable_scope('crf_layer'):
        trans_matrix = tf.get_variable(name='transition_matrix', shape=[gl.ann_num_tags, gl.ann_num_tags],
                                       initializer=tf.initializers.glorot_normal)

        log_likelihood, trans_matrix = tf.contrib.crf.crf_log_likelihood(inputs=projection_out,
                                                                              tag_indices=labels,
                                                                              sequence_lengths=lengths,
                                                                              transition_params=trans_matrix)
        predictions, _ = tf.contrib.crf.crf_decode(projection_out, trans_matrix, lengths)



    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(predictions=predictions, mode=mode)
    else:
        loss = tf.reduce_sum(-log_likelihood)

        # loss =tf.contrib.seq2seq.sequence_loss(projection_out, labels, weights=tf.ones(shape=[gl.batch_size, gl.MAX_SEQ_LENGTH]))

        logger.debug('loss_fp_finished')

        with tf.variable_scope('metric'):
            # predictions = tf.cast(tf.argmax(projection_out, -1), tf.int32)

            # labels = tf.reshape(labels, [gl.batch_size*gl.MAX_SEQ_LENGTH])
            # predictions = tf.reshape(predictions, [gl.batch_size*gl.MAX_SEQ_LENGTH])
            weights = tf.sequence_mask(lengths, gl.MAX_SEQ_LENGTH)
            #pos_indices = list(range(1,gl.ann_num_tags))
            recall = eval.recall(predictions=predictions, labels=labels, weights=weights,
                                 average='macro', num_classes=gl.ann_num_tags)
            f1 = eval.f1(predictions=predictions, labels=labels, num_classes=gl.ann_num_tags, weights=weights,
                         average='macro',)
            precision = eval.precision(predictions=predictions, labels=labels, num_classes=gl.ann_num_tags,
                                       weights=weights, average='macro')

            # out = tf.equal(predictions, labels)
            # bool_to_int = tf.cast(out, dtype=tf.int32)
            # hits = tf.reduce_sum(bool_to_int)
            # recall = tf.metrics.recall(predictions=predictions, labels=labels)
            # acc = tf.metrics.accuracy(predictions=predictions, labels=labels)
            # precision = tf.metrics.precision(predictions=predictions, labels=labels)
            metrics = {'recall': recall, 'precision': precision, 'f1': f1}

    if mode == tf.estimator.ModeKeys.TRAIN:

        # vars_to_load = [i[0] for i in tf.train.list_variables(gl.INITIAL_CKPT)]
        # print(vars_to_load)
        # # print('bert/embeddings/gather' in vars_to_load)
        # # print('bert/embeddings/gather' in [var.op.name for var in tf.global_variables()])
        # # global_var = [var.op.name for var in tf.global_variables()]
        # assignment_map = {variable.op.name: variable for variable in tf.global_variables() if
        #                   variable.op.name in vars_to_load}
        # tvars = tf.trainable_variables()
        # (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, gl.INITIAL_CKPT)
        # tf.train.init_from_checkpoint(gl.INITIAL_CKPT, assignment_map)
        # logger.debug('graph is initialized!')
        optimizer = tf.train.AdamOptimizer(learning_rate=gl.LR)

        tvars = tf.trainable_variables()
        (assignment_map, initialized_vars) = get_assignment_map_from_checkpoint(init_checkpoint=gl.INITIAL_CKPT, tvars=tvars)
        tf.train.init_from_checkpoint(gl.INITIAL_CKPT, assignment_map)
        train_vars = []
        for var in tvars:
            if var.name in initialized_vars:
                continue
            else:
                train_vars.append(var)
        grads = tf.gradients(loss, train_vars)
        train_op = optimizer.apply_gradients(zip(grads, train_vars), global_step=tf.train.get_global_step())
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(predictions=projection_out, loss=loss, train_op=train_op, eval_metric_ops=metrics,
                                      mode=mode)


def model_estimator(params):
    tf.reset_default_graph()
    config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': gl.is_GPU}),
        log_step_count_steps=gl.log_step_count_steps,
        save_checkpoints_steps=gl.save_checkpoints_steps,
        keep_checkpoint_max=gl.keep_checkpoint_max,
        save_summary_steps=gl.save_summary_steps,

    )

    # ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=gl.INITIAL_CKPT, vars_to_warm_start=['bert.+kernel[^/]'])
    # model = tf.estimator.Estimator(model_fn=model_fn, config=config, model_dir=gl.MODEL_CKPT, params=params, warm_start_from=tf.estimator.WarmStartSettings(ckpt_to_initialize_from=gl.INITIAL_CKPT,
    #                                                                                                                                                         vars_to_warm_start="(bert.*)"))
    model = tf.estimator.Estimator(model_fn=model_fn, config=config, model_dir=gl.MODEL_CKPT, params=params)

    return model


