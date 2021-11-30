import os
import re
import codecs
import numpy as np
from datetime import datetime
from config import global_var as gl
from bert import tokenization
import tensorflow as tf
import re


tokenizer = tokenization.FullTokenizer(vocab_file=gl.ROOT_PATH + '/' + 'chinese_L-12_H-768_A-12/vocab.txt',
                                       do_lower_case=True)


def parse_ann_data(raw_data_dir):
    files = os.listdir(raw_data_dir)
    sentences_num = len(files) // 2

    def read_single_sentence(ann_path, txt_path):
        t_f = open(txt_path, 'r', encoding='utf-8')
        sentence = t_f.readline()
        sentence_length = len(sentence)
        t_f.close()
        mapping_list = ['O'] * sentence_length

        a_f = open(ann_path, 'r', encoding='utf-8')
        ann_records = a_f.readlines()
        # entity_pos_lists = []
        for ann_record in ann_records:
            ann_record = ann_record.split()
            entity_begin = int(ann_record[gl.ANN_FLAGS_BEGIN])
            entity_end = int(ann_record[gl.ANN_FLAGS_END])
            entity_name = str(ann_record[gl.ANN_FLAGS_ENTITY_NAME])
            for j in range(entity_begin, entity_end):
                mapping_list[j] = 'B' + '-' + entity_name if j == entity_begin else 'I' + '-' + entity_name
            # entity_pos_pair = [entity_begin, entity_end]
            # entity_pos_lists.extend(entity_pos_pair)
        # print(entity_pos_lists)
        # print(mapping_list)
        a_f.close()
        return sentence, mapping_list,

    def write_single_sentence_to_txt(sentence, mapping_list,
                                     output_dir=gl.NER_TRANSFORMED_DATA_DIR + '/transformed.txt'):
        with open(output_dir, 'a', encoding='utf-8') as fw:
            for i in range(len(mapping_list)):
                s_old = sentence[i] + ' ' + mapping_list[i] + '\n'

                if s_old.split(' ')[0] in ['\u3000', '\u00A0', '\u0020', '\xa0', '\t', '']:
                    pass

                else:
                    fw.write(s_old)
                    if i == int(len(mapping_list) - 1):
                        fw.write('\n')

    for i in range(1, sentences_num):
        ann_path, txt_path = os.path.join(raw_data_dir, '%i.ann' % i), os.path.join(raw_data_dir, '%i.txt' % i)
        sentence, mapping_list = read_single_sentence(ann_path, txt_path)
        write_single_sentence_to_txt(sentence=sentence, mapping_list=mapping_list)


def zero_digits(text):
    re.sub('\d', '0', text)
    return text


def load_sentences(path, lower, zeros):
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', encoding='utf8'):
        num += 1
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word = line.split()
            assert len(word) >= 2, print([word])
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    return sentences


def tag_mapping(sentences):
    # print(len(sentences))
    tags = [[char[-1] for char in s] for s in sentences]
    mapping_dict = create_disco(tags)
    mapping_dict['[SEP]'] = len(mapping_dict) + 1
    mapping_dict['[CLS]'] = len(mapping_dict) + 2

    id_to_tag, tag_to_id = create_mapping(mapping_dict)

    return id_to_tag, tag_to_id


def create_disco(item_list):
    dict = {}
    assert type(item_list) is list
    for item in item_list:
        for i in item:
            if i not in dict:
                dict[i] = 1
            else:
                dict[i] += 1
    print(len(dict.keys()))
    return dict


def create_mapping(dict):
    sorted_dict = sorted(dict.items(), key=lambda x: (-x[1], x[0]))

    id_to_tag = {i: v[0] for i, v in enumerate(sorted_dict)}
    tag_to_id = {v: k for k, v in id_to_tag.items()}

    return id_to_tag, tag_to_id


def prepare_dataset(sentences, max_seq_length, tag_to_id, lower, train=True):
    def f(x):
        return x.lower() if lower else x

    print(len(sentences))
    ids_list = []
    mask_list = []
    segment_ids_list = []
    label_list = []
    for s in sentences:
        string = [w[0].strip() for w in s]
        char_line = ' '.join(string)
        text = tokenization.convert_to_unicode(char_line)
        if train:
            tags = [w[-1] for w in s]
        else:
            tags = ['O' for _ in s]

        labels = ' '.join(tags)
        labels = tokenization.convert_to_unicode(labels)
        ids, mask, segment_ids, label_ids = convert_single_example(char_line=text, tag_to_id=tag_to_id,
                                                                   max_seq_length=max_seq_length, tokenizer=tokenizer,
                                                                   label_line=labels)
        ids_list.append(ids)
        mask_list.append(mask)
        segment_ids_list.append(segment_ids)
        label_list.append(label_ids)

    return ids_list, mask_list, segment_ids_list, label_list


def convert_single_example(char_line,
                           tag_to_id,
                           max_seq_length,
                           tokenizer,
                           label_line):
    text_list = char_line.split(' ')
    label_list = label_line.split(' ')

    tokens, labels = [], []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label)
            else:
                labels.append('X')

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]

    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append('[CLS]')
    segment_ids.append(0)
    label_ids.append(tag_to_id['[CLS]'])
    for i in range(len(tokens)):
        ntokens.append(tokens[i])
        label_ids.append(tag_to_id[labels[i]])
        segment_ids.append(0)
    ntokens.append('[SEP]')
    segment_ids.append(0)
    label_ids.append(tag_to_id['[SEP]'])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    masks = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)

        # we don't concerned about it!
        masks.append(0)
        segment_ids.append(0)

        label_ids.append(0)
        ntokens.append("**NULL**")

    return input_ids, masks, segment_ids, label_ids


def get_Float_ListFeature(value):
    if not isinstance(value, np.ndarray):
        value = np.asarray(value)
        value = value.astype(np.float32).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(bytes_list=float_list)
    else:
        value = value.astype(np.float32).tostring()
        value = [value]
        float_list = tf.train.BytesList(value=value)
        return tf.train.Feature(float_list)


def list_to_tfrecords(lists_dict=None, rows_count=0, output_dir=None):
    assert not (lists_dict is None)

    if not os.path.exists(os.path.join(output_dir)):
        os.mkdir(output_dir)
    filename = datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.tfrecords'
    with tf.io.TFRecordWriter(path=output_dir + '/' + filename) as wr:
        for i in range(int(rows_count)):
            single_row_dict = {}
            for k, v in lists_dict.items():
                    single_row_dict[k] = get_Float_ListFeature(v[i])
            features = tf.train.Features(feature=single_row_dict)
            example = tf.train.Example(features=features)
            wr.write(example.SerializeToString())
        wr.close()


def build_tfrecords_from_list(ids_list, mask_list, segment_ids_list, label_list, output_dir):
    assert type(ids_list) is list

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    _get = lambda x, l: [x[i] for i in l]
    n_example = len(mask_list)
    n_train = int(gl.TRAIN_VALID_RATIO / (1 + gl.TRAIN_VALID_RATIO) * n_example)
    n_valid = n_example - n_train
    train_idx = np.random.choice(list(range(n_example)), n_train, replace=False)
    valid_idx = np.random.choice(list(range(n_example)), n_valid, replace=False)

    ids_list_train, mask_list_train, segment_ids_list_train, label_list_train = _get(ids_list, train_idx), _get(
        mask_list, train_idx), _get(segment_ids_list, train_idx), _get(label_list, train_idx)
    ids_list_valid, mask_list_valid, segment_ids_list_valid, label_list_valid = _get(ids_list, valid_idx), _get(
        mask_list, valid_idx), _get(segment_ids_list, valid_idx), _get(label_list, valid_idx)

    train_dict = {'ids': ids_list_train, 'masks': mask_list_train, 'segment_ids':segment_ids_list_train,
                  'labels': label_list_train}
    valid_dict = {'ids': ids_list_valid, 'masks': mask_list_valid, 'segment_ids':segment_ids_list_valid,
                  'labels': label_list_valid}

    list_to_tfrecords(lists_dict=train_dict, rows_count=n_train, output_dir=output_dir + '/' + 'train')
    list_to_tfrecords(lists_dict=valid_dict, rows_count=n_valid, output_dir=output_dir + '/' + 'valid')


# parse_ann_data(gl.NER_ANN_DATA_DIR)
sentences = load_sentences(gl.NER_TRANSFORMED_DATA_DIR + '/transformed.txt', gl.LOWER, gl.ZEROS)
id_to_tag, tag_to_id = tag_mapping(sentences)
ids_list, mask_list, segment_ids_list, label_list = prepare_dataset(sentences=sentences,
                                                                    max_seq_length=gl.MAX_SEQ_LENGTH,
                                                                    tag_to_id=tag_to_id, lower=gl.LOWER,
                                                                    train=True)
print(1)
#build_tfrecords_from_list(ids_list, mask_list, segment_ids_list, label_list, gl.TFRECORDS_ROOT)

