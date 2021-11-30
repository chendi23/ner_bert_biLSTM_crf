import os
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = ROOT_PATH + '/data'
NER_ANN_DATA_DIR = ROOT_PATH + '/ner_ann'
NER_TRANSFORMED_DATA_DIR = ROOT_PATH+ '/ner_transformed'
TFRECORDS_ROOT = DATA_PATH + '/tfrecords'
INITIAL_CKPT = ROOT_PATH+'/chinese_L-12_H-768_A-12'+'/bert_model.ckpt'
LOG_DIR = DATA_PATH + '/log'
MODEL_PB  = DATA_PATH + '/model_pb'
MODEL_CKPT = DATA_PATH + '/ckpt'
# .ann flags
#ANN_FLAGS_ENTITY_NAME = 1
ANN_FLAGS_BEGIN = 2
ANN_FLAGS_END = 3
ANN_FLAGS_ENTITY_NAME = 1
ann_num_tags = 55


ZEROS = False
LOWER = True
MAX_SEQ_LENGTH = 120

RANDOM_SEED = 2021
TRAIN_VALID_RATIO = 4
K_FOLDS = 3
batch_size = 20
LR = 0.001
EPOCH = 100

is_GPU = True
log_step_count_steps = 10000
save_checkpoints_steps = 10000
keep_checkpoint_max = 10000
save_summary_steps = 10000

