import os

# These paths are relative to the user's DATA_DIR

TRAINING_DATA_DIR_NAME = 'training'
EVALUATION_DATA_DIR_NAME = 'evaluation'
EVALUATION_CONFLICTS_DATA_DIR_NAME = 'evaluation_conflicts'
TRAINING_CONFLICTS_DATA_DIR_NAME = 'training_conflicts'
TESTING_CONFLICTS_DATA_DIR_NAME = 'testing_conflicts'

DATABASE_DATASET_PICKLE_FILE_NAME = 'database_dataset.pkl'
FILTERED_DATASET_PICKLE_FILE_NAME = 'filtered_dataset.pkl'
SESSIONS_SCANS_DICT_FILE_NAME = 'sessions_scans_dict.json'

TRAINING_FULL_DATA_SET_PATH = os.path.join(TRAINING_DATA_DIR_NAME,'tiles_dataset.json')
TRAINING_TFRECORDS_BLUEPRINTS_PATH = os.path.join(TRAINING_DATA_DIR_NAME,'tf_blueprints.json')
EVALUATION_DATA_SET_PATH = os.path.join(EVALUATION_DATA_DIR_NAME,'tiles_dataset.json')
EVALUATION_CONFLICTS_DATA_SET_PATH = os.path.join(EVALUATION_CONFLICTS_DATA_DIR_NAME,'tiles_dataset.json')
TRAINING_CONFLICTS_DATA_SET_PATH = os.path.join(TRAINING_CONFLICTS_DATA_DIR_NAME,'tiles_dataset.json')
TESTING_CONFLICTS_DATA_SET_PATH = os.path.join(TESTING_CONFLICTS_DATA_DIR_NAME,'tiles_dataset.json')

CHECKPOINTS_ANALYSIS_PATH = 'checkpoints'

TRAINED_MODEL_PATH = '../trained_model'
ITERATIVE_TRAINING_LOG_FILE_NAME = 'iterative_training.log'

TRAINING_END_FILE_NAME = 'end_of_training.txt'