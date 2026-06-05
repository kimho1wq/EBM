import os
import itertools

def get_args():
    """
    Returns
        system_args (dict): path, log setting
        experiment_args (dict): hyper-parameters
        args (dict): system_args + experiment_args
    """
    system_args = {
        # expeirment info
        'project'       : 'CM',
        'name'          : 'EBM',
        'tags'          : ['LA'],
        'description'   : '',
        # log
        'path_log'      : '/results',
        'neptune_user'  : '',
        'neptune_token' : '',

        # dataset
        'path_19LA'    : '/data/ASVspoof2019',
        'path_21LA'  : '/data/ASVspoof2021_LA_eval',
        'path_21DF'  : '/data/ASVspoof2021_DF',

        # others
        'num_workers'   : 4,
        'usable_gpu'    : None,
    }

    experiment_args = {
        # huggingface model
        'PLM_name'          : 'facebook/wav2vec2-xls-r-300m',

        # experiment
        'attack_type'       : 'LA', 
        'epoch'             : 150,
        'batch_size'        : 40,
        'rand_seed'		    : 1,
        'ratio_bona'        : 0.1,
        'ratio_spoof'       : 0.9,

        # SSL model
        'num_layers'        : 25,
        'hidden_size'       : 1024,
        'agg_size'          : 128,

        # Linear
        'output_size'       : 160, 
        'linear_units'      : 256,
        'dropout_rate'      : 0.35,
        'num_blocks'        : 6,
        'embd_dropout_rate' : 0.5,
        'downsample_layer'  : [1,3],
        'asp'               : True,
        'r_real'            : 0.9,
        'r_fake'            : 0.2,
        'alpha'             : 20.0,
        'alpha_cl'          : 30.0,
        'loss_weight'       : [1,1,1,1,1],
        'use_class_weight'  : True,

        # data processing
        'num_train_frames'  : 404,
        'num_train_frames_short'  : 303,
        'num_test_frames'   : 404,
        'num_seg'           : 1,

        # learning rate
        'lr'                : 1e-6,
        'lr_min'            : 1e-6,
        'weight_decay'      : 0.0001,
        'T_0'               : 150,
        'T_mult'            : 1,
    }

    args = {}
    for k, v in itertools.chain(system_args.items(), experiment_args.items()):
        args[k] = v
    args['path_scripts'] = os.path.dirname(os.path.realpath(__file__))

    return args, system_args, experiment_args
