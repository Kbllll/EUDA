from random import shuffle

import utils

if __name__ == '__main__':
    r_path = './out'

    template_args = utils.parse_args()
    template_args.enhance = ''
    template_args.comment = 'base'

    combine_space = {
        'uci_id': [17, 42, 43, 53, 109, 697],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hidden': [4],
        'layers': [2],
        'dropout': [0.01]
    }

    base_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.enhance = 'eua'
    template_args.comment = 'eua'

    combine_space = {
        'uci_id': [17, 42, 43, 53, 109, 697],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hidden': [4],
        'layers': [2],
        'dropout': [0.01],
        'r': [0.001, 0.005, 0.01, 0.1, 0.2]
    }

    eua_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.enhance = 'white_noise'
    template_args.comment = 'white_noise'

    combine_space = {
        'uci_id': [17, 42, 43, 53, 109, 697],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hidden': [4],
        'layers': [2],
        'dropout': [0.01],
        'white_noise_level': [0.01, 0.02, 0.03, 0.04, 0.05]
    }

    white_noise_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.enhance = 'salt_noise'
    template_args.comment = 'salt_noise'

    combine_space = {
        'uci_id': [17, 42, 43, 53, 109, 697],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hidden': [4],
        'layers': [2],
        'dropout': [0.01],
        'salt_noise_ratio': [0.01, 0.02, 0.03, 0.04, 0.05]
    }

    salt_noise_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.enhance = 'specaugment'
    template_args.comment = 'specaugment'

    combine_space = {
        'uci_id': [17, 42, 43, 53, 109, 697],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hidden': [4],
        'layers': [2],
        'dropout': [0.01],
        'mask_ratio': [0.01, 0.05, 0.1, 0.15, 0.2]
    }

    specaugment_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    template_args = utils.parse_args()
    template_args.enhance = 'cutout'
    template_args.comment = 'cutout'

    combine_space = {
        'uci_id': [17, 42, 43, 53, 109, 697],
        'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'hidden': [4],
        'layers': [2],
        'dropout': [0.01],
        'mask_ratio': [0.01, 0.05, 0.1, 0.15, 0.2]
    }

    cutout_args_list = utils.get_arg_list(args=template_args, combine_space=combine_space, r_path=r_path, sf=True)

    all_lists = (white_noise_args_list + base_args_list + eua_args_list + salt_noise_args_list
                 + specaugment_args_list + cutout_args_list)
    shuffle(all_lists)
    utils.parallel_run(all_lists, workers=8)
