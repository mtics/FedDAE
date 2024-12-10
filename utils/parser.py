import argparse


def loadParser():
    """
    Load parser
    """

    # Training settings
    parser = argparse.ArgumentParser()

    ######################################################
    #                       Method                       #
    ######################################################
    parser.add_argument('--alias', type=str, default='FedMAE')
    parser.add_argument('--is_federated', type=bool, default=True)

    ######################################################
    #                      Dataset                       #
    ######################################################
    parser.add_argument('--data_path', type=str, default='../datasets/')
    parser.add_argument('--dataset', type=str, default='movielens')
    parser.add_argument('--data_file', type=str, default='ml-100k.dat')
    parser.add_argument('--num_negative', type=int, default=4)

    parser.add_argument('--split', type=str, default='leave_one_out')
    # parser.add_argument('--split', type=str, default='holdout')
    parser.add_argument('--holdout_rates', type=float, default=0.1)

    parser.add_argument('--data_type', type=str, default='implicit')

    ######################################################
    #           Evaluation & Client Sampling             #
    ######################################################
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--clients_sample_ratio', type=float, default=1.0)
    parser.add_argument('--clients_sample_strategy', type=str, default='random')

    ######################################################
    #                     Training                       #
    ######################################################
    parser.add_argument('--early_stop', type=bool, default=True)

    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--hardware', type=str, default='gpu')
    parser.add_argument('--num_gpus', type=int, default=1)

    ######################################################
    #                  Hyperparameters                   #
    ######################################################
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--l2_reg', type=float, default=1e-8)
    parser.add_argument('--decay_rate', type=float, default=0.9)

    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--q_dims', type=int, nargs='+', default=None)
    parser.add_argument('--affine_type', type=str, default='mlp')
    parser.add_argument('--anneal_cap', type=float, default=0.2)

    # The following parameters are used for implementing the differential privacy mechanism
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--noise_scale', type=float, default=0.1)

    parser.add_argument('--weight', type=float, default=0.25) # for FedMAE with fixed weights

    ######################################################
    #                        Others                      #
    ######################################################
    parser.add_argument('--type', type=str, default='seed')
    parser.add_argument('--comment', type=str, default='default')

    parser.add_argument('--on_server', type=bool, default=False)
    parser.add_argument('--notice', type=bool, default=False)

    return parser.parse_args()
