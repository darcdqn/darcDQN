
def arg_parser():
    """
    Create an empty argparse.ArgumentParser
    """
    import argparse
    return argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser with arguments
    """
    parser = arg_parser()
    parser.add_argument('--env', help='Environment ID', type=str, default=None)
    parser.add_argument('--agent', help='Agent ID', type=str, default=None)
    parser.add_argument('--num_episodes', type=float, default=0.0)
    parser.add_argument('--num_timesteps', type=int, default=None)
    parser.add_argument('--mode', help='Render mode (human, rgb_array, none)',
                        type=str, default='human')
    parser.add_argument('--play', default=False, action='store_true')
    parser.add_argument('--save-path', help='Path to save trained agent to',
                        type=str, default=None)
    parser.add_argument('--load-path', help='Path to load trained agent from',
                        type=str, default=None)
    return parser


def parse_unknown_args(args):
    """
    Parse args not parsed by the common arg parser to a dictionary
    """
    retval = {}
    preceded_by_key = False
    for arg in args:
        if arg.startswith('--'):
            if '=' in arg:
                key = arg.split('=')[0][2:]
                value = arg.split('=')[1]
                retval[key] = value
            else:
                key = arg[2:]
                preceded_by_key = True
        elif preceded_by_key:
            retval[key] = arg
            preceded_by_key = False

    return retval
