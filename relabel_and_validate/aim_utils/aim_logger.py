import os
import time

"""
common args:
    parser.add_argument('--use_aim', default=True, action='store_false', help='use aim logger')
    parser.add_argument('--run_name', type=str, default="default", help='name of the run')
    parser.add_argument('--exp_name', type=str, default="test", help='name of the experiment')
    parser.add_argument('--tag_name', type=str, default=None, help='tag of the run')
    parser.add_argument('--cfg_yaml', type=str, default=None, help='path to config file')
"""

# export all
__all__ = ['aim_log', 'aim_hyperparam_log', 'aim_terminal_log', 'Logger']

def aim_log(run, metrics, step, epoch=0):
    """
    metrics: dict
        'train/loss' : 0.1
    """
    for key, value in metrics.items():
        if '/' in key:
            run.track(value, name=key.split('/')[1], step=step, context={'subset' : key.split('/')[0]})
        else:
            run.track(value, name=key, step=step)

def aim_hyperparam_log(run, args):
    """
    log hyperparameters to aim
    """
    hyperparams = dict()
    for key, value in vars(args).items():
        hyperparams.update({key: value})
    run["hparams"] = hyperparams

    if hasattr(args, 'tag_name') and args.tag_name:
        run.add_tag(args.tag_name)

def aim_terminal_log(run, custom_dir, args):
    time.sleep(5)   # wait for 5 seconds to make sure the run is finished
    assert args is not None # store args
    
    path = os.path.join(custom_dir, f'{run.name}_{args.cur_time}.txt')
    # create if not exists
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'w') as f:
        print(f"Writing to file {path}")
        f.write(f"created_at: {str(run.created_at)}")
        f.write(str(args.__dict__))
        f.write('\n')
        for line in run.get_terminal_logs().values.tolist():
            f.write(str(line))
            f.write('\n')

class Logger():
    def __init__(self, path):
        self.logger = open(os.path.join(path, 'log.txt'), 'w')

    def __call__(self, string, end='\n', print_=True):
        if print_:
            print("{}".format(string), end=end)
            if end == '\n':
                self.logger.write('{}\n'.format(string))
            else:
                self.logger.write('{} '.format(string))
            self.logger.flush()