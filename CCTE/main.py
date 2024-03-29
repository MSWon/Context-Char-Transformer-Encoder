import argparse
import yaml
from trainer import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", "-c", help="path of config file", required=True)
    args = parser.parse_args()

    hyp_args = yaml.load(open(args.config_path))
    print('========================')
    for key,value in hyp_args.items():
        print('{} : {}'.format(key, value))
    print('========================')
    ## Build model
    model = Trainer(hyp_args)
    ## Train model
    model.train()
