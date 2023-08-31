import yaml
import os.path as path

from lightning import Trainer
from argparse import ArgumentParser

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from src.data import CIFAR10DM
from src.vdm import VariationalDiffusion

def main(args):

    config = path.join('conf', args.config)

    with open(config, 'r') as f:
        conf_file = yaml.safe_load(f)

    # Create the variational diffusion model
    vdm = VariationalDiffusion.from_conf(conf_file)
    cifar = CIFAR10DM(**conf_file['DATASET'])

    ckpt_dir = conf_file['MISC']['ckpt_dir']
    logs_dir = conf_file['MISC']['logs_dir']
    run_name = conf_file['MISC']['run_name']
    monitor  = conf_file['MISC']['monitor']
    resume   = conf_file['MISC']['resume_ckpt']

    ckpter = ModelCheckpoint(dirpath=ckpt_dir, monitor=monitor)
    logger = TensorBoardLogger(logs_dir, name =run_name)

    args = {**vars(args), **conf_file['TRAINER'], 'logger' : logger, 'callbacks' : ckpter}
    args.pop('config')

    trainer = Trainer(**args)

    trainer.fit(vdm, datamodule=cifar, ckpt_path=resume)


if __name__ == '__main__':
    parser = ArgumentParser(
        prog = 'Variational-Diffusion Model Training Script',
        description = 'Training of the Variational-Diffusion Model on the CIFAR-10 Dataset',
    )
    
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-config', type = str, default = 'vdm_cifar10.yaml', help='Configuration file name')

    args = parser.parse_args()
    
    main(args)