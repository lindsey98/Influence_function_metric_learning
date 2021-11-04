"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import os
import argparse
import shutil

from tensorboardX import SummaryWriter

from deprecated.AlignMix.utils import get_config, make_result_folders
from deprecated.AlignMix.utils import write_loss, Timer
from trainer import Trainer

import torch.backends.cudnn as cudnn
from deprecated.AlignMix import dataset
import json

# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True
os.environ["CUDA_VISIBLE_DEVICES"]="1, 0"

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/funit_cub.yaml',
                    help='configuration file for training and testing')
parser.add_argument('--output_path', type=str,
                    default='checkpoints/cub200', help="outputs path")
parser.add_argument('--multigpus', default=True, action="store_true")
parser.add_argument("--resume", default=False, action="store_true")
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--test_batch_size', type=int, default=32)
parser.add_argument('--dataset', type=str, default='cub')
parser.add_argument('--workers', default = 4, type=int, dest = 'nb_workers')
parser.add_argument('--recall_list', default=[1,2,4,8], type=list)
opts = parser.parse_args()

def load_config(config_name = 'config.json'):
    '''
        Load config.json file
    '''
    config = json.load(open(config_name))
    def eval_json(config):
        for k in config:
            if type(config[k]) != dict:
                config[k] = eval(config[k])
            else:
                eval_json(config[k])
    eval_json(config)
    return config



if __name__ == '__main__':

    # Load experiment setting
    config = get_config(opts.config)
    print(config)
    max_iter = config['max_iter']

    # Override the batch size if specified.
    if opts.batch_size != 0:
        config['batch_size'] = opts.batch_size

    trainer = Trainer(config)
    trainer.cuda()
    if opts.multigpus:
        ngpus = torch.cuda.device_count()
        config['gpus'] = ngpus
        print("Number of GPUs: %d" % ngpus)
        trainer.model = torch.nn.DataParallel(
            trainer.model, device_ids=range(ngpus))
    else:
        config['gpus'] = 1

    # TODO classbalanced dataloader
    dataset_config = load_config('dataset/config.json')
    transform_key = 'transform_parameters'
    train_transform = deprecated.AlignMix.dataset.utils.make_transform(
                **dataset_config[transform_key]
    )
    tr_dataset = dataset.load(
        name=opts.dataset,
        root=dataset_config['dataset'][opts.dataset]['root'],
        source=dataset_config['dataset'][opts.dataset]['source'],
        classes=dataset_config['dataset'][opts.dataset]['classes']['trainval'],
        transform=train_transform,
    )
    num_class_per_batch = 8
    batch_sampler = deprecated.AlignMix.dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_class_per_batch,
                                                                           int(opts.batch_size / num_class_per_batch))

    # training loader
    dl_tr = torch.utils.data.DataLoader(
        tr_dataset,
        batch_sampler = batch_sampler,
        num_workers = opts.nb_workers,
    )

    dl_tr_noshuffle = torch.utils.data.DataLoader(
            dataset=dataset.load(
                    name=opts.dataset,
                    root=dataset_config['dataset'][opts.dataset]['root'],
                    source=dataset_config['dataset'][opts.dataset]['source'],
                    classes=dataset_config['dataset'][opts.dataset]['classes']['trainval'],
                    transform=deprecated.AlignMix.dataset.utils.make_transform(
                        **dataset_config[transform_key],
                        is_train=False
                    )
                ),
            num_workers = opts.nb_workers,
            shuffle=False,
            batch_size=64,
    )

    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name=opts.dataset,
            root=dataset_config['dataset'][opts.dataset]['root'],
            source=dataset_config['dataset'][opts.dataset]['source'],
            classes=dataset_config['dataset'][opts.dataset]['classes']['eval'],
            transform=deprecated.AlignMix.dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train=False
            )
        ),
        batch_size=64,
        shuffle=False,
        num_workers=opts.nb_workers,
    )

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    train_writer = SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = make_result_folders(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))

    iterations = trainer.resume(checkpoint_directory,
                                hp=config,
                                multigpus=opts.multigpus) if opts.resume else 0
    # nmi, recall = trainer.evaluate(dl_ev, opts.multigpus)
    # print(recall)

    while True:
        for it, (x, y, indices) in enumerate(dl_tr):
            with Timer("Elapsed time in update: %f"):
                xa, la = x[:opts.batch_size//2], y[:opts.batch_size//2]
                xb, lb = x[opts.batch_size//2:], y[opts.batch_size//2:]
                trainer.gen_update(xa, la, xb, lb, config,
                                           opts.multigpus)
                torch.cuda.synchronize()

            if iterations == 0 or (iterations + 1) % config['log_iter'] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                nmi, recall = trainer.evaluate(dl_ev, opts.multigpus)
                write_loss(iterations, trainer, train_writer)
                train_writer.add_scalar('nmi', nmi, iterations + 1)
                for j, i in enumerate(opts.recall_list):
                    train_writer.add_scalar('recall@{}'.format(i), recall[j], iterations + 1)

            if iterations == 0 or (iterations + 1) % config['snapshot_save_iter'] == 0 or (iterations + 1) >= max_iter:
                trainer.save(checkpoint_directory, iterations, opts.multigpus)
                print('Saved model at iteration %d' % (iterations + 1))

            iterations += 1
            if iterations >= max_iter:
                print("Finish Training")
                exit()
