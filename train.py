
import logging
import dataset
import utils
import loss

import os

import torch
import numpy as np
import matplotlib
matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
import time
import argparse
import json
import random
from tqdm import tqdm
# from apex import amp
from utils import JSONEncoder, json_dumps
from utils import predict_batchwise, inner_product_sim
from dataset.base import SubSampler
from hard_detection import hard_potential, split_potential
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"]="1"

parser = argparse.ArgumentParser(description='Training ProxyNCA++')
parser.add_argument('--embedding-size', default = 2048, type=int, dest = 'sz_embedding')
parser.add_argument('--epochs', default = 40, type=int, dest = 'nb_epochs')
parser.add_argument('--log-filename', default = 'example')
parser.add_argument('--workers', default = 16, type=int, dest = 'nb_workers')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--lr_steps', default=[1000], nargs='+', type=int)
parser.add_argument('--source_dir', default='', type=str)
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--eval_nmi', default=False, action='store_true')
parser.add_argument('--recall', default=[1, 2, 4, 8], nargs='+', type=int)
parser.add_argument('--init_eval', default=False, action='store_true')
parser.add_argument('--apex', default=False, action='store_true')
parser.add_argument('--warmup_k', default=5, type=int)

parser.add_argument('--dataset', default='cub')
parser.add_argument('--config', default='config/cub_dist.json')
parser.add_argument('--mode', default='trainval', choices=['train', 'trainval', 'test', 'testontrain', 'testontrain_super'],
                    help='train with train data or train with trainval')
parser.add_argument('--batch-size', default = 32, type=int, dest = 'sz_batch')
parser.add_argument('--dynamic_proxy', default=False, action='store_true')
parser.add_argument('--initial_proxy_num', default=1, type=int)
parser.add_argument('--tau', default=0.0, type=float)
parser.add_argument('--proxy_update_schedule', default=[0.5, 0.75], nargs='+', type=float)
parser.add_argument('--no_warmup', default=True, action='store_true')

args = parser.parse_args()

def save_best_checkpoint(model):
    torch.save(model.state_dict(), 'results/' + args.log_filename + '.pt')

def load_best_checkpoint(model):
    try:
        model.load_state_dict(torch.load('results/' + args.log_filename + '.pt'))
    except FileNotFoundError:
        model.load_state_dict(torch.load('results/' + args.log_filename + '.pth'))
    model = model.cuda()
    return model

if __name__ == '__main__':

    # set random seed for all gpus
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs('results', exist_ok=True)
    os.makedirs('log', exist_ok=True)

    curr_fn = os.path.basename(args.config).split(".")[0]
    config = utils.load_config(args.config)
    dataset_config = utils.load_config('dataset/config.json')

    if args.source_dir != '':
        bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['source'])
        dataset_config['dataset'][args.dataset]['source'] = os.path.join(args.source_dir, bs_name)
    if args.root_dir != '':
        bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['root'])
        dataset_config['dataset'][args.dataset]['root'] = os.path.join(args.root_dir, bs_name)

    #set NMI or recall accordingly depending on dataset. note for cub and cars R=1,2,4,8
    if (args.mode =='trainval' or args.mode == 'test' or args.mode == 'testontrain'):
        if args.dataset == 'sop' or args.dataset == 'sop_h5':
            args.recall = [1, 10, 100, 1000]
        elif 'cub' in args.dataset or 'cars' in args.dataset: # FIXME: logo2k didnt evaluate NMI cuz it's time-consuming
            args.eval_nmi = True

    args.nb_epochs = config['nb_epochs']
    args.sz_batch = config['sz_batch']
    args.sz_embedding = config['sz_embedding']
    if 'warmup_k' in config:
        args.warmup_k = config['warmup_k']

    transform_key = 'transform_parameters'
    if 'transform_key' in config.keys():
        transform_key = config['transform_key']
    print('Transformation: ', transform_key)


    out_results_fn = "log/%s_%s_%s_%d_%s_loss%s_proxy%d_tau%0.2f.json" % (args.dataset, curr_fn, args.mode,
                                                                        args.seed,
                                                                        args.dynamic_proxy,
                                                                        str(config['criterion']['type']).split('.')[1],
                                                                        args.initial_proxy_num, args.tau)

    args.log_filename = '%s_%s_%s_%d_%d_%s_loss%s_proxy%d_tau%0.2f' % (args.dataset, curr_fn, args.mode, args.sz_embedding,
                                                                     args.seed,
                                                                     args.dynamic_proxy,
                                                                     str(config['criterion']['type']).split('.')[1],
                                                                     args.initial_proxy_num, args.tau)

    if args.mode == 'test':
        args.log_filename = args.log_filename.replace('test', 'trainval')
    elif args.mode == 'testontrain':
        args.log_filename = args.log_filename.replace('testontrain', 'trainval')
    elif args.mode == 'testontrain_super':
        args.log_filename = args.log_filename.replace('testontrain_super', 'trainval')
    best_epoch = args.nb_epochs

    '''Dataloader'''
    if args.mode == 'trainval':
        train_results_fn = "log/%s_%s_%s_%d_%d_%s_loss%s_proxy%d_tau%0.2f.json" % (args.dataset, curr_fn, 'train',
                                                                                 args.sz_embedding, args.seed,
                                                                                 args.dynamic_proxy,
                                                                                 str(config['criterion']['type']).split('.')[1],
                                                                                 args.initial_proxy_num, args.tau)
        # train_results_fn = "log/%s_kd.json" % (args.dataset)

        if os.path.exists(train_results_fn):
            with open(train_results_fn, 'r') as f:
                train_results = json.load(f)
            args.lr_steps = train_results['lr_steps']
            best_epoch = train_results['best_epoch']

    train_transform = dataset.utils.make_transform(
                **dataset_config[transform_key]
            )
    print('best_epoch', best_epoch)

    results = {}

    if ('inshop' not in args.dataset ):
        dl_ev = torch.utils.data.DataLoader(
            dataset.load(
                name = args.dataset,
                root = dataset_config['dataset'][args.dataset]['root'],
                source = dataset_config['dataset'][args.dataset]['source'],
                classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
                transform = dataset.utils.make_transform(
                    **dataset_config[transform_key],
                    is_train = False
                )
            ),
            batch_size = args.sz_batch,
            shuffle = False,
            num_workers = args.nb_workers,
            #pin_memory = True
        )
    else:
        #inshop trainval mode
        dl_query = torch.utils.data.DataLoader(
            dataset.load_inshop(
                name = args.dataset,
                root = dataset_config['dataset'][args.dataset]['root'],
                source = dataset_config['dataset'][args.dataset]['source'],
                classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
                transform = dataset.utils.make_transform(
                    **dataset_config[transform_key],
                    is_train = False
                ),
                dset_type = 'query'
            ),
            batch_size = args.sz_batch,
            shuffle = False,
            num_workers = args.nb_workers,
            #pin_memory = True
        )
        dl_gallery = torch.utils.data.DataLoader(
            dataset.load_inshop(
                name = args.dataset,
                root = dataset_config['dataset'][args.dataset]['root'],
                source = dataset_config['dataset'][args.dataset]['source'],
                classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
                transform = dataset.utils.make_transform(
                    **dataset_config[transform_key],
                    is_train = False
                ),
                dset_type = 'gallery'
            ),
            batch_size = args.sz_batch,
            shuffle = False,
            num_workers = args.nb_workers,
            #pin_memory = True
        )

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler("{0}/{1}.log".format('log', args.log_filename)),
            logging.StreamHandler()
        ]
    )

    if args.mode == 'train':
        tr_dataset = dataset.load(
                name = args.dataset,
                root = dataset_config['dataset'][args.dataset]['root'],
                source = dataset_config['dataset'][args.dataset]['source'],
                classes = dataset_config['dataset'][args.dataset]['classes']['train'],
                transform = train_transform
            )

    elif args.mode == 'trainval' or args.mode == 'test' or args.mode == 'testontrain' or args.mode == 'testontrain_super':
        # print(dataset_config['dataset'][args.dataset]['root'])
        tr_dataset = dataset.load(
                name = args.dataset,
                root = dataset_config['dataset'][args.dataset]['root'],
                source = dataset_config['dataset'][args.dataset]['source'],
                classes = dataset_config['dataset'][args.dataset]['classes']['trainval'],
                transform = train_transform
            )

    num_class_per_batch = config['num_class_per_batch']
    num_gradcum = config['num_gradcum']
    is_random_sampler = config['is_random_sampler']
    if is_random_sampler:
        batch_sampler = dataset.utils.RandomBatchSampler(tr_dataset.ys, args.sz_batch, True, num_class_per_batch, num_gradcum)
    else:

        batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_class_per_batch,
                                                           int(args.sz_batch / num_class_per_batch))


    dl_tr = torch.utils.data.DataLoader(
        tr_dataset,
        batch_sampler = batch_sampler,
        num_workers = args.nb_workers,
        #pin_memory = True
    )

    # training dataloader without shuffling and without transformation
    dl_tr_noshuffle = torch.utils.data.DataLoader(
            dataset=dataset.load(
                    name=args.dataset,
                    root=dataset_config['dataset'][args.dataset]['root'],
                    source=dataset_config['dataset'][args.dataset]['source'],
                    classes=dataset_config['dataset'][args.dataset]['classes']['trainval'],
                    transform=dataset.utils.make_transform(
                        **dataset_config[transform_key],
                        is_train=False
                    )
                ),
            num_workers = args.nb_workers,
            shuffle=False,
            batch_size=64,
    )


    print("===")
    if args.mode == 'train':
        dl_val = torch.utils.data.DataLoader(
            dataset.load(
                name = args.dataset,
                root = dataset_config['dataset'][args.dataset]['root'],
                source = dataset_config['dataset'][args.dataset]['source'],
                classes = dataset_config['dataset'][args.dataset]['classes']['val'],
                transform = dataset.utils.make_transform(
                    **dataset_config[transform_key],
                    is_train = False
                )
            ),
            batch_size = args.sz_batch,
            shuffle = False,
            num_workers = args.nb_workers,
            #drop_last=True
            #pin_memory = True
        )

    '''Model'''
    feat = config['model']['type']()
    feat.eval()
    in_sz = feat(torch.rand(1, 3, 256, 256)).squeeze().size(0)
    feat.train()
    emb = torch.nn.Linear(in_sz, args.sz_embedding)
    model = torch.nn.Sequential(feat, emb)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    '''Loss'''
    # TODO
    criterion = config['criterion']['type'](
        nb_classes = dl_tr.dataset.nb_classes(),
        sz_embed = args.sz_embedding,
        initial_proxy_num = args.initial_proxy_num,
        **config['criterion']['args']
    ).cuda()

    # options for warmup
    opt_warmup = config['opt']['type'](
        [
            {
                **{'params': list(feat.parameters()
                    )
                },
                'lr': 0
            },
            {
                **{'params': list(emb.parameters()
                    )
                },
                **config['opt']['args']['embedding']

            },

            {
                **{'params': criterion.proxies}
                ,
                **config['opt']['args']['proxynca']

            },
            {
                **{'params': criterion.sigmas_inv},
                **config['opt']['args']['proxynca_sigma']
            },

        ],
        **config['opt']['args']['base']
    )

    # options for model and loss
    opt = config['opt']['type'](
        [
            {
                **{'params': list(feat.parameters()
                    )
                },
                **config['opt']['args']['backbone']
            },
            {
                **{'params': list(emb.parameters()
                    )
                },
                **config['opt']['args']['embedding']
            },

            {
                **{'params': criterion.proxies},
                **config['opt']['args']['proxynca']
            },
            {
                **{'params': criterion.sigmas_inv},
                **config['opt']['args']['proxynca_sigma']
            },

        ],
        **config['opt']['args']['base']
    )

    if args.mode == 'test':
        with torch.no_grad():
            logging.info("**Evaluating...(test mode)**")
            model = load_best_checkpoint(model)
            if 'inshop' in args.dataset:
                utils.evaluate_inshop(model, dl_query, dl_gallery)
            else:
                utils.evaluate(model, dl_ev, args.eval_nmi, args.recall)
        exit() # exit the program
    if args.mode == 'testontrain':
        with torch.no_grad():
            logging.info("**Evaluating...(test mode, test on training set)**")
            model = load_best_checkpoint(model)
            utils.evaluate(model, dl_tr_noshuffle, args.eval_nmi, args.recall)
        exit() # exit the program
    if args.mode == 'testontrain_super':
        with torch.no_grad():
            logging.info("**Evaluating with gt super 100 classes...(test mode, test on training set)**")
            model = load_best_checkpoint(model)
            with open("mnt/datasets/logo2ksuperclass0.01.json", 'rt') as handle:
                labeldict = json.load(handle)
            utils.evaluate_super(model, dl_tr_noshuffle, labeldict, args.eval_nmi, args.recall)
        exit() # exit the program

    if args.mode == 'train':
        scheduler = config['lr_scheduler']['type'](
            opt, **config['lr_scheduler']['args']
        )
    elif args.mode == 'trainval':
        scheduler = config['lr_scheduler2']['type'](
            opt,
            milestones=args.lr_steps,
            gamma=0.1
            #opt, **config['lr_scheduler2']['args']
        )

    logging.info("Training parameters: {}".format(vars(args)))
    logging.info("Training for {} epochs.".format(args.nb_epochs))
    losses = []
    scores = []
    scores_tr = []

    t1 = time.time()


    if args.init_eval:
        logging.info("**Evaluating initial model...**")
        with torch.no_grad():
            if args.mode == 'train':
                c_dl = dl_val
            else:
                c_dl = dl_ev

            utils.evaluate(model, c_dl, args.eval_nmi, args.recall) #dl_val

    it = 0
    best_val_hmean = 0
    best_val_nmi = 0
    best_val_epoch = 0
    best_val_r1 = 0
    best_test_nmi = 0
    best_test_r1 = 0
    best_test_r2 = 0
    best_test_r5 = 0
    best_test_r8 = 0
    best_tnmi = 0

    prev_lr = opt.param_groups[0]['lr']
    lr_steps = []

    logging.info('Number of training: {}'.format(len(dl_tr.dataset)))
    logging.info('Number of original training: {}'.format(len(dl_tr_noshuffle.dataset)))
    logging.info('Number of testing: {}'.format(len(dl_ev.dataset)))

    '''Warmup training'''
    if not args.no_warmup:
        #warm up training for 5 epochs
        logging.info("**warm up for %d epochs.**" % args.warmup_k)
        for e in range(0, args.warmup_k):
            for ct, (x,y,_) in tqdm(enumerate(dl_tr)):
                opt_warmup.zero_grad()
                m = model(x.cuda())
                loss = criterion(m, None, y.cuda())
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 10)
                opt_warmup.step()
            logging.info('warm up ends in %d epochs' % (args.warmup_k-e))


    '''training loop'''
    for e in range(0, args.nb_epochs):
        len_training = len(dl_tr_noshuffle.dataset)  # training
        cached_sim = np.zeros(len_training)  # cache the similarity to ground-truth proxy
        cached_cls = np.zeros(len_training) # cache the gt-class

        if args.dynamic_proxy:
            if e == 0:
                loss_recorder = {}
                with open("{0}/{1}_ip.json".format('log', args.log_filename), 'wt') as handle:
                    json.dump(loss_recorder, handle)
                label_recorder = {}
                with open("{0}/{1}_cls.json".format('log', args.log_filename), 'wt') as handle:
                    json.dump(label_recorder, handle)

            with open("{0}/{1}_ip.json".format('log', args.log_filename), 'rt') as handle:
                loss_recorder = json.load(handle)
            with open("{0}/{1}_cls.json".format('log', args.log_filename), 'rt') as handle:
                label_recorder = json.load(handle)

        # lr decay
        if args.mode == 'train':
            curr_lr = opt.param_groups[0]['lr']
            print(prev_lr, curr_lr)
            if curr_lr != prev_lr:
                prev_lr = curr_lr
                lr_steps.append(e)

        time_per_epoch_1 = time.time()
        losses_per_epoch = []
        tnmi = []

        for ct, (x, y, indices) in tqdm(enumerate(dl_tr)):
            it += 1
            x, y = x.cuda(), y.cuda()
            m = model(x)
            loss = criterion(m, indices, y)
            opt.zero_grad()
            loss.backward() # backprop
            torch.nn.utils.clip_grad_value_(model.parameters(), 10) # clip gradient?
            opt.step() # gradient descent

            losses_per_epoch.append(loss.data.cpu().numpy())

        time_per_epoch_2 = time.time()
        losses.append(np.mean(losses_per_epoch[-20:]))

        # save proxy-similarity and class labels
        if args.dynamic_proxy:
            train_embs, train_cls, *_ = predict_batchwise(model, dl_tr_noshuffle)
            cached_sim, cached_cls = inner_product_sim(X=train_embs, P=criterion.proxies, T=train_cls,
                                                       mask=criterion.mask,
                                                       nb_classes=criterion.nb_classes,
                                                       max_proxy_per_class=criterion.max_proxy_per_class)
            loss_recorder[e] = cached_sim.tolist()
            with open("{0}/{1}_ip.json".format('log', args.log_filename), 'wt') as handle:
                json.dump(loss_recorder, handle)
            label_recorder[e] = cached_cls.tolist()
            with open("{0}/{1}_cls.json".format('log', args.log_filename), 'wt') as handle:
                json.dump(label_recorder, handle)

        print('it: {}'.format(it))
        print(opt)
        logging.info(
            "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
                e,
                losses[-1],
                time_per_epoch_2 - time_per_epoch_1
            )
        )

        model.losses = losses
        model.current_epoch = e

        if e == best_epoch:
            break

        if args.mode == 'train':
            with torch.no_grad():
                logging.info("**Validation...**")
                nmi, recall = utils.evaluate(model, dl_val, args.eval_nmi, args.recall)

            chmean = (2 * nmi * recall[0]) / (nmi + recall[0])

            scheduler.step(chmean)

            if chmean > best_val_hmean:
                best_val_hmean = chmean
                best_val_nmi = nmi
                best_val_r1 = recall[0]
                best_val_r2 = recall[1]
                best_val_r4 = recall[2]
                best_val_r8 = recall[3]
                best_val_epoch = e
                best_tnmi = torch.Tensor(tnmi).mean()

            if e == (args.nb_epochs - 1):
                #saving last epoch
                results['last_NMI'] = nmi
                results['last_hmean'] = chmean
                results['best_epoch'] = best_val_epoch
                results['last_R1'] = recall[0]
                results['last_R2'] = recall[1]
                results['last_R4'] = recall[2]
                results['last_R8'] = recall[3]

                #saving best epoch
                results['best_NMI'] = best_val_nmi
                results['best_hmean'] = best_val_hmean
                results['best_R1'] = best_val_r1
                results['best_R2'] = best_val_r2
                results['best_R4'] = best_val_r4
                results['best_R8'] = best_val_r8

            logging.info('Best val epoch: %s', str(best_val_epoch))
            logging.info('Best val hmean: %s', str(best_val_hmean))
            logging.info('Best val nmi: %s', str(best_val_nmi))
            logging.info('Best val r1: %s', str(best_val_r1))
            logging.info(str(lr_steps))

        # add new proxy
        if args.dynamic_proxy:
            update_epoch_schedule = [int(x * args.nb_epochs) for x in args.proxy_update_schedule]
            if e in update_epoch_schedule:
                update, bad_indices = hard_potential(loss_recorder,
                                                     label_recorder,
                                                     current_t=e,
                                                     rolling_t=5,
                                                     ts_sim=config['ts_sim'], # larger ts_sim catches more hard examples
                                                     ts_ratio=config['ts_ratio'], # higher lower bound catches less hard examples
                                                     ) #FIXME: rolling_t=5 is ok, you need to adjust ts_ratio, ts_sim
                if update == True:
                    for k, v in bad_indices.items():
                        sampler = SubSampler(v)
                        tr_loader_temp = DataLoader(
                                            dataset=dataset.load(
                                                        name=args.dataset,
                                                        root=dataset_config['dataset'][args.dataset]['root'],
                                                        source=dataset_config['dataset'][args.dataset]['source'],
                                                        classes=dataset_config['dataset'][args.dataset]['classes']['trainval'],
                                                        transform=dataset.utils.make_transform(
                                                            **dataset_config[transform_key],
                                                            is_train=False
                                                      )
                                                   ),
                                            batch_size=64,
                                            shuffle=False,
                                            sampler=sampler,
                                            drop_last=False)
                        feature_emb = predict_batchwise(model, tr_loader_temp)[0]  # shape (N, nz_embedding)
                        centroid_emb = F.normalize(torch.mean(feature_emb, dim=0), p=2, dim=-1) # shape (nz_embedding,)
                        criterion.add_proxy(k, centroid_emb.to(criterion.proxies.device))
                        logging.info('Class {} update no. proxies to be {}'.format(k, criterion.current_proxy[k]))

        #TODO: this is for umap visualization -- save intermediate models and proxies
        save_dir = 'dvi_data_{}_{}_loss{}_proxy{}_tau{}/ResNet_{}_Model'.format(args.dataset,
                                                                              args.dynamic_proxy,
                                                                              str(config['criterion']['type']).split('.')[1],
                                                                              str(args.initial_proxy_num),
                                                                              str(args.tau),
                                                                              str(args.sz_embedding))
        os.makedirs('{}'.format(save_dir), exist_ok=True)
        os.makedirs('{}/Epoch_{}'.format(save_dir, e+1), exist_ok=True)
        with open('{}/Epoch_{}/index.json'.format(save_dir, e + 1), 'wt') as handle:
            handle.write(json.dumps(list(range(len(dl_tr_noshuffle.dataset)))))
        torch.save(model.state_dict(), '{}/Epoch_{}/{}_{}_{}_{}_{}.pth'.format(save_dir, e+1, args.dataset,
                                                                               args.dataset, args.mode,
                                                                               str(args.sz_embedding), str(args.seed)))
        # torch.save({"proxies": criterion.proxies, "mask": criterion.mask}, '{}/Epoch_{}/proxy.pth'.format(save_dir, e+1))
        # # TODO
        torch.save({"proxies": criterion.proxies}, '{}/Epoch_{}/proxy.pth'.format(save_dir, e+1))
        ######################################################################################

        if args.mode == 'trainval':
            scheduler.step(e) # adjust learning rate

    if args.mode == 'trainval':
        save_best_checkpoint(model)

        with torch.no_grad():
            logging.info("**Evaluating...**")
            model = load_best_checkpoint(model)
            if 'inshop' in args.dataset:
                best_test_nmi, (best_test_r1, best_test_r10, best_test_r20, best_test_r30, best_test_r40, best_test_r50) = utils.evaluate_inshop(model, dl_query, dl_gallery)
            else:
                best_test_nmi, (best_test_r1, best_test_r2, best_test_r4, best_test_r8) = utils.evaluate(model, dl_ev, args.eval_nmi, args.recall)
            #logging.info('Best test r8: %s', str(best_test_r8))
        if 'inshop' in args.dataset:
            results['NMI'] = best_test_nmi
            results['R1']  = best_test_r1
            results['R10'] = best_test_r10
            results['R20'] = best_test_r20
            results['R30'] = best_test_r30
            results['R40'] = best_test_r40
            results['R50'] = best_test_r50
        else:
            results['NMI'] = best_test_nmi
            results['R1'] = best_test_r1
            results['R2'] = best_test_r2
            results['R4'] = best_test_r4
            results['R8'] = best_test_r8

    if args.mode == 'train':
        print('lr_steps', lr_steps)
        results['lr_steps'] = lr_steps

    with open(out_results_fn,'w') as outfile:
        json.dump(results, outfile)

    t2 = time.time()
    logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
