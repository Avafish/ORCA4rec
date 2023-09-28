import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
from utils import get_data, get_optimizer_scheduler, count_params, count_trainable_params, save_state, load_state
from model import get_tgt_model, train_one_epoch, evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--model_name', default='roberta-base', type=str) # roberta-base gpt2
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--maxlen', default=512, type=int)
parser.add_argument('--dataset', default='wikipedia')
parser.add_argument('--path', default='/data0/longzewen/LM/ORCA4rec/datasets/')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--target_seq_len', default=512, type=int)
parser.add_argument('--drop_out', default=0.0, type=float)
parser.add_argument('--epochs', default=25, type=int)
parser.add_argument('--maxsamples', default=128, type=int)
parser.add_argument('--predictor_epochs', default=5, type=int)
parser.add_argument('--embedder_epochs', default=60, type=int)
parser.add_argument('--finetune_method', default='all', type=str)
parser.add_argument('--optimizer', default={'name':'AdamW','params':{'lr':2e-5,'betas':[0.9, 0.98],'weight_decay':0.000003,'momentum':0.9}})
parser.add_argument('--scheduler', default={'name':'WarmupLR','params':{'warmup_epochs':10,'decay_epochs':25,'sched':[20, 40, 60],'base':0.2}})
parser.add_argument('--no_warmup_scheduler', default={'name':'StepLR','params':{'warmup_epochs':10,'decay_epochs':60,'sched':[20, 40, 60],'base':0.2}})
parser.add_argument('--accum', default=1, type=int)
parser.add_argument('--clip', default=1, type=int)
parser.add_argument('--validation_freq', default=1, type=int)
parser.add_argument('--eval_batch_size', default=1024, type=int)
args = parser.parse_args()

default_timer = time.perf_counter
torch.cuda.empty_cache()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed) 
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
topks = [10]

loss = nn.CrossEntropyLoss()
sample_shape = (1, 1, args.maxlen)
num_classes, train_loader, valid_loader, test_loader = get_data(args)
n_train, n_val, n_test = len(train_loader), len(valid_loader), len(test_loader)
model, embedder_stats = get_tgt_model(args, sample_shape, num_classes, train_loader)
compare_metrics = np.max
decoder, transform = None, None
args, model, optimizer, scheduler = get_optimizer_scheduler(args, model, module=None if args.predictor_epochs == 0 else 'predictor', n_train=n_train)
train_full = args.predictor_epochs == 0
if args.device == 'cuda':
    model.cuda()
    try:
        loss.cuda()
    except:
        pass
    if decoder is not None:
        decoder.cuda()
        
print("\n------- Experiment Summary --------")
print("dataset:", args.dataset, "\tbatch size:", args.batch_size, "\tlr:", args.optimizer['params']['lr'])
print("num train batch:", n_train, "\tnum validation batch:", n_val, "\tnum test batch:", n_test)
print("finetune method:", args.finetune_method)
print("param count:", count_params(model), count_trainable_params(model))
print(model)

print("\n------- Start Training --------")
train_time, train_losses, train_score = [], [], []
for ep in range(args.epochs + args.predictor_epochs):
    time_start = default_timer()
    train_loss = train_one_epoch(args, model, optimizer, scheduler, train_loader, loss, n_train, decoder, transform)
    train_time_ep = default_timer() -  time_start
    
    if ep % args.validation_freq == 0 or ep == args.epochs + args.predictor_epochs - 1:
        val_loss, val_score = evaluate(num_classes, args, model, valid_loader, loss, n_val, topks, decoder, transform)
        train_losses.append(train_loss)
        train_score.append(val_score)
        train_time.append(train_time_ep)
        print("[train", "full" if ep >= args.predictor_epochs else "predictor", ep, "%.6f" % optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (train_time[-1]), "\ttrain loss:", "%.4f" % train_loss, "\tval loss:", "%.4f" % val_loss, "\tval score:", "%.4f" % val_score, "\tbest val score:", "%.4f" % compare_metrics(train_score))
        
        if compare_metrics(train_score) == val_score:
            id_current = save_state(args, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats)
            id_best = id_current
        
       
    if ep == args.epochs + args.predictor_epochs - 1:
        print("\n------- Start Test --------")
        test_scores = []
        test_model = model
        test_time_start = default_timer()
        test_loss, test_score = evaluate(num_classes, args, test_model, test_loader, loss, n_test, topks, decoder, transform)
        test_time_end = default_timer()
        test_scores.append(test_score)
        print("[test last]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
        
        test_model, _, _, _, _, _ = load_state(args, test_model, optimizer, scheduler, n_train, id_best, test=True)
        test_time_start = default_timer()
        test_loss, test_score = evaluate(num_classes, args, test_model, test_loader, loss, n_test, topks, decoder, transform)
        test_time_end = default_timer()
        test_scores.append(test_score)

        print("[test best-validated]", "\ttime elapsed:", "%.4f" % (test_time_end - test_time_start), "\ttest loss:", "%.4f" % test_loss, "\ttest score:", "%.4f" % test_score)
       
print(args)  