import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
from collections import defaultdict
from timm.models.layers import trunc_normal_
import copy
import math
import os
import random
import operator
from functools import reduce, partial

class embedder_placeholder(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is not None:
            return x

        return inputs_embeds
    
class adaptive_pooler(torch.nn.Module):
    def __init__(self, out_channel=1, output_shape=None, dense=False):
        super().__init__()
        self.pooler = nn.AdaptiveAvgPool1d(out_channel)
        self.out_channel = out_channel
        self.output_shape = output_shape
        self.dense = dense

    def forward(self, x):
        if len(x.shape) == 3:
            if self.out_channel == 1 and not self.dense:
                x = x.transpose(1, 2)
            pooled_output = self.pooler(x)
            if self.output_shape is not None:
                pooled_output = pooled_output.reshape(x.shape[0], *self.output_shape)
            else:
                pooled_output = pooled_output.reshape(x.shape[0], -1)
            
        else:
            b, c, h, w = x.shape
            x = x.reshape(b, c, -1)
            pooled_output = self.pooler(x.transpose(1, 2))
            pooled_output = pooled_output.transpose(1, 2).reshape(b, self.out_channel, h, w)
            if self.out_channel == 1:
                pooled_output = pooled_output.reshape(b, h, w)

        return pooled_output
    
def count_params(model):
    c = 0
    for p in model.parameters():
        try:
            c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c

def count_trainable_params(model):
    c = 0
    for p in model.parameters():
        try:
            if p.requires_grad:
                c += reduce(operator.mul, list(p.size()))
        except:
            pass

    return c

def set_param_grad(model, finetune_method):

    if finetune_method == "layernorm":
        for n, m in model.named_parameters():
            if 'layer' in n:
                if 'layernorm' in n or 'LayerNorm' in n:
                    continue
                else:
                    m.requires_grad = False

    elif finetune_method == "non-attn":
        for n, m in model.named_parameters():
            if 'layer' in n:
                if 'query' in n or 'key' in n or 'value' in n:
                    m.requires_grad = False

def get_optimizer(name, params):
    if name == 'SGD':
        return partial(torch.optim.SGD, lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    elif name == 'Adam':
        return partial(torch.optim.Adam, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])
    elif name == 'AdamW':
        return partial(torch.optim.AdamW, lr=params['lr'], betas=tuple(params['betas']), weight_decay=params['weight_decay'])

def get_scheduler(name, params, epochs=200, n_train=None):
    if name == 'StepLR':
        sched = params['sched']

        def scheduler(epoch):    
            optim_factor = 0
            for i in range(len(sched)):
                if epoch > sched[len(sched) - 1 - i]:
                    optim_factor = len(sched) - i
                    break
                    
            return math.pow(params['base'], optim_factor)  

        lr_sched_iter = False

    elif name == 'WarmupLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return f  

    elif name == 'ExpLR':
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))

            current_decay_steps = total_steps - step
            total_decay_steps = total_steps - warmup_steps
            f = (current_decay_steps / total_decay_steps)

            return params['base'] * f  

    elif name == 'SinLR':

        cycles = 0.5
        warmup_steps = int(params['warmup_epochs'] * n_train)
        total_steps = int(params['decay_epochs'] * n_train)
        lr_sched_iter = True

        def scheduler(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            # progress after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1. + math.cos(math.pi * float(cycles) * 2.0 * progress)))

    return scheduler, lr_sched_iter

def get_params_to_update(model, finetune_method):

    params_to_update = []
    name_list = ''
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            name_list += "\t" + name

    print("Params to learn:", name_list)
    
    return params_to_update

def get_optimizer_scheduler(args, model, module=None, n_train=1):
    if module is None:
        set_grad_state(model, True)
        set_param_grad(model, args.finetune_method)
        optimizer = get_optimizer(args.optimizer['name'], args.optimizer['params'])(get_params_to_update(model, ""))
        lr_lambda, args.lr_sched_iter = get_scheduler(args.scheduler['name'], args.scheduler['params'], args.epochs, n_train)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return args, model, optimizer, scheduler

    elif module == 'embedder':
        embedder_optimizer_params = copy.deepcopy(args.optimizer['params'])
        if embedder_optimizer_params['lr'] <= 0.001:
            embedder_optimizer_params['lr'] = 0.01
        embedder_optimizer = get_optimizer(args.optimizer['name'], embedder_optimizer_params)(get_params_to_update(model, ""))
        lr_lambda, _ = get_scheduler(args.no_warmup_scheduler['name'], args.no_warmup_scheduler['params'], args.embedder_epochs, 1)
        embedder_scheduler = torch.optim.lr_scheduler.LambdaLR(embedder_optimizer, lr_lambda=lr_lambda)

        return args, model, embedder_optimizer, embedder_scheduler

    elif module == 'predictor':

        try:
            predictor = model.predictor
            set_grad_state(model, False)
            for n, m in model.embedder.named_parameters():
                m.requires_grad = True
            for n, m in model.predictor.named_parameters():
                m.requires_grad = True

            predictor_optimizer_params = copy.deepcopy(args.optimizer['params'])
            if predictor_optimizer_params['lr'] <= 0.001:
                predictor_optimizer_params['lr'] = 0.01
            predictor_optimizer = get_optimizer(args.optimizer['name'], predictor_optimizer_params)(get_params_to_update(model, ""))
            lr_lambda, args.lr_sched_iter = get_scheduler(args.no_warmup_scheduler['name'], args.no_warmup_scheduler['params'], args.predictor_epochs, 1)
            predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(predictor_optimizer, lr_lambda=lr_lambda)

            return args, model, predictor_optimizer, predictor_scheduler
        except:
            print("No predictor module.")

def set_grad_state(module, state):
    for n, m in module.named_modules():
        if len(n) == 0: continue
        if not state and 'position' in n: continue
        if not state and 'tunable' in n: continue
        for param in m.parameters():
            param.requires_grad = state

def embedder_init(source, target, train_embedder=False, match_stats=False):
    if train_embedder:
        if hasattr(source, 'patch_embeddings'):
            if match_stats:
                weight_mean, weight_std = source.patch_embeddings.projection.weight.mean(), source.patch_embeddings.projection.weight.std()
                nn.init.normal_(target.projection.weight, weight_mean, weight_std)
                
                bias_mean, bias_std = source.patch_embeddings.projection.bias.mean(), source.patch_embeddings.projection.bias.std()
                nn.init.normal_(target.projection.bias, bias_mean, bias_std)
            else:
                rep_num = target.projection.in_channels // source.patch_embeddings.projection.in_channels + 1
                rep_weight = torch.cat([source.patch_embeddings.projection.weight.data] * rep_num, 1)
                
                target.projection.weight.data.copy_(rep_weight[:, :target.projection.in_channels, :target.projection.kernel_size[0], :target.projection.kernel_size[1]])        
                target.projection.bias.data.copy_(source.patch_embeddings.projection.bias.data)

            target.norm.weight.data.copy_(source.norm.weight.data)
            target.norm.bias.data.copy_(source.norm.bias.data)

        else:
            target.norm.weight.data.copy_(source.LayerNorm.weight.data)
            target.norm.bias.data.copy_(source.LayerNorm.bias.data)
     
            target.position_embeddings = copy.deepcopy(source.position_embeddings)

    else:
        for n, m in target.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)  

        try:
            target.position_embeddings = copy.deepcopy(source.position_embeddings)
        except:
            pass

def create_position_ids_from_inputs_embeds(inputs_embeds, padding_idx=1):
    input_shape = inputs_embeds.size()[:-1]
    sequence_length = input_shape[1]

    position_ids = torch.arange(padding_idx + 1, sequence_length + padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
    return position_ids.unsqueeze(0).expand(input_shape)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def get_data(args):
    path = args.path
    dataset = args.dataset + '.txt'
    maxlen = args.maxlen
    batch_size = args.batch_size
    f = open(path + dataset, 'r')
    User = defaultdict(list)
    #usernum = 0
    itemnum = 0
    user_train_x, user_valid_x, user_test_x, user_train_y, user_valid_y, user_test_y = [], [], [], [], [], []
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)-1
        #usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    for u in User:
        seqlen = len(User[u])
        if seqlen > 3:
            user_train_x.append(User[u][:-3])
            user_train_y.append(User[u][-3])
            user_valid_x.append(User[u][:-2])
            user_valid_y.append(User[u][-2])
            user_test_x.append(User[u][:-1])
            user_test_y.append(User[u][-1])
        else:
            user_train_x.append(User[u])
            user_train_y.append(User[u][-1])
            user_valid_x.append(User[u])
            user_valid_y.append(User[u][-1])
            user_test_x.append(User[u])
            user_test_y.append(User[u][-1])
    for data in [user_train_x, user_valid_x, user_test_x]:
        for i,t in enumerate(data):
            if len(t) < maxlen:
                t.reverse()
                while len(t) < maxlen:
                    t.append(-1)
                t.reverse()
            else:
                data[i] = t[-maxlen:]
        data = np.array(data)
    user_train_x = torch.from_numpy(np.expand_dims(user_train_x, 1))
    user_valid_x = torch.from_numpy(np.expand_dims(user_valid_x, 1))
    user_test_x = torch.from_numpy(np.expand_dims(user_test_x, 1))
    user_train_y = torch.tensor(user_train_y)
    user_valid_y = torch.tensor(user_valid_y)
    user_test_y = torch.tensor(user_test_y)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(user_train_x, user_train_y), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(user_valid_x, user_valid_y), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(user_test_x, user_test_y), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return itemnum+1, train_loader, valid_loader, test_loader

def get_text(path, batch_size):
    train_data = np.load(os.path.join(path, 'text_xs_32.npy'), allow_pickle =True)
    train_labels = np.load(os.path.join(path, 'text_ys_32.npy'), allow_pickle =True)
    maxsize = len(train_data)
    train_data = torch.from_numpy(train_data[:maxsize]).float()#.mean(1)
    train_labels = torch.from_numpy(train_labels[:maxsize]).long()
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader

def find_negs(outputs, i, t, num_classes):
    negs, final, temp = [], [], []
    negs.append(t.item())
    for _ in range(100):
        neg = np.random.randint(0, num_classes)
        while neg in negs:
            neg = np.random.randint(0, num_classes)
        negs.append(neg)
    random.shuffle(negs)
    output = outputs[i].tolist()
    for n in negs:
        idx = output.index(n)
        temp.append(idx)
    temp.sort()
    for tmp in temp:
        final.append(output[tmp])
    return final
        
def accuracy(num_classes, outputs, target, topk):
    hr = 0
    maxk = max(topk)
    outputs = outputs.topk(outputs.size(1), 1, True, True).indices # 1024,1000
    for i, t in enumerate(target):
        final = find_negs(outputs, i, t, num_classes)
        if t in final[:maxk]:
            hr += 1
    batch_size = target.size(0)
    hr = hr / batch_size
    return hr

def save_with_path(ep, path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats):
    np.save(os.path.join(path, 'ep-' + str(ep) + '-hparams.npy'), args)
    np.save(os.path.join(path, 'train_score.npy'), train_score)
    np.save(os.path.join(path, 'train_losses.npy'), train_losses)
    np.save(os.path.join(path, 'embedder_stats.npy'), embedder_stats)

    model_state_dict = {
                'network_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
    torch.save(model_state_dict, os.path.join(path, 'state_dict.pt'))

    rng_state_dict = {
                'cpu_rng_state': torch.get_rng_state(),
                'gpu_rng_state': torch.cuda.get_rng_state(),
                'numpy_rng_state': np.random.get_state(),
                'py_rng_state': random.getstate()
            }
    torch.save(rng_state_dict, os.path.join(path, 'rng_state.ckpt'))

def save_state(args, model, optimizer, scheduler, ep, n_train, train_score, train_losses, embedder_stats):
    path = 'results/' + args.dataset + '/maxlen' + str(args.maxlen) + "-target_seq_len" + str(args.target_seq_len) + "-embedder_epochs" + str(args.embedder_epochs)
    if not os.path.exists(path):
        os.makedirs(path)
    save_with_path(ep, path, args, model, optimizer, scheduler, train_score, train_losses, embedder_stats)
    return ep
    
def load_state(args, model, optimizer, scheduler, n_train, id_best, test=True):
    path = 'results/' + args.dataset +'/maxlen' + str(args.maxlen) + "-target_seq_len" + str(args.target_seq_len) + "-embedder_epochs" + str(args.embedder_epochs)
    if not os.path.isfile(os.path.join(path, 'state_dict.pt')):
        return model, 0, 0, [], [], None
    train_score = np.load(os.path.join(path, 'train_score.npy'))
    train_losses = np.load(os.path.join(path, 'train_losses.npy'))
    embedder_stats = np.load(os.path.join(path, 'embedder_stats.npy'))
    epochs = len(train_score)
    checkpoint_id = epochs - 1
    model_state_dict = torch.load(os.path.join(path, 'state_dict.pt'))
    model.load_state_dict(model_state_dict['network_state_dict'])
    
    return model, epochs, checkpoint_id, list(train_score), list(train_losses), embedder_stats