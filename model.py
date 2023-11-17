import os
import sys
import time
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
#from otdd.pytorch.distance import DatasetDistance
from Otdd.pytorch.distance import DatasetDistance
from utils import get_text, accuracy, conv_init, create_position_ids_from_inputs_embeds, embedder_init, set_grad_state, get_optimizer_scheduler, embedder_placeholder, adaptive_pooler
from print_log import print_log, nvidia_smi_print
default_timer = time.perf_counter

def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)

    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    maxsamples = len(src_train_dataset)
    d = dist.distance(maxsamples)
    return d

class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim, target_seq_len, log_location):
        super().__init__()
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)
        self.projection = nn.Conv1d(input_shape[1], embed_dim, kernel_size=self.stack_num, stride=self.stack_num)
        self.log_location = log_location
        self.k = 0
        conv_init(self.projection)
        
    def get_stack_num(self, input_len, target_seq_len):
        if self.embed_dim == 768:
            for i in range(1, input_len + 1):
                if input_len % i == 0 and input_len // i <= target_seq_len:
                    break
            return i
        else:
            for i in range(1, input_len + 1):
                root = np.sqrt(input_len // i)
                if input_len % i == 0 and input_len // i <= target_seq_len and int(root + 0.5) ** 2 == (input_len // i):
                    break
            return i
        
    def forward(self, x=None, inputs_embeds=None):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        if x is None:
            x = inputs_embeds
        x = x.float()
        x = self.projection(x).transpose(1, 2)
        
        # import subprocess
        # # Define the shell command as a string
        # command = "nvidia-smi"
        # # Execute the command and capture its output
        # output = subprocess.check_output(command, shell=True, universal_newlines=True)
        # # Print the output
        # print_log(self.log_location, output)
        
        # print_log(self.log_location, f"{torch.cuda.memory_allocated()}")
        # print_log(self.log_location, f"{torch.cuda.memory_reserved()}")

        x = self.norm(x)
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps
        return x

class wrapper1D(torch.nn.Module):
    def __init__(self, modelname, input_shape, output_shape, target_seq_len=512, train_epoch=0, drop_out=None, log_location = None):
        super().__init__()
        self.modelname = modelname
        self.log_location = log_location
        self.output_shape = output_shape
        self.target_seq_len = target_seq_len
        #modelname = 'roberta-base'
        configuration = AutoConfig.from_pretrained(modelname)
        if drop_out is not None:
            configuration.hidden_dropout_prob = drop_out
            configuration.attention_probs_dropout_prob = drop_out
        self.model = AutoModel.from_pretrained(modelname, config = configuration)
        self.embedder = Embeddings1D(input_shape, embed_dim=768, target_seq_len=target_seq_len, log_location=log_location)
        embedder_init(self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
        set_grad_state(self.embedder, True)
        self.model.embeddings = embedder_placeholder()
        self.model.pooler = adaptive_pooler()
        self.predictor = nn.Linear(in_features=768, out_features=output_shape)
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)
        self.output_raw = True
        
    def forward(self, x):
        if self.output_raw:
            # print("11111")
            return self.embedder(x)
        x = self.embedder(x)
        x = self.model(inputs_embeds=x)
        x = x['pooler_output']
        #x = x['last_hidden_state']
        #x = F.softmax(x, dim=-1)
        x = self.predictor(x)
        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)
        return x

def get_tgt_model(args, sample_shape, num_classes, tgt_train_loader, log_location):
    if args.use_parallel:
        if torch.cuda.device_count() > 1:
            num = torch.cuda.device_count()
            device_list = list(range(num))
            device = "cuda:0"
            args.device = device
    modelname = args.model_name
    src_train_loader = get_text(args.path, args.batch_size)
    src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
    wrapper_func = wrapper1D
    tgt_model = wrapper_func(modelname, sample_shape, num_classes, target_seq_len=args.target_seq_len, train_epoch=args.embedder_epochs, drop_out=args.drop_out, log_location=log_location)
    tgt_model = torch.nn.DataParallel(tgt_model,device_ids=device_list)
    tgt_model = tgt_model.to(args.device).train()
    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()
    score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    total_losses, times, embedder_stats = [], [], []
    
    for ep in range(args.embedder_epochs):
        total_loss = 0
        time_start = default_timer()
        Loss = []
        #_feats = []
        # _feats_tensor = 
        # torch.zeros([32, 512, 768], dtype = torch.float32).to(args.device)
        # _
        #datanum = 0
        for i, data in enumerate(tgt_train_loader):
            x, y = data
            x = x.to(args.device)
            out = tgt_model(x)
            out_mean = out.mean(1).cpu()
            loss = score_func(out_mean)
            Loss.append(loss)

            #_feats.append(out_mean)
            #datanum += x.shape[0]
            #if datanum > args.maxsamples: break
            #del x, out, out_mean
            #torch.cuda.empty_cache()
            # nvidia_smi_print(log_location)
        # nvidia_smi_print(log_location)
        #feats = torch.cat(_feats, 0).to(args.device)
        #del _feats
        """ nvidia_smi_print(log_location)
        import sys
        sys.exit()
        torch.cuda.empty_cache() """
        
        """ if feats.shape[0] > 1:
            loss = score_func(feats)
            loss.backward()
            total_loss += loss.item() """
        final_loss = torch.stack(Loss, 0).mean(0)
        final_loss.backward()
        total_loss += final_loss.item()
        time_end = default_timer()
        times.append(time_end - time_start)
        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\totdd loss:", "%.4f" % total_losses[-1])
        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()
    torch.cuda.empty_cache()
    tgt_model.output_raw = False
    return tgt_model, embedder_stats

def train_one_epoch(args, model, optimizer, scheduler, loader, loss, temp, decoder=None, transform=None):

    model.train()            
    train_loss = 0
    optimizer.zero_grad()
    for i, data in enumerate(loader):
        
        if transform is not None:
            x, y, z = data
            z = z.to(args.device)
        else:
            x, y = data
        x, y = x.to(args.device), y.to(args.device)
        out = model(x)
        
        if isinstance(out, dict):
            out = out['out']
        
        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)
            
        if transform is not None:
            out = transform(out, z)
            y = transform(y, z)
            
        l = loss(out, y)
        l.backward()
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        if args.lr_sched_iter:
            scheduler.step()
            
        train_loss += l.item()
        if i >= temp - 1:
            break
        
    if (not args.lr_sched_iter):
        scheduler.step()
    torch.cuda.empty_cache()
    return train_loss / temp

def evaluate(num_classes, args, model, loader, loss, n_eval, topks, decoder=None, transform=None):
    model.eval()
    eval_loss, eval_score = 0, 0
    ys, outs, n_eval, n_data = [], [], 0, 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if transform is not None:
                x, y, z = data
                z = z.to(args.device)
            else:
                x, y = data
                                
            x, y = x.to(args.device), y.to(args.device)

            out = model(x)
            
            if isinstance(out, dict):
                out = out['out']
                
            if decoder is not None:
                out = decoder.decode(out).view(x.shape[0], -1)
                y = decoder.decode(y).view(x.shape[0], -1)
                                
            if transform is not None:
                out = transform(out, z)
                y = transform(y, z)
                
            outs.append(out)
            ys.append(y)
            n_data += x.shape[0]
            
            if n_data >= args.eval_batch_size or i == len(loader) - 1:
                outs = torch.cat(outs, 0)
                ys = torch.cat(ys, 0)

                
                eval_loss += loss(outs, ys).item()
                eval_score += accuracy(num_classes, outs, ys, topks)
                n_eval += 1

                ys, outs, n_data = [], [], 0

        eval_loss /= n_eval
        eval_score /= n_eval
        
    return eval_loss, eval_score