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
from utils import find_p_n, evl, cross_entropy_loss, negs, find_negs, get_text, accuracy, conv_init, create_position_ids_from_inputs_embeds, embedder_init, set_grad_state, get_optimizer_scheduler, embedder_placeholder, adaptive_pooler
default_timer = time.perf_counter

def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)

    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    maxsamples = len(src_train_dataset)
    d = dist.distance(maxsamples)
    return d

""" class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim, target_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)
        self.projection = nn.Conv1d(input_shape[1], embed_dim, kernel_size=self.stack_num, stride=self.stack_num)
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
        if x.shape[1] == 101:
            x = x.float()
            x = x.unsqueeze(1)
            x = self.projection(x).transpose(1, 2)
            x = F.softmax(x, dim=-1)
            #x = self.norm(x)
            return x
        x = x.float()
        x = self.projection(x).transpose(1, 2)
        x = self.norm(x)
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps
        return x """
        
class Embeddings1D(nn.Module):
    def __init__(self, num_labels, embed_dim=768, target_seq_len=512):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.padding_idx = 0
        self.embeddings = nn.Embedding(num_labels+1, embed_dim, padding_idx=self.padding_idx)
        self.norm = nn.LayerNorm(embed_dim)
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx) 
        
    def forward(self, x=None, inputs_embeds=None, emb_only=False):
        if x is None:
            x = inputs_embeds
        if (x.shape[1] == 101) | emb_only:
            x = self.embeddings(x)
            #x = F.softmax(x, dim=-1)
            x = self.norm(x)
            return x
        x = x.squeeze(1)
        x = self.embeddings(x)
        x = self.norm(x)
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps
        return x

class wrapper1D(torch.nn.Module):
    def __init__(self, args, modelname, input_shape, output_shape, target_seq_len=512, train_epoch=0, drop_out=None):
        super().__init__()
        self.args = args
        self.modelname = modelname
        self.output_shape = output_shape
        self.target_seq_len = target_seq_len
        #modelname = 'roberta-base'
        configuration = AutoConfig.from_pretrained(modelname)
        if drop_out is not None:
            configuration.hidden_dropout_prob = drop_out
            configuration.attention_probs_dropout_prob = drop_out
        self.model = AutoModel.from_pretrained(modelname, config = configuration)
        #self.embedder = Embeddings1D(input_shape, embed_dim=768, target_seq_len=target_seq_len)
        self.embedder = Embeddings1D(output_shape, embed_dim=768, target_seq_len=target_seq_len)
        embedder_init(self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
        set_grad_state(self.embedder, True)
        self.model.embeddings = embedder_placeholder()
        self.model.pooler = adaptive_pooler()
        self.predictor = nn.Linear(in_features=768, out_features=output_shape)
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)
        self.output_raw = True
        
    def forward(self, x):
        if self.output_raw | (x.shape[1]==101):
            return self.embedder(x)
        x = self.embedder(x)
        x = self.model(inputs_embeds=x)
        x = x['last_hidden_state'] # batch_size, maxlen, 768
        #x = x['pooler_output'] # batch_size, 768
        #x = F.softmax(x, dim=-1)
        if self.args.use_predictor:
            x = self.predictor(x)
        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)
        return x

def get_tgt_model(args, sample_shape, num_classes, tgt_train_loader):
    if args.use_parallel:
        if torch.cuda.device_count() > 1:
            device_list = [0]
            device = "cuda:0"
            args.device = device
    modelname = args.model_name
    src_train_loader = get_text(args.path, args.batch_size)
    src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
    src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
    wrapper_func = wrapper1D
    tgt_model = wrapper_func(args, modelname, sample_shape, num_classes, target_seq_len=args.target_seq_len, train_epoch=args.embedder_epochs, drop_out=args.drop_out)
    if args.use_parallel:
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
        for i, data in enumerate(tgt_train_loader):
            x, y = data
            x = x.to(args.device)
            out = tgt_model(x)
            out_mean = out.mean(1).cpu() # bs, 768
            loss = score_func(out_mean)
            Loss.append(loss)
        
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

def train_one_epoch(num_classes, args, model, optimizer, scheduler, loader, loss, temp, decoder=None, transform=None):

    model.train()            
    train_loss = 0
    optimizer.zero_grad()
    #pos_weight = torch.ones([101])
    #pos_weight[0] = 100
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    for i, data in enumerate(loader):
        
        x, y = data
        x, y = x.to(args.device), y.to(args.device)
        if args.use_predictor:
            out = model(x)
            batch_loss = loss(out, y)
        else:
            pos, neg = find_p_n(x, y, num_classes)
            pos, neg = pos.to(args.device), neg.to(args.device)
            pos_emb = model.embedder(pos, emb_only=True)
            neg_emb = model.embedder(neg, emb_only=True)
            out = model(x) # batch_size, (1, 768) | num_labels
            pos_logits = (out * pos_emb).sum(dim=-1)
            neg_logits = (out * neg_emb).sum(dim=-1)
            pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
            batch_loss = 0
            indices = torch.where(pos != 0)
            batch_loss += bce_criterion(pos_logits[indices], pos_labels[indices])
            batch_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
        
        """ if isinstance(out, dict):
            out = out['out']
        
        if decoder is not None:
            out = decoder.decode(out).view(x.shape[0], -1)
            y = decoder.decode(y).view(x.shape[0], -1)
            
        if transform is not None:
            out = transform(out, z)
            y = transform(y, z) """
            
        batch_loss.backward()
        
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        if (i + 1) % args.accum == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        if args.lr_sched_iter:
            scheduler.step()
            
        train_loss += batch_loss.item()
        if i >= temp - 1:
            break
        
    if (not args.lr_sched_iter):
        scheduler.step()
    #torch.cuda.empty_cache()
    return train_loss / temp

def evaluate(num_classes, args, model, loader, loss, n_eval, topks, decoder=None, transform=None):
    model.eval()
    hr, ndcg, num_eval = 0, 0, 0
    eval_score, eval_loss, eval_hr, eval_ndcg, n_eval, n_data = 0, 0, 0, 0, 0, 0
    ys, outs, ys_outs = [], [], []
    with torch.no_grad():
        for i, data in enumerate(loader):
            
            x, y = data
                                
            if args.use_predictor:
                x, y = x.to(args.device), y.to(args.device)
                out = model(x)
            else:
                nys = negs(y, num_classes, 100)
                x, y, nys = x.to(args.device), y.to(args.device), nys.to(args.device)

                ys_out = model(nys).transpose(1, 2) # batch_size, 768, 101
                out = model(x)[:, -1, :].unsqueeze(1) # batch_size, (1, 768) | num_labels
                ys_outs.append(ys_out)
            
            """ if isinstance(out, dict):
                out = out['out']
                
            if decoder is not None:
                out = decoder.decode(out).view(x.shape[0], -1)
                y = decoder.decode(y).view(x.shape[0], -1)
                                
            if transform is not None:
                out = transform(out, z)
                y = transform(y, z) """
                
            outs.append(out)
            ys.append(y)
            n_data += x.shape[0]
            
            if n_data >= args.eval_batch_size or i == len(loader) - 1:
                #default_timer = time.perf_counter
                #time_start = default_timer()
                outs = torch.cat(outs, 0)
                ys = torch.cat(ys, 0)
                if ys_outs:
                    ys_outs = torch.cat(ys_outs, 0)

                if args.use_predictor:
                    eval_loss += loss(outs, ys).item()
                    eval_score, eval_size = accuracy(num_classes, outs, ys, topks)
                    hr += eval_score
                    num_eval += eval_size
                    n_eval += 1
                else:
                    logits = torch.matmul(outs, ys_outs).squeeze(1)
                    logits = F.softmax(logits, dim=-1)
                    #eval_loss += loss(outs, ys).item()
                    #eval_score += accuracy(num_classes, outs, ys, topks)
                    eval_hr, eval_ndcg, eval_size = evl(logits) # hr , batch_size
                    hr += eval_hr
                    ndcg += eval_ndcg
                    num_eval += eval_size
                    n_eval += 1

                #ti = default_timer() - time_start
                #print(ti)
                ys, outs, ys_outs, n_data = [], [], [], 0

        eval_loss = 0
        eval_loss /= n_eval
        #eval_score /= n_eval
        eval_hr = hr / num_eval
        eval_ndcg = ndcg / num_eval
        
    return eval_loss, eval_hr, eval_ndcg