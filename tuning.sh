#!/bin/bash
data=$1
resultpath="tuning_results/""/$data""/"
mkdir -p "$resultpath"
python main.py --device "cuda:2" --dataset "$data" --embedder_epochs "0" --finetune_method "all" &> "$resultpath"0_all.txt &
python main.py --device "cuda:2" --dataset "$data" --embedder_epochs "60" --finetune_method "all" &> "$resultpath"60_all.txt &
python main.py --device "cuda:3" --dataset "$data" --embedder_epochs "0" --finetune_method "layernorm" &> "$resultpath"0_ln.txt &
python main.py --device "cuda:3" --dataset "$data" --embedder_epochs "60" --finetune_method "layernorm" &> "$resultpath"60_ln.txt
done