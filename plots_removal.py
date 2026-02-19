from track_block_activations import get_acts
from block_drop import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import numpy as np
import torch.nn as nn
from numpy.linalg import norm
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from evaluation.effectiveness import *
import json
import pandas as pd
from datasets import load_dataset
    
n_samples = 8
LMs=['meta-llama/Llama-2-7b-hf',"mistralai/Mistral-7B-v0.1"]

device = torch.device("cpu")
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    device = torch.device("cuda:0")
    print("using gpu")
else:
    print("using cpu")



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


for model_name in LMs:
    path_unpruned_model = "./saved_models/unpruned/" + model_name
    
    base_results_path = "./acts/activations/"+ model_name+"/"

    
    if os.path.exists(path_unpruned_model):
        tokenizer = AutoTokenizer.from_pretrained(path_unpruned_model)
        model = AutoModelForCausalLM.from_pretrained(path_unpruned_model, device_map=device)
        model.name = model_name
        
    
    if not os.path.isfile(base_results_path+"removal_list.npy"): 
    
        get_acts(model, tokenizer, n_samples=n_samples, save_path=base_results_path, device=device)
        blocks_info = retrieve_blocks(model, ["self_attn"], base_results_path)
        
        scores = get_blocks_scores_online(model, cosine, block_types=["self_attn"], acts_folder=base_results_path)
        
        ys = [1-out for out in scores]
        order_to_remove = np.argsort(ys)
        
        np.save(base_results_path+"removal_list.npy", order_to_remove)
        np.save(base_results_path+"final_scores.npy", scores)
    else:
        order_to_remove = np.load(base_results_path+"removal_list.npy")
        scores = np.load(base_results_path+"final_scores.npy")
        
        ys = [1-out for out in scores]
    
    
    tmp = "./results/pruned/"+"/".join(model_name.split("/")[:-1])
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    
    
    plt.figure()
    plt.plot(ys, lw=2.0)
    plt.grid()
    plt.xlabel("Layer")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig("./results/pruned/"+ model_name+"_scores.pdf", dpi=600, bbox_inches="tight")
    
    
    
    xs = list(range(len(scores)))
    

    plt.figure(figsize=(10, 2))
    cmap = plt.get_cmap('CMRmap_r')
    labels = ["C4"]
    df_cm = pd.DataFrame([ys], index = labels)

    sns.heatmap(df_cm, cmap=truncate_colormap(cmap, 0.2, 0.9), vmin=0, vmax=0.3)
    plt.xlabel("Layer")
    plt.ylabel("Dataset")

    plt.tight_layout()
    plt.savefig("./results/pruned/"+ model_name+"_scores_heatmap.pdf", dpi=600, bbox_inches="tight")
    
    