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
from evaluation.faithfulness import *
import json
import pandas as pd
from datasets import load_dataset
from captum.attr import (
    Lime,
    KernelShap
)
import random

def set_random_seed(seed):
    torch.manual_seed(seed)   
    random.seed(seed)
    np.random.seed(seed)
    

print("FA ks")

n_samples = 8
LMs=["mistralai/Mistral-7B-v0.1",'meta-llama/Llama-2-7b-hf']

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
        if "gemma" in model_name:
            model = AutoModelForCausalLM.from_pretrained(path_unpruned_model, device_map=device)
        else:
            print("start loading")
            model = AutoModelForCausalLM.from_pretrained(path_unpruned_model, device_map=device, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, revision="float16")
            print("loaded")
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
    
    
    for num, i in enumerate(order_to_remove[:int(len(order_to_remove)*0.5)]):
        block_data = {
                        "layer_number":i,
                        "block_name": "self_attn",
                    }

        replace_block(model, block_data)
        print("Removed "+str(num+1)+" blocks")
        
        if not os.path.isfile("./results/pruned/"+ model_name+"_"+str(num)+'.json'):
        
            out = eval_zero_shot(model_name, model, tokenizer, device=device)
            print(out)
            
            tmp = "./results/pruned/"+"/".join(model_name.split("/")[:-1])
            if not os.path.exists(tmp):
                os.makedirs(tmp)
            with open("./results/pruned/"+ model_name+"_"+str(num)+'.json', 'w+') as f:
                json.dump(out["results"],f)
        
        tl_faith=["rte","boolq","arc_challenge","arc_easy","openbookqa"]
        
        for seed in [0,1,2]:
            
            
            
            if not os.path.isfile("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_fa_kernel_shap_'+str(seed)+'.json'):
                path_res_lm_eval = "./results/pruned/"+ model_name+"_"+str(num)+'.json'
                
                out_file = open("./results/samples/partitions.json", "r")
                sampled = json.load(out_file)
                out_file.close()
    
    
                set_random_seed(seed)
                out = compute_fa_seeds(model, tokenizer, path_res_lm_eval, sampled, num, fa_approach = KernelShap, task_list=tl_faith, device=device)

                tmp = "./results/pruned/seeds/"+"/".join(model_name.split("/")[:-1])
                if not os.path.exists(tmp):
                    os.makedirs(tmp)
                
                with open("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_fa_kernel_shap_'+str(seed)+'.json', 'w+') as f:
                    json.dump(out,f)
                    