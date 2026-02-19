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
    

n_samples = 8
LMs=[sys.argv[1]]
print(LMs)

if sys.argv[2] == "1":
    print("Lime")
else:
    print("Kernel shap")
                


device = torch.device("cpu")
if torch.cuda.is_available():
    print(torch.cuda.device_count())
    device = torch.device("cuda:0")
    print("using gpu")
else:
    print("using cpu")



for model_name in LMs:
    path_unpruned_model = "./saved_models/unpruned/" + model_name
    
    base_results_path = "./acts/activations/"+ model_name+"/"

    
    if os.path.exists(path_unpruned_model):
        tokenizer = AutoTokenizer.from_pretrained(path_unpruned_model)
        model = AutoModelForCausalLM.from_pretrained(path_unpruned_model, device_map=device, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16, revision="float16")      
        model.name = model_name
        
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
            set_random_seed(seed)
            print(sys.argv[2])
            
            if sys.argv[2] == "1":
                print("Lime")
        
                if not os.path.isfile("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_lime_res_3d_'+str(seed)+'.json'):
                    
                    path_res_lm_eval = "./results/pruned/"+ model_name+"_"+str(num)+'.json'
                    
                    
                    if os.path.isfile("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_fa_fa_'+str(seed)+'.json'):
                        
                        path_fa = "./results/pruned/seeds/"+ model_name+"_"+str(num)+'_fa_fa_'+str(seed)+'.json'

                    
                        out = compute_comp_suff(model, tokenizer, path_res_lm_eval, path_fa, task_list=tl_faith, device=device)
                        
                        tmp = "./results/pruned/seeds/"+"/".join(model_name.split("/")[:-1])
                        if not os.path.exists(tmp):
                            os.makedirs(tmp)
                        with open("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_lime_res_3d_'+str(seed)+'.json', 'w+') as f:
                            json.dump(out,f)
            else:
                print("Kernel Shap")
                
                
                if not os.path.isfile("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_ks_res_3d_'+str(seed)+'.json'):
                
                    path_res_lm_eval = "./results/pruned/"+ model_name+"_"+str(num)+'.json'
                    
                    
                    if os.path.isfile("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_fa_kernel_shap_'+str(seed)+'.json'):
                        
                        path_fa = "./results/pruned/seeds/"+ model_name+"_"+str(num)+'_fa_kernel_shap_'+str(seed)+'.json'

                    
                        out = compute_comp_suff(model, tokenizer, path_res_lm_eval, path_fa, task_list=tl_faith, device=device)

                        tmp = "./results/pruned/seeds/"+"/".join(model_name.split("/")[:-1])
                        if not os.path.exists(tmp):
                            os.makedirs(tmp)
                        with open("./results/pruned/seeds/"+ model_name+"_"+str(num)+'_ks_res_3d_'+str(seed)+'.json', 'w+') as f:
                            json.dump(out,f)
                
                    
                
                
                    
                    
            
        