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
    
print("experiment_unpruned")
    
LMs=["mistralai/Mistral-7B-v0.1",'meta-llama/Llama-2-7b-hf']

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
        

        if not os.path.isfile("./results/unpruned/"+ model_name+'.json'):
        
            out = eval_zero_shot(model_name, model, tokenizer, device=device, log_res=True)
            
            out_preds = {"results":out['results'],"configs":out["configs"],'versions':out['versions'],'samples':out["samples"]}
            
            tmp = "./results/unpruned/"+"/".join(model_name.split("/")[:-1])
            if not os.path.exists(tmp):
                os.makedirs(tmp)
            with open("./results/unpruned/"+ model_name+'.json', 'w+') as f:
                json.dump(out_preds,f)
                
                
                
        tl_faith=["rte","boolq","arc_challenge","arc_easy","openbookqa"]
        out_file = open("./results/samples/partitions.json", "r")
        sampled = json.load(out_file)
        out_file.close()
        
        num = 200 # Not used anymore, remove from function

    
        for seed in [0,1,2]:
            
            if not os.path.isfile("./results/unpruned/seeds/"+ model_name+'_fa_fa_'+str(seed)+'.json'):
                path_res_lm_eval = "./results/unpruned/"+ model_name+'.json'
                set_random_seed(seed)
                out = compute_fa_seeds(model, tokenizer, path_res_lm_eval, sampled, num, fa_approach = Lime, task_list=tl_faith, device=device)
                tmp = "./results/unpruned/seeds/"+"/".join(model_name.split("/")[:-1])
                if not os.path.exists(tmp):
                    os.makedirs(tmp)    
                with open("./results/unpruned/seeds/"+ model_name+'_fa_fa_'+str(seed)+'.json', 'w+') as f:
                    json.dump(out,f)
            
            if not os.path.isfile("./results/unpruned/seeds/"+ model_name+'_fa_kernel_shap_'+str(seed)+'.json'):
                path_res_lm_eval = "./results/unpruned/"+ model_name+'.json'
                set_random_seed(seed)
                out = compute_fa_seeds(model, tokenizer, path_res_lm_eval, sampled, num, fa_approach = KernelShap, task_list=tl_faith, device=device)
                tmp = "./results/unpruned/seeds/"+"/".join(model_name.split("/")[:-1])
                if not os.path.exists(tmp):
                    os.makedirs(tmp)
                with open("./results/unpruned/seeds/"+ model_name+'_fa_kernel_shap_'+str(seed)+'.json', 'w+') as f:
                    json.dump(out,f)
            
            
            if not os.path.isfile("./results/unpruned/seeds/"+ model_name+'_lime_res_3d_'+str(seed)+'.json'):
                path_res_lm_eval = "./results/unpruned/"+ model_name+'.json'
                if os.path.isfile("./results/unpruned/seeds/"+ model_name+'_fa_fa_'+str(seed)+'.json'):
                    path_fa = "./results/unpruned/seeds/"+ model_name+'_fa_fa_'+str(seed)+'.json'
                    out = compute_comp_suff(model, tokenizer, path_res_lm_eval, path_fa, task_list=tl_faith, device=device)
                    tmp = "./results/unpruned/seeds/"+"/".join(model_name.split("/")[:-1])
                    if not os.path.exists(tmp):
                        os.makedirs(tmp)
                    with open("./results/unpruned/seeds/"+ model_name+'_lime_res_3d_'+str(seed)+'.json', 'w+') as f:
                        json.dump(out,f)
    
            if not os.path.isfile("./results/unpruned/seeds/"+ model_name+'_ks_res_3d_'+str(seed)+'.json'):
                path_res_lm_eval = "./results/unpruned/"+ model_name+'.json'   
                if os.path.isfile("./results/unpruned/seeds/"+ model_name+'_fa_kernel_shap_'+str(seed)+'.json'):
                    path_fa = "./results/unpruned/seeds/"+ model_name+'_fa_kernel_shap_'+str(seed)+'.json'
                    out = compute_comp_suff(model, tokenizer, path_res_lm_eval, path_fa, task_list=tl_faith, device=device)
                    tmp = "./results/unpruned/seeds/"+"/".join(model_name.split("/")[:-1])
                    if not os.path.exists(tmp):
                        os.makedirs(tmp)
                    with open("./results/unpruned/seeds/"+ model_name+'_ks_res_3d_'+str(seed)+'.json', 'w+') as f:
                        json.dump(out,f)
                                        
            
                    
                
                
                    
                    
            
        
            
            