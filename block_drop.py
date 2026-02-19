import torch.nn as nn
import numpy as np
from numpy.linalg import norm
import os
import torch.nn.functional as F
import torch

class Identity_att(nn.Module):
    def __init__(self,layer_idx):
        super(Identity_att, self).__init__()
        self.layer_idx=layer_idx
        
    def forward(self, **args):
        args["past_key_value"].update(torch.zeros(2,2,2), torch.zeros(2,2,2), self.layer_idx, None)
        return 0, None, args["past_key_value"]


class Identity_mlp(nn.Module):
    def __init__(self,layer_idx):
        super(Identity_mlp, self).__init__()
        self.layer_idx=layer_idx
        
    def forward(self, *args):
        return 0

def replace_block(model, block_info):
    if block_info["block_name"] == "self_attn":
        setattr(model.model.layers[block_info["layer_number"]],block_info["block_name"], Identity_att(block_info["layer_number"]))
    else:
        setattr(model.model.layers[block_info["layer_number"]],block_info["block_name"], Identity_mlp(block_info["layer_number"]))



def retrieve_blocks(model, block_types=["self_attn","mlp"], acts_folder=None):
    blocks = []
    for i, layer in enumerate(list(model.model.layers)):
        
        for block_type in block_types:
            block_data = {
                "layer_number":i,
                "block_name": block_type,
                "input": None,
                "output": None
            }
            if acts_folder is not None:
                if block_type == "self_attn":
                    block_data["input"] = np.load(acts_folder+str(i)+"_input_layernorm_inputs.npy")
                    block_data["output"] = np.load(acts_folder+str(i)+"_post_att_inputs.npy")
                elif block_type == "mlp":
                    block_data["input"] = np.load(acts_folder+str(i)+"_post_att_inputs.npy")
                    block_data["output"] = block_data["input"] + np.load(acts_folder+str(i)+"_mlp_outputs.npy")               
        
            blocks.append((getattr(layer,block_type), block_data))

    return blocks





def get_blocks_scores_online(model, scoring, block_types=["self_attn","mlp"], acts_folder=None):
    results = []
    for i, layer in enumerate(list(model.model.layers)):
        
        for block_type in block_types:
            block_data = {
                "layer_number":i,
                "block_name": block_type,
                "input": None,
                "output": None
            }
            if acts_folder is not None:
                if block_type == "self_attn":
                    block_data["input"] = np.load(acts_folder+str(i)+"_input_layernorm_inputs.npy")
                    block_data["output"] = block_data["input"] + np.load(acts_folder+str(i)+"_self_attn_outputs.npy")
                elif block_type == "mlp":
                    block_data["input"] = np.load(acts_folder+str(i)+"_post_att_inputs.npy")
                    block_data["output"] = block_data["input"] + np.load(acts_folder+str(i)+"_mlp_outputs.npy")               
        
            results.append(get_block_score((getattr(layer,block_type), block_data),cosine,acts_folder))
            

    return results


def cosine_similarity(A, B):
    cos_sim = F.cosine_similarity(torch.tensor(A), torch.tensor(B), dim=-1)
    return cos_sim.mean().item()


def cosine(block, filepath = None):
    if os.path.isfile(filepath+"scores/"+str(block[1]["layer_number"])+"_"+block[1]["block_name"]+".npy"):
        return np.load(filepath+"scores/"+str(block[1]["layer_number"])+"_"+block[1]["block_name"]+".npy")
    
    cs = cosine_similarity(block[1]["input"], block[1]["output"])
    
    if not os.path.exists(filepath+"scores/"):
        os.makedirs(filepath+"scores/")
                
    np.save(filepath+"scores/"+str(block[1]["layer_number"])+"_"+block[1]["block_name"]+".npy",cs)
    return cs


def get_block_score(block, scoring, cache_path=None):
    return scoring(block, cache_path)

    
