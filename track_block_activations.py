import torch
import torch.nn as nn
from data import get_loaders 
import numpy as np
import os

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def get_model_specific_variables(model):
    seqlen = 2048
    hidden_size = 0
    
    if model.name in ['meta-llama/Llama-2-7b-hf']:
        seqlen = model.config.max_position_embeddings
        hidden_size = model.config.hidden_size
        layers = model.model.layers
    elif model.name in  ["mistralai/Mistral-7B-v0.1"]:
        seqlen = model.config.max_position_embeddings
        hidden_size = model.config.hidden_size
        layers = model.model.layers
    seqlen = min(seqlen, 2048)
    return seqlen, hidden_size, layers


def find_layers(model, block_types=["attention", "MLP"]):
    blocks = []
    for name, bl in model.named_modules():
        if name in block_types:
            bl.name = name
            blocks.append(bl)
    return blocks


def prepare_calibration_input(model, dataloader, device):
    
    seqlen, hidden_size, layers = get_model_specific_variables(model)

    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    cache = {'inputs': [], 'attention_mask': [], "position_ids": [], "position_ids": [], "cache_position": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):       
            cache["inputs"].append(inp)
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_ids'].append(kwargs['position_ids'])
            cache['cache_position'].append(kwargs['cache_position'] if 'cache_position' in kwargs else None)
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = [None] * len(cache['inputs'])

    model.config.use_cache = use_cache

    return cache["inputs"], outs, cache['attention_mask'], cache['position_ids'], cache['cache_position']


class Wrapper:
    def __init__(self, layer, layer_name="none"):
        self.layer = layer
        self.stored = {"input": [], "output": []} 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if inp is not None:
            self.stored["input"].append(inp.squeeze(0).clone().cpu())
        self.stored["output"].append(out.squeeze(0).clone().cpu())

       
def get_acts(model, tokenizer, block_types=["self_attn","mlp"], seed=0, n_samples=8, save_path="", device=torch.device("cuda:0"), verbose=False):
    if not os.path.exists(save_path):
        
        use_cache = model.config.use_cache 
        model.config.use_cache = False 

        seqlen, hidden_size, layers = get_model_specific_variables(model)

        dataloader, _ = get_loaders("c4_fast",nsamples=n_samples,seed=seed,seqlen=seqlen,tokenizer=tokenizer)
        with torch.no_grad():
            inps, outs, attention_mask, position_ids, cache_position = prepare_calibration_input(model, dataloader, device)

        for i in range(len(layers)):
            if verbose:
                print("Starting layer: "+str(i))
                
            layer = layers[i]
            
            subset = {str(i)+"_"+bt: getattr(layer,bt) for bt in block_types}
            if "mlp" in block_types:
                subset[str(i)+"_post_att"] = layer.post_attention_layernorm
            if "self_attn" in block_types:
                subset[str(i)+"_input_layernorm"] = layer.input_layernorm

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = Wrapper(subset[name], name)

            def add_batch_func(name):
                
                def tmp2(_, inp, out):
                        wrapped_layers[name].add_batch(inp[0].data, out[0].data)
                to_return = tmp2
                
                
                if "self_attn" in name:
                    def tmp(_, inp, out):
                        wrapped_layers[name].add_batch(None, out[0].data)
                    to_return = tmp
                    
                    
                return to_return

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch_func(name)))
                
            for j in range(n_samples):
                if verbose:
                    print("Sample number "+str(j+1))
                    
                with torch.no_grad():
                    outs[j] = layer(inps[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
                    
            for h in handles:
                h.remove()

            for name in subset:
                
                path_to_save = save_path+name
                
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                acts_inp = np.array(wrapped_layers[name].stored["input"])
                acts_out = np.array(wrapped_layers[name].stored["output"])
                
                np.save(path_to_save+"_inputs.npy",acts_inp)
                np.save(path_to_save+"_outputs.npy",acts_out)
                
            inps, outs = outs, inps

        model.config.use_cache = use_cache

        torch.cuda.empty_cache()

