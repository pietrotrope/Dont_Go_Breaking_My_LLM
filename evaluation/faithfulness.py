import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from captum.attr import (
    KernelShap,
    Lime,    
    TextTokenInput, 
    TextTemplateInput,
    LLMAttribution
)
import json
import numpy as np
import math

warnings.filterwarnings("ignore", ".*past_key_values.*")
warnings.filterwarnings("ignore", ".*Skipping this token.*")



def get_attribution(model, tokenizer, full_templ, vals, target, fa_approach):
    fa = fa_approach(model)
    llm_attr = LLMAttribution(fa, tokenizer)

    inp = TextTemplateInput(
        template=full_templ, 
        values=vals
    )
    attr_res = llm_attr.attribute(inp, target=target)
    return attr_res

def get_ranking(attr_res):
    ranking = np.argsort(-abs(attr_res.seq_attr.cpu()), )
    rs = list(range(len(attr_res.input_tokens)))
    
    num = []
    ranked = []
    for k in ranking:
        ranked.append(attr_res.input_tokens[k])
        num.append(rs[k])
    return ranked, num





def get_loglikelihood(model, tokenizer, prompt, continuation, device="cpu"):
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    continuation_ids = tokenizer(continuation, return_tensors="pt").input_ids[:,1:]

    full_input_ids = torch.cat([input_ids, continuation_ids], dim=-1).to(device)

    with torch.no_grad():
        logits = model(full_input_ids).logits.cpu()
  
    continuation_start_idx = input_ids.shape[-1]-1
    continuation_logits = logits[:, continuation_start_idx:, :]

   
    continuation_ids = continuation_ids.squeeze(0)

    log_likelihood = 0
    for i, token_id in enumerate(continuation_ids):
        token_logit = continuation_logits[0, i]
        token_log_prob = torch.nn.functional.log_softmax(token_logit, dim=-1)
        token_log_prob = token_log_prob[token_id]
        log_likelihood += token_log_prob

    return log_likelihood.item()
    
    
def get_probs(model, tokenizer, prompt, continuations, answer, device="cpu"):
    prob = 0
    ll_answer = 0
    
    for continuation in continuations:
        tmp = math.exp(get_loglikelihood(model, tokenizer, prompt, continuation, device=device))
        prob+=tmp
        
        if continuation == answer or " "+continuation == answer:
            ll_answer=tmp

    return ll_answer/prob
            

def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def get_prediction(out, sample_number):
    choices = out[sample_number]['filtered_resps']
    pred = argmax([choice[0] for choice in choices])
    return pred, out[sample_number]["arguments"][pred][1] # choices[pred][1]

def get_template(out, dataset_name):
    return out["configs"][dataset_name]["doc_to_text"]

def apply_template(dataset_name, resps, template):
    out = template
    
    if dataset_name == "openbookqa":
        return resps["question_stem"]
    
    for name in resps:
        if "{{"+name+"}}" in template:
            out = out.replace("{{"+name+"}}",resps[name])
            
    return out

def get_sample_text_and_schoices(out, dataset_name, sample_number):
    sample_texts = out["samples"][dataset_name][sample_number]['doc']
    possible_choices = None
    
    if dataset_name == "rte":
        possible_choices = [" True"," False"]
        sample_texts={"sentence1":sample_texts["sentence1"],
                      "sentence2":sample_texts["sentence2"]}
        
    elif dataset_name == "openbookqa":
        possible_choices = sample_texts["choices"]["text"]
        sample_texts={"question_stem":sample_texts["question_stem"]}
    
    elif dataset_name == "arc_challenge" or dataset_name == "arc_easy":
        possible_choices = sample_texts["choices"]["text"]
        sample_texts={"question":sample_texts["question"]}

    elif dataset_name == "boolq":
        possible_choices = [" no", " yes"]
        sample_texts={"passage":sample_texts["passage"],
                      "question":sample_texts["question"]}
 
    return sample_texts, possible_choices



def compute_fa_seeds(model, tokenizer, path_res_lm_eval, sampled, num, fa_approach = KernelShap, task_list=tl_f, device="cpu"):
    
    skip_tokens = [tokenizer.bos_token_id]
    
    faithfulness_res = {}
    
    out_file = open(path_res_lm_eval, "r")
    out = json.load(out_file)
    out_file.close()

    scores = {}
    
    for dataset_name in task_list:
        scores[dataset_name] = {}
        
        if dataset_name in out["versions"]:
            print(dataset_name)
            faithfulness_res[dataset_name]={}
            
            template = get_template(out, dataset_name)
            print(template)

            for i in sampled[dataset_name]:    
                

                
                tgt_idx = out["samples"][dataset_name][i]['target']
                label = out["samples"][dataset_name][i]["arguments"][tgt_idx][1]
                
                _, target = get_prediction(out["samples"][dataset_name], i)
                target = str(target)
                    
                sample_texts, possible_choices = get_sample_text_and_schoices(out, dataset_name, i)
                
                sample_texts_processed = {}
                vals = []
                for sample_text in sample_texts:
                    v, t = prepare_text(sample_texts[sample_text])
                    sample_texts_processed[sample_text] = t
                    vals += v
                    
                prompt = apply_template(dataset_name, sample_texts, template)
                
                fa = fa_approach(model)
                llm_attr = LLMAttribution(fa, tokenizer)

                inp = TextTokenInput(
                    prompt, 
                    tokenizer,
                    skip_tokens=skip_tokens
                )
                
                
                
                l = tokenizer(prompt, add_special_tokens=False)
                
                try:
                    attr_res = llm_attr.attribute(inp, target=target, n_samples=3*len(l['input_ids']))

                
                    scores[dataset_name][i]=attr_res.seq_attr.cpu().tolist()
                except Exception as e: 
                    scores[dataset_name][i]="error"
    return scores

def remove_from_index_f(tokenized, idxs_to_remove):
    out = []
    for i in range(len(tokenized)):
        if i not in idxs_to_remove:
            out.append(tokenized[i])
    return out

def compute_faith_f(model, tokenizer, prompt, continuations, answer, ranked, bins_perc = [0.1, 0.2, 0.3, 0.4, 0.5], s=1, device="cpu"):
    aggregated_comp = 0
    den = 0
    
    orig_prob = get_probs(model, tokenizer, prompt, continuations, answer, device=device)
    
    for perc in bins_perc:
        to_remove = int(len(ranked)*perc)
        if to_remove>0:
            den+=1
            
            if s:
                words = ranked[:to_remove]
                idxs_to_replace = ranked[:to_remove]
            else:
                idxs_to_replace = ranked[to_remove:]
                
            tokenized = (tokenizer(prompt)["input_ids"])[1:]
            
            tmp_prompt = tokenizer.decode(remove_from_index_f(tokenized, idxs_to_replace))

            tmp_prob = get_probs(model, tokenizer, tmp_prompt, continuations, answer, device=device)

            aggregated_comp+= orig_prob - tmp_prob
        
    return aggregated_comp/(1+den)

def comprehesiveness_f(model, tokenizer, prompt, continuations, answer, ranked, bins_perc = [0.1, 0.2, 0.3, 0.4, 0.5], device="cpu"):
    return compute_faith_f(model, tokenizer, prompt, continuations, answer, ranked, bins_perc, True, device=device)

def sufficiency_f(model, tokenizer, prompt, continuations, answer, ranked, bins_perc = [0.1, 0.2, 0.3, 0.4, 0.5], device="cpu"):
    return compute_faith_f(model, tokenizer, prompt, continuations, answer, ranked, bins_perc, False, device=device)

def compute_comp_suff(model, tokenizer, path_res_lm_eval, path_fa, task_list=tl_f, device="cpu"):
    
    faithfulness_res = {}
    
    out_file = open(path_res_lm_eval, "r")
    out = json.load(out_file)
    out_file.close()
    
    out_file = open(path_fa, "r")
    scores = json.load(out_file)
    out_file.close()

    for dataset_name in task_list:
        
        if dataset_name in out["versions"]:
            print(dataset_name)
            faithfulness_res[dataset_name]={}
            
            template = get_template(out, dataset_name)
            print(template)
            
            num_samples = len(out["samples"][dataset_name])
            
            step = 0

            cumulated_comp = []
            cumulated_suff = []
            cc = []
            cs = []
            
            for i in range(num_samples):
                
                if int((i/num_samples)*100)>step:
                    print(str(i)+"/"+str(num_samples))
                    step+=5
                

                _, target = get_prediction(out["samples"][dataset_name], i)
                target = str(target)
                
                tgt_idx = out["samples"][dataset_name][i]['target']
                label = out["samples"][dataset_name][i]["arguments"][tgt_idx][1]

                if str(i) in scores[dataset_name]:
                    
                    sample_texts, possible_choices = get_sample_text_and_schoices(out, dataset_name, i)
                        
                    prompt = apply_template(dataset_name, sample_texts, template)
                    
                    
                    if scores[dataset_name][str(i)] != "error":
                        scores_cur = np.array(scores[dataset_name][str(i)])
                        
                        ranked = np.argsort(-scores_cur) 
                    
                            
                        comp = comprehesiveness_f(model, tokenizer, prompt, possible_choices, target, ranked, device=device)

                        suf = sufficiency_f(model, tokenizer, prompt, possible_choices, target, ranked, device=device)
                        
                        cumulated_comp.append(comp)
                        cc.append(comp)
                        cumulated_suff.append(suf)
                        cs.append(suf)
                    else:
                        cumulated_comp.append("error")
                        cumulated_suff.append("error")
                        


            faithfulness_res[dataset_name]["cumulated_comp"]=cumulated_comp
            faithfulness_res[dataset_name]["cumulated_suff"]=cumulated_suff
            
            faithfulness_res[dataset_name]["comp"]=sum(cc)/len(cc)
            faithfulness_res[dataset_name]["suff"]=sum(cs)/len(cs)
            
    return faithfulness_res


def compute_faith_f_i(model, tokenizer, prompt, continuations, answer, ranked, bins_perc = [0.1, 0.2, 0.3, 0.4, 0.5], s=1, device="cpu"):
    aggregated_comp = 0
    den = 0
    
    orig_prob = get_probs(model, tokenizer, prompt, continuations, answer, device=device)
    tmp_results = {}
    
    for perc in bins_perc:
        to_remove = int(len(ranked)*perc)
        if to_remove>0:
            den+=1
            
            if s:
                words = ranked[:to_remove]
                idxs_to_replace = ranked[:to_remove]
            else:
                idxs_to_replace = ranked[to_remove:]

            tokenized = (tokenizer(prompt)["input_ids"])[1:]
            
            tmp_prompt = tokenizer.decode(remove_from_index_f(tokenized, idxs_to_replace))

            tmp_prob = get_probs(model, tokenizer, tmp_prompt, continuations, answer, device=device)
            
            tmp_results[perc] = orig_prob - tmp_prob

            aggregated_comp+= tmp_results[perc]
            
        else:
            tmp_results[perc] = None
            
    return aggregated_comp/(1+den), tmp_results

def comprehesiveness_f_i(model, tokenizer, prompt, continuations, answer, ranked, bins_perc = [0.1, 0.2, 0.3, 0.4, 0.5], device="cpu"):
    return compute_faith_f_i(model, tokenizer, prompt, continuations, answer, ranked, bins_perc, True, device=device)

def sufficiency_f_i(model, tokenizer, prompt, continuations, answer, ranked, bins_perc = [0.1, 0.2, 0.3, 0.4, 0.5], device="cpu"):
    return compute_faith_f_i(model, tokenizer, prompt, continuations, answer, ranked, bins_perc, False, device=device)


def compute_comp_suff_i(model, tokenizer, path_res_lm_eval, path_fa, task_list=tl_f, device="cpu"):
    
    faithfulness_res = {}
    
    out_file = open(path_res_lm_eval, "r")
    out = json.load(out_file)
    out_file.close()
    
    out_file = open(path_fa, "r")
    scores = json.load(out_file)
    out_file.close()

    for dataset_name in task_list:
        
        if dataset_name in out["versions"]:
            print(dataset_name)
            faithfulness_res[dataset_name]={}
            
            template = get_template(out, dataset_name)
            print(template)
            
            num_samples = len(out["samples"][dataset_name])
            
            step = 0

            cumulated_comp = []
            cumulated_suff = []
            cc = []
            cs = []
            
            partial_comp = []
            partial_suf = []
            
            for i in range(num_samples):
                
                if int((i/num_samples)*100)>step:
                    print(str(i)+"/"+str(num_samples))
                    step+=5
                

                _, target = get_prediction(out["samples"][dataset_name], i)
                target = str(target)
                
                tgt_idx = out["samples"][dataset_name][i]['target']
                label = out["samples"][dataset_name][i]["arguments"][tgt_idx][1]
                
                #if label == target and str(i) in scores[dataset_name]:
                if str(i) in scores[dataset_name]:
                    
                    sample_texts, possible_choices = get_sample_text_and_schoices(out, dataset_name, i)
                        
                    prompt = apply_template(dataset_name, sample_texts, template)
                    
                    
                    if scores[dataset_name][str(i)] != "error":
                        scores_cur = np.array(scores[dataset_name][str(i)])
                        
                        ranked = np.argsort(-scores_cur) 
                    
                            
                        comp, add_to_partial_comp = comprehesiveness_f_i(model, tokenizer, prompt, possible_choices, target, ranked, device=device)

                        suf, add_to_partial_suf = sufficiency_f_i(model, tokenizer, prompt, possible_choices, target, ranked, device=device)
                        
                        cumulated_comp.append(comp)
                        cc.append(comp)
                        cumulated_suff.append(suf)
                        cs.append(suf)
                        
                        partial_comp.append(add_to_partial_comp)
                        partial_suf.append(add_to_partial_suf)
                        
                    else:
                        cumulated_comp.append("error")
                        cumulated_suff.append("error")
                        partial_comp.append(None)
                        partial_suf.append(None)
                        


            faithfulness_res[dataset_name]["cumulated_comp"]=cumulated_comp
            faithfulness_res[dataset_name]["cumulated_suff"]=cumulated_suff

            faithfulness_res[dataset_name]["partial_comp"]=partial_comp
            faithfulness_res[dataset_name]["partial_suff"]=partial_suf
            
            faithfulness_res[dataset_name]["comp"]=sum(cc)/len(cc)
            faithfulness_res[dataset_name]["suff"]=sum(cs)/len(cs)
            
    return faithfulness_res


