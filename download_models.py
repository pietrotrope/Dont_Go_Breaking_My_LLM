from transformers import AutoTokenizer, AutoModelForCausalLM
from tokens import *
from huggingface_hub import login
import os

login(token=TOKEN_huggingface)

LMs = ["mistralai/Mistral-7B-v0.1",'meta-llama/Llama-2-7b-hf']

for model_name in LMs:

    if not os.path.isdir("saved_models/unpruned/" + model_name):
        print("Downloading "+model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_name)

        model.save_pretrained("saved_models/unpruned/" + model_name, safe_serialization=False)
        tokenizer.save_pretrained("saved_models/unpruned/" + model_name)
        print("saved_models/unpruned/" + model_name)