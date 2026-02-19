import fnmatch
from lm_eval import tasks, evaluator, models 



tl=["boolq","rte","arc_challenge","arc_easy","openbookqa"]
def eval_zero_shot(model_name, model, tokenizer, task_list=tl, 
        num_fewshot=0, use_accelerate=True, add_special_tokens=False, device=None, log_res=False):

    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.TaskManager().all_tasks)
    print(task_names)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights,trust_remote_code=True"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True,trust_remote_code=True"
    
    my_model = models.huggingface.HFLM(pretrained=model,tokenizer=tokenizer,trust_remote_code=True)
    
    results = evaluator.simple_evaluate(
        model=my_model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=device,
        limit=limit,
        check_integrity=False,
        cache_requests=True,
        log_samples=log_res
    )

    return results 
