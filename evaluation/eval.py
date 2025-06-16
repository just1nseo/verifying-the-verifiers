from collections import Counter
import json
import math
import os
import random
import re
import subprocess
import time
from typing import List, Dict

import fire
import numpy as np

from common import utils
from evaluation.factory import load_fact_verification_dataset, load_model
from evaluation.tasks import FactVerificationLabel
from models import FactVerificationResult

def run(
    dataset_name: str,
    model_type: str,
    model_name: str,
    num_processes: int = 1,
    category: str = None,
    category_wise: bool = False,
    topic_wise: bool = False,
    is_debug: bool = False,
):
    output_dir = f"output/{dataset_name}-{model_type}"
    if model_name:
        output_dir += "-" + "_".join(model_name.split())
    if category:
        output_dir += f"-{category}"
    if is_debug:
        output_dir += "-debug"

    if os.path.exists(os.path.join(output_dir, "results.jsonl")):
        result = FactVerificationResult()
        result.load(output_dir)
        
        # Use unified reporting system
        result.report(category_wise=category_wise, topic_wise=topic_wise)
        return

    dataset = load_fact_verification_dataset(dataset_name)
    if category:
        dataset = [d for d in dataset if d.category == category]
    if is_debug:
        utils.print_color("Running in debug mode", "red")
        random.seed(42)
        supported_dataset = [
            data for data in dataset if data.label == FactVerificationLabel.SUPPORTED
        ]
        not_supported_dataset = [
            data
            for data in dataset
            if data.label == FactVerificationLabel.NOT_SUPPORTED
        ]
        dataset = supported_dataset[:10] + not_supported_dataset[:10]

    model = load_model(model_type, model_name, num_processes)

    result = model.run(dataset)
    
    # Use unified reporting system
    result.report(category_wise=category_wise, topic_wise=topic_wise)

    # Save the results
    result.save(output_dir)


def run_shard(
    dataset_name: str,
    num_gpus: int,
    shard_id: int,
    model_type: str,
    model_name: str,
    num_subshards: int | None = None,
    sub_shard_id: int | None = None,
    sub_shard_num_gpus: int | None = None,
) -> None:
    model_name_str = "_".join(model_name.split())
    output_dir = f"output/{dataset_name}-{model_type}-{model_name_str}/shard_{shard_id}of{num_gpus}"
    if num_subshards is None:
        dataset = load_fact_verification_dataset(dataset_name)
        dataset = dataset[shard_id * (len(dataset) // num_gpus + 1):
                         (shard_id + 1) * (len(dataset) // num_gpus + 1)]
        if sub_shard_num_gpus is not None:
            assert sub_shard_id is not None, "sub_shard_id should be not None if sub_shard_num_gpus is not None."
            output_dir += f"/subshard_{sub_shard_id}of{sub_shard_num_gpus}"
            dataset = dataset[sub_shard_id * (len(dataset) // sub_shard_num_gpus + 1):
                             (sub_shard_id + 1) * (len(dataset) // sub_shard_num_gpus + 1)]

        model = load_model(model_type, model_name, num_processes=1)
        result = model.run(dataset)
        result.save(output_dir)
    else:
        processes = []
        for i in range(num_subshards):
            cmd = f"CUDA_VISIBLE_DEVICES={i} python evaluation/eval.py run_shard {dataset_name} {num_gpus} {shard_id} {model_type} {model_name} --sub_shard_id {i} --sub_shard_num_gpus {num_subshards}"
            process = subprocess.Popen(cmd, shell=True)
            processes.append(process)
            
        for process in processes:
            process.wait()
            
        # now, merge the results.
        all_results = []
        for i in range(num_subshards):
            result = FactVerificationResult()
            result.load(f"output/{dataset_name}-{model_type}-{model_name_str}/shard_{shard_id}of{num_gpus}/subshard_{i}of{num_subshards}")
            all_results.extend(result.results)
            
        result = FactVerificationResult()
        result.results = all_results
        result.save(output_dir)
        

def merge_results(
    num_gpus: int,
    dataset_name: str,
    model_type: str,
    model_name: str,
):
    model_name_str = "_".join(model_name.split())
    output_dir = f"output/{dataset_name}-{model_type}-{model_name_str}"

    # now, merge the results.
    all_results = []
    for i in range(num_gpus):
        result = FactVerificationResult()
        result.load(f"output/{dataset_name}-{model_type}-{model_name_str}/shard_{i}of{num_gpus}")
        all_results.extend(result.results)
    result = FactVerificationResult()
    result.results = all_results
    result.save(output_dir)
    
    # Use unified reporting system
    result.report()


def run_multigpu(
    num_gpus: int,
    dataset_name: str,
    model_type: str,
    model_name: str,
    category_wise: bool = False,
    topic_wise: bool = False,
):
    model_name_str = "_".join(model_name.split())
    output_dir = f"output/{dataset_name}-{model_type}-{model_name_str}"

    if os.path.exists(os.path.join(output_dir, "results.jsonl")):
        result = FactVerificationResult()
        result.load(output_dir)
        
        # Use unified reporting system
        result.report(category_wise=category_wise, topic_wise=topic_wise)
        return

    processes = []
    for i in range(num_gpus):
        # allocate GPU i and do run.
        cmd = f"CUDA_VISIBLE_DEVICES={i} python evaluation/eval.py run_shard {dataset_name} {num_gpus} {i} {model_type} {model_name}"
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    # now, merge the results.
    all_results = []
    for i in range(num_gpus):
        result = FactVerificationResult()
        result.load(f"output/{dataset_name}-{model_type}-{model_name_str}/shard_{i}of{num_gpus}")
        all_results.extend(result.results)
    result = FactVerificationResult()
    result.results = all_results
    result.save(output_dir)
    
    # Use unified reporting system
    result.report(category_wise=category_wise, topic_wise=topic_wise)


commands = {
    "run": run,
    "run_multigpu": run_multigpu,
    "run_shard": run_shard,
    "merge_results": merge_results,
}

if __name__ == "__main__":
    fire.Fire(commands)
