import glob
import os
import json


def get_hellaswag(num_lines=5):
    # TODO: Provide path to HellaSWAG jsonl below
    with open("/home/gridsan/fkossmann/ensemble_serve/workloads/llama/hellaswag/data/hellaswag_train.jsonl", "r") as f:
        lines = f.readlines()

    cnt = 0
    prompts = []
    labels = []
    for json_line in lines:
        # Parse JSON line
        parsed_json = json.loads(json_line)

        # Extract desired fields
        labels.append(parsed_json['label'])
        ctx_a = parsed_json['ctx']
        endings = parsed_json['endings']

        # make list of all prompts from this eval sample
        sample_prompts = []
        for e in endings:
            sample_prompts.append((ctx_a, e))
        prompts.append(sample_prompts)
        # prompts += sample_prompts

        cnt += 1
        if cnt >= num_lines:
            break

    return prompts, labels
