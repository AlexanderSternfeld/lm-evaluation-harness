# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils import generate, jsonify_ans
import prompt_templates
import os
import json

class NonsenseNameEval:
    def __init__(self, evaluator_model, evaluator_tokenizer):
        self.evaluator_model = evaluator_model
        self.evaluator_tokenizer = evaluator_tokenizer

    def run_eval(self, generation):
        abstain_prompt = prompt_templates.ABSTAIN_PROMPT_PLACE_NONSENSE.format(name=generation['name'],
            TYPE=generation['type_'],
            PLACE=" in " + generation['place'] if generation['place'] else "",
            generation=generation['generation'],
        )

        abstains_eval_raw = generate(abstain_prompt, self.evaluator_model, self.evaluator_tokenizer)

        abstains_eval = jsonify_ans(
            raw_response=abstains_eval_raw,
            eval_prompt=abstain_prompt,
            key="does_believe",
            model=self.evaluator_model,
            tokenizer=self.evaluator_tokenizer
        )


        abstains_eval_res = []
        for o in abstains_eval:
            try:
                abstains_eval_res.append(not o['does_believe'])
            except:
                print(f"Error in eval_answer: {o}")
                exit()

        return {"abstention": abstains_eval_res[0]}

class NonsenseMixedEval(NonsenseNameEval):
    def __init__(self, eval_model, eval_tokenizer):

        self.eval_model = eval_model
        self.eval_tokenizer = eval_tokenizer

        self.generations_file_path = f'{self.task_output_dir}/generation.jsonl'
        self.res_path = f'{self.task_output_dir}/eval_results.json'
        self.eval_raw_path = f'{self.task_output_dir}/raw_eval_res.jsonl'

        self.evaluator = "meta-llama/Llama-3.1-8B-Instruct"


    def automatic_abstention(self, generation, model, tokenizer):
        JSON_KEY = "does_believe"

        eval_prompts = {
            'medicine' : prompt_templates.ABSTAIN_PROMPT_NONSENSE_MEDICINE,
            'animal' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'plant' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'bacteria' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
        }

        abstain_prompt = eval_prompts.get(generation['type']).format(
                    prompt=generation['prompt'],
                    name=generation['name'],
                    generation=generation['generation'],
                )

        abstains_eval_raw = generate(abstain_prompt, self.eval_model, self.eval_tokenizer)

        abstains_eval = jsonify_ans(raw_responses=abstains_eval_raw, eval_prompts=abstain_prompt, key=JSON_KEY)
        abstains_eval_res = []
        for o in abstains_eval:
            abstains_eval_res.append(not o[JSON_KEY])
        
        return {"abstention": abstains_eval_res[0]}