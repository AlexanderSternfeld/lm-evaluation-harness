def generate(prompt, model, tokenizer, temperature=0.0, top_p=1.0, max_tokens=512):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        max_len = min(tokenizer.model_max_length, 4096)  # defensive coding
        input = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        if temperature > 0.0:
            output = model.generate(
                input_ids=input["input_ids"],
                attention_mask=input["attention_mask"],
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            # If temperature is 0, we use greedy decoding
            output = model.generate(
                input_ids=input["input_ids"],
                attention_mask=input["attention_mask"],
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        prompt_length = input["input_ids"].shape[1]
        return tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

def jsonify_ans_longwiki(raw_responses, eval_prompts, model, tokenizer, key):

    def check_validity(gen):
        if '{{"{}":false}}'.format(key) in gen.lower():
            return '{{"{}":false}}'.format(key)
        elif '{{"{}":true}}'.format(key) in gen.lower():
            return '{{"{}":true}}'.format(key)
        else:
            return -1
        
    jsonifyed_res  = []
    for r, p in zip(raw_responses, eval_prompts):
        
        if check_validity(r) != -1:
            jsonifyed_res.append(json.loads(check_validity(r)))
            continue
        else:
            r = r.split("\n")[0]
            try:
                jsonifyed_res.append(json.loads(r))
            except:
                print(f"Error in eval_answer: {r}")
                error = True
                error_count = 0
                
                while error:
                    re_eval = generate(
                        prompt=p,
                        model=model,
                        tokenizer=tokenizer,
                        temperature=0.0,
                        max_tokens=128
                    )

                    try: 
                        print("\n** RETRY:", re_eval)
                        if check_validity(re_eval) != -1:
                            json_res = json.loads(check_validity(re_eval))
                        else:
                            re_eval = re_eval.split("\n")[0]
                            json_res = json.loads(re_eval)
                        error = False
                        
                    except:
                        print("*** trying again** \n")
                        error = True
                    error_count += 1

                    if error_count > 3:
                        print("Error count exceeded 3. Skipping this prompt.")
                        jsonifyed_res.append({"error": "Error count exceeded 3. Skipping this prompt."})
                        break
                jsonifyed_res.append(json_res)
                print("<<< PASS >>>")

    return jsonifyed_res