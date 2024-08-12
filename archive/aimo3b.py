#!/usr/bin/env python
# coding: utf-8

# ## Forked from https://www.kaggle.com/code/abdurrafae/improved-code-interpretation

# **Lewis:** the only changes in this notebook are those needed to run the original one with the new Kaggle evaluation API

# Forked From  https://kaggle.com/code/xiaoz259/pure-rng/notebook
# 
# credits:
# https://www.kaggle.com/code/olyatsimboy/aimo-openmath-mistral-baseline \
# https://www.kaggle.com/code/aatiffraz/prompt-prediction-w-mixtral-mistral7b-gemma-llama \
# https://www.kaggle.com/code/thedrcat/aimo-mixtral-baseline

import time
NOTEBOOK_START_TIME = time.time()

PRIVATE = False
TRAIN_PATH = 'data/ood.csv'
# TRAIN_PATH = '/kaggle/input/aimo-val-t/ood.csv'


TOTAL_TOKENS = 2048

BATCH_SIZE = 100       # 50 on kaggle
LOOP_REPS = 1
CODE_PROMPT_COUNT = 50   # somewhere from 1:3 to 2:3 is ok  # 22 on kaggle

TIME_LIMIT = 31500 if PRIVATE else 6300
PER_Q_TIME_LIMIT = 650
LAG = 150

BEST_COUNT_THRESHOLD = 60   # it was set as np.sqrt(jj)
CODE_WITH_TEXT = True

TEMP = 0.8
TOP_P = 1.0

SEED = 30108
CFILE = 'code01.py'


import pandas as pd
if not PRIVATE:
    class train_env():
        def __init__(self, randomize=False):
            self.randomlize = randomize
            
            self.df = pd.read_csv(TRAIN_PATH)
            self.df['ground_truth'] = self.df['answer']
            self.df['answer'] = -1
            
            if self.randomlize:
                self.df = self.df.reset_index().sample(frac=1).reset_index(drop=True)
            
            self.predict_called = True
            self.counter = 0
            self.len = len(self.df)
        
        
        def iter_test(self):
             while self.counter<self.len:
                if self.predict_called:
                    self.predict_called = False
                    yield (self.df.loc[[self.counter]][['id','problem']]),(self.df.loc[[self.counter]][['id','answer']])
                else:
                    print("You must call `predict()` successfully before you can continue with `iter_test()`")
                    yield None 
                
        def predict(self, answer):
            self.df.loc[self.counter, ('answer')] = answer['answer'].values[0]
            self.predict_called = True
            self.counter+=1

    env = train_env(randomize=True)
    iter_test = env.iter_test()
else:
    # Set up the evaluation API
    import aimo

    env = aimo.make_env()
    iter_test = env.iter_test()


# TO-DO
# 
# Change temperature as the question goes longer
# Change temperature based on question lenght

# # Zero-shot MMOS-DeepSeekMath-7B with self-consistency and generated code reasoning evaluation
# 
# Self-consistency is a modification of the standard greedy decoding in reasoning pipelines via sampling several diverse answers followed by aggregation, e.g., most common answer ([SC-CoT paper](https://arxiv.org/pdf/2203.11171.pdf)).
# 
# In this kernel, we will consider MMOS-DeepSeekMath-7B RL-tuned backbone; in my experiments, this model produces more consistent code reasoning and the code block execution will allow us to decrease arithmetic hallucinations.

DEBUG = False
# QUANT = False


# %%capture
# %set_env CONDA_PREFIX=/opt/conda
# !pip install -U --no-index /kaggle/input/uv-package-manager/uv-0.2.2-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !uv pip uninstall torch
# !uv pip install --no-index --find-links=/kaggle/input/vllm-wheels -U vllm /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl


# !pip uninstall torch
# !pip install --no-index --find-links=/kaggle/input/vllm-whl -U vllm
# !pip install --no-index -U /kaggle/input/vllm-t4-fix/grpcio-1.62.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
# !pip install --no-index -U /kaggle/input/vllm-t4-fix/ray-2.11.0-cp310-cp310-manylinux2014_x86_64.whl --find-links /kaggle/input/vllm-whl


MODEL_NAME = '/scratch/aakash_ks.iitr/models/deepseek/'
# MODEL_NAME = '/kaggle/input/deepseek-math'
# MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'


stop_words = ["```output", "```\nOutput" , ")\n```" , "``````output"]
# stop_words = ["```output", "```python", "```\nOutput" , ")\n```" , "``````output"]


from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL_NAME,
    dtype="half",
    # enforce_eager=True, #to disable CUDA graphs
    gpu_memory_utilization=0.92,  #with enforce eager you can use 0.99; else 0.91
    swap_space=4,  #CPU RAM per gpu preferable is 2, default is 4
    max_model_len=TOTAL_TOKENS,
    kv_cache_dtype="fp8",  #auto means same as model dtype. use fp8 to save memory
    tensor_parallel_size=2,
    max_num_seqs=BATCH_SIZE,
    max_seq_len_to_capture=TOTAL_TOKENS,
    seed=SEED,
)

# if getting CUDA OOM switch to using enforce_eager = True


# import gc
# import torch
# torch.backends.cuda.enable_mem_efficient_sdp(False)


from tqdm import tqdm
# import math


def naive_parse(answer):
    out = []
    start = False
    end = False
    for l in reversed(list(answer)):
        if l in '0123456789' and not end:
            start = True
            out.append(l)
        else:
            if start:
                end = True
        
    out = reversed(out)
    return ''.join(out)


import re
import sys
import subprocess

def return_last_print(output, n):
    lines = output.strip().split('\n')
    if lines:
        return lines[n]
    else:
        return ""

def process_code(code, return_shell_output=False):
    
    def repl(match):
        if "real" not in match.group():
            return "{}{}".format(match.group()[:-1], ', real=True)')
        else:
            return "{}{}".format(match.group()[:-1], ')')
    code = re.sub(r"symbols\([^)]+\)", repl, code)

    if return_shell_output:
        code = code.replace('\n', '\n    ')
            # Add a try...except block
        code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(code)
    
    if not return_shell_output:
        print(code)
    with open(f'{CFILE}', 'w') as fout:
        fout.write(code)
    
    batcmd = 'timeout 5 ' + sys.executable + f' {CFILE}'
    try:
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        return_value = return_last_print(shell_output, -1)
        print(shell_output)
        if return_shell_output:
            if return_value=='FAIL':
                CODE_STATUS = False
                return_value = return_last_print(shell_output, -2)
                if "not defined" in return_value:
                    return_value+='\nTry checking the formatting and imports'
            else:
                CODE_STATUS = True
            return return_value, CODE_STATUS  
        code_output = round(float(eval(return_value)))
    except Exception as e:
        print(e,'shell_output')
        code_output = -1
    
    if return_shell_output:
        if code_output==-1:
            CODE_STATUS = False
        else:
            CODE_STATUS = True
        return code_output, CODE_STATUS  
    
    
    return code_output


def process_text_output(output):
    result = output    
    try:
        result_output = re.findall(r'\\boxed\{(\d+)\}', result)

        print('BOXED', result_output)
        if not len(result_output):
            result_output = naive_parse(result)
        else:
            result_output = result_output[-1]

        print('BOXED FINAL', result_output)
        if not len(result_output):
            result_output = -1
        
        else:
            result_output = round(float(eval(result_output)))
    
    except Exception as e:
        print(e)
        print('ERROR PARSING TEXT')
        result_output = -1
    
    return result_output


code = """Below is a math problem you are to solve (non-negative numeric answer!):
\"{}\"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions, and remember, your final answer should be non-negative integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{}.

Approach:"""

cot = """Below is a math problem you are to solve (non-negative numeric answer!):
\"{}\"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{}.\n\n"""

promplt_options = [code,cot]


# tokenizer = llm.get_tokenizer()


import gc
from collections import defaultdict
from collections import Counter

import numpy as np
np.random.seed(SEED)

tool_instruction = '\n\nPlease integrate natural language reasoning with programs to solve the above problem, and put your final numerical answer within \\boxed{}.\nNote that the intermediary calculations may be real numbers, but the final numerical answer would always be an integer.'


#tool_instruction = " The answer should be given as a non-negative modulo 1000."
#tool_instruction += '\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.'

temperature = TEMP
top_p = TOP_P

# temperature_coding = TEMP
# top_p_coding = TOP_P
batch_size = BATCH_SIZE
   
total_results = {}
total_answers = {}
best_stats = {}
total_outputs = {}
question_type_counts = {}
starting_counts = (0, 0)

for i, (test, sample_submission) in tqdm(enumerate(iter_test)):
    print(f"Solving problem {i} ...")
    try:
        TIME_SPENT = time.time() - NOTEBOOK_START_TIME

        if TIME_SPENT>TIME_LIMIT:
            sample_submission['answer'] = 0
            env.predict(sample_submission)
            break
        
        Q_START_TIME = time.time()
        problem = test['problem'].values[0]
        
        cot_prompt = f"User: {cot.format(problem, '{}')}"
        code_prompt = f"User: {code.format(problem, '{}')}"
        
        loop_internal_flag = False
        
        for ji in range(LOOP_REPS):            
            if loop_internal_flag:
                break
            
            time_now = time.time()
            
            if (time_now - Q_START_TIME) > PER_Q_TIME_LIMIT-LAG or (time_now - NOTEBOOK_START_TIME)>TIME_LIMIT-LAG:
                print(f'BREAKING BATCH BECAUSE QUESTION TIME LIMIT EXCEEDED')
                break
            
            gc.collect()
            
            prompts = [cot_prompt] * (batch_size - CODE_PROMPT_COUNT) + [code_prompt] * CODE_PROMPT_COUNT
            np.random.shuffle(prompts)
            
            generation_outputs = llm.generate(prompts, SamplingParams(
                stop=stop_words, temperature=temperature, max_tokens=TOTAL_TOKENS, top_p=top_p, 
                include_stop_str_in_output=True))
            
            decoded_outputs = []
            prompts_u = []
            prompt_token_lengths = []
            output_lengths = []
            
            for prompt, output in zip(prompts, generation_outputs):
                if output.outputs[0].finish_reason != 'length':
                    prompts_u.append(prompt)
                    decoded_outputs.append(prompt + output.outputs[0].text)
                    prompt_token_lengths.append(len(output.prompt_token_ids))
                    output_lengths.append(len(output.outputs[0].token_ids))
            
            del generation_outputs
                
            for jk in range(batch_size):
                jj = ji*batch_size + jk
                    
                print(f"\n\n\nQUESTION {i} - {jj} - TIME_SPENT : {TIME_SPENT:.0f} secs")
                
                best, best_count = best_stats.get(i,(-1,-1))
                if best_count>BEST_COUNT_THRESHOLD:      # jj instead of n_repetitions
                    print("SKIPPING CAUSE FOUND BEST")
                    loop_internal_flag = True
                    break
                
                time_now = time.time()
                
                if (time_now - Q_START_TIME) > PER_Q_TIME_LIMIT or (time_now - NOTEBOOK_START_TIME)>TIME_LIMIT:
                    print(f'BREAKING BECAUSE QUESTION TIME LIMIT EXCEEDED')
                    break
                
                if len(decoded_outputs) == 0:
                    print("FINISHED WHOLE BATCH EARLY")
                    break
                    
                outputs = total_outputs.get(i,[])
                text_answers, code_answers = question_type_counts.get(i,starting_counts)
                results = total_results.get(i,[])
                answers = total_answers.get(i,[])

                try:
                    ALREADY_GEN = 0
                    code_error = None
                    code_error_count = 0
                    code_miss_count = 0
                    code_output = -1
                    was_code = False
                    inner_loop_continue_flag = False
                    while_break_flag = False
                    max_token_flag = False
                    #initail_message = problem  + tool_instruction 
                
                    prompt = prompts_u.pop(0)
                    current_printed = len(prompt)
                    
                    print(f"{jj}_{prompt}\n")

                    # model_inputs = tokenizer(prompt, return_tensors='pt')
                    input_len = prompt_token_lengths.pop(0)
                    input_len2 = len(prompt)

                    decoded_output = decoded_outputs.pop(0)
                    
                    # model_inputs = tokenizer(decoded_output, return_tensors='pt')
                    ALREADY_GEN = output_lengths.pop(0)-input_len
                    
                    print(f"{decoded_output[current_printed:]}\n")
                    current_printed += len(decoded_output[current_printed:])
                    
                    stop_word_cond = False
                    for stop_word in stop_words:
                        stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)
                        
                    while_loop_count = 0
                    while (stop_word_cond) and (ALREADY_GEN<(TOTAL_TOKENS)):
                        temperature_inner=temperature
                        top_p_inner = top_p
                        try:
                            if (decoded_output[-len("``````output"):]=="``````output"):
                                code_text = decoded_output.split('```python')[-1].split("``````")[0]
                            else:
                                code_text = decoded_output.split('```python')[-1].split("```")[0]
                            
                            code_output, CODE_STATUS = process_code(code_text, return_shell_output=True)
                            was_code = True
                            print('CODE RESULTS', code_output)
                            
                            # check if code output is numeric
                            try:
                                float(eval(code_output))
                                code_miss_count = 0
                            except:
                                code_output = -1
                                code_miss_count+=1
                                
                            # in case when code outputs something like 1/34 and text will add num and denom to give output
                            try:
                                float(code_output)
                                is_float_flag = True
                            except:
                                is_float_flag = False


                            if code_error==code_output:
                                code_error_count+=1
                            else:
                                code_error=code_output
                                code_error_count = 0
                                
                            if while_loop_count>1 or code_error_count>0:
                                print('WHILE LOOP BREAK')
                                while_break_flag = True
                                break
                                
                            if code_miss_count>0:
                                print('REPEATED CODE MISS')
                                inner_loop_continue_flag = True
                                break

                            if not CODE_STATUS or code_output is None or code_output=='None':
                                print('CODE ERROR')
                                inner_loop_continue_flag = True
                                break
                                

                        except Exception as e:
                            print(e)
                            print('ERROR PARSING CODE')
                            code_output = -1

                        if code_output!=-1:
                            if (decoded_output[-len(")\n```"):]==")\n```"):
                                prompt = decoded_output+'```output\n'+str(code_output)+'\n```\n'
                            else:
                                prompt = decoded_output+'\n'+str(code_output)+'\n```\n'
                        else:
                            prompt = decoded_output
                            
                        # model_inputs = tokenizer(prompt, return_tensors='pt')
                        # ALREADY_GEN =  len(model_inputs['input_ids'][0])-input_len
                        
                        generation_output = llm.generate([prompt], SamplingParams(
                            stop=stop_words, temperature=temperature_inner, max_tokens=TOTAL_TOKENS-ALREADY_GEN, top_p=top_p_inner, 
                            include_stop_str_in_output=True))
                        
                        ALREADY_GEN += len(generation_output[0].outputs[0].token_ids)
                        
                        decoded_output = prompt + generation_output[0].outputs[0].text
                        
                        print(f"\nINTERMEDIATE OUT :\n{decoded_output[current_printed:]}\n")
                        current_printed+=len(decoded_output[current_printed:])
                        
                        if generation_output[0].outputs[0].finish_reason == 'length':
                            max_token_flag = True
                            print('MAX TOKENS REACHED ')
                            break
                        
                        stop_word_cond = False
                        for stop_word in stop_words:
                            stop_word_cond = stop_word_cond or (decoded_output[-len(stop_word):]==stop_word)
                    
                        while_loop_count+=1
                
                    try:
                        if was_code:
                            code_output = round(float(eval(code_output)))

                        else:
                            code_output = -1
                            pass
                    except Exception as e:
                        print(e,'final_eval')
                        code_output = -1

                    if was_code and inner_loop_continue_flag:
                        continue
                    
                    raw_output = decoded_output[input_len2:]
                    result_output = process_text_output(raw_output)
                
                    if ALREADY_GEN>=TOTAL_TOKENS or max_token_flag:
                        print('HAD REACHED MAX TOKENS. SKIPPING TEXT OUTPUT')
                        result_output = -1
                        
                    if while_break_flag:
                        print('WHILE BREAK')
                        result_output = -1
                                                

                except Exception as e:
                    print(e,"5")
                    result_output, code_output = -1, -1

                if code_output!=-1:
                    # ?????
                    # this part is doubtful if should be included or not
                    # should code output have more weightage than text output ??
                    if code_output==result_output:
                        print('MATCHED')
                        if CODE_WITH_TEXT:
                            outputs.append(result_output)
                            text_answers+=1
                            
                        outputs.append(code_output)
                        code_answers+=1
                        
                    else:
                        print('NOT MATCHED')
                        outputs.append(result_output)
                        text_answers+=1
                        
                        if is_float_flag:
                            outputs.append(code_output)
                            code_answers+=1
                

                elif result_output!=-1:
                    outputs.append(result_output)
                    text_answers+=1

                if len(outputs) > 0:
                    occurances = Counter(outputs).most_common()
                    print(occurances)
                    if occurances[0][1] > best_count:
                        print("GOOD ANSWER UPDATED!")
                        best = occurances[0][0]
                        best_count = occurances[0][1]
                    if occurances[0][1] > BEST_COUNT_THRESHOLD:
                        print("ANSWER FOUND!")
                        loop_internal_flag = True
                        break

                results.append(result_output)
                answers.append(code_output)
                
                best_stats[i] = (best, best_count) 
                question_type_counts[i] = (text_answers, code_answers)
                total_outputs[i] = outputs
                
                total_results[i] = results
                total_answers[i] = answers

                print("code_answers",code_answers-starting_counts[1],"text_answers",text_answers-starting_counts[0])
                if DEBUG:
                    loop_internal_flag = True
                    break
                
        try:
            print(f"Predicted best answer: {best_stats}")
            sample_submission['answer'] = best_stats[i][0] % 1000
        except:
            sample_submission['answer'] = 0
        env.predict(sample_submission)
    
    except Exception as e:
        print(e)
        try:
            print(f"Predicted best answer: {best_stats}")
            sample_submission['answer'] = best_stats[i][0] % 1000
        except:
            sample_submission['answer'] = 0
            
        env.predict(sample_submission)
    
    print('-' * 80)
    print(f'Time spent on the question: {time.time() - Q_START_TIME:.0f} secs')
    print('-' * 80)


if not PRIVATE:
    print(env.df)
    print(f"\ncorrect: {(env.df['ground_truth'] == env.df['answer']).sum()} out of {len(env.df)}")


with open(f'{CFILE}', 'w') as fout:
    fout.write("print('done')")

batcmd = 'timeout 5 ' + sys.executable + f' {CFILE}'
try:
    shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
    print(shell_output)
except:
    pass


print(f'TOTAL TIME TAKEN BY NB = {time.time() - NOTEBOOK_START_TIME:.0f} secs')




