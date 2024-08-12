import time
import pandas as pd
import re
import sys
import subprocess
import gc
from collections import Counter
import numpy as np
from vllm import LLM, SamplingParams
import torch
from tqdm import tqdm

NOTEBOOK_START_TIME = time.time()


class Config:
    PRIVATE = False
    TOTAL_TOKENS = 2048
    TRAIN_PATH = 'data/ood.csv'
    BATCH_SIZE = 100
    LOOP_REPS = 1
    CODE_PROMPT_COUNT = 50
    TIME_LIMIT = 31500
    PER_Q_TIME_LIMIT = 650
    LAG = 150
    BEST_COUNT_THRESHOLD = 50
    CODE_WITH_TEXT = True
    TEMP = 0.8
    TOP_P = 1.0
    SEED = 30108
    CFILE = 'code01.py'
    DEBUG = False
    MODEL_NAME = '/scratch/aakash_ks.iitr/models/deepseek/'


class TrainEnvironment:
    def __init__(self, randomize=False):
        self.randomize = randomize
        self.df = pd.read_csv(Config.TRAIN_PATH)
        self.df['ground_truth'] = self.df['answer']
        self.df['answer'] = -1

        if self.randomize:
            self.df = self.df.reset_index().sample(frac=1).reset_index(drop=True)

        self.predict_called = True
        self.counter = 0
        self.len = len(self.df)

    def iter_test(self):
        while self.counter < self.len:
            if self.predict_called:
                self.predict_called = False
                yield (self.df.loc[[self.counter]][['id', 'problem']]), (self.df.loc[[self.counter]][['id', 'answer']])
            else:
                print("You must call `predict()` successfully before you can continue with `iter_test()`")
                yield None

    def predict(self, answer):
        self.df.loc[self.counter, ('answer')] = answer['answer'].values[0]
        self.predict_called = True
        self.counter += 1


class AIModelWrapper:
    def __init__(self):
        self.llm = LLM(
            model=Config.MODEL_NAME,
            dtype="half",
            gpu_memory_utilization=0.92,
            swap_space=4,
            max_model_len=Config.TOTAL_TOKENS,
            kv_cache_dtype="fp8",
            tensor_parallel_size=torch.cuda.device_count(),
            max_num_seqs=Config.BATCH_SIZE,
            max_seq_len_to_capture=Config.TOTAL_TOKENS,
            seed=Config.SEED,
        )
        self.stop_words = ["```output", "```\nOutput", ")\n```", "``````output"]

    def generate(self, prompts, max_tokens):
        return self.llm.generate(prompts, SamplingParams(
            stop=self.stop_words, temperature=Config.TEMP, max_tokens=max_tokens,
            top_p=Config.TOP_P, include_stop_str_in_output=True))


class CodeProcessor:
    @staticmethod
    def process_code(code, return_shell_output=False):
        def repl(match):
            if "real" not in match.group():
                return "{}{}".format(match.group()[:-1], ', real=True)')
            else:
                return "{}{}".format(match.group()[:-1], ')')

        code = re.sub(r"symbols\([^)]+\)", repl, code)

        if return_shell_output:
            code = code.replace('\n', '\n    ')
            code = "\ntry:\n    from sympy import *\n{}\nexcept Exception as e:\n    print(e)\n    print('FAIL')\n".format(
                code)

        if not return_shell_output:
            print(code)
        with open(f'{Config.CFILE}', 'w') as fout:
            fout.write(code)

        batcmd = f'timeout 4 {sys.executable} {Config.CFILE}'
        try:
            shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
            return_value = CodeProcessor.return_last_print(shell_output, -1)
            print(shell_output)
            if return_shell_output:
                if return_value == 'FAIL':
                    CODE_STATUS = False
                    return_value = CodeProcessor.return_last_print(shell_output, -2)
                    if "not defined" in return_value:
                        return_value += '\nTry checking the formatting and imports'
                else:
                    CODE_STATUS = True
                return return_value, CODE_STATUS
            code_output = round(float(eval(return_value)))
        except Exception as e:
            print(e, 'shell_output')
            code_output = -1

        if return_shell_output:
            CODE_STATUS = code_output != -1
            return code_output, CODE_STATUS

        return code_output

    @staticmethod
    def return_last_print(output, n):
        lines = output.strip().split('\n')
        return lines[n] if lines else ""


class OutputProcessor:
    @staticmethod
    def process_text_output(output):
        result = output
        try:
            result_output = re.findall(r'\\boxed\{(\d+)\}', result)
            print('BOXED', result_output)
            if not result_output:
                result_output = OutputProcessor.naive_parse(result)
            else:
                result_output = result_output[-1]
            print('BOXED FINAL', result_output)
            if not result_output:
                result_output = -1
            else:
                result_output = round(float(eval(result_output)))
        except Exception as e:
            print(e)
            print('ERROR PARSING TEXT')
            result_output = -1
        return result_output

    @staticmethod
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


class AIModelRunner:
    def __init__(self):
        self.model = AIModelWrapper()
        self.code_processor = CodeProcessor()
        self.output_processor = OutputProcessor()
        self.total_results = {}
        self.total_answers = {}
        self.best_stats = {}
        self.total_outputs = {}
        self.question_type_counts = {}
        self.starting_counts = (0, 0)

    def run(self, env, iter_test):
        for i, (test, sample_submission) in tqdm(enumerate(iter_test)):
            print(f"Solving problem {i} ...")
            try:
                TIME_SPENT = time.time() - NOTEBOOK_START_TIME

                if TIME_SPENT > Config.TIME_LIMIT:
                    sample_submission['answer'] = 0
                    env.predict(sample_submission)
                    break

                Q_START_TIME = time.time()
                problem = test['problem'].values[0]

                self.process_single_question(problem, i, TIME_SPENT, Q_START_TIME)

            except Exception as e:
                print(e)


            try:
                print(f"Predicted best answer: {self.best_stats}")
                sample_submission['answer'] = self.best_stats[i][0] % 1000
            except:
                sample_submission['answer'] = 0

            env.predict(sample_submission)

            print('-' * 80)
            print(f'Time spent on the question: {time.time() - Q_START_TIME:.0f} secs')
            print('-' * 80)

    def process_single_question(self, problem, i, TIME_SPENT, Q_START_TIME):
        cot_prompt = f"User: {self.get_cot_prompt(problem)}"
        code_prompt = f"User: {self.get_code_prompt(problem)}"

        loop_internal_flag = False

        for ji in range(Config.LOOP_REPS):
            if loop_internal_flag:
                break

            time_now = time.time()

            if (time_now - Q_START_TIME) > Config.PER_Q_TIME_LIMIT - Config.LAG or (
                    time_now - NOTEBOOK_START_TIME) > Config.TIME_LIMIT - Config.LAG:
                print(f'BREAKING BATCH BECAUSE QUESTION TIME LIMIT EXCEEDED')
                break

            gc.collect()

            prompts = [cot_prompt] * (Config.BATCH_SIZE - Config.CODE_PROMPT_COUNT) + [
                code_prompt] * Config.CODE_PROMPT_COUNT
            np.random.shuffle(prompts)

            generation_outputs = self.model.generate(prompts, Config.TOTAL_TOKENS)

            decoded_outputs, prompts_u, prompt_token_lengths, output_lengths = self.process_outputs(prompts,
                                                                                                    generation_outputs)

            del generation_outputs

            for jk in range(Config.BATCH_SIZE):
                jj = ji * Config.BATCH_SIZE + jk

                print(f"\n\n\nQUESTION {i} - {jj} - TIME_SPENT : {TIME_SPENT:.0f} secs")

                best, best_count = self.best_stats.get(i, (-1, -1))
                if best_count > Config.BEST_COUNT_THRESHOLD:
                    print("SKIPPING CAUSE FOUND BEST")
                    loop_internal_flag = True
                    break

                time_now = time.time()

                if (time_now - Q_START_TIME) > Config.PER_Q_TIME_LIMIT or (
                        time_now - NOTEBOOK_START_TIME) > Config.TIME_LIMIT:
                    print(f'BREAKING BECAUSE QUESTION TIME LIMIT EXCEEDED')
                    break

                if len(decoded_outputs) == 0:
                    print("FINISHED WHOLE BATCH EARLY")
                    break

                outputs = self.total_outputs.get(i, [])
                text_answers, code_answers = self.question_type_counts.get(i, self.starting_counts)
                results = self.total_results.get(i, [])
                answers = self.total_answers.get(i, [])

                try:
                    result_output, code_output = self.process_single_output(decoded_outputs, prompts_u,
                                                                            prompt_token_lengths, output_lengths)
                except Exception as e:
                    print(e, "5")
                    result_output, code_output = -1, -1

                outputs, text_answers, code_answers = self.update_outputs(outputs, text_answers, code_answers,
                                                                          result_output, code_output)

                if len(outputs) > 0:
                    occurances = Counter(outputs).most_common()
                    print(occurances)
                    if occurances[0][1] > best_count:
                        print("GOOD ANSWER UPDATED!")
                        best = occurances[0][0]
                        best_count = occurances[0][1]
                    if occurances[0][1] > Config.BEST_COUNT_THRESHOLD:
                        print("ANSWER FOUND!")
                        loop_internal_flag = True
                        break

                results.append(result_output)
                answers.append(code_output)

                self.best_stats[i] = (best, best_count)
                self.question_type_counts[i] = (text_answers, code_answers)
                self.total_outputs[i] = outputs

                self.total_results[i] = results
                self.total_answers[i] = answers

                print("code_answers", code_answers - self.starting_counts[1], "text_answers",
                      text_answers - self.starting_counts[0])

                if Config.DEBUG:
                    loop_internal_flag = True
                    break

    def get_cot_prompt(self, problem):
        return f"""Below is a math problem you are to solve (non-negative numeric answer!):
"{problem}"
Analyze this problem and think step by step to come to a solution with programs. After solving the problem, output the final numerical answer within \\boxed{{}}.\n\n"""

    def get_code_prompt(self, problem):
        return f"""Below is a math problem you are to solve (non-negative numeric answer!):
"{problem}"
To accomplish this, first determine a sympy-based approach for solving the problem by listing each step to take and what functions need to be called in each step. Be clear so even an idiot can follow your instructions, and remember, your final answer should be non-negative integer, not an algebraic expression!
Write the entire script covering all the steps (use comments and document it well) and print the result. After solving the problem, output the final numerical answer within \\boxed{{}}."""

    def process_outputs(self, prompts, generation_outputs):
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

        return decoded_outputs, prompts_u, prompt_token_lengths, output_lengths

    def process_single_output(self, decoded_outputs, prompts_u, prompt_token_lengths, output_lengths):
        ALREADY_GEN = 0
        code_error = None
        code_error_count = 0
        code_miss_count = 0
        code_output = -1
        was_code = False
        inner_loop_continue_flag = False
        while_break_flag = False
        max_token_flag = False

        prompt = prompts_u.pop(0)
        current_printed = len(prompt)

        print(f"{prompt}\n")

        input_len = prompt_token_lengths.pop(0)
        input_len2 = len(prompt)

        decoded_output = decoded_outputs.pop(0)

        ALREADY_GEN = output_lengths.pop(0) - input_len

        print(f"{decoded_output[current_printed:]}\n")
        current_printed += len(decoded_output[current_printed:])

        stop_word_cond = any(decoded_output.endswith(stop_word) for stop_word in self.model.stop_words)

        while_loop_count = 0
        while stop_word_cond and (ALREADY_GEN < Config.TOTAL_TOKENS):
            code_output, was_code, inner_loop_continue_flag, while_break_flag = self.process_code_output(decoded_output)

            if inner_loop_continue_flag or while_break_flag:
                break

            prompt, decoded_output, ALREADY_GEN, max_token_flag = self.generate_next_output(prompt, decoded_output,
                                                                                            code_output, ALREADY_GEN)

            print(f"\nINTERMEDIATE OUT :\n{decoded_output[current_printed:]}\n")
            current_printed += len(decoded_output[current_printed:])

            if max_token_flag:
                print('MAX TOKENS REACHED ')
                break

            stop_word_cond = any(decoded_output.endswith(stop_word) for stop_word in self.model.stop_words)
            while_loop_count += 1

        result_output = self.finalize_output(decoded_output, input_len2, was_code, code_output, ALREADY_GEN,
                                             max_token_flag, while_break_flag)

        return result_output, code_output

    def process_code_output(self, decoded_output):
        try:
            if decoded_output.endswith("``````output"):
                code_text = decoded_output.split('```python')[-1].split("``````")[0]
            else:
                code_text = decoded_output.split('```python')[-1].split("```")[0]

            code_output, CODE_STATUS = self.code_processor.process_code(code_text, return_shell_output=True)
            was_code = True
            print('CODE RESULTS', code_output)

            try:
                float(eval(code_output))
                code_miss_count = 0
            except:
                code_output = -1
                code_miss_count += 1

            try:
                float(code_output)
                is_float_flag = True
            except:
                is_float_flag = False

            if code_error == code_output:
                code_error_count += 1
            else:
                code_error = code_output
                code_error_count = 0

            if code_error_count > 0:
                print('WHILE LOOP BREAK')
                return code_output, was_code, False, True

            if code_miss_count > 0:
                print('REPEATED CODE MISS')
                return code_output, was_code, True, False

            if not CODE_STATUS or code_output is None or code_output == 'None':
                print('CODE ERROR')
                return code_output, was_code, True, False

        except Exception as e:
            print(e)
            print('ERROR PARSING CODE')
            code_output = -1
            was_code = False

        return code_output, was_code, False, False

    def generate_next_output(self, prompt, decoded_output, code_output, ALREADY_GEN):
        if code_output != -1:
            if decoded_output.endswith(")\n```"):
                prompt = decoded_output + '```output\n' + str(code_output) + '\n```\n'
            else:
                prompt = decoded_output + '\n' + str(code_output) + '\n```\n'
        else:
            prompt = decoded_output

        generation_output = self.model.generate([prompt], Config.TOTAL_TOKENS - ALREADY_GEN)

        ALREADY_GEN += len(generation_output[0].outputs[0].token_ids)
        decoded_output = prompt + generation_output[0].outputs[0].text
        max_token_flag = generation_output[0].outputs[0].finish_reason == 'length'

        return prompt, decoded_output, ALREADY_GEN, max_token_flag

    def finalize_output(self, decoded_output, input_len2, was_code, code_output, ALREADY_GEN, max_token_flag,
                        while_break_flag):
        try:
            if was_code:
                code_output = round(float(eval(code_output)))
            else:
                code_output = -1
        except Exception as e:
            print(e, 'final_eval')
            code_output = -1

        raw_output = decoded_output[input_len2:]
        result_output = self.output_processor.process_text_output(raw_output)

        if ALREADY_GEN >= Config.TOTAL_TOKENS or max_token_flag:
            print('HAD REACHED MAX TOKENS. SKIPPING TEXT OUTPUT')
            result_output = -1

        if while_break_flag:
            print('WHILE BREAK')
            result_output = -1

        return result_output

    def update_outputs(self, outputs, text_answers, code_answers, result_output, code_output):
        if code_output != -1:
            if code_output == result_output:
                print('MATCHED')
                if Config.CODE_WITH_TEXT:
                    outputs.append(result_output)
                    text_answers += 1

                outputs.append(code_output)
                code_answers += 1

            else:
                print('NOT MATCHED')
                outputs.append(result_output)
                text_answers += 1

                if isinstance(code_output, (int, float)):
                    outputs.append(code_output)
                    code_answers += 1

        elif result_output != -1:
            outputs.append(result_output)
            text_answers += 1

        return outputs, text_answers, code_answers


def main():
    np.random.seed(Config.SEED)

    if not Config.PRIVATE:
        env = TrainEnvironment(randomize=True)
        iter_test = env.iter_test()
    else:
        import aimo
        env = aimo.make_env()
        iter_test = env.iter_test()

    runner = AIModelRunner()
    runner.run(env, iter_test)

    if not Config.PRIVATE:
        print(env.df)
        print(f"\ncorrect: {(env.df['ground_truth'] == env.df['answer']).sum()} out of {len(env.df)}")

    with open(f'{Config.CFILE}', 'w') as fout:
        fout.write("print('done')")

    batcmd = f'timeout 4 {sys.executable} {Config.CFILE}'
    try:
        shell_output = subprocess.check_output(batcmd, shell=True).decode('utf8')
        print(shell_output)
    except:
        pass

    print(f'TOTAL TIME TAKEN BY NB = {time.time() - NOTEBOOK_START_TIME:.0f} secs')


if __name__ == "__main__":
    main()
