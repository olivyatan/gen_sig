import json
import traceback
#Image: hub.tess.io/gen-ai/ellement:latest

import torch
from ellement.transformers import AutoModelForCausalLM, AutoTokenizer
# from ellement.peft import PeftConfig, PeftModel

from dev.generation.generators.base_sigs_generator import BaseSignalGenerator


class SignalsGeneratorMistral(BaseSignalGenerator):
    def __init__(self,  cols_for_prompt=['title', 'aspects', 'desc'], prompt: str = None, model_path="mms://athena-mvp/Mistral-7B-Instruct-v02",  mistral_gen_func=None):
        BaseSignalGenerator.__init__(self, prompt)
        self.cols_for_prompt = cols_for_prompt
        self.model_path = model_path
        if mistral_gen_func is not None:
            self.mistral_gen_func = mistral_gen_func
        else:
            # Initialize tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="balanced",
                                                         torch_dtype=torch.bfloat16)  # On V100 switch to torch.float16
            def mistral_gen(prompt_input):
                # print(f"Prompt: {prompt_input}")
                args = {'temperature': 0.8, 'max_new_tokens': 256}
                prompt_input_hf_format = [{"role": "user", "content": f"{prompt_input}"}]
                prompt = tokenizer.apply_chat_template(prompt_input_hf_format, tokenize=False,
                                                       add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
                output = model.generate(**inputs, max_new_tokens=args['max_new_tokens'], do_sample=True,
                                        temperature=args['temperature'])
                gen_text = tokenizer.decode(output[0][len(inputs[0]):], skip_special_tokens=True)
                # print(f"Generated text: {gen_text}")
                return gen_text


            self.mistral_gen_func = mistral_gen

    async def generate_sig(self, item,  debug=False):
        try:
            item = item.iloc[0]
            # prompt = self.prompt.format(item["title"], item["aspects"], item["desc"])
            prompt = self.prompt.format(*item[self.cols_for_prompt])
            # print(prompt)
            # print(f"Prompt: {prompt}")
            gen_text = self.mistral_gen_func(prompt)

        except Exception as e:
            if debug:
                print(e)
                traceback.print_exc()
            return None
        return gen_text


# def run_items_sigs_generation_mistral(item, mistral_gen_func, basic_prompt):
#     # print (f"Basic prompt: {basic_prompt}")
#     # print(f"Item: {item}")
#     try:
#         # print(f"in generation_mistral Title: {item['title']}")
#         prompt = basic_prompt.format(item["title"], item["aspects"], item["desc"])
#         gen_text = mistral_gen_func(prompt)
#         # print(f"item['gen_sigs']: {item['gen_sigs']}")
#     except Exception as e:
#         if debug:
#             print(e)
#             traceback.print_exc()
#         return None
#     return gen_text