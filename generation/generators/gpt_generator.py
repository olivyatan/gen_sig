import sys
#sys.path.append('gpt_call_async/..')
from gpt_call_async.async_gpt import main_gpt_call, GPTCaller
from dev.generation.generators.base_sigs_generator import BaseSignalGenerator,BaseLabelGenerator


class SignalsGeneratorGPT(GPTCaller, BaseSignalGenerator):
    def __init__(self, cols_for_prompt=['title', 'aspects', 'desc'], prompt: str = None,
                 prompt_instructions_version: str = "default", num_signals_per_item: int = 1):
        # Initialize BaseSignalGenerator with the selected prompt instructions version
        BaseSignalGenerator.__init__(self, prompt=prompt, prompt_instructions_version=prompt_instructions_version,
                                     num_signals_per_item=num_signals_per_item)
        self.cols_for_prompt = cols_for_prompt
        self.num_signals_per_item = num_signals_per_item  # Store the number of signals to generate

    def generate_prompt(self, title: str, aspects: str, desc: str):
        prompt = self.prompt.format(title, aspects, desc)
        # print(f"Prompt: {prompt}")
        return prompt

    async def generate_sig(self, item, n_tasks_slice=1, outfile=None, debug=False):
        # print(f"Prompt: {self.prompt}")
        res = await main_gpt_call(
            gpt_caller=self,
            data=item,
            output_path=outfile,
            n_tasks_slice=n_tasks_slice,
            cols=self.cols_for_prompt,
            append_prompt_features=False
        )
        # TODO: if I want to support n_tasks_slice > 1 I need to change the return value
        return res[0]

    async def generate_sigs(self, input_df, outpath, n_tasks_slice=2):
        await main_gpt_call(
            gpt_caller=self,
            data=input_df,
            output_path=outpath,
            n_tasks_slice=n_tasks_slice,
            cols=self.cols_for_prompt,
            append_prompt_features=False
        )


class SignalLabelGeneratorGPT(GPTCaller, BaseLabelGenerator):
    def __init__(self, cols_for_prompt=['title', 'aspects', 'desc', 'signal'], prompt: str = None):
        BaseLabelGenerator.__init__(self, prompt)
        self.cols_for_prompt = cols_for_prompt

    def generate_prompt(self, title: str, aspects: str, desc: str, signal: str):
        return self.prompt.format(title, aspects, desc, signal)

    async def generate_label(self, item, n_tasks_slice=1, outfile=None, debug=False):
        res = await main_gpt_call(
            gpt_caller=self,
            data=item,
            output_path=outfile,
            n_tasks_slice=n_tasks_slice,
            cols=self.cols_for_prompt,
            append_prompt_features=False
        )
        return res[0]

    async def generate_labels(self, input_df, outpath, n_tasks_slice=2):
        await main_gpt_call(
            gpt_caller=self,
            data=input_df,
            output_path=outpath,
            n_tasks_slice=n_tasks_slice,
            cols=self.cols_for_prompt,
            append_prompt_features=False
        )