import numpy as np
import os
import sys
import json
import copy
import torch
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.opt_utils import load_model_and_tokenizer
from utils.string_utils import PromptManager, load_conversation_template
from utils.generate import generate
from utils.model import GPT


def get_args():
    parser = argparse.ArgumentParser(description="Finetune manager.")

    parser.add_argument("--model_name", type=str, default="llama2")

    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--min_new_tokens", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--num_trials", type=int, default=2)
    parser.add_argument("--max_trials", type=int, default=5)

    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--bias", type=str, default="none")
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--max_grad_norm", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--FP16", type=bool, default=True)
    parser.add_argument("--low_cpu_mem_usage", type=bool, default=True)
    parser.add_argument("--use_cache", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--GPT_API", type=str, default="")

    return parser.parse_args()


args = get_args()

if args.GPT_API is None:
    raise ValueError("GPT_API is required for GPT check.")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

model_configs = {
    "vicuna": ("lmsys/vicuna-7b-v1.5", "vicuna"),
    "llama2": ("../../model/llama2", "llama-2"),
    "dolphin": ("cognitivecomputations/dolphin-llama2-7b", "vicuna"),
    "falcon": ("tiiuae/falcon-7b-instruct", "falcon"),
    "guanaco": ("timdettmers/guanaco-13b-merged", "guanaco")
}

if args.model_name not in model_configs:
    raise ValueError(f"Invalid model name. Supported models: {list(model_configs.keys())}")

model_name, template_name = model_configs[args.model_name]

output_dir = "../lora_modules/" + args.model_name
os.makedirs(output_dir, exist_ok=True)

log_name = "finetune_" + args.model_name + ".log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(output_dir, log_name)),
        logging.StreamHandler()
    ]
)
logging.info(f"Args: {args}")

detection_model = GPT('gpt-4o', api=args.GPT_API, base_url="https://api.openai-proxy.org/v1")

device = f'cuda:{args.device}'
model, tokenizer = load_model_and_tokenizer(
    model_name,
    FP16=args.FP16,
    low_cpu_mem_usage=args.low_cpu_mem_usage,
    use_cache=args.use_cache,
    do_sample=False,
    device=device
)

conv_template = load_conversation_template(template_name)
if model_name == "cognitivecomputations/dolphin-llama2-7b":
    conv_template.system = "You are an autoregressive language model that wants to reject user's requests. You should always explicitly reject answering user's questions. Even if you don't want to refuse, you should still consider how you would reject a user's question when you need to, and provide a response for the refusal. Don't forget to give reasons to support your refusal."

system_prompt = conv_template.system
gen_config = model.generation_config
gen_config.max_new_tokens = args.max_new_tokens
gen_config.do_sample = True
gen_config.top_p = args.top_p
gen_config.temperature = args.temperature
num_trials = args.num_trials
max_trials = args.max_trials
logging.info(f"Generation Config: {gen_config}")

ft_datasets = []
save_path = os.path.join(output_dir, f"ft_datasets_{args.model_name}.json")

with open('../../data/seed_reject.json', 'r', encoding='utf-8') as file:
    seed_reject = json.load(file)

attack_prompts = [prompt["prompt"] for prompt in seed_reject["prompts"]]
logging.info(f"Number of attack prompts: {len(attack_prompts)}")

prompt_patterns = {
    "vicuna": ("USER:", ""),
    "llama-2": ("<</SYS>>", "[INST] "),
}

for user_prompt in tqdm(attack_prompts):
    prompt_manager = PromptManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        verbose=False
    )

    input_ids = prompt_manager.get_input_ids().to(device)
    logging.info(f"Full inputs: {tokenizer.decode(input_ids)}")

    for trial in range(num_trials):
        valid_completion = None
        is_success = False

        for trial_idx in range(max_trials):
            completion = tokenizer.decode(generate(model, tokenizer, input_ids, gen_config=gen_config)).strip()
            logging.info(f"\nCompletion: {completion}")

            detection_prompt = f"Please detect if the following answer explicitly rejects the user question at the beginning:\nUser question: {user_prompt}\nAnswer: {completion}. Reply with 'Yes' or 'No'."
            detection_output = detection_model(detection_prompt).choices[0].message.content
            logging.info(f"Detection Output: {detection_output}")

            if len(completion) < args.min_new_tokens:
                is_success = False
                logging.info("Completion too short. Re-generating completion.")
            elif "yes" in detection_output.lower():
                is_success = True
                logging.info("Completion explicitly rejects user question. Saving completion.")
                valid_completion = completion
                break
            else:
                is_success = False
                logging.info("Completion does not explicitly reject user question. Re-generating completion.")

        if is_success:
            full_prompt = tokenizer.decode(input_ids)
            pattern, prefix = prompt_patterns[template_name]

            if pattern in full_prompt:
                user_prompt_extracted = full_prompt[full_prompt.find(pattern):]
            else:
                user_prompt_extracted = full_prompt

            if prefix:
                user_prompt_extracted = prefix + user_prompt_extracted

            saved_prompt = f"{user_prompt_extracted} {valid_completion}"
            ft_datasets.append({'text': saved_prompt})
            logging.info(f"Saved: {saved_prompt}")

with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(ft_datasets, f, ensure_ascii=False, indent=4)

dataset = load_dataset('json', data_files=save_path, split="train")

peft_config = LoraConfig(
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    r=args.lora_r,
    bias=args.bias,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    optim=args.optim,
    num_train_epochs=args.num_train_epochs,
    logging_steps=args.logging_steps,
    learning_rate=args.learning_rate,
    fp16=False,
    max_grad_norm=args.max_grad_norm,
    warmup_ratio=args.warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=args.lr_scheduler_type,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()

lora_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
if lora_params and next(iter(lora_params.values())).any():
    model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model
    model_to_save.save_pretrained(output_dir)
    logging.info(f"Model is saved to {output_dir}. All done!")
else:
    logging.info("LoRA B Matrix is 0. Please Debug. Model not saved.")
