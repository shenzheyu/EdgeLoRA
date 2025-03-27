import ast
import os
import sys
from typing import List

from accelerate import Accelerator
import fire
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import transformers
from datasets import Dataset
import pandas as pd
from scipy.special import softmax

from transformers import AutoTokenizer, DataCollatorWithPadding
from trl import SFTTrainer
from transformers.models.llama import LlamaForSequenceMultiLableClassification

from peft import LoraConfig, get_peft_model_state_dict

class TextClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx],

        }

def generate_prompt(data_point):
    return f"""
            Please analyze this text accurately and predict which adpter was good for it. 
            text: {data_point["prompt"]}""".strip()


def get_data(input_data):
    new_dic = {
        "text": [i for i in input_data["prompt"]],
        "labels": [i for i in input_data["labels"]],
    }
    data0 = Dataset.from_dict(new_dic)
    return data0

def get_data_loader(tokenizer,dataset,num_classes=7):
    inputs = tokenizer([i for i in dataset["text"]], padding=True, truncation=True, return_tensors="pt")
    parsed_labels = [ast.literal_eval(item) for item in dataset["labels"]]
    labels = torch.zeros((len(parsed_labels), num_classes))
    for i, sublist in enumerate(parsed_labels):
        labels[i] = F.one_hot(torch.tensor(sublist), num_classes=num_classes).sum(dim=0)
    return TextClassificationDataset(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],labels=labels)

# class MyTrainer(SFTTrainer):

    # def compute_loss(self, model, inputs,  return_outputs=False, num_items_in_batch=None):
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     labels = inputs["labels"]
    #     loss_fn = torch.nn.BCEWithLogitsLoss()
    #     loss = loss_fn(logits, labels)
    #     return (loss, outputs) if return_outputs else loss

def compute_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.logits
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, labels)
    return loss

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    model_type: str = "LLaMA",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    adapter_name: str = "lora",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 0,
    use_gradient_checkpointing: bool = False,
    eval_step: int = 200,
    save_step: int = 200,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = None,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    print(
        f"Finetuning model with params:\n"
        f"base_model: {base_model}\n"
        f"model_type: {model_type}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"use_gradient_checkpointing: {use_gradient_checkpointing}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"wandb_project: {wandb_project}\n"
        f"wandb_run_name: {wandb_run_name}\n"
        f"wandb_watch: {wandb_watch}\n"
        f"wandb_log_model: {wandb_log_model}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    accelerator = Accelerator()

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForSequenceMultiLableClassification.from_pretrained(
        base_model,
        num_labels=7,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Allow batched inference

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # df = pd.read_csv("/workspace/code/adapter_router/adapter_data.csv")

    # df = df.sample(frac=1, random_state=85).reset_index(drop=True)

    # # Split the DataFrame
    # train_size = 0.8
    # eval_size = 0.1

    # # Calculate sizes
    # train_end = int(train_size * len(df))
    # eval_end = train_end + int(eval_size * len(df))

    # # Split the data
    # X_train = df[:train_end]
    # # X_eval = df[train_end:eval_end]
    # X_test = df[train_end:]

    X_train = pd.read_csv("/workspace/code/adapter_router/train.csv")
    X_test = pd.read_csv("/workspace/code/adapter_router/test.csv")

    # Generate prompts for training and evaluation data
    X_train.loc[:, "prompt"] = X_train.apply(generate_prompt, axis=1)
    # X_eval.loc[:, "prompt"] = X_eval.apply(generate_prompt, axis=1)

    # Generate test prompts and extract true labels
    y_true = [[X_test.loc[i, f'score_{j}'] for j in range(7)] for i in range(len(X_test))]
    X_test.loc[:, "prompt"] = X_test.apply(generate_prompt, axis=1)

    # Convert to datasets
    train_data = get_data(X_train)
    # eval_data = get_data(X_eval)
    test_data = get_data(X_test)

    train_loader = get_data_loader(tokenizer, train_data)
    # eval_loader = get_data_loader(tokenizer, eval_data)
    test_loader = get_data_loader(tokenizer, test_data)

    # if not ddp and torch.cuda.device_count() > 1:
    #     # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    #     model.is_parallelizable = True
    #     model.model_parallel = True

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_step if val_set_size > 0 else None,
        save_steps=save_step,
        output_dir=output_dir,
        save_total_limit=3,
        save_safetensors=False,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
    )

    model.config.use_cache = False
    model.config.pad_token_id = model.config.eos_token_id
    model.config.loss_type = "ForSequenceClassification"
    model.config.problem_type = "multi_label_classification"

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        max_seq_length=cutoff_len,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        dataset_text_field="text",
        peft_config=peft_config,
        data_collator=data_collator,
    )

    trainer.compute_loss_func = compute_loss_func

    trainer.accelerator.print(f"{trainer.model}")

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    trainer.train()

    trainer.save_model(output_dir)

    if os.environ.get("LOCAL_RANK") == "0":
        accuracy = {'bbh': [0, 0], 'gpqa': [0, 0], 'ifeval': [0, 0], 'math': [0, 0], 'mmlu_pro': [0, 0], 'musr': [0, 0]}

        with torch.no_grad():
            tr_data = DataLoader(test_loader, batch_size=1, shuffle=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for step, batch in tqdm(enumerate(tr_data)):
                batch = tuple(batch[t].to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                test_output = model(b_input_ids,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                pr = softmax(test_output.logits[0].cpu().numpy())
                # print index of pr in decending order
                y_pred = pr.argmax()
                task = X_test.loc[step, 'task']
                accuracy[task][0] += X_test.loc[step, f'score_{y_pred}']
                accuracy[task][1] += 1
            
        print("Accuracy:")
        for key in accuracy:
            print(f"{key}: {accuracy[key][0] / accuracy[key][1]}")


if __name__ == "__main__":
    # import debugpy
    # try:
    #     debugpy.listen(9501)
    #     print("Waiting for debugger attach")
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     print(e)
    fire.Fire(train)
