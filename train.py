import torch
from torch.optim.lr_scheduler import OneCycleLR

# https://huggingface.co/transformers/v3.0.2/_modules/transformers/trainer.html
# https://huggingface.co/course/chapter3/4
import transformers
from transformers import DataCollatorWithPadding, EarlyStoppingCallback
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoConfig
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast, GPT2ForSequenceClassification

import data_loaders.data_loader as dataloader
import utils.util as utils
import model.model as model_arch
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import mlflow
import mlflow.sklearn
from azureml.core import Workspace

from datetime import datetime
import re
import os
from omegaconf import OmegaConf
from collections import defaultdict


class MyDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        max_len = 0
        for i in features:
            if len(i["input_ids"]) > max_len:
                max_len = len(i["input_ids"])

        batch = defaultdict(list)
        for item in features:
            for k in item:
                if "label" not in k:
                    padding_len = max_len - item[k].size(0)
                    if k == "input_ids":
                        item[k] = torch.cat((item[k], torch.tensor([self.tokenizer.pad_token_id] * padding_len)), dim=0)
                    else:
                        item[k] = torch.cat((item[k], torch.tensor([0] * padding_len)), dim=0)
                batch[k].append(item[k])

        for k in batch:
            batch[k] = torch.stack(batch[k], dim=0)
            batch[k] = batch[k].to(torch.long)
        return batch


def start_mlflow(experiment_name):
    # Enter details of your AzureML workspace
    subscription_id = "0275dc6c-996d-42d1-8263-8f7b4e81f271"
    resource_group = "basburger"
    workspace_name = "basburger"
    ws = Workspace.get(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group)

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    # https://learn.microsoft.com/ko-kr/azure/machine-learning/how-to-log-view-metrics?tabs=interactive
    mlflow.set_experiment(experiment_name)
    # Start the run
    mlflow_run = mlflow.start_run()


def train(conf):
    # 실행 시간을 기록합니다.
    now = datetime.now()
    train_start_time = now.strftime("%d-%H-%M")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = conf.model.model_name
    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token="<s>", eos_token="</s>", unk_token="<unk>", mask_token="<mask>")

    data_collator = MyDataCollatorWithPadding(tokenizer=tokenizer)

    experiment_name = model_name + "_" + conf.model.model_class_name + "_bs" + str(conf.train.batch_size) + "_ep" + str(conf.train.max_epoch) + "_lr" + str(conf.train.learning_rate)
    start_mlflow(experiment_name)  # 간단한 실행을 하는 경우 주석처리를 하시면 더 빠르게 실행됩니다.

    # load dataset
    RE_train_dataset = dataloader.load_dataset(tokenizer, conf.path.train_path, conf)
    RE_dev_dataset = dataloader.load_dataset(tokenizer, conf.path.dev_path, conf)
    RE_test_dataset = dataloader.load_dataset(tokenizer, conf.path.test_path, conf)

    model = model_arch.PTuneForGPT2()
    model.parameters

    for layer_name, param in model.named_parameters():
        if "score.weight" in layer_name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.to(device)
    # 다른 옵티마이저를 사용하고 싶으신 경우 이 부분을 바꿔주세요.
    optimizer = transformers.AdamW(model.parameters(), lr=conf.train.learning_rate)

    # 이등변 삼각형 형태로 lr이 서서히 증가했다가 감소하는 스케줄러입니다.
    # 첫시작 lr: learning_rate/div_factor, 마지막 lr: 첫시작 lr/final_div_factor
    # 학습과정 step수를 계산해 스케줄러에 입력해줍니다. -> steps_per_epoch * epochs / 2 지점 기준으로 lr가 상승했다가 감소
    steps_per_epoch = len(RE_train_dataset) // conf.train.batch_size + 1 if len(RE_train_dataset) % conf.train.batch_size != 0 else len(RE_train_dataset) // conf.train.batch_size
    scheduler = OneCycleLR(
        optimizer,
        max_lr=conf.train.learning_rate,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.5,
        epochs=conf.train.max_epoch,
        anneal_strategy="linear",
        div_factor=1e100,
        final_div_factor=1,
    )

    training_args = TrainingArguments(
        # output directory 경로 step_saved_model/실행모델/실행시각(일-시-분)
        # -> ex. step_saved_model/klue-roberta-latge/18-12-04(표준시각이라 9시간 느림)
        # 모델이 같더라도 실행한 시간에 따라 저장되는 경로가 달라집니다. 서버 용량 관리를 잘해주세요.
        # step_saved_model 폴더에 저장됩니다.
        output_dir=f"./step_saved_model/{re.sub('/', '-', model_name)}/{train_start_time}",
        save_total_limit=conf.utils.top_k,  # save_steps에서 저장할 모델의 최대 개수
        save_steps=conf.train.save_steps,  # 이 step마다 eval_steps에서 계산한 값을 기준으로 모델을 저장합니다.
        num_train_epochs=conf.train.max_epoch,  # 학습 에포크 수
        learning_rate=conf.train.learning_rate,  # learning_rate
        per_device_train_batch_size=conf.train.batch_size,  # train batch size
        per_device_eval_batch_size=conf.train.batch_size,  # valid batch size
        # weight_decay=0.01,               # strength of weight decay 이거 머하는 건지 모르겠어요.
        logging_dir="./logs",  # directory for storing logs 로그 경로 설정인데 폴더가 안생김?
        logging_steps=conf.train.logging_steps,  # 해당 스탭마다 loss, lr, epoch가 cmd에 출력됩니다.
        evaluation_strategy="steps",
        eval_steps=conf.train.eval_steps,  # 해당 스탭마다 valid set을 이용해서 모델을 평가합니다. 이 값을 기준으로 save_steps 모델이 저장됩니다.
        load_best_model_at_end=True,
        metric_for_best_model=conf.utils.monitor,
    )
    trainer = Trainer(
        model=model,
        args=training_args,  # 위에서 설정한 training_args를 가져옵니다.
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=utils.compute_metrics,  # utils에 있는 평가 매트릭을 가져옵니다.
        data_collator=data_collator,
        optimizers=(optimizer, scheduler),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=conf.utils.patience)],
    )

    trainer.train()
    trainer.save_model(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}")

    mlflow.end_run()  # 간단한 실행을 하는 경우 주석처리를 하시면 더 빠르게 실행됩니다.
    model.eval()
    metrics = trainer.evaluate(RE_test_dataset)
    print("Training is complete!")
    print("==================== Test metric score ====================")
    print("eval loss: ", metrics["eval_loss"])
    print("eval auprc: ", metrics["eval_auprc"])
    print("eval micro f1 score: ", metrics["eval_micro f1 score"])

    # best_model 저장할 때 사용했던 config파일도 같이 저장합니다.
    if not os.path.exists(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}"):
        os.makedirs(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}")
    with open(f"./best_model/{re.sub('/', '-', model_name)}/{train_start_time}/config.yaml", "w+") as fp:
        OmegaConf.save(config=conf, f=fp.name)
