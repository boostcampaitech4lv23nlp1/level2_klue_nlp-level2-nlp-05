import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, EarlyStoppingCallback
# https://huggingface.co/transformers/v3.0.2/_modules/transformers/trainer.html
import data_loaders.data_loader as dataloader
import utils.util as utils

import model.model as custom
from transformers import DataCollatorWithPadding
# https://huggingface.co/course/chapter3/4

import mlflow
import mlflow.sklearn

from ray import air, tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
# from ray.air import session
# from ray.air.callbacks.mlflow import MLflowLoggerCallback
# from ray.tune.integration.mlflow import mlflow_mixin

from azureml.core import Workspace
import argparse


def start_mlflow(experiment_name):
  #Enter details of your AzureML workspace
  subscription_id = "0275dc6c-996d-42d1-8263-8f7b4e81f271"
  resource_group = "basburger"
  workspace_name = "basburger"
  ws = Workspace.get(name=workspace_name,
                    subscription_id=subscription_id,
                    resource_group=resource_group)

  mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

  # https://learn.microsoft.com/ko-kr/azure/machine-learning/how-to-log-view-metrics?tabs=interactive
  mlflow.set_experiment(experiment_name)
  # Start the run
  mlflow_run = mlflow.start_run()

# for ray_tune har
def ray_hp_space(trial):
  """_summary_
  for ray_tune hyperparameter setting
  Args:
      trial (_type_): _description_

  Returns:
      _type_: _description_
  """
  return {
      "learning_rate": tune.loguniform(1e-6, 1e-4),
      "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
  }

# 여기서 하드코딩 말고 config에서 model_name, num_label을 가져올 방법은 없을까....?
# def model_init(trial) :
#     print('😀 This is trial :', trial)
#     return custom.CustomModel('klue/roberta-base', 30)

def train(args):
  #huggingface-cli login  #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

  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  MODEL_NAME = args.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

  experiment_name = args.model_name+'_bs'+str(args.batch_size)+'_ep'+str(args.max_epoch)+'_lr'+str(args.learning_rate)+'_dt_'+args.train_data
  start_mlflow(experiment_name)

  # load dataset
  RE_train_dataset = dataloader.load_train_dataset(tokenizer, args.train_data)
  RE_dev_dataset = dataloader.load_dev_dataset(tokenizer, args.dev_data)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  print(device)
  # setting model hyperparameter
  #model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  #model_config.num_labels = 30
  num_labels = 30

  #model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  model = custom.CustomModel(MODEL_NAME, num_labels)
  print(model.config)
  model.parameters
  model.to(device)

  # model_init을 여기다가 선언해도 되나???
  def model_init(trial) :
    print('😀 This is trial :', trial)
    return custom.CustomModel(MODEL_NAME, num_labels)

  # optimizer와 learning rate scheduler를 지정해줍니다.
  '''optimizer=transformers.AdamW(model.parameters(), lr=5e-5)
  scheduler=transformers.get_cosine_schedule_with_warmup(
      optimizer=optimizer,
      num_warmup_steps=500,
      num_training_steps=60000)
  optimizers = optimizer, scheduler'''


  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    hub_model_id="jstep750/basburger",
    output_dir='./output',          # output directory
    save_total_limit=5,              # number of total save model.
    save_steps=500,                 # model saving step.
    # num_train_epochs=args.max_epoch,              # total number of training epochs
    # learning_rate=args.learning_rate,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,            # evaluation step.
    load_best_model_at_end = True,
    push_to_hub=True
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=utils.compute_metrics,         # define metrics function
    data_collator=data_collator,
    #optimizers=optimizers,
    model_init=model_init,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
  )

  # For Hugging Face hyperparameter_search
  trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    hp_space=ray_hp_space,
    n_trials=2,
  )

  # # train model
  # trainer.train()
  #model.save_pretrained('./best_model')
  mlflow.end_run()
  trainer.push_to_hub()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', default='klue/roberta-base', type=str)
  parser.add_argument('--batch_size', default=16, type=int)
  parser.add_argument('--max_epoch', default=5, type=int)
  parser.add_argument('--shuffle', default=True)

  parser.add_argument('--learning_rate', default=5e-5, type=float)
  parser.add_argument('--train_data', default='train')
  parser.add_argument('--dev_data', default='dev')
  parser.add_argument('--test_data', default='dev')
  parser.add_argument('--predict_data', default='test')
  parser.add_argument('--optimizer', default='AdamW')
  parser.add_argument('--loss_function', default='L1Loss')

  parser.add_argument('--preprocessing', default=False)
  parser.add_argument('--precision', default=32, type=int)
  parser.add_argument('--dropout', default=0.1, type=float)
  args = parser.parse_args()

  # check hyperparameter arguments
  # print(args)
  train(args)