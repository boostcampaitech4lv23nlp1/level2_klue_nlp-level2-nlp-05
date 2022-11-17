# Relation Extraciton(RE, 문장 내 개체간 관계 추출)

## Project Description

관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 

이번 대회에서는 문장, 단어에 대한 정보를 통해 ,문장 속에서 단어 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 단어들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 


<br/>

## 데이터셋 
| Dataset            | train                    | test |
| ------------------ | ----------------------- |--------------- |
| **문장 수**        | 32470      |     7765   |
| **비율**        | 95      |     5 |

<br/>

### Columns
* **id** (문자열) : 문장 고유 ID 

* **sentence** (문자열) : 주어진 문장

* **subject_entity** (딕셔너리) : 주체 entity

* **object_entity** (딕셔너리) : 객체 entity

* **label** : (문자열) 30가지 label에 해당하는 주체와 객체간 관계

* **source** : (문자열) 문장의 출처

    * **wikipedia** (위키피디아)

    * **wikitree** (위키트리)

    * **policy_briefing** (정책 보도 자료?)

<br>

## Folder Structure (*수정 예정!*)

```
📁data
├─📁raw_data
| ├─train.csv
| ├─dev.csv
| └─test.csv
├─README.md
└─submission_format.csv

```



## Set up

### 1. Requirements

```bash
$ pip install -r requirements.txt
```

### 2. Prepare Dataset
train split -> train + dev

<br>

# How to Run

## How to train

```bash
$ python train.py
```

```bash
$ sh train.sh
```

## How to Inference

```bash
$ python inference.py
```
