from transformers import PreTrainedTokenizerFast, GPT2ForSequenceClassification
from torch import nn
import torch
import model.loss as loss_module
from torch.cuda.amp import autocast
import torch


class PromptEncoder(nn.Module):
    def __init__(self, prompt_token_len, hidden_size, device, lstm_dropout):
        super().__init__()
        print("[#] Init prompt encoder...")
        # Input [0,1,2,3,4,5,6,7,8]
        self.seq_indices = torch.LongTensor(list(range(prompt_token_len))).to(device)
        # Embedding
        self.embedding = nn.Embedding(prompt_token_len, hidden_size)
        # LSTM
        self.lstm_head = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=2, dropout=lstm_dropout, bidirectional=True, batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds


class PTuneForGPT2(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.model = GPT2ForSequenceClassification.from_pretrained("skt/kogpt2-base-v2", num_labels=30)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>", mask_token="<mask>")

        # GPT2 모델에서는 마지막 출력층만 학습을 하고 본체는 학습을 하지 않습니다.
        for layer_name, param in self.model.named_parameters():
            if "score.weight" in layer_name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.embeddings = self.model.get_input_embeddings()
        self.embedding_dim = self.embeddings.embedding_dim  # 768

        # prompt token으로 사용할 토큰 아이디 선택
        self.prompt_token_id = self.tokenizer.get_vocab()["<unused1>"]
        # prompt 인풋에서 사용할 prompt token의 개수
        self.prompt_token_len = 17
        self.prompt_encoder = PromptEncoder(self.prompt_token_len, self.embedding_dim, 0.2)

    def embed_input(self, input_ids):
        bs = input_ids.shape[0]  # batch size
        embeds = self.embeddings(input_ids)

        # prompt token의 인덱스 위치 ex) [16, 17]
        prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
        # prompt의 임베딩 ex) prompt을 2개 사용하는 경우 [2, 768]
        prompt_embeds = self.prompt_encoder()
        for bidx in range(bs):
            for i in range(self.prompt_token_len):
                # nth_batch, 토큰들 중에서 prompt 토큰 위치, 임베딩 차원 / [1, 16, 768] = [768]
                # 1번째 배치에서 16번째 토큰이 prompt니까 해당 위치에 prompt embedding 정보를 전달
                embeds[bidx, prompt_token_position[bidx, i], :] = prompt_embeds[i, :]

        return embeds

    @autocast()
    def forward(self, input_ids, labels=None):
        inputs_embeds = self.embed_input(input_ids)
        output = self.model(inputs_embeds=inputs_embeds)
        logits = output.logits

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if self.conf.train.rdrop:
                loss = self.rdrop(logits, input_ids, labels)
            return loss, logits
        return logits

    def rdrop(self, logits, input_ids, labels, alpha=0.1):
        logits2 = self.forward(input_ids, labels)
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits.view(-1, self.num_labels)

        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss
