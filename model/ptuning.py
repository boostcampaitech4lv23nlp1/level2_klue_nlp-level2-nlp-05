from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, RobertaForSequenceClassification
from transformers import BertPreTrainedModel, AutoModel
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


class MSAModel(torch.nn.Module):
    """main model"""

    def __init__(self, conf):
        super().__init__()

        self.model = BertForSequenceClassification.from_pretrained("monologg/kobert", num_labels=30)
        self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

        self.embeddings = self.model.bert.get_input_embeddings()
        self.embedding_dim = self.embeddings.embedding_dim  # 768

        self.prompt_token_id = self.tokenizer.get_vocab()["<unused1>"]
        self.prompt_token_len = 15
        self.prompt_encoder = PromptEncoder(self.prompt_token_len, self.embedding_dim, 0.2)

    def embed_input(self, input_ids):
        bs = input_ids.shape[0]  # batch size
        embeds = self.embeddings(input_ids)

        prompt_token_position = torch.nonzero(input_ids == self.prompt_token_id).reshape((bs, self.prompt_token_len, 2))[:, :, 1]
        prompt_embeds = self.prompt_encoder()
        for bidx in range(bs):
            for i in range(self.prompt_token_len):
                embeds[bidx, prompt_token_position[bidx, i], :] = prompt_embeds[i, :]

        return embeds

    def forward(self, input_ids, attention_mask, labels):
        inputs_embeds = self.embed_input(input_ids)
        output = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        loss, logits = output.loss, output.logits

        pred = logits[labels != -100]
        probs = pred[:, self.label_id_list]
        pred_labels_idx = torch.argmax(probs, dim=-1).tolist()
        y_ = [self.label_id_list[i] for i in pred_labels_idx]

        y = labels[labels != -100]

        return loss, y_, y.tolist()
