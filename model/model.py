from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, RobertaForSequenceClassification
from torch import nn
import model.loss as loss_module
from torch.cuda.amp import autocast
import torch

class Model(nn.Module):
    def __init__(self, conf, new_vocab_size):
        super().__init__()
        self.num_labels = 30
        self.conf = conf
        self.model_name = conf.model.model_name
        # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        self.model.resize_token_embeddings(new_vocab_size)
        self.loss_fct = loss_module.loss_config[conf.train.loss]

    @autocast()  
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            if(self.conf.train.rdrop):
                loss = self.rdrop(logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return outputs

    def rdrop(self, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        logits2 = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits.view(-1, self.num_labels)
        
        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss


class CustomModel(nn.Module):
    def __init__(self, conf, new_vocab_size):
        super(CustomModel, self).__init__()
        self.num_labels = 30
        self.model_name = conf.model.model_name
        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.resize_token_embeddings(new_vocab_size)
        self.hidden_dim = self.model.config.hidden_size
        self.loss_fct = loss_module.loss_config[conf.train.loss]

        self.activation = torch.tanh
        self.dense = nn.Linear(self.hidden_dim, self.hidden_dim * 4)
        self.dropout = nn.Dropout(conf.train.dropout)
        self.out_proj = nn.Linear(self.hidden_dim * 4, self.num_labels)

    def process(self, input_ids=None, attention_mask=None, token_type_ids=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Add custom layers
        features = outputs[0]  # outputs[0]=last hidden state
        x = features[:, 0, :] # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        return logits
    
    @autocast() 
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        logits = self.process(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        loss = None
        if labels is not None:
            loss_fct = self.loss_fct
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            if(self.conf.train.rdrop):
                loss = self.rdrop(logits, labels, input_ids, attention_mask, token_type_ids)
            return loss, logits
        return logits

    def rdrop(self, logits, labels, input_ids, attention_mask, token_type_ids, alpha=0.1):
        logits2 = self.process(input_ids, attention_mask, token_type_ids)
        # cross entropy loss for classifier
        logits = logits.view(-1, self.num_labels)
        logits2 = logits.view(-1, self.num_labels)
        
        ce_loss = 0.5 * (self.loss_fct(logits, labels.view(-1)) + self.loss_fct(logits2, labels.view(-1)))
        kl_loss = loss_module.compute_kl_loss(logits, logits2)
        # carefully choose hyper-parameters
        loss = ce_loss + alpha * kl_loss
        return loss