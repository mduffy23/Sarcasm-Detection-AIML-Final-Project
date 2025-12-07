import torch
from torch import nn
from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Model defintion with Lexical Features
class RobertaWithLexical(nn.Module):
    def __init__(self, model_name="roberta-base", feature_dim=5, num_labels=2, dropout=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + feature_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, lexical_features=None, labels=None, **kwargs):
        # Roberta embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Lexical features
        if lexical_features is None:
            lexical_features = torch.zeros(cls_emb.size(0), 0, device=cls_emb.device, dtype=cls_emb.dtype)
        else:
            lexical_features = lexical_features.to(cls_emb.device)
            if lexical_features.dtype != cls_emb.dtype:
                lexical_features = lexical_features.to(dtype=cls_emb.dtype)

        # Concatenate
        combined = torch.cat([cls_emb, lexical_features], dim=1)
        pooled = self.dropout(combined)
        logits = self.classifier(pooled)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return proper HF output
        return SequenceClassifierOutput(loss=loss, logits=logits)