###generator.py
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer

class Generator(nn.Module):
    def __init__(self, hidden_size, num_topics, max_length):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.num_topics = num_topics
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.topic_embeddings = nn.Embedding(num_topics, hidden_size)
        self.linear = nn.Linear(hidden_size*2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, self.max_length)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, topic_ids):
        input_ids = input_ids.squeeze()
        topic_embeddings = self.topic_embeddings(topic_ids)
        comment_embeddings = self.bert(input_ids)[1] 
        concat = torch.cat((comment_embeddings, topic_embeddings), dim=1)
        hidden_state = self.linear(concat)
        output = self.output_layer(hidden_state)
        output = self.softmax(output)
        return output
