import torch
import numpy as np
import torch.nn as nn
import string
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

class CommentDataset(Dataset):
    def __init__(self, filename):
        self.topics = ['社会', '生活', '文娱', '性别', '政治', '职场']
        self.topic_ids = list(range(len(self.topics)))
        self.tokenizer = None
        self.input_ids = []
        self.topic_ids_list = []
        self.created_at = []
        self.load_data(filename)

    def load_data(self, filename):
        print('loading data from:', filename)
        lines = []
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        for line in tqdm(lines, ncols=100):
            topic, comment, created_at = line.strip().split('\t')
            topic_id = self.topics.index(topic)
            tokens = tokenizer.tokenize(comment)
            # Convert tokens to input IDs
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
            self.input_ids.append(torch.tensor(input_ids))
            #self.input_ids.append(torch.tensor(comment))
            self.topic_ids_list.append(topic_id)
            self.created_at.append(created_at)

    def __getitem__(self, index):
        return self.input_ids[index], self.topic_ids_list[index]

    def __len__(self):
        return len(self.input_ids)

    def comment_collate(self, batch):
        input_ids, topic_ids_list = zip(*batch)
        input_ids = [np.array(comment).astype(str).translate(str.maketrans('', '', string.punctuation)).encode('ascii', 'ignore').decode() for comment in input_ids]
        input_ids = [torch.tensor([ord(c) for c in comment]) for comment in input_ids]
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        topic_ids_list = torch.tensor(topic_ids_list)
        return input_ids, topic_ids_list

       

class BehaviorDataset(Dataset):
    def __init__(self, filename):
        self.topics = ['社会', '生活', '文娱', '性别', '政治', '职场']
        self.tokenizer = None
        self.history_topic = []
        self.comment_total_count = []
        self.comment_sentiment = []
        self.comment_publish_frequency = []
        self.comment_reply_percentage = []
        self.load_data(filename)

    def load_data(self, filename):
        print('loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        for line in tqdm(lines, ncols=100):
            data = line.strip().split('\t')
            history_topic = self.topics.index(data[0])
            comment_total_count = int(data[1])
            comment_sentiment = float(data[2])
            comment_publish_frequency = float(data[3])
            comment_reply_percentage = float(data[4])
            self.history_topic.append(history_topic)
            self.comment_total_count.append(comment_total_count)
            self.comment_sentiment.append(comment_sentiment)
            self.comment_publish_frequency.append(comment_publish_frequency)
            self.comment_reply_percentage.append(comment_reply_percentage)

    def __getitem__(self, index):
        return (torch.tensor(self.history_topic[index]), torch.tensor(self.comment_total_count[index]), 
                torch.tensor(self.comment_sentiment[index]), torch.tensor(self.comment_publish_frequency[index]), 
                torch.tensor(self.comment_reply_percentage[index]))

    def __len__(self):
        return len(self.history_topic)

    def behavior_collate(batch):
        history_topic, comment_total_count, comment_sentiment, comment_publish_frequency, comment_reply_percentage = zip(*batch)
        history_topic = torch.tensor(history_topic)
        comment_total_count = torch.tensor(comment_total_count)
        comment_sentiment = torch.tensor(comment_sentiment)
        comment_publish_frequency = torch.tensor(comment_publish_frequency)
        comment_reply_percentage = torch.tensor(comment_reply_percentage)
        return history_topic, comment_total_count, comment_sentiment, comment_publish_frequency, comment_reply_percentage
