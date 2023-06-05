import torch
import torch.nn as nn
import torch.optim as optim
import string
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CommentDataset, BehaviorDataset
from generator import Generator
from discriminator import Discriminator

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
num_epochs = 10
lr = 0.0002
latent_dim = 100
n_discriminator = 1
clip_value = 0.01
hidden_size = 256
num_topics = 5
max_length = 50

# Load data
'''
comment_dataset = CommentDataset('comment.train.txt')
comment_dataloader = DataLoader(comment_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

behavior_dataset = BehaviorDataset('behavior.train.txt')
behavior_dataloader = DataLoader(behavior_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
'''
comment_dataset = CommentDataset('comment.train.txt')

def comment_collate(batch):
        input_ids, topic_ids_list = zip(*batch)
        #input_ids = [np.array(comment).astype(str).translate(str.maketrans('', '', string.punctuation)).encode('ascii', 'ignore').decode() for comment in input_ids]
        input_ids = [comment.tolist() for comment in input_ids]
        input_ids = ["".join([str(c) for c in comment if str(c) not in string.punctuation]) for comment in input_ids]
        input_ids = [torch.tensor([ord(c) for c in comment]) for comment in input_ids]
        input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        topic_ids_list = torch.tensor(topic_ids_list)
        return input_ids, topic_ids_list

comment_dataloader = torch.utils.data.DataLoader(comment_dataset, batch_size=32, shuffle=True, collate_fn=comment_collate)

behavior_dataset = BehaviorDataset('behavior.train.txt')

def behavior_collate(batch):
        '''
        history_topic = torch.tensor(history_topic, device=device)
        comment_total_count = torch.tensor(comment_total_count, device=device)
        comment_sentiment = torch.tensor(comment_sentiment, device=device)
        comment_publish_frequency = torch.tensor(comment_publish_frequency, device=device)
        comment_reply_percentage = torch.tensor(comment_reply_percentage, device=device)
        '''
        history_topic, comment_total_count, comment_sentiment, comment_publish_frequency, comment_reply_percentage = zip(*batch)
        history_topic = torch.tensor(history_topic)
        comment_total_count = torch.tensor(comment_total_count)
        comment_sentiment = torch.tensor(comment_sentiment)
        comment_publish_frequency = torch.tensor(comment_publish_frequency)
        comment_reply_percentage = torch.tensor(comment_reply_percentage)
        return history_topic, comment_total_count, comment_sentiment, comment_publish_frequency, comment_reply_percentage

behavior_dataloader = torch.utils.data.DataLoader(behavior_dataset, batch_size=32, shuffle=True, collate_fn=behavior_collate)


# Initialize generator and discriminator
generator = Generator(hidden_size, num_topics, max_length).to(device)
discriminator = Discriminator(input_dim=len(behavior_dataset[0]), hidden_dim=128, output_dim=64, num_classes=10).to(device)

# Initialize optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr)
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

# Define loss functions
adversarial_loss = nn.BCELoss()

# Training
best_D_loss = float('inf')
best_G_loss = float('inf')

for epoch in range(num_epochs):

    G_loss_running = 0
    D_loss_running = 0

    for i, (comments, topics) in tqdm(enumerate(comment_dataloader), total=len(comment_dataloader)):

        # Adversarial ground truths
        real_labels = torch.ones((topics.size(0), 1), requires_grad=False, device=device)
        fake_labels = torch.zeros((topics.size(0), 1), requires_grad=False, device=device)

        # Configure input
        
        real_behaviors = []
        for t in topics:
            real_behaviors.append(behavior_dataset.history_topic[t.item()])
        real_behaviors = torch.tensor(real_behaviors, dtype=torch.float, device=device)
        comments = comments.to(device)
        '''
        real_behaviors = []
        for t in topics:
            # Check if the topic exists in the behavior dataset
            if t.item() not in behavior_dataset.history_topic:
                print(f"Warning: topic {t.item()} not found in behavior dataset")
                continue
            # Split the string into a list of floats
            behavior = behavior_dataset.history_topic[t.item()].strip().split(',')
            try:
                behavior = [float(x) for x in behavior]
                real_behaviors.append(behavior)
            except ValueError:
                print(f"Warning: topic {t.item()} contains non-numeric characters")
        real_behaviors = torch.tensor(real_behaviors, dtype=torch.float, device=device)
        comments = comments.to(device)
        '''
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        
        for _ in range(n_discriminator):
            # Sample noise as generator input
            z = torch.randn((topics.size(0), latent_dim), device=device)
            
            # Generate a batch of behaviors
            fake_behaviors = generator(z, topics)
            #fake_behaviors = fake_behaviors.to(device)

            # Reset gradients
            discriminator.zero_grad()

            # Discriminator loss on real data
            real_preds = discriminator(real_behaviors, topics)
            d_real_loss = adversarial_loss(real_preds, real_labels)

            # Discriminator loss on fake data
            fake_preds = discriminator(fake_behaviors.detach(), topics)
            d_fake_loss = adversarial_loss(fake_preds, fake_labels)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss

            # Backward pass
            d_loss.backward()

            # Clip weights
            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # Optimize discriminator
            optimizer_D.step()

            D_loss_running += d_loss.item()

        # -----------------
        #  Train Generator
        # -----------------

        # Reset gradients
        generator.zero_grad()

        # Sample noise as generator input
        z = torch.randn((topics.size(0), latent_dim), device=device)

        # Generate a batch of behaviors
        fake_behaviors = generator(z, topics)

        # Discriminator loss on fake data
        fake_preds = discriminator(fake_behaviors, topics)
        g_loss = adversarial_loss(fake_preds, real_labels)

        # Backward pass
        g_loss.backward()

        # Optimize generator
        optimizer_G.step()

        G_loss_running += g_loss.item()

    # Average losses for epoch
    G_loss_avg = G_loss_running / len(comment_dataloader)
    D_loss_avg = D_loss_running / (len(comment_dataloader) * n_discriminator)

    # Print losses for epoch
    print(f"Epoch {epoch+1}/{num_epochs}, G_loss={G_loss_avg:.4f}, D_loss={D_loss_avg:.4f}")

    # Save best model
    if D_loss_avg < best_D_loss:
        torch.save(discriminator.state_dict(), "best_D_model.pt")
        best_D_loss = D_loss_avg
    if G_loss_avg < best_G_loss:
        torch.save(generator.state_dict(), "best_G_model.pt")
        best_G_loss = G_loss_avg

#Print model evaluation report
print("Training complete!")
print(f"Best D_loss={best_D_loss:.4f}, Best G_loss={best_G_loss:.4f}")
