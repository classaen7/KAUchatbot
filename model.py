# model.py

from transformers import AutoTokenizer, ElectraForSequenceClassification
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator", num_labels=4,problem_type="multi_label_classification").to(device)

model.load_state_dict(torch.load('C:/Users/CHOI/Downloads/intent_model.pt',map_location=torch.device('cpu')))