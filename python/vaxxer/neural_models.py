import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

from transformers.file_utils import ModelOutput
from typing import Optional, Tuple

from transformers import AutoModel, BertModel
from transformers import get_scheduler

import time
from tqdm import tqdm
import numpy as np
import pandas as pd

from vaxxer.utils import calculate_metrics, metrics2df

##################
# Network architecture
##################  

class NeuralModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    states: Optional[Tuple[torch.FloatTensor]] = None 
    
class NeuralNet(nn.Module):
    """ Configurable NeuralNet for optional BERT, LSTM, inter layer and a non-optional fully connected classifier head."""
    def __init__(self, num_labels: int, #NeuralModel arguments
                 input_dim: int, #dimension of input if bert=None, or dimension of appendices to stack to bert if provided
                 mode: str="simple", inter_dim: int=128, dropout: float=0.0, #fully connected arguments
                 #Bert arguments
                 bert: BertModel=None, bert_lr: float=7e-5, bert_dropout: float=0.0, train_bert: bool=False,
                 #LSTM arguments
                 use_lstm: bool=False, lstm_dim: int=128, maintain_lstm_state: bool=True, 
                 verbose: bool=True):
        super(NeuralNet, self).__init__()
        self.batch_norm = False
        self.output_layer_dim = num_labels# if num_labels > 2 else 1
        self.mode = mode
        self.description = ""
        # 'dim' variable is keeping the current size of the output tensor 
        dim = self._init_bert(input_dim, train_bert, bert, bert_lr, bert_dropout)
        self._init_bn1(dim)
        print("Neural network input dimension:", dim)
        dim = self._init_lstm(dim, use_lstm, lstm_dim, maintain_lstm_state)
        dim = self._init_inter_layer(dim, inter_dim, dropout)
        # set last layer
        self.classifier = nn.Linear(dim, self.output_layer_dim)
        self.description += "FC(%i,%i)" % (dim, self.output_layer_dim)
        print(self.description)
        
    def _init_bert(self, input_dim, train_bert, bert, bert_lr, bert_dropout):
        if bert is not None:
            self.bert_lr = bert_lr
            self.bert = bert
            for param in bert.parameters():
                param.requires_grad = train_bert
            self.bert_dropout = bert_dropout
            bert_dim = bert.config.hidden_size
            dim = bert_dim + input_dim
            self.use_bert = True
            self.description += "BERT(%s,%f,%f,%i)->" % (train_bert, bert_lr, bert_dropout, bert_dim)
            if input_dim > 0:
                self.description += "concat(%i,%i)->" % (bert_dim, input_dim)
        else:
            dim = input_dim
            self.use_bert = False
            self.description += "input(%i)->" % input_dim
        return dim
            
    def _init_bn1(self, dim):
        self.use_bn1 = False
        if self.batch_norm and (dim > 0):
            self.use_bn1 = True
            self.bn1 = nn.BatchNorm1d(dim)
            self.description += "BN(%i)->" % dim
        
    def _init_lstm(self, dim, use_lstm, lstm_dim, maintain_lstm_state):
        self.use_lstm = False
        self.maintain_lstm_state = maintain_lstm_state
        self.states = None
        if use_lstm:
            self.use_lstm = True
            self.lstm_dim = lstm_dim
            self.lstm = nn.LSTM(dim, lstm_dim, batch_first=True)
            self.description += "LSTM(%i,%s,%i)->" % (dim, maintain_lstm_state, lstm_dim)
            dim = lstm_dim
        return dim
            
    def _init_inter_layer(self, dim, inter_dim, dropout):
        self.use_inter = False
        self.use_bn2 = False
        if self.mode == "inter" and inter_dim != None and inter_dim > 0:
            self.inter = nn.Linear(dim, inter_dim)
            self.dropout = dropout
            self.description += "FC(%i,%f,%i)->" % (dim, dropout, inter_dim)
            dim = inter_dim
            self.use_inter = True
            if self.batch_norm:
                self.use_bn2 = True
                self.bn2 = nn.BatchNorm1d(dim)
                self.description += "BN(%i)->" % dim
            self.description += "RELU->"
        return dim
    
    def forward(self, data, labeled=False, states=None, return_dict=True):
        if self.use_bert:
            # data components: 0-input_ids, 1-mask
            x = self.bert(data[0], attention_mask=data[1],
                          token_type_ids=None, 
                          output_hidden_states=False, output_attentions=False)
            x = F.dropout(x.pooler_output, self.bert_dropout)
            #if there is more data (e.g. network, history, tfidf components)
            if len(data) > 3: # label is always present at the last index
                # data components: 2-modalities
                x = torch.cat([x, data[2]], dim=1)
        else:
            x = data[0]
        if self.use_bn1:
            x = self.bn1(x)
        if self.use_lstm:
            if len(x.size()) < 3: #lstm expects 3d array as input: (batch, seq_len, feature)
                x = x.unsqueeze(1)
            dim_match = 0 #at last step states.size() might not match with shape of x
            if states is not None and x.size(0) != states[0].size(1):
                dim_match = states[0].size(1) - x.size(0) #cat zeros to x to match size of states
                if dim_match > 0:
                    augmentation = torch.zeros(dim_match, x.size(1), x.size(2)).to(x.device)
                    x = torch.cat([x, augmentation], dim=0)
                else:
                    augmentation = torch.zeros(1, -dim_match, states[0].size(2)).to(states[0].device)
                    states = [torch.cat([state, augmentation], dim=1) for state in states]
            x, states = self.lstm(x, states)
            states = [state.detach() for state in states]
            x = x.view(-1, self.lstm_dim) #shape back to normal form
        if self.use_inter:
            x = self.inter(x)
            x = F.dropout(x, self.dropout)
            if self.use_bn2:
                x = self.bn2(x)
            x = F.relu(x)
        logits = self.classifier(x)
        if self.use_lstm and dim_match > 0:
            logits = logits[:-dim_match]
        loss = None #calculate loss when labels are present
        if labeled:
            if self.output_layer_dim == 1:
                criterion = nn.BCEWithLogitsLoss()
                labels = torch.flatten(data[-1]).float()
                logits_for_loss = torch.flatten(logits).float()
                loss = criterion(logits_for_loss, labels)
            else:
                criterion = nn.CrossEntropyLoss()
                labels = data[-1].view(-1).long()
                loss = criterion(logits.view(-1, self.output_layer_dim), labels)
        if return_dict:
            return NeuralModelOutput(loss=loss, logits=logits, states=states)
        else:
            return (loss, logits, states)

##################
# Utility functions
##################  

def set_device(device: str=None, verbose: bool=True):
    """Set device for torch models to use. If no GPU is found then CPU will be used automatically."""
    if torch.cuda.is_available():      
        device = torch.device("cuda" if device == None else device)
        if verbose:
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use GPU:', torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        if verbose:
            print('No GPU available, using the CPU instead.')
    return device

def get_dataloader(dataset: TensorDataset, batch_size: int=32, shuffle: bool=True):
    """Returns a dataloader for dataset. For training data, you should use a random sampler, while for prediction and evaluation a sequential one."""
    if shuffle:
        dataloader = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = batch_size)
    else:
        dataloader = DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = batch_size)
    return dataloader
        
##################
# Neural models
##################  
        
class NeuralModels():
    """Absrtact class for handling different PyTorch models."""
    def __init__(self, num_labels: int, batch_size: int, epochs: int, lr_rate: float, schedule_type: str, device: str=None, show_confusion_matrix: bool=False, verbose: bool=False):
        self.device = set_device(device, True)
        print(self.device)
        self.output_layer_dim = num_labels
        self.multiclass = num_labels > 2
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr_rate = lr_rate
        self.schedule_type = schedule_type
        self.show_confusion_matrix = show_confusion_matrix
        self.verbose = verbose
        
    def close(self):
        """Close model"""
        del self.model
        del self.optimizer
        del self.scheduler
        
    def _load_model(self, train: bool=True):
        device = self.device
        model = self.model      
        if train:
            optimizer = self.optimizer
            scheduler = self.scheduler
            model.train()
        else:
            model.eval()
            optimizer = None
            scheduler = None
        states = self.model.states
        return device, model, states, optimizer, scheduler
    
    def _save_model(self, result, model, optimizer, scheduler, probas, truth, train: bool=True, labeled: bool=False):
        self.model = model
        self.probas = np.concatenate(probas, axis=0)
        if train:
            self.optimizer = optimizer
            self.scheduler = scheduler
        if self.model.maintain_lstm_state:
            self.model.states = result.states
        if labeled:
            self.truth = np.concatenate(truth, axis=0)
        else:
            self.truth = None
        
    def _epoch(self, dataloader: DataLoader, train: bool=True, labeled: bool=False):
        """
        Modes: 
        train=True is training. 
        train=False is evaluating or predicting, depending on the presence of labels.
        """
        device, model, states, optimizer, scheduler = self._load_model(train)
        labeled = True if train else labeled
        total_loss = 0
        truth, probas = [], []
        start = time.time()
        for step, batch in enumerate(dataloader):
            batch = [r.to(device) for r in batch]
            if train:
                model.zero_grad()
                result = self.pass_to_model(batch, labeled, states) 
            else:
                with torch.no_grad():
                    result = self.pass_to_model(batch, labeled, None)    
            logits = result.logits
            logits = logits.detach().cpu().numpy()
            if len(logits.shape) < 2:
                logits = logits.reshape((1,-1))
            probas.append(logits)
            if labeled:
                loss = result.loss
                total_loss += loss.item()
                # batch index: the last element (-1) is the label
                labels = batch[-1].to('cpu').numpy()
                truth.append(labels)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            if self.model.maintain_lstm_state:
                states = result.states
        self._save_model(result, model, optimizer, scheduler, probas, truth, train, labeled)
        avg_loss = total_loss / len(dataloader) 
        duration = time.time() - start
        return avg_loss, duration
    
    def pass_to_model(self, batch, labeled=False, states=None):
        """
        How to pass forward data through model. 
        BertHuggingface's model is not a NeuralNet instance, so it will override this method.
        """
        return self.model(batch, labeled, states)      
        
    def fit(self, data: TensorDataset, shuffle: bool=True):
        """Fit model for the provided training data"""
        self.optimizer = AdamW(self.model.parameters(), lr = self.lr_rate, eps = 1e-8)
        dataloader = get_dataloader(data, batch_size=self.batch_size, shuffle=shuffle)
        num_tr_steps = len(dataloader) * self.epochs
        self.scheduler = get_scheduler(self.schedule_type, self.optimizer, num_warmup_steps = 0, num_training_steps = num_tr_steps)
        start = time.time()
        for epoch in tqdm(range(self.epochs)):
            self._epoch(dataloader, train=True)
        duration = time.time() - start
        print("Training time: %f seconds" % duration)
    
    def predict(self, data: TensorDataset, labeled=False):
        """Return prediction for the provided data. Note that the 'fit()' function must be executed in advance!"""
        dataloader = get_dataloader(data, batch_size=self.batch_size, shuffle=False)
        self._epoch(dataloader, train=False, labeled=labeled)
        probas = np.copy(self.probas)
        return probas
    
    def fit_eval_predict(self, train_data: TensorDataset, test_data: TensorDataset, return_probas=False):
        """First train a new model then evaluate performance for both training and testing set. Predictions are also returned if needed."""
        self.fit(train_data, shuffle=True)
        train_metrics = calculate_metrics(self.truth, self.probas, self.show_confusion_matrix, self.verbose)
        _ = self.predict(test_data, labeled=True)
        test_metrics = calculate_metrics(self.truth, self.probas, self.show_confusion_matrix, self.verbose)
        if return_probas:
            train_probas = self.predict(train_data, labeled=False)
            test_probas = self.predict(test_data, labeled=False)
        else:
            train_probas, test_probas = None, None
        self.close()
        return metrics2df({"train": train_metrics, "test": test_metrics}), train_probas, test_probas
        
class TextClassifier(NeuralModels):
    def __init__(self, num_labels: int, batch_size:int, epochs:int, lr_rate:float, schedule_type: str, 
                 input_dim: int, mode: str, inter_dim: int, dropout: float, device: str=None,
                 show_confusion_matrix: bool=False, verbose: bool=False):
        super(TextClassifier, self).__init__(num_labels, batch_size, epochs, lr_rate, schedule_type, device, show_confusion_matrix, verbose)
        self.model = NeuralNet(num_labels, input_dim, mode, inter_dim, dropout)
        self.model.to(self.device)
        
class BertClassifier(NeuralModels):
    def __init__(self, num_labels, batch_size, epochs, lr_rate, schedule_type, 
                 input_dim: int, mode: str, inter_dim: int, dropout: float, 
                 bert: str=None, bert_dropout: float=0.0, train_bert: bool=False, device: str=None,
                 show_confusion_matrix: bool=False, verbose: bool=False):
        super(BertClassifier, self).__init__(num_labels, batch_size, epochs, lr_rate, schedule_type, device, show_confusion_matrix, verbose)
        print(bert)
        bert_obj = AutoModel.from_pretrained(bert)
        self.model = NeuralNet(num_labels, input_dim, mode, inter_dim, dropout,
                               bert=bert_obj, bert_dropout=bert_dropout, train_bert=train_bert)
        self.model.to(self.device)

class LSTMClassifier(NeuralModels):
    def __init__(self, num_labels, batch_size, epochs, lr_rate, schedule_type, 
                 input_dim: int, mode: str, inter_dim: int, dropout: float,
                 use_lstm: bool=False, lstm_dim: int=128, maintain_lstm_state: bool=True, device: str=None,
                 show_confusion_matrix: bool=False, verbose: bool=False):
        super(LSTMClassifier, self).__init__(num_labels, batch_size, epochs, lr_rate, schedule_type, device, show_confusion_matrix, verbose)
        self.model = NeuralNet(num_labels, input_dim, mode, inter_dim, dropout,
                               use_lstm=use_lstm, lstm_dim=lstm_dim, maintain_lstm_state=maintain_lstm_state)
        self.model.to(self.device)
        
class BertLSTMClassifier(NeuralModels):
    def __init__(self, num_labels, batch_size, epochs, lr_rate, schedule_type, 
                 input_dim: int, mode: str, inter_dim: int, dropout: float, 
                 bert: str=None, bert_dropout: float=0.0, train_bert: bool=False,
                 use_lstm: bool=False, lstm_dim: int=128, maintain_lstm_state: bool=True, device: str=None,
                 show_confusion_matrix: bool=False, verbose: bool=False):
        super(BertLSTMClassifier, self).__init__(num_labels, batch_size, epochs, lr_rate, schedule_type, device, show_confusion_matrix, verbose)
        print(bert)
        bert_obj = AutoModel.from_pretrained(bert)
        self.model = NeuralNet(num_labels, input_dim, mode, inter_dim, dropout,
                               bert=bert_obj, bert_dropout=bert_dropout, train_bert=train_bert,
                               use_lstm=use_lstm, lstm_dim=lstm_dim, maintain_lstm_state=maintain_lstm_state)
        self.model.to(self.device)
        
class EmbeddingClassifier(NeuralModels):
    pass