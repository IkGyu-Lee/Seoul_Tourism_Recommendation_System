import torch
import torch.nn as nn
from model_visitor.GMF import GMF
from MLP import MLP

# NCF with RNN
class VanillaRNN(nn.Module):
    def __init__(self,
                 num_factor,
                 num_layer,
                 num_date,
                 num_dayofweek,
                 num_time,
                 num_sex,
                 num_age,
                 num_destination,
                 num_hidden,
                 rnn_layer=2):
        super(VanillaRNN, self).__init__()
        self.hidden_size = num_hidden
        self.num_layer = rnn_layer
        self.date_embedding = nn.Embedding(num_embeddings=num_date,
                                           embedding_dim=num_factor)
        self.GMF = GMF(num_factor=num_factor,
                       num_dayofweek=num_dayofweek,
                       num_time=num_time,
                       num_sex=num_sex,
                       num_age=num_age,
                       num_destination=num_destination)
        self.MLP = MLP(num_factor=num_factor,
                       num_layer=num_layer,
                       num_dayofweek=num_dayofweek,
                       num_time=num_time,
                       num_sex=num_sex,
                       num_age=num_age,
                       num_destination=num_destination)
        self.VanillaRNN_layer = nn.RNN(input_size=num_factor*3,   # (batchsize, sequence len(time step), input_size), 1개의 batch를 한 번에 넣으려면 1/ 3번에 쪼개서 넣으려면 3
                                       hidden_size=num_hidden,    # size(dimension) of hidden state
                                       dropout=0.2,
                                       num_layers=rnn_layer,      # 각 cell에 layer를 몇 개씩 쌓을건지, overfitting과 관련있음
                                       batch_first=True)

    def forward(self, batch, hidden, dayofweek, time, sex, age, destination, date):
        output_NeuMF = torch.cat([self.GMF(dayofweek, time, sex, age, destination), self.MLP(dayofweek, time, sex, age, destination)], dim=-1)
        output_NeuMF = output_NeuMF.unsqueeze(1)
        # print(f'output_NeuMF:{output_NeuMF.shape}')
        date_embedded = self.date_embedding(date)
        date_embedded = date_embedded.unsqueeze(1)
        # print(f'date_embedded:{date_embedded.shape}')

        input_rnn = torch.cat([date_embedded, output_NeuMF], dim=2)
        # print(f'input_rnn:{input_rnn.shape}')
        input_rnn = input_rnn.view(batch//7, 7, -1)
        # print(f'input_rnn:{input_rnn.shape}')
        outputs, hidden = self.VanillaRNN_layer(input_rnn, hidden)
        # print(f'outputs:{outputs.shape}')
        outputs = outputs.sum(2)
        # print(f'outputs.sum(2):{outputs.shape}')
        outputs = outputs.view(-1)
        # print(f'outputs:{outputs.shape}')
        return outputs

    def initHidden(self, batch):
        return torch.zeros(self.num_layer, batch//7, self.hidden_size)