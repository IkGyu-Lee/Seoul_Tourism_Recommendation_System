import torch
import torch.nn as nn
from model_visitor.Create_userId import Create_userId

# target: visitor
class MLP(nn.Module):
    def __init__(self,
                 num_factor,
                 num_layer,
                 num_dayofweek,
                 num_time,
                 num_sex,
                 num_age,
                 num_month,
                 num_day,
                 num_destination,
                 use_pretrain,
                 use_NeuMF,
                 pretrained_MLP):
        super(MLP, self).__init__()
        self.use_pretrain = use_pretrain
        self.use_NeuMF = use_NeuMF
        self.pretrained_MLP = pretrained_MLP

        # user info로 userId output 만들 객체 생성
        self.Create_userId = Create_userId(num_factor=num_factor,
                                           num_dayofweek=num_dayofweek,
                                           num_time=num_time,
                                           num_sex=num_sex,
                                           num_age=num_age,
                                           num_month=num_month,
                                           num_day=num_day,
                                           use_pretrain=use_pretrain,
                                           pretrained_GMF=None,
                                           pretrained_MLP=pretrained_MLP,
                                           GMF_model=False)
        # itemId Embedding
        self.item_embedding = nn.Embedding(num_embeddings=num_destination,
                                           embedding_dim=num_factor * (2 ** (num_layer - 2)))

        # MLP layer module lsit 생성
        MLP_modules = []
        for i in range(1, num_layer):
          input_size = num_factor * (2 ** (num_layer - i))
          MLP_modules.append(nn.Linear(input_size, input_size // 2))
          MLP_modules.append(nn.BatchNorm1d(input_size // 2))
          MLP_modules.append(nn.LeakyReLU())
        # list안에 있는 module list Sequential로 연결시켜주기
        self.MLP_layers = nn.Sequential(*MLP_modules)

        # 최종 predictive_size는 parser로 받은 num_factor로 맞춘다.
        predict_size = num_factor
        self.predict_layer = nn.Linear(predict_size, 1)
        # activation function은 LeakyReLU
        self.relu = nn.LeakyReLU()

        # NeuMF(with pretraining)을 사용할 때, embedding & layer weight, bias들 불러오기
        if use_pretrain:
            self.item_embedding.weight.data.copy_(
                self.pretrained_MLP.item_embedding.weight)
            for layer, pretrained_layer in zip(self.MLP_layers, self.pretrained_MLP.MLP_layers):
                if isinstance(layer, nn.Linear) and isinstance(pretrained_layer, nn.Linear):
                    layer.weight.data.copy_(pretrained_layer.weight)
                    layer.bias.data.copy_(pretrained_layer.bias)
        # NeuMF(with pretraining)을 사용할 때, embedding & layer weight initialization
        else:
            # Layer weight initialization(NeuMF용이 아니라, MLP용 일때만)
            if not use_NeuMF:
                # nn.init.uniform_(self.predict_layer.weight)
                # nn.init.normal_(self.predict_layer.weight)
                # nn.init.xavier_uniform_(self.predict_layer.weight)
                nn.init.kaiming_uniform_(self.predict_layer.weight)

            # Embedding weight initialization(normal|uniform)
            nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)
            # nn.init.uniform_(self.item_embedding.weight, a=-1, b=1)

            # Layer weight initialization
            for layer in self.MLP_layers:
                if isinstance(layer, nn.Linear):
                    # nn.init.uniform_(layer.weight)
                    # nn.init.normal_(layer.weight)
                    # nn.init.xavier_uniform_(layer.weight)
                    nn.init.kaiming_uniform_(layer.weight)

    def forward(self, dayofweek, time, sex, age, month, day, destination):
        destination_embedded = self.item_embedding(destination)
        vector = torch.cat([self.Create_userId(dayofweek, time, sex, age, month, day), destination_embedded], dim=-1)

        if self.use_NeuMF == False:
            output_MLP = self.MLP_layers(vector)
            output_MLP = self.predict_layer(output_MLP)
            output_MLP = self.relu(output_MLP)
            output_MLP = output_MLP.view(-1)
        else:
            output_MLP = self.MLP_layers(vector)
        # print('MLP ouput shape', output_MLP.shape)
        return output_MLP