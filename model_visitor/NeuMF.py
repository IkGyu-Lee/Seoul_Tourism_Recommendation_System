import torch
import torch.nn as nn
from model_visitor.GMF import GMF
from model_visitor.MLP import MLP

# target: visitor
class NeuMF(nn.Module):
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
                 pretrained_GMF,
                 pretrained_MLP):
        super(NeuMF, self).__init__()
        self.use_pretrain = use_pretrain
        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

        # GMF 객체 생성
        self.GMF = GMF(num_factor=num_factor,
                       num_dayofweek=num_dayofweek,
                       num_time=num_time,
                       num_sex=num_sex,
                       num_age=num_age,
                       num_month=num_month,
                       num_day=num_day,
                       num_destination=num_destination,
                       use_pretrain=use_pretrain,
                       use_NeuMF=use_NeuMF,
                       pretrained_GMF=pretrained_GMF)
        # MLP 객체 생성
        self.MLP = MLP(num_factor=num_factor,
                       num_layer=num_layer,
                       num_dayofweek=num_dayofweek,
                       num_time=num_time,
                       num_sex=num_sex,
                       num_age=num_age,
                       num_month=num_month,
                       num_day=num_day,
                       num_destination=num_destination,
                       use_pretrain=use_pretrain,
                       use_NeuMF=use_NeuMF,
                       pretrained_MLP=pretrained_MLP)

        # GMF와 MLP의 output을 합쳤기 때문에 num_factor의 2배
        self.predict_layer = nn.Linear(num_factor*2, 1)
        # activation function은 LeakyReLU
        self.relu = nn.LeakyReLU()

        if use_pretrain:
            predict_weight = torch.cat([
                self.pretrained_GMF.predict_layer.weight,
                self.pretrained_MLP.predict_layer.weight], dim=1)
            predict_bias = self.pretrained_GMF.predict_layer.bias + \
                           self.pretrained_MLP.predict_layer.bias
            self.predict_layer.weight.data.copy_(0.5 * predict_weight)
            self.predict_layer.bias.data.copy_(0.5 * predict_bias)
        else:
            # Layer weight initialization
            # nn.init.uniform_(self.predict_layer.weight)
            # nn.init.normal_(self.predict_layer.weight)
            # nn.init.xavier_uniform_(self.predict_layer.weight)
            nn.init.kaiming_uniform_(self.predict_layer.weight)


    def forward(self, dayofweek, time, sex, age, month, day, destination):
        output_NeuMF = torch.cat([self.GMF(dayofweek, time, sex, age, month, day, destination),
                                  self.MLP(dayofweek, time, sex, age, month, day, destination)],dim=-1)
        output_NeuMF = self.predict_layer(output_NeuMF)
        output_NeuMF = self.relu(output_NeuMF)
        output_NeuMF = output_NeuMF.view(-1)
        # print('NeuMF ouput shape', NeuMF_output.shape)
        return output_NeuMF