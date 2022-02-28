import torch
import torch.nn as nn

# target: visitor
class Create_userId(nn.Module):
    def __init__(self,
                 num_factor,
                 num_dayofweek,
                 num_time,
                 num_sex,
                 num_age,
                 num_month,
                 num_day,
                 use_pretrain,
                 pretrained_GMF,
                 pretrained_MLP,
                 GMF_model=True):
        super(Create_userId, self).__init__()
        self.use_pretrain = use_pretrain
        self.pretrained_GMF = pretrained_GMF
        self.pretrained_MLP = pretrained_MLP

        # userId Embedding
        if GMF_model:
            # GMF: item(destination) dimension과 동일해야 하기 때문에, user info에 해당하는 6개의 feature들의 num_factor를 6으로 나눔
            num_dim = int(num_factor / 6)
            self.dayofweek_embedding = nn.Embedding(num_embeddings=num_dayofweek,
                                                      embedding_dim=num_dim)
            self.time_embedding = nn.Embedding(num_embeddings=num_time,
                                                      embedding_dim=num_dim)
            self.sex_embedding = nn.Embedding(num_embeddings=num_sex,
                                                      embedding_dim=num_dim)
            self.age_embedding = nn.Embedding(num_embeddings=num_age,
                                                      embedding_dim=num_dim)
            self.month_embedding = nn.Embedding(num_embeddings=num_month,
                                                      embedding_dim=num_dim)
            self.day_embedding = nn.Embedding(num_embeddings=num_day,
                                                      embedding_dim=num_dim)
        else:
            # MLP: item(destination) dimension이 num_factor * (2 ** (num_layer - 2))에 해당하기 때문에,
            # user info에 해당하는 6개의 feature들의 num_factor를 3으로 나눔
            num_dim = int(num_factor / 3)
            self.dayofweek_embedding = nn.Embedding(num_embeddings=num_dayofweek,
                                                      embedding_dim=num_dim)
            self.time_embedding = nn.Embedding(num_embeddings=num_time,
                                                      embedding_dim=num_dim)
            self.sex_embedding = nn.Embedding(num_embeddings=num_sex,
                                                      embedding_dim=num_dim)
            self.age_embedding = nn.Embedding(num_embeddings=num_age,
                                                      embedding_dim=num_dim)
            self.month_embedding = nn.Embedding(num_embeddings=num_month,
                                                      embedding_dim=num_dim)
            self.day_embedding = nn.Embedding(num_embeddings=num_day,
                                                      embedding_dim=num_dim)

        # userId Network(create userId by MLP layer)
        # if GMF_model:
        #     self.User_Net = nn.Sequential(nn.Linear(num_dim * 6, num_factor),
        #                                   nn.BatchNorm1d(num_factor),
        #                                   nn.LeakyReLU()
        #                                   )
        # else:
        #     self.User_Net = nn.Sequential(nn.Linear(num_dim * 6, num_factor * 2),
        #                                   nn.BatchNorm1d(num_factor * 2),
        #                                   nn.LeakyReLU()
        #                                   )

        # NeuMF(with pretraining)을 사용할 때, pretrained GMF/MLP에서 각 embedding의 weight 불러오기
        if use_pretrain:
            # pretrained GMF model에 해당하는 weight 불러오기
            if GMF_model:
                self.dayofweek_embedding.weight.data.copy_(
                    self.pretrained_GMF.Create_userId.dayofweek_embedding.weight)
                self.time_embedding.weight.data.copy_(
                    self.pretrained_GMF.Create_userId.time_embedding.weight)
                self.sex_embedding.weight.data.copy_(
                    self.pretrained_GMF.Create_userId.sex_embedding.weight)
                self.age_embedding.weight.data.copy_(
                    self.pretrained_GMF.Create_userId.age_embedding.weight)
                self.month_embedding.weight.data.copy_(
                    self.pretrained_GMF.Create_userId.month_embedding.weight)
                self.day_embedding.weight.data.copy_(
                    self.pretrained_GMF.Create_userId.day_embedding.weight)
            # pretrained MLP model에 해당하는 weight 불러오기
            else:
                self.dayofweek_embedding.weight.data.copy_(
                    self.pretrained_MLP.Create_userId.dayofweek_embedding.weight)
                self.time_embedding.weight.data.copy_(
                    self.pretrained_MLP.Create_userId.time_embedding.weight)
                self.sex_embedding.weight.data.copy_(
                    self.pretrained_MLP.Create_userId.sex_embedding.weight)
                self.age_embedding.weight.data.copy_(
                    self.pretrained_MLP.Create_userId.age_embedding.weight)
                self.month_embedding.weight.data.copy_(
                    self.pretrained_MLP.Create_userId.month_embedding.weight)
                self.day_embedding.weight.data.copy_(
                    self.pretrained_MLP.Create_userId.day_embedding.weight)
        # NeuMF(with pretraining)을 사용하지 않을 때, embedding initialization
        else:
            # Embedding weight initialization(normal|uniform)
            nn.init.normal_(self.dayofweek_embedding.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.time_embedding.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.sex_embedding.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.age_embedding.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.month_embedding.weight, mean=0.0, std=0.1)
            nn.init.normal_(self.day_embedding.weight, mean=0.0, std=0.1)

            # nn.init.uniform_(self.dayofweek_embedding.weight, a=-1, b=1)
            # nn.init.uniform_(self.time_embedding.weight, a=-1, b=1)
            # nn.init.uniform_(self.sex_embedding.weight, a=-1, b=1)
            # nn.init.uniform_(self.age_embedding.weight, a=-1, b=1)
            # nn.init.uniform_(self.month_embedding.weight, a=-1, b=1)
            # nn.init.uniform_(self.day_embedding.weight, a=-1, b=1)

            # Layer weight initialization
            # for layer in self.User_Net:
                # if isinstance(layer, nn.Linear):
                    # nn.init.uniform_(layer.weight)
                    # nn.init.normal_(layer.weight)
                    # nn.init.xavier_uniform_(layer.weight)
                    # nn.init.kaiming_uniform_(layer.weight)

    def forward(self, dayofweek, time, sex, age, month, day):
        # Embedding
        dayofweek_embedded = self.dayofweek_embedding(dayofweek)
        time_embedded = self.time_embedding(time)
        sex_embedded = self.sex_embedding(sex)
        age_embedded = self.age_embedding(age)
        month_embedded = self.month_embedding(month)
        day_embedded = self.day_embedding(day)

        # dayofweek, time, sex, age, month, day concat
        output_userId = torch.cat([dayofweek_embedded, time_embedded, sex_embedded, age_embedded, month_embedded, day_embedded], dim=-1)
        # output_userId = self.User_Net(output_userId)
        # print('userId ouput shape:', output_userId.shape)

        return output_userId
