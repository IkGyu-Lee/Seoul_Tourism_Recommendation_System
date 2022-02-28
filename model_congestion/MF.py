import torch
import torch.nn as nn

# target: congestion_1, congestion_2
class MatrixFactorization(nn.Module):
    def __init__(self,
                 num_dayofweek,
                 num_time,
                 num_month,
                 num_day,
                 num_destination,
                 num_factor):
        super(MatrixFactorization, self).__init__()

        # item(destination)과의 dimension을 맞춰주기 위해, user info로 볼 수 있는 것들의 dimension을 4로 나눈다.
        num_dim = int(num_factor / 4)
        self.dayofweek_embedding = nn.Embedding(num_embeddings=num_dayofweek,
                                                embedding_dim=num_dim)
        self.time_embedding = nn.Embedding(num_embeddings=num_time,
                                           embedding_dim=num_dim)
        self.month_embedding = nn.Embedding(num_embeddings=num_month,
                                            embedding_dim=num_dim)
        self.day_embedding = nn.Embedding(num_embeddings=num_day,
                                          embedding_dim=num_dim)
        # item(destination)의 dimension은 parser로 받은 num_factor
        self.item_embedding = nn.Embedding(num_embeddings=num_destination,
                                           embedding_dim=num_factor)

        # 각 embedding을 평균 0, 표준편차 0.1로 하는 정규분포로 initialization
        nn.init.normal_(self.dayofweek_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.time_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.month_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.day_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)

    def forward(self, dayofweek, time, month, day, destination):
        # User Embedding
        dayofweek_embedded = self.dayofweek_embedding(dayofweek)
        time_embedded = self.time_embedding(time)
        month_embedded = self.month_embedding(month)
        day_embedded = self.day_embedding(day)
        user_embedding = torch.cat([dayofweek_embedded, time_embedded, month_embedded, day_embedded], dim=-1)
        # Item Embedding
        item_embedding = self.item_embedding(destination)

        # MF의 dot product
        output_MF = torch.mm(user_embedding,
                             torch.transpose(item_embedding, 0, 1))

        output_MF = torch.diagonal(output_MF, 0)
        output_MF = output_MF.view(-1)
        return output_MF