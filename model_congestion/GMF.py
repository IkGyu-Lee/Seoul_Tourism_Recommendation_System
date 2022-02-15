import torch
import torch.nn as nn

# target: congestion_1, congestion_2
class GeneralizedMatrixFactorization(nn.Module):
    def __init__(self,
                 num_dayofweek,
                 num_time,
                 num_month,
                 num_day,
                 num_destination,
                 num_factor):
        super(GeneralizedMatrixFactorization, self).__init__()

        num_dim = int(num_factor / 4)
        self.dayofweek_embedding = nn.Embedding(num_embeddings=num_dayofweek,
                                                embedding_dim=num_dim)
        self.time_embedding = nn.Embedding(num_embeddings=num_time,
                                           embedding_dim=num_dim)
        self.month_embedding = nn.Embedding(num_embeddings=num_month,
                                            embedding_dim=num_dim)
        self.day_embedding = nn.Embedding(num_embeddings=num_day,
                                          embedding_dim=num_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_destination,
                                           embedding_dim=num_factor)

        nn.init.normal_(self.dayofweek_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.time_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.month_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.day_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.1)

        predictive_size = num_factor
        self.predict_layer = nn.Linear(predictive_size, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, dayofweek, time, month, day, destination):
        # User Embedding
        dayofweek_embedded = self.dayofweek_embedding(dayofweek)
        time_embedded = self.time_embedding(time)
        month_embedded = self.month_embedding(month)
        day_embedded = self.day_embedding(day)
        user_embedding = torch.cat([dayofweek_embedded, time_embedded, month_embedded, day_embedded], dim=-1)
        # Item Embedding
        item_embedding = self.item_embedding(destination)

        output_GMF = user_embedding * item_embedding
        output_GMF = self.predict_layer(output_GMF)
        output_GMF = self.relu(output_GMF)
        output_GMF = output_GMF.view(-1)

        return output_GMF