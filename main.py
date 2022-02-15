import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from parser import args
from data_utils import Tourism, Preprocessing, Congestion_Dataset
from model_congestion.MF import MatrixFactorization
from model_congestion.GMF import GeneralizedMatrixFactorization
from model_visitor.GMF import GMF
from model_visitor.MLP import MLP
from model_visitor.NeuMF import NeuMF
from evaluate import RMSE, RMSE_con
import warnings
warnings.filterwarnings('ignore')

############################# CONFIGURATION #############################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
print('Model Name:', format(args.model_name))
if args.shuffle == 1:
    print('Stratify Split')
else:
    print("Each Year Split")

# argparse doesn't support boolean type
use_pretrain = True if args.use_pretrain == 'True' else False
save_model = True if args.save_model == 'True' else False

# select rating
rating_name = 'visitor' if args.target == 'v' else('congestion_1' if args.target == 'c1' else 'congestion_2')
print(f'Selected target is {rating_name}')

pretrain_dir = 'pretrain_model'
if not os.path.exists(pretrain_dir):
    os.mkdir(pretrain_dir)

############################## PREPARE DATASET ##########################
data = Preprocessing(shuffle=args.shuffle)
num_destination, num_time, num_sex, num_age, num_dayofweek, num_month, num_day = data.get_num()

# normalization되어 있는 train/test df 불러오기
train_df, test_df = data.preprocessing()

if rating_name == 'visitor':
    train_dataset = Tourism(train_df, rating_name)
    test_dataset = Tourism(test_df, rating_name)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False, drop_last=True)
else:
    train_dataset = Congestion_Dataset(train_df, rating_name)
    test_dataset = Congestion_Dataset(test_df, rating_name)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch, shuffle=False, drop_last=True)
########################### CREATE MODEL #################################
# target: visitor
if rating_name == 'visitor':
    if args.model_name == 'GMF':
        model = GMF(num_factor=args.num_factors,
                    num_dayofweek=num_dayofweek,
                    num_time=num_time,
                    num_sex=num_sex,
                    num_age=num_age,
                    num_month=num_month,
                    num_day=num_day,
                    num_destination=num_destination,
                    use_pretrain=use_pretrain,
                    use_NeuMF=False,
                    pretrained_GMF=None)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    elif args.model_name == 'MLP':
        model = MLP(num_factor=args.num_factors,
                    num_layer=args.num_layers,
                    num_dayofweek=num_dayofweek,
                    num_time=num_time,
                    num_sex=num_sex,
                    num_age=num_age,
                    num_month=num_month,
                    num_day=num_day,
                    num_destination=num_destination,
                    use_pretrain=use_pretrain,
                    use_NeuMF=False,
                    pretrained_MLP=None)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.model_name == 'NeuMF':
        if use_pretrain:
            GMF_dir = os.path.join(pretrain_dir, f'{args.shuffle}_False_GMF_{args.epochs}_{args.batch}_{rating_name}.pth')
            MLP_dir = os.path.join(pretrain_dir, f'{args.shuffle}_False_MLP_{args.epochs}_{args.batch}_{rating_name}.pth')

            pretrained_GMF = GMF(num_factor=args.num_factors,
                                 num_dayofweek=num_dayofweek,
                                 num_time=num_time,
                                 num_sex=num_sex,
                                 num_age=num_age,
                                 num_month=num_month,
                                 num_day=num_day,
                                 num_destination=num_destination,
                                 use_pretrain=False,
                                 use_NeuMF=False,
                                 pretrained_GMF=None)
            pretrained_MLP = MLP(num_factor=args.num_factors,
                                 num_layer=args.num_layers,
                                 num_dayofweek=num_dayofweek,
                                 num_time=num_time,
                                 num_sex=num_sex,
                                 num_age=num_age,
                                 num_month=num_month,
                                 num_day=num_day,
                                 num_destination=num_destination,
                                 use_pretrain=False,
                                 use_NeuMF=False,
                                 pretrained_MLP=None)

            pretrained_GMF.load_state_dict(torch.load(GMF_dir))
            pretrained_MLP.load_state_dict(torch.load(MLP_dir))

            # 신경망의 모든 매개변수를 고정합니다
            for param in pretrained_GMF.parameters():
                param.requires_grad = False

            for param in pretrained_MLP.parameters():
                param.requires_grad = False
        else:
            pretrained_GMF = None
            pretrained_MLP = None

        model = NeuMF(num_factor=args.num_factors,
                      num_layer=args.num_layers,
                      num_dayofweek=num_dayofweek,
                      num_time=num_time,
                      num_sex=num_sex,
                      num_age=num_age,
                      num_month=num_month,
                      num_day=num_day,
                      num_destination=num_destination,
                      use_pretrain=use_pretrain,
                      use_NeuMF=True,
                      pretrained_GMF=pretrained_GMF,
                      pretrained_MLP=pretrained_MLP)
        if not use_pretrain:
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
# target: congestion
else:
    if args.model_name == 'MF':
        model = MatrixFactorization(num_dayofweek=num_dayofweek,
                                    num_time=num_time,
                                    num_month=num_month,
                                    num_day=num_day,
                                    num_destination=num_destination,
                                    num_factor=args.num_factors)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    elif args.model_name == 'GMF':
        model = GeneralizedMatrixFactorization(num_dayofweek=num_dayofweek,
                                               num_time=num_time,
                                               num_month=num_month,
                                               num_day=num_day,
                                               num_destination=num_destination,
                                               num_factor=args.num_factors)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

model.to(device)
criterion = nn.MSELoss()
########################### TRAINING #####################################
# Training visitor
if rating_name == 'visitor':
    print('--------------------Train Start---------------------')
    start = datetime.now()
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for destination, time, sex, age, dayofweek, month, day, visitor in train_dataloader:
            # itemId
            destination = destination.to(device)
            # user information(userId)
            dayofweek,time,sex,age,month,day = dayofweek.to(device),time.to(device),sex.to(device),age.to(device),month.to(device),day.to(device)
            # rating(target)
            visitor = visitor.to(device)

            # gradient 초기화
            optimizer.zero_grad()
            prediction = model(dayofweek, time, sex, age, month, day, destination)
            loss = criterion(prediction, visitor)
            # RMSE LOSS
            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            # batch마다의 loss
            total_loss += loss

        # evaluation
        model.eval()
        rmse = RMSE(model, criterion, test_dataloader, device)

        print("Epoch: {}\tTRAIN Average RMSE Loss: {}\tTEST RMSE: {}".format(epoch+1, total_loss/len(train_dataloader), rmse))

    end = datetime.now()
    print(f'Training Time: {end-start}')
    print('-------------------Train Finished-------------------')

# Training congestion_1 & congestion_2
else:
    print('--------------------Train Start---------------------')
    start = datetime.now()
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for destination, time, dayofweek, month, day, congestion in train_dataloader:
            # itemId
            destination = destination.to(device)
            # user information(userId)
            dayofweek, time, month, day = dayofweek.to(device), time.to(device), month.to(device), day.to(device)
            # rating(target)
            congestion = congestion.to(device)

            # gradient 초기화
            optimizer.zero_grad()
            prediction = model(dayofweek, time, month, day, destination)
            loss = criterion(prediction, congestion)
            # RMSE LOSS
            loss = torch.sqrt(loss)
            loss.backward()
            optimizer.step()
            # batch마다의 loss
            total_loss += loss

        # evaluation
        model.eval()
        rmse = RMSE_con(model, criterion, test_dataloader, device)

        print("Epoch: {}\tTRAIN Average RMSE Loss: {}\tTEST RMSE: {}".format(epoch + 1,
                                                                             total_loss / len(train_dataloader), rmse))

    end = datetime.now()
    print(f'Training Time: {end - start}')
    print('-------------------Train Finished-------------------')

if save_model:
    MODEL_PATH = os.path.join(pretrain_dir,
                              f'{args.shuffle}_{use_pretrain}_{args.model_name}_{args.epochs}_{args.batch}_{rating_name}.pth')
    torch.save(model.state_dict(), MODEL_PATH)