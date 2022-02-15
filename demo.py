import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_utils import Preprocessing, Input_Dataset
from model_visitor.GMF import GMF
from model_visitor.MLP import MLP
from model_visitor.NeuMF import NeuMF

'''
    date : 2018-01-01 ~ 2020-12-31
    destination : 관광지 코드
    time : 방문한 시간대
    sex : 성
    age : 나이
    visitor : 방문객수
    year : 년도
    month : 월
    day : 일
    dayofweek : 요일
    total_num : 총 수용인원수
    area : 관광지 면적
    date365 : 0~365
    congestion_1 : 방문객수 / 총수용인원수
    congestion_2 : 방문객수 / 관광지면적
'''


def input_filterchar(userinfo:str):
    str=""
    for token in userinfo:
        if ord(token)<48 or ord(token)>57:
            break
        str+=token
    return int(str)

def char2num(chr):
    day_dict ={'월':0,'화':1,'수':2,'목':3,'금':4,'토':5,'일':6}
    return day_dict[chr]

def str2datetime(date_info:list):
    month, day, dayofweek = None, None, None
    for i,token in enumerate(date_info):
        str = ""
        if i==0 :
            month = int(input_filterchar(token))
        if i==1:
            day = int(input_filterchar(token))
        if i==2:
            for c in token:
                if c!='요':
                    str+=c
                else:
                    break
            dayofweek = char2num(str)
    return month, day, dayofweek

def time2range(time):
    if time<6:
        return 1
    if time<11:
        return 2
    if time<14:
        return 3
    if time<18:
        return 4
    if time<21:
        return 5
    if time<24:
        return 6

def age2range(age):
    return ((age//10)*10)*100+ ((age//10)*10+9)

def sex2int(sex):
    return 1 if sex[0]=='남'else 0

def define_args():
    shuffle = 0
    use_pretrain = False
    model_name = 'GMF'
    epochs = 10
    batch = 512
    num_factors = 16
    num_layers = 3
    return shuffle, model_name, epochs, batch, num_factors, num_layers, use_pretrain

if __name__ == '__main__' :
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # print GPU information
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    # print("몇명이서 관광할 계획이신가요? ex) 3명")
    # num_people = input_filterchar(input())
    # print("몇월 몇일 무슨 요일에 놀러갈 계획이신가요? ex) 1월 3일 수요일")
    # date = list(input().split())
    # print("시간대는 언제가 좋으신가요? ex) 13시")
    # timezone = time2range(input_filterchar(input()))
    # month, day, dayofweek = str2datetime(date)
    # total_user_info =[]
    # for i in range(1, num_people + 1):
    #     li=[month,day,dayofweek,timezone]
    #     print(f"{i}번째 분의 어떤 연령대 인가요?. ex) 20대")
    #     li.append(age2range(input_filterchar(input())))
    #     print(f"{i}번째 분의 성별은 무엇이신가요?. ex) 남성/여성")
    #     li.append(sex2int(input()))
    #     total_user_info.append(li)
    # RecSys_total_input = total_user_info

    # for sample testcase
    RecSys_total_input =[]
    with open('sample_input.txt',mode='r') as f:
        for line in f:
            RecSys_total_input.append([int(x) for x in line.split(',')])

    # print converted user info
    print("\n변환된 user info는 다음과 같습니다.")
    for i in RecSys_total_input:
        print(i)

    # check for congestion
    print("혼잡도를 고려한 관광지 추천 리스트를 원하시나요?")
    check_congestion = True if input().lower()=='yes' else False

    # input for topk
    # print("총 몇개의 관광지가 포함된 추천 리스트를 원하시요?")
    # topk = input()
    topk = 10

    print("-------------------Load Destination_info-------------------\n")
    shuffle, model_name, epochs, batch, num_factors, num_layers, use_pretrain = define_args()
    data = Preprocessing(shuffle=shuffle)
    num_destination, num_time, num_sex, num_age, num_dayofweek, num_month, num_day = data.get_num()
    destination_id_name_df, destination_list = data.destination_list()
    batch_candidate = 100
    print("Load Destination_info complete\n")
    print("-------------------Load Model-------------------\n")
    FOLDER_PATH = 'pretrain_model'
    MODEL_PATH_VISITOR = os.path.join(FOLDER_PATH,f'{shuffle}_{use_pretrain}_{model_name}_{epochs}_{batch}_visitor.pth')
    MODEL_PATH_CONGESTION = os.path.join(FOLDER_PATH,f'{shuffle}_{use_pretrain}_{model_name}_{epochs}_{batch}_congestion_1.pth')
    if not os.path.exists(MODEL_PATH_VISITOR) or not os.path.exists(MODEL_PATH_CONGESTION):
        print("Model doesn't exist.\n")
        sys.exit()

    if model_name == 'GMF':
        model_visitor = GMF(num_factor=num_factors,
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
        model_congestion = GMF(num_factor=num_factors,
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
    elif model_name == 'MLP':
        model_visitor = MLP(num_factor=num_factors,
                            num_layer=num_layers,
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
        model_congestion = MLP(num_factor=num_factors,
                                num_layer=num_layers,
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
    elif model_name == 'NeuMF':
        if use_pretrain:
            # visitor pretrain load
            GMF_dir_visitor = os.path.join(FOLDER_PATH, f'{shuffle}_{use_pretrain}_{model_name}_{epochs}_{batch}_visitor.pth')
            MLP_dir_visitor = os.path.join(FOLDER_PATH, f'{shuffle}_{use_pretrain}_{model_name}_{epochs}_{batch}_visitor.pth')
            pretrained_GMF_visitor = torch.load(GMF_dir_visitor)
            pretrained_MLP_visitor = torch.load(MLP_dir_visitor)
            # congestion pretrain load
            GMF_dir_congestion = os.path.join(FOLDER_PATH, f'{shuffle}_{use_pretrain}_{model_name}_{epochs}_{batch}_congestion_1.pth')
            MLP_dir_congestion = os.path.join(FOLDER_PATH, f'{shuffle}_{use_pretrain}_{model_name}_{epochs}_{batch}_congestion_1.pth')
            pretrained_GMF_congestion = torch.load(GMF_dir_congestion)
            pretrained_MLP_congestion = torch.load(MLP_dir_congestion)
        else:
            pretrained_GMF_visitor = None
            pretrained_MLP_visitor = None
            pretrained_GMF_congestion = None
            pretrained_MLP_congestion = None

        model_visitor = NeuMF(num_factor=num_factors,
                              num_layer=num_layers,
                              num_dayofweek=num_dayofweek,
                              num_time=num_time,
                              num_sex=num_sex,
                              num_age=num_age,
                              num_month=num_month,
                              num_day=num_day,
                              num_destination=num_destination,
                              use_pretrain=use_pretrain,
                              use_NeuMF=True,
                              pretrained_GMF=pretrained_GMF_visitor,
                              pretrained_MLP=pretrained_MLP_visitor)
        model_congestion = NeuMF(num_factor=num_factors,
                                  num_layer=num_layers,
                                  num_dayofweek=num_dayofweek,
                                  num_time=num_time,
                                  num_sex=num_sex,
                                  num_age=num_age,
                                  num_month=num_month,
                                  num_day=num_day,
                                  num_destination=num_destination,
                                  use_pretrain=use_pretrain,
                                  use_NeuMF=True,
                                  pretrained_GMF=pretrained_GMF_congestion,
                                  pretrained_MLP=pretrained_MLP_congestion)

    model_visitor.load_state_dict(torch.load(MODEL_PATH_VISITOR,map_location=device))
    model_congestion.load_state_dict(torch.load(MODEL_PATH_CONGESTION,map_location=device))
    print("Load Model complete\n")

    total_ranking = {}

    for i,user_input in enumerate(RecSys_total_input):
        user_df = destination_id_name_df
        RecSys_dataset = Input_Dataset(destination_list=destination_list, RecSys_input=user_input)
        RecSys_dataloader = DataLoader(dataset=RecSys_dataset,
                                       batch_size=batch_candidate,
                                       shuffle=False)
        for destination, time, sex, age, dayofweek, month, day in RecSys_dataloader:
            destination = destination.to(device)
            dayofweek, time, sex, age, month, day = dayofweek.to(device), time.to(device), sex.to(device),\
                                                    age.to(device), month.to(device), day.to(device)
            pred_visitor = model_visitor(dayofweek, time, sex, age, month, day, destination)
            pred_congestion = model_congestion(dayofweek, time, sex, age, month, day, destination)
        pred_visitor = pred_visitor.view(-1)
        pred_congestion = pred_congestion.view(-1)
        pred_visitor = pred_visitor.tolist()
        pred_congestion = pred_congestion.tolist()
        user_df['visitor'] = pred_visitor
        user_df['congestion'] = pred_congestion
        user_df = user_df.sort_values(by='visitor', ascending=False)

        print(f'\n-------------------{i + 1}번째 사람을 위한 Top {topk}등 추천지 입니다.-------------------\n')

        for k in range(topk):
            destionation_name = user_df.iloc[k, 1]
            pred_target = user_df.iloc[k, 2]
            pred_congestion = user_df.iloc[k, 3]
            print(f'{k + 1}등:\t{pred_target}\t{destionation_name}')
            if (rank_weight := total_ranking.get(destionation_name)) is None:
                total_ranking[destionation_name] = []
            total_ranking[destionation_name][0] += topk - k
            total_ranking[destionation_name][1] += pred_congestion

    sorted_total_ranking = sorted(total_ranking.items(), key=lambda item: item[1][0], reverse=True)

    if check_congestion:
        print(f'-------------------혼잡도를 고려한 랭킹을 다시 하겠습니다.-------------------')
        total_ranking_congest = {}
        for i, (dest, _) in enumerate(sorted_total_ranking):
            total_ranking_congest[dest] = total_ranking[dest][1] * np.reciprocal(np.log2(i + 2))

        sorted_total_ranking = sorted(total_ranking_congest.items(), key=lambda item: item[1], reverse=True)

    print(f'\n-------------------전체 Top {topk}등 추천지 입니다.-------------------\n')
    print("전체 랭킹리스트 개수: ", len(sorted_total_ranking))
    for k in range(topk):
        print(f'{k + 1}등 :\t{sorted_total_ranking[k]}')