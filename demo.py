import os
import sys
import torch
import numpy as np
import pandas as pd
import datetime as dt
import time as ti
from haversine import haversine
import math
import tqdm
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
    middle_category(middle_category_name) : 관광지 중분류 유형 코드(이름)
    small_category(small_category_name) : 관광지 소분류 유형 코드(이름)
    x,y : 경도,위도
    city,gu,dong : 시,구,동
'''

def define_args():
    use_pretrain = False
    model_name = 'NeuMF'  # Choice GMF, MLP, NeuMF
    epochs = 10           # Choice 20,  10,  10
    num_factors = 36      # Choice 36,  24,  36
    return use_pretrain, model_name, epochs, num_factors

def input_filterchar(userinfo:str):
    str=""
    for token in userinfo:
        if ord(token)<48 or ord(token)>57:
            break
        str+=token
    return int(str)

def str2datetime(date_info:list):
    year, month, day = None, None, None
    for i, token in enumerate(date_info):
        if i==0 :
            year = int(input_filterchar(token))
        if i==1:
            month = int(input_filterchar(token))
        if i==2:
            day = int(input_filterchar(token))
    return year, month, day

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
    return ((age//10)*10)*100 + ((age//10)*10+9)

def sex2int(sex):
    return 1 if sex[0]=='남'else 0

def destint2str(li):
    dest_dict = {'1':'역사관광지','2':'휴양관광지','3':'체험관광지','4':'문화시설','5':'건축/조형물','6':'자연관광지','7':'쇼핑'}
    dest_li=[]
    for val in li:
        dest_li.append(dest_dict[val])
    return dest_li

# destination_id_name_df: 3가지 장르 des id name, 경도 위도
# df: congestion total
def load_congestion(destination_id_name_df, df, dayofweek, time, month, day):
    dayofweek, time, day, month = dayofweek.item(), time.item(), day.item(), month.item()
    new_df = df[((df['month'] == month) & (df['day'] == day)) & ((df['dayofweek'] == dayofweek) & (df['time'] == time))]

    new_df = new_df[new_df.destination.isin(destination_id_name_df.destination)]

    return new_df['congestion_1']

def filter_destination(DEST_PATH,genre_list):
    df = pd.read_pickle(DEST_PATH)
    new_df = pd.DataFrame(columns=df.columns)

    for i in genre_list:
        temp_df = df[(df['middle_category_name'] == i)]
        new_df = pd.concat([new_df, temp_df], ignore_index=True)
    des_list = new_df['destination'].to_list()
    return new_df, des_list

# just for visualization for laoding
def progress_bar(text):
    ti.sleep(0.01)
    t = tqdm.tqdm(total=10, ncols=100, desc=text)
    for i in range(5):
        ti.sleep(0.2)
        t.update(2)
    t.close()

############################## Main ##############################
if __name__ == '__main__' :
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # print GPU information
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

############################## User Input ##############################
    # user info
    print("몇 명이서 관광할 계획이신가요? ex) 3명")
    num_people = input_filterchar(input())

    user_info =[]
    for i in range(num_people):
        tem_list = []
        print(f"{i+1}번째 분의 어떤 연령대 인가요?. ex) 20대")
        tem_list.append(age2range(input_filterchar(input())))
        print(f"{i+1}번째 분의 성별은 무엇이신가요?. ex) 남성/여성")
        tem_list.append(sex2int(input()))
        user_info.append(tem_list)

    # time info
    print("관광을 시작할 연 월 일이 언제인가요? ex) 2022년 5월 21일")
    date = list(input().split())
    year, month, day = str2datetime(date)

    print("며칠동안 관광하시나요? ex) 4일(max:7일, min:1일)")
    num_term = input_filterchar((input()))

    user_df = pd.DataFrame()
    for i in range(num_term):
        print(f"{i+1}번째 날은 어떤 시간대가 좋으신가요? ex) 13시")
        timezone = time2range(input_filterchar(input()))
        datetime = dt.date(year, month, day) + dt.timedelta(days=i)

        m, d, w, t = datetime.month, datetime.day, datetime.weekday(), timezone
        tem_df = pd.DataFrame(user_info, columns=['age', 'sex'])
        tem_df['month'] = m
        tem_df['day'] = d
        tem_df['dayofweek'] = w
        tem_df['time'] = t

        user_df = pd.concat([user_df, tem_df], axis=0, ignore_index=True)

    # print(user_df)

    # input for staring point
    print("어디서 출발하시나요? 행정구와 동을 입력해주세요. ex) 종로구 삼청동")
    start_info = input().split(' ')

    # select destination genre
    print("어떤 장르의 관광지를 원하시나요? (3개 이상 골라주세요) ex) 1,2,3"
          "\n1.역사관광지 \t2.휴양관광지\t3.체험관광지\t4.문화시설\t5.건축/조형물\t6.자연관광지\t7.쇼핑")
    genre_list = destint2str(input().split(','))

    # select ratio visitor Vs congestion Vs distance
    print("관광지를 선택하는 과정에서 '선호도/혼잡도/거리' 순별로 중요시하는 비율을 부여해주세요. ex) 0.6:0.2:0.2")
    ratio = list(map(float, (input().split(':'))))

############################## Operation ##############################
    total_start = ti.time()

    progress_bar('Loading Dataset')
    ROOT_DIR = 'dataset'
    DEST_INFO_PATH = os.path.join(ROOT_DIR, 'destination_id_name_genre_coordinate.pkl')
    PREDICTED_CONGEST_PATH = os.path.join(ROOT_DIR, 'congestion_1_2.pkl')
    CITY_INFO_PATH = os.path.join(ROOT_DIR, 'seoul_gu_dong_coordinate.pkl')

    # 3개 장르에 포함된 dataframe, list
    destination_id_name_df, destination_list = filter_destination(DEST_INFO_PATH, genre_list)
    batch_candidate = len(destination_list)
    # load file
    congestion_df = pd.read_pickle(PREDICTED_CONGEST_PATH)
    city_df = pd.read_pickle(CITY_INFO_PATH)
    # user 출발지 경도 위도 tuple
    start_df = city_df[(city_df['gu'] == start_info[0]) & (city_df['dong'] == start_info[1])]
    user_pos = (start_df['y'], start_df['x'])

    print(f"\nLoad Destination_info complete\n")
    progress_bar('Loading Model')


    use_pretrain, model_name, epochs, num_factors = define_args()
    FOLDER_PATH = 'pretrain_model'
    MODEL_PATH_VISITOR = os.path.join(FOLDER_PATH,f'0_{use_pretrain}_{model_name}_{epochs}_512_{num_factors}_visitor.pth')
    if not os.path.exists(MODEL_PATH_VISITOR):
        print("Model doesn't exist.\n")
        sys.exit()

    if model_name == 'GMF':
        model_visitor = GMF(num_factor=36,
                            num_dayofweek=7,
                            num_time=7,
                            num_sex=2,
                            num_age=7001,
                            num_month=13,
                            num_day=32,
                            num_destination=2505928,
                            use_pretrain=False,
                            use_NeuMF=False,
                            pretrained_GMF=None)
    elif model_name == 'MLP':
        model_visitor = MLP(num_factor=24,
                            num_layer=3,
                            num_dayofweek=7,
                            num_time=7,
                            num_sex=2,
                            num_age=7001,
                            num_month=13,
                            num_day=32,
                            num_destination=2505928,
                            use_pretrain=False,
                            use_NeuMF=False,
                            pretrained_MLP=None)
    elif model_name == 'NeuMF':
        model_visitor = NeuMF(num_factor=36,
                              num_layer=3,
                              num_dayofweek=7,
                              num_time=7,
                              num_sex=2,
                              num_age=7001,
                              num_month=13,
                              num_day=32,
                              num_destination=2505928,
                              use_pretrain=False,
                              use_NeuMF=True,
                              pretrained_GMF=None,
                              pretrained_MLP=None)

    model_visitor.load_state_dict(torch.load(MODEL_PATH_VISITOR,map_location=device))
    print("Load Model complete\n")

    All_day_ranking = [0] * num_term
    total_ranking = {}
    for i in user_df.index:
        tem_df = destination_id_name_df.copy()
        user_input = list(user_df.iloc[i,:])
        RecSys_dataset = Input_Dataset(destination_list=destination_list, RecSys_input=user_input)
        RecSys_dataloader = DataLoader(dataset=RecSys_dataset,
                                       batch_size=batch_candidate,
                                       shuffle=False)
        for destination, time, sex, age, dayofweek, month, day in RecSys_dataloader:
            destination = destination.to(device)
            dayofweek, time, sex, age, month, day =\
                dayofweek.to(device), time.to(device), sex.to(device), age.to(device), month.to(device), day.to(device)
            pred_visitor = model_visitor(dayofweek, time, sex, age, month, day, destination)
            saved_congestion = load_congestion(destination_id_name_df=destination_id_name_df,
                                               df=congestion_df,
                                               dayofweek=dayofweek[0],
                                               time=time[0],
                                               month=month[0],
                                               day=day[0])
            pred_visitor = pred_visitor.tolist()
            saved_congestion = saved_congestion.tolist()
            tem_df['visitor'] = pred_visitor
            tem_df['congestion'] = saved_congestion
            tem_df = tem_df.sort_values(by='visitor', ascending=False)

            for k in range(batch_candidate):
                dest_pos = (tem_df.iloc[k, 8], tem_df.iloc[k, 9])
                destination_name = tem_df.iloc[k, 1]
                small_genre = tem_df.iloc[k, 7]
                middle_genre = tem_df.iloc[k, 6]
                visitor = tem_df.iloc[k, 10]
                congestion = tem_df.iloc[k, 11]
                # 출발지-관광지 haversine 거리
                distance = haversine(user_pos, dest_pos)
                # destination name의 key로 설정
                rank_weight = total_ranking.get(destination_name)
                if rank_weight is None:
                    total_ranking[destination_name] = [0, 0, distance, middle_genre, small_genre]
                # 해당 user info들과 날짜에 대한 visitor, congestion 합 구하기
                total_ranking[destination_name][0] += visitor
                total_ranking[destination_name][1] += congestion
            if i % num_people == num_people - 1:
                All_day_ranking[i // num_people] = total_ranking
                total_ranking = {}

    result_ranking = [0] * num_term
    total_dest = {}
    for day, each_rank in enumerate(All_day_ranking):
        sorted_visitor_ranking = sorted(each_rank.items(), key=lambda item: item[1][0], reverse=True)
        sorted_congestion_ranking = sorted(each_rank.items(), key=lambda item: item[1][1], reverse=True)
        sorted_distance_ranking = sorted(each_rank.items(), key=lambda item: item[1][2], reverse=True)
        tmp_rank = {}
        for i, dest in enumerate(sorted_visitor_ranking):
            tmp_rank[dest[0]] = ratio[0] * (batch_candidate - i)
        for i, dest in enumerate(sorted_congestion_ranking):
            tmp_rank[dest[0]] += ratio[1] * (batch_candidate - i)
        for i, dest in enumerate(sorted_congestion_ranking):
            tmp_rank[dest[0]] += ratio[2] * (batch_candidate - i)
            val = total_dest.get(dest[0])
            if val is None:
                total_dest[dest[0]] = []
            total_dest[dest[0]].append(tmp_rank[dest[0]])

        sorted_total_ranking = sorted(tmp_rank.items(), key=lambda item: item[1], reverse=True)
        result_ranking[day] = sorted_total_ranking

    # calculate median ranking
    median_dest = {}
    for k, v in total_dest.items():
        v.sort(reverse=True)
        median_dest[k] = math.floor(v[len(v) // 2])

    for i, day_ranking in enumerate(result_ranking):
        print(f'\n{i + 1}일 째 추천 관광지')
        rank = 1
        for dest in day_ranking:
            if median_dest[dest[0]] > dest[1]:
                continue
            print(f'{rank}등: {dest[0]}')
            rank += 1

    end_time = ti.time()
    print(f'추천하는데 총 걸린 시간 : {end_time - total_start}')


