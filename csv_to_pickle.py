import os
import pandas as pd

# csv file -> pickle file
ROOT_DIR = 'dataset'
DEST_INFO_PATH = os.path.join(ROOT_DIR, 'destination_id_name_genre_coordinate.csv')
PREDICTED_CONGEST_PATH = os.path.join(ROOT_DIR, 'congestion_1_2.csv')
CITY_INFO_PATH = os.path.join(ROOT_DIR, 'seoul_gu_dong_coordinate.csv')

li = [DEST_INFO_PATH,PREDICTED_CONGEST_PATH,CITY_INFO_PATH]
for i in li:
    df = pd.read_csv(i)
    df.to_pickle(i.rstrip('csv')+'pkl')
