from itertools import islice
from youtube_comment_downloader import *
from youtube_search_requests import YoutubeSearch
import pandas as pd
from tqdm import tqdm
import time

y = YoutubeSearch()
downloader = YoutubeCommentDownloader()
data =  pd.read_csv('./genre_11_tempo_4.csv')
data3 = data.iloc[6666:]
com = []
for idx,value in tqdm(data3.iterrows()):
    li = []
    count = 10
    try:
        videos = y.search_videos(f'{value['song_name']} {value['singer']}', max_results=10, timeout=1 )
    except:
        continue
    if not videos:
        continue
    url = videos[0]['url']
    comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
    for comment in islice(comments, count):
        li.append(comment['text'])
    for i in range(count-len(li)):
        li.append(None)
    li = [value['singer'], value['song_name'], url] + li
    com.append(li)
    time.sleep(0.1)
column = ['singer','song_name','url']
for i in range(count):
    column.append(f'comment{i+1}')
data_comments = pd.DataFrame(com,columns=column)
data_comments.to_csv('./data_comments3.csv',index =False)