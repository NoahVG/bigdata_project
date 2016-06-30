# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:11:02 2016

@author: PBM887

reads in all json files from a folder as python dictionaries
"""

import os
import json
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
# constants
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'data'))
MOJO_DIR = os.path.join(DATA_DIR, 'boxofficemojo')
META_DIR = os.path.join(DATA_DIR, 'metacritic')



def load_movie_data():
    movie_list_mojo=[]
    movie_list_meta=[]

    for file1 in os.listdir(MOJO_DIR):
        path=MOJO_DIR+'\\'+file1
        with open(path, 'r') as target_file:
            movie = json.load(target_file)
            movie_list_mojo.append(movie)
            
    for file1 in os.listdir(META_DIR):
        if '_parsed' in str(file1):
            path=META_DIR+'\\'+file1
            with open(path, 'r') as target_file:
                movie = json.load(target_file)
                movie_list_meta.append(movie)
    #print 6
    return movie_list_mojo,movie_list_meta

mojo,meta=load_movie_data()[0],load_movie_data()[1]



for i in meta:
    i['num_critic_reviews']=str(i['num_critic_reviews'])
    i['num_user_reviews']=str(i['num_user_reviews'])
    
meta_df=pd.DataFrame(meta)
mojo_df=pd.DataFrame(mojo)

both_df=pd.merge(mojo_df,meta_df, on='title',how='left')
##################
both_df2=both_df[pd.notnull(both_df['num_critic_reviews'])]
both_df2.reset_index(drop=True,inplace=True)
critic_reviews=pd.DataFrame(data=both_df2['num_critic_reviews'].str.split(',').tolist(),columns=['critic_positive','critic_negative','critic_neutral','critic_total'])
critic_reviews=critic_reviews.replace('\[|\]','',regex=True)
critic_reviews=critic_reviews[critic_reviews.columns].astype(float)
both_df3=both_df2.join(critic_reviews)
critic_reviews1=both_df3[['title','critic_positive','critic_negative','critic_neutral','critic_total']]
##################
both_df2=both_df[pd.notnull(both_df['num_user_reviews'])]
both_df2.reset_index(drop=True,inplace=True)
user_reviews=pd.DataFrame(both_df2['num_user_reviews'].str.split(',').tolist(),columns=['user_positive','user_negative','user_neutral','user_total'])
user_reviews=user_reviews.replace('\[|\]','',regex=True)
user_reviews=user_reviews[user_reviews.columns].astype(float)
both_df3=both_df2.join(user_reviews)
user_reviews1=both_df3[['title','user_positive','user_negative','user_neutral','user_total']]

#######################
both_df['release_date_wide']=both_df['release_date_wide'].astype('datetime64[ns]')
abc=both_df.groupby('director_x')
for i in abc:
    #df=i[1]
    if len(i[1])>1:
        df=i[1].copy()
        df.sort_values('release_date_wide',ascending=False,inplace=True)
        df.reset_index(drop=True,inplace=True)
        #q=df.index.values
        df['rolling_mean']=np.nan
        for z in range(0,(len(df)-1)):
            df.ix[z,'rolling_mean']= df.ix[(z+1):,'domestic_gross'].mean()
    else:
        df=i[1].copy()
        df['rolling_mean']=np.nan
        #break
    try:
        new_df3=pd.concat([new_df3,df])
    except:
        new_df3=df
    rolling_avg=new_df3[['title','rolling_mean']]


##############
final_df=pd.merge(both_df,critic_reviews1,on='title',how='left')
final_df=pd.merge(final_df,user_reviews1,on='title',how='left')
final_df=pd.merge(final_df,rolling_avg,on='title',how='left')

final_df.columns.values.tolist()

final_df2=final_df.dropna()
X = final_df2[['critic_positive',
 'critic_negative',
 'critic_neutral',
 'critic_total',
 'user_positive',
 'user_negative',
 'user_neutral',
 'user_total',
 'rolling_mean']]
Y = final_df2.domestic_gross
X=sm.add_constant(X)


linmodel = sm.OLS(Y,X).fit()

predicted_gross = linmodel.predict(X)
plt.scatter(predicted_gross, predicted_gross-final_df2.domestic_gross, color='gray')
linmodel.summary()