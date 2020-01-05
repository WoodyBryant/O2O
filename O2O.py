##本地结果0.8303459801781878
##提交结果0.7334
import pandas as pd 
import numpy as np
from dateutil.parser import parse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb
#from sklearn.model_selection import cross_val_score
df1 = pd.read_csv("ccf_offline_stage1_train.csv",keep_default_na=False)
df2 = pd.read_csv("ccf_offline_stage1_test_revised.csv",keep_default_na=False)
df2_Date_received = df2['Date_received']
df1 = df1[df1['Coupon_id'] != 'null']##加上这句话结果会提升
df2['Date'] = '$'
df3 = pd.concat([df1,df2])

def label(row):
    if row['Date_received'] == 'null':
        return -1
    elif row['Date'] == 'null':
        return 0
    elif (row['Date']-row['Date_received']).days <=15:
        return  1
    else:
        return 0
    
def get_discount_type(row):
    if row == 'null':
        return np.nan
    elif':' in row:
        return 0
    else:
        return 1

def rate(line):
    if ':' in str(line):
        x = str(line).split(':')
        return float(int(x[1])/int(x[0]))
    elif line != 'null':
        return float(line)
    else:
        return line
    
def get_discount_man(row):
    if':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0
    
def get_discount_jian(row):
    if':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0
##统计商家的券被领的次数
Count_Merchant_id1 = df3.groupby('Merchant_id').size()
df1['Count_Merchant_id'] = df1['Merchant_id'].apply(lambda x:Count_Merchant_id1[x])

##Count_Merchant_id2 = df2.groupby('Merchant_id').size()
df2['Count_Merchant_id'] = df2['Merchant_id'].apply(lambda x:Count_Merchant_id1[x])

##每个用户在不同时间领取的所有优惠券的总个数
User_Date_received1= df3[['User_id','Date_received']]
User_Date_received1['User_Date_received'] = 1
User_Date_received1 = User_Date_received1.groupby(['User_id','Date_received']).count().reset_index()
df1 = pd.merge(df1,User_Date_received1,how = 'left',on = ['User_id','Date_received'])

##User_Date_received2 = df2[['User_id','Date_received']]
##User_Date_received2['User_Date_received'] = 1
##User_Date_received2 = User_Date_received2.groupby(['User_id','Date_received']).count().reset_index()
df2 = pd.merge(df2,User_Date_received1,how = 'left',on = ['User_id','Date_received'])

##每个用户在不同时间领取的每种优惠券各自的个数
User_Date_received_Coupon1= df3[['User_id','Date_received','Coupon_id']]
User_Date_received_Coupon1['User_Date_received_Coupon'] = 1
User_Date_received_Coupon1 = User_Date_received_Coupon1.groupby(['User_id','Date_received','Coupon_id']).count().reset_index()

##User_Date_received_Coupon2= df2[['User_id','Date_received','Coupon_id']]
##User_Date_received_Coupon2['User_Date_received_Coupon'] = 1
##User_Date_received_Coupon2 = User_Date_received_Coupon2.groupby(['User_id','Date_received','Coupon_id']).count().reset_index()

df1 = pd.merge(df1,User_Date_received_Coupon1,how = 'left',on = ['User_id','Date_received','Coupon_id'])
df2 = pd.merge(df2,User_Date_received_Coupon1,how = 'left',on = ['User_id','Date_received','Coupon_id'])

##每个用户领取每种优惠券各自的个数    
Count_Userid_same_Couponid1 = df3[['User_id','Coupon_id']]
Count_Userid_same_Couponid1['Count_Userid_same_Couponid'] = 1
Count_Userid_same_Couponid1 =Count_Userid_same_Couponid1.groupby(['User_id','Coupon_id']).count().reset_index()

##Count_Userid_same_Couponid2 = df2[['User_id','Coupon_id']]
##Count_Userid_same_Couponid2['Count_Userid_same_Couponid'] = 1
##Count_Userid_same_Couponid2 =Count_Userid_same_Couponid2.groupby(['User_id','Coupon_id']).count().reset_index()

df1 = pd.merge(df1,Count_Userid_same_Couponid1,how = 'left',on = ['User_id','Coupon_id'])
df2 = pd.merge(df2,Count_Userid_same_Couponid1,how = 'left',on = ['User_id','Coupon_id'])

df1['Distance'] = df1['Distance'].apply(lambda x:-1 if x == 'null' else x)
df2['Distance'] = df2['Distance'].apply(lambda x:-1 if x == 'null' else x)

df1['get_discount_type'] = df1['Discount_rate'].apply(get_discount_type)
df2['get_discount_type'] = df2['Discount_rate'].apply(get_discount_type)

df1['get_discount_man'] = df1['Discount_rate'].apply(get_discount_man)
df2['get_discount_man'] = df2['Discount_rate'].apply(get_discount_man)

df1['get_discount_jian'] = df1['Discount_rate'].apply(get_discount_jian)
df2['get_discount_jian'] = df2['Discount_rate'].apply(get_discount_jian)

##每个用户ID总共领取优惠券数目
count_User1= df3.groupby('User_id').size()
##count_User2 = df2.groupby('User_id').size()
df1['count_User'] = df1['User_id'].apply(lambda x:count_User1[x])
df2['count_User'] = df2['User_id'].apply(lambda x:count_User1[x])


##每个用户领取不同商家的优惠券的个数
count_User_Merchant = df3[['User_id','Merchant_id']]
count_User_Merchant['count_User_Merchant'] = 1
count_User_Merchant = count_User_Merchant.groupby(['User_id','Merchant_id']).count().reset_index()
df1 = pd.merge(df1,count_User_Merchant,on = ['User_id','Merchant_id'],how = 'left')
df2 = pd.merge(df2,count_User_Merchant,on = ['User_id','Merchant_id'],how = 'left')
df1['count_User_Merchant'].fillna(0,inplace = True)
df2['count_User_Merchant'].fillna(0,inplace = True)

##每个用户领取不同商家的不同优惠券的个数
count_User_Merchant_Coupon = df3[['User_id','Merchant_id','Coupon_id']]
count_User_Merchant_Coupon['count_User_Merchant_Coupon'] = 1
count_User_Merchant_Coupon = count_User_Merchant_Coupon.groupby(['User_id','Merchant_id','Coupon_id']).count().reset_index()
df1 = pd.merge(df1,count_User_Merchant_Coupon,on = ['User_id','Merchant_id','Coupon_id'],how = 'left')
df2 = pd.merge(df2,count_User_Merchant_Coupon,on = ['User_id','Merchant_id','Coupon_id'],how = 'left')
df1['count_User_Merchant_Coupon'].fillna(0,inplace = True)
df2['count_User_Merchant_Coupon'].fillna(0,inplace = True)

##每个优惠券出现的次数
count_Coupon = df3[['Coupon_id']]
count_Coupon['count_Coupon'] = 1
count_Coupon = count_Coupon.groupby('Coupon_id').count().reset_index()
df1 = pd.merge(df1,count_Coupon,on = 'Coupon_id',how = 'left')
df2 = pd.merge(df2,count_Coupon,on = 'Coupon_id',how = 'left')
df1['count_Coupon'].fillna(0,inplace = True)
df2['count_Coupon'].fillna(0,inplace = True)

df1['Date_received'] = df1['Date_received'].apply(lambda x:str(x)[:8]).apply(lambda x:parse(x) if x!='null' else x)
df2['Date_received'] = df2['Date_received'].apply(lambda x:str(x)[:8]).apply(lambda x:parse(x) if x!='null' else x)

df1['Date'] = df1['Date'].apply(lambda x:str(x)[:8]).apply(lambda x:parse(x) if x!='null' else x)

df1['Discount_rate'] = df1['Discount_rate'].apply(rate)
df2['Discount_rate'] = df2['Discount_rate'].apply(rate)

df1['weekday_received'] = df1['Date_received'].apply(lambda x : x.weekday()+1 if x !='null' else x)
df2['weekday_received'] = df2['Date_received'].apply(lambda x : x.weekday()+1 if x !='null' else x)

df1['weekdaytype'] = df1['weekday_received'].apply(lambda x:0 if x in [1,2,3,4,5] else 1)
df2['weekdaytype'] = df2['weekday_received'].apply(lambda x:0 if x in [1,2,3,4,5] else 1)

weekdaycols = ['received_on_weekday_'+str(i) for i in range(1,8)]
#weekdaycols.append('null')

weekdays1 = pd.get_dummies(df1['weekday_received'])
weekdays1.columns = weekdaycols
df1[weekdaycols] = weekdays1
#
weekdays2 = pd.get_dummies(df1['weekday_received'])
weekdays2.columns = weekdaycols
df2[weekdaycols] = weekdays2


df1['label'] = df1.apply(label,axis = 1)
df1 = df1[df1['label']!=-1]

####用到label的特征基本会导致过拟合
#df1_temp = df1[df1['label']==1]
#df1_temp['Discount_rate'] = df1_temp['Discount_rate'].astype('float')
#df1_temp['Distance'] = df1_temp['Distance'].astype('float')

########这三个特征容易过拟合
###用户核销的的平均折扣率
#mean_User_rate = df1_temp.groupby('User_id')['Discount_rate'].mean()
#df1['mean_User_rate'] = df1['User_id'].apply(lambda x:mean_User_rate[x] if x in mean_User_rate.index else 0)
#df2['mean_User_rate'] = df2['User_id'].apply(lambda x:mean_User_rate[x] if x in mean_User_rate.index else 0)
#
###用户核销的最大折扣率
#max_User_rate = df1_temp.groupby('User_id')['Discount_rate'].max()
#df1['max_User_rate'] = df1['User_id'].apply(lambda x:max_User_rate[x] if x in max_User_rate.index else 0)
#df2['max_User_rate'] = df2['User_id'].apply(lambda x:max_User_rate[x] if x in max_User_rate.index else 0)
#
###用户核销的的最小折扣率
#min_User_rate = df1_temp.groupby('User_id')['Discount_rate'].max()
#df1['min_User_rate'] = df1['User_id'].apply(lambda x:min_User_rate[x] if x in min_User_rate.index else 0)
#df2['min_User_rate'] = df2['User_id'].apply(lambda x:min_User_rate[x] if x in min_User_rate.index else 0)


##这个特征也容易过拟合
##用户核销过的不同商家的数量
#User_Merchant_label_1 = df1_temp[['User_id','Merchant_id']]
#User_Merchant_label_1['User_Merchant_label1'] = 1
#User_Merchant_label_1 = User_Merchant_label_1.groupby(['User_id','Merchant_id']).count().reset_index()
#df1 = pd.merge(df1,User_Merchant_label_1,how = 'left',on = ['User_id','Merchant_id'])
#df2 = pd.merge(df2,User_Merchant_label_1,how = 'left',on = ['User_id','Merchant_id'])
#df1['User_Merchant_label1'].fillna(0,inplace = True)
#df2['User_Merchant_label1'].fillna(0,inplace = True)


#df1.drop(labels = 'null',axis = 1,inplace = True)
#df2.drop(labels = 'null',axis = 1,inplace = True)

#####用户id核销的次数(用户核销所有商家的次数),应用到了label的信息，容易过拟合
#Count_User_Date = df1_temp[['User_id']]
#Count_User_Date['Count_label'] = 1
#Count_User_Date = Count_User_Date.groupby('User_id').count().reset_index()
#df1 = pd.merge(df1,Count_User_Date,on = 'User_id',how = 'left')
#df2 = pd.merge(df2,Count_User_Date,on = 'User_id',how = 'left')
#df1['Count_label'].fillna(0,inplace = True)
#df2['Count_label'].fillna(0,inplace = True)

##用户核销过的不同商家的次数占所用核销过的商家总次数的比例
#df1['User_Merchant_rate'] = df1.apply(lambda x:x['User_Merchant_label_1']/x['Count_label'] \
#   if x['Count_label'] !=0 else 0,axis = 1)

########这三个特征容易过拟合
###用户核销过的商家的平均距离
#mean_User_Distance = df1_temp.groupby('User_id')['Distance'].mean()
#df1['mean_User_Distance'] = df1['User_id'].apply(lambda x:mean_User_Distance[x] if x in mean_User_Distance.index else 0)
#df2['mean_User_Distance'] = df2['User_id'].apply(lambda x:mean_User_Distance[x] if x in mean_User_Distance.index else 0)
#
###用户核销过的商家的最大距离
#max_User_Distance = df1_temp.groupby('User_id')['Distance'].max()
#df1['max_User_Distance'] = df1['User_id'].apply(lambda x:max_User_Distance[x] if x in max_User_Distance.index else 0)
#df2['max_User_Distance'] = df2['User_id'].apply(lambda x:max_User_Distance[x] if x in max_User_Distance.index else 0)
#
###用户核销过的商家的最小距离
#min_User_Distance = df1_temp.groupby('User_id')['Distance'].mean()
#df1['min_User_Distance'] = df1['User_id'].apply(lambda x:min_User_Distance[x] if x in min_User_Distance.index else 0)
#df2['min_User_Distance'] = df2['User_id'].apply(lambda x:min_User_Distance[x] if x in min_User_Distance.index else 0)

features = list(df1.columns)
remove_features = ['label','Date_received','Date']
for i in remove_features:
    features.remove(i)
    
print(df1.isnull().sum(),df1.info())
model = lgb.LGBMRegressor(n_estimators=45,learning_rate=0.05,num_leaves=30,is_sparse = True,metric = 'auc',\
                          train_metric = True)
df1['Coupon_id']  = df1['Coupon_id'].astype('int')
df1['Discount_rate'] = df1['Discount_rate'].astype('float')
df1['Distance'] = df1['Distance'].astype('int')
df1['weekday_received'] = df1['weekday_received'].astype('int')

df2['Coupon_id']  = df2['Coupon_id'].astype('int')
df2['Discount_rate'] = df2['Discount_rate'].astype('float')
df2['Distance'] = df2['Distance'].astype('int')
df2['weekday_received'] = df2['weekday_received'].astype('int')
x_train, x_test, y_train, y_test = train_test_split(df1[features],df1['label'], test_size = 0.3,random_state=44)

model.fit(x_train,y_train)
y_pred1 = model.predict(x_test)

df_test = x_test
df_test['label'] = y_test
df_test['label_predict'] = y_pred1
scores = []
for name,group in df_test.groupby('Coupon_id'):
    df_temp = pd.DataFrame(group)
    if len(df_temp['label'].unique()) != 1:
        score1 =  roc_auc_score(df_temp['label'], df_temp['label_predict'])
        scores.append(score1)
scores = np.array(scores)
auc_score = scores.mean()
print("{}在测试集上auc为：{}".format(model,auc_score))
print("特征为{}".format(features))
model.fit(df1[features],df1['label'])
y_pred2 = model.predict(df2[features])
df4 = df2[['User_id','Coupon_id']]
df4['Date_received'] = df2_Date_received 
df4['Probability'] = y_pred2
df4.to_csv('submission1.csv',header = False,index = False)


