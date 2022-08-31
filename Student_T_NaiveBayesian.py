from sklearn . metrics import confusion_matrix
from scipy import stats
import numpy as np
import pandas as pd
import os
pd.options.mode.chained_assignment = None

goog_path = os.path.abspath('GOOG_weekly_return_volatility.csv')
df_goog = pd.read_csv(goog_path)
df_googvol_2019 = df_goog[df_goog.Year.isin([2019])]
df_googvol_2020 = df_goog[df_goog.Year.isin([2020])]
df_googvol_2yrs = df_goog[df_goog.Year.isin([2019,2020])]

print("Files Read")

df_googvol_2020 = df_goog[df_goog.Year.isin([2020])]
X2020_mr = df_googvol_2020 [["mean_return"]]
X2020_vol = df_googvol_2020 [["volatility"]]
X2020_Label = df_googvol_2020 ['Label']


print('############## Q1 to Q5 ################')


Label0_2019 = df_googvol_2019[df_googvol_2019.Label.isin([0])]
# print(Label0_2019)
Label1_2019 = df_googvol_2019[df_googvol_2019.Label.isin([1])]

df_red = df_googvol_2019.groupby(['Label']).agg({'mean_return': 'count'}).reset_index()
# print(df_red)
red = (df_red.loc[df_red.Label == 0, 'mean_return']).values
green = (df_red.loc[df_red.Label == 1, 'mean_return']).values
totr = df_red.sum()
totr = totr.to_numpy()
p_red = (red/totr[1])
print("The probability for red is:" ,p_red)
totg = df_red.sum()
totg = totg.to_numpy()
p_green= (green/totg[1])
print("The probability for green is:" ,p_green)

print("####################################################")

error_rate = []
d_lst = [0.5,1,5]
    
# print('For Label 1 2019')
X11 = Label1_2019 [["mean_return"]]
df11 , location11 , scale11 = stats .t.fit(X11)
X12 = Label1_2019 [["volatility"]]
df12 , location12 , scale12 = stats .t.fit(X12)  
    
# print('For Label 0 2019')
X01 = Label0_2019 [["mean_return"]]
df01 , location01 , scale01 = stats .t.fit(X01)
X02 = Label0_2019 [["volatility"]]
df02 , location02 , scale02 = stats .t.fit(X02)

def Student_t(degree):
    for d in degree:
    
        value11 = stats .t. pdf (X2020_mr, d, location11 , scale11 )
        value11 = value11.tolist()   
    
        value12 = stats .t. pdf (X2020_vol,d, location12 , scale12 )
        value12 = value12.tolist()
    
        value01 = stats.t.pdf (X2020_mr, d , location01 , scale01 )
        value01 = value01.tolist()
    
        value02 = stats .t. pdf (X2020_vol, d, location02 , scale02 )
        value02 = value02.tolist()
        
        pred_lst = []
        
        for i in range (0, len(value11)):
            posterior_red = p_red * value01[i] * value02[i] 
            a_red  = list(posterior_red)
            post_red = a_red[0]
            posterior_green = p_green * value11[i] * value12[i]
            a_green  = list(posterior_green)
            post_green = a_green[0]
            normalized_red = post_red /( post_red + post_green )
            normalized_green = post_green /( post_red + post_green )
            if(normalized_red >= normalized_green):
                pred_lst.append(0)
            else:
                pred_lst.append(1)
        accuracy = np. mean ( pred_lst == X2020_Label )
        error_rate.append(accuracy)
        print("Prediction list for d{} is {}: " .format(d, pred_lst))
        print("Accuracy for d{} is {}: " .format(d, accuracy))
        
        ###Q1( part 2,3)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
        cf_1 = confusion_matrix( X2020_Label , pred_lst )
        print("Confusion matrix for year 2020 with d {} is {} ".format(d,cf_1))
        tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
        tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
        print("TPR  for year 2020 with d {} is {}  and TNR is {}".format( d,tpr, tnr))
        print("####################################################")
    
    max_value = max(error_rate)
    max_index = error_rate.index(max_value)
    maxd = d_lst[max_index]
    print('Best value for d is : ', d_lst[max_index])
    return maxd

maxd = Student_t(d_lst)
    
value11 = stats .t. pdf (X2020_mr, maxd, location11 , scale11 )
value11 = value11.tolist()   
value12 = stats .t. pdf (X2020_vol,maxd, location12 , scale12 )
value12 = value12.tolist()
value01 = stats.t.pdf (X2020_mr, maxd , location01 , scale01 )
value01 = value01.tolist()
value02 = stats .t. pdf (X2020_vol, maxd, location02 , scale02 )
value02 = value02.tolist()
pred_lst_maxdf = []
for i in range (0, len(value11)):
    posterior_red = p_red * value01[i] * value02[i] 
    a_red  = list(posterior_red)
    post_red = a_red[0]
    posterior_green = p_green * value11[i] * value12[i]
    a_green  = list(posterior_green)
    post_green = a_green[0]
    normalized_red = post_red /( post_red + post_green )
    normalized_green = post_green /( post_red + post_green )
    if(normalized_red >= normalized_green):
        pred_lst_maxdf.append(0)
    else:
        pred_lst_maxdf.append(1)
# print(pred_lst_maxdf)

print('################# Labels buy and hold and trading Strategy ###########')

googd_path = os.path.abspath('GOOG_weekly_return_volatility_detailed.csv')

df_googvold = pd.read_csv(googd_path,parse_dates=["Date"],dayfirst=True).drop(['High','Low','Open','Close','Volume', 'mean_return','volatility'], axis=1).sort_values('Date')
df_googvold['Open'] = df_googvold['Adj Close'].shift(1)
df_googvold['Close'] = df_googvold['Adj Close']
df_googvold = df_googvold.drop(['Adj Close'], axis = 1)

df_googvold = df_googvold[df_googvold.Year.isin([2020])]
df_goog = df_googvold.groupby(['Year','Week_Number']).agg({'Date': ['min','max']}).reset_index()
df_goog.columns = ['Year','Week_Number','OpenDate','CloseDate']
df_goog = (df_goog.merge(df_googvold[['Date','Open']], left_on = 'OpenDate', right_on = 'Date')
      .drop('Date',axis=1)
      .merge(df_googvold[['Date','Close']], left_on = 'CloseDate', right_on = 'Date')
      .drop('Date',axis=1))


df_goog = df_goog.merge(df_googvol_2yrs[['Week_Number','Year','Label']],how='left',left_on=['Week_Number','Year'],right_on = ['Week_Number','Year'])
df_goog['Label'] = pred_lst_maxdf
df_goog['NexLabel'] = df_goog['Label'].shift(-1)


cap = 100 + 100*(df_goog.loc[52,'Close'] - df_goog.loc[0,'Open'])/df_goog.loc[0,'Open']
buynhold = round(cap,2)
print("GOOG buy-hold  cap for 2020 : {}".format(buynhold))

cap  = 100
op = 0
for index, row in df_goog.iterrows():
    if row[6] == 1 and op == 0:
        op = row[4]
    if row[6] == 1 and row[7] == 0:
        cap = cap + cap * ((row[5] - op)/op)
        op = 0

strategy = round(cap,2)
print("GOOG trading strategy based on label cap for 2020 : {}".format(strategy))
