from sklearn.model_selection import train_test_split
from sklearn . preprocessing import StandardScaler
from sklearn . metrics import confusion_matrix
from sklearn . naive_bayes import GaussianNB
from sklearn . preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
pd.options.mode.chained_assignment = None

goog_path = os.path.abspath('GOOG_weekly_return_volatility.csv')
df_goog = pd.read_csv(goog_path)
df_googvol_2yrs = df_goog[df_goog.Year.isin([2019,2020])]

print("Files Read")

print('##############Q1################')

X = df_googvol_2yrs [["mean_return", "volatility"]]
y = df_googvol_2yrs["Label"]

def naive_bayesian(a,b):
    
    X_train ,X_test , Y_train , Y_test = train_test_split (a,b,test_size =0.5,shuffle = False)
    NB_classifier = GaussianNB (). fit ( X_train , Y_train )
    pred_log = NB_classifier . predict ( X_test )
    accuracy = np. mean ( pred_log == Y_test )
    print("The Accuracy is {}  with Naive Bayesian for year 2020".format(accuracy))
    return pred_log,Y_test

pred_log,Y_test = naive_bayesian(X,y)

###Q1( part 2,3)  -> Confusion matrix , TNR and TPR for year 2 ie 2020 ####
cf_1 = confusion_matrix( Y_test , pred_log )
print("Confusion matrix for year 2020  is {} ".format(cf_1))
tpr = cf_1[1][1]/(cf_1[1][1] + cf_1[1][0])
tnr = cf_1[0][0]/(cf_1[0][0] + cf_1[0][1])
print("TPR  for year 2020 is {}  and TNR for year 2020 is {}".format( tpr, tnr))

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
df_goog['Label'] = pred_log
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
