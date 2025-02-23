from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import joblib

def get_scaler(data,type='StandardScaler',save=True,path='scaler.joblib'):
        #concatenate data for scaler fitting
        data_concat = [point for roast in data for point in roast]
        data_concat = pd.DataFrame(data_concat,columns=['bt','et','burner'])
        data_concat.dropna(inplace=True)

        #scaler preparation
        X = data_concat[['bt', 'et', 'burner']]
        if type == 'StandardScaler':
                scaler = StandardScaler()
        elif type == 'MinMaxScaler':
                scaler = MinMaxScaler()
        scaler.fit(X)

        if save:
                joblib.dump(scaler,path)
        return scaler