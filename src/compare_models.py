"""Leakage-aware avocado model comparison: MLP and voting ensembles."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

FEATURES = ['price_1week','price_2week','price_4week','volume_1week','volume_2week','volume_4week','sin_year','cos_year']

def prepare(df, kind, geography):
    sub = df.loc[(df['type'].str.lower()==kind)&(df['geography']==geography)].copy()
    sub['date']=pd.to_datetime(sub['date'])
    sub=sub.sort_values('date').drop_duplicates('date').set_index('date')
    # No backward filling or interpolation using future prices.
    sub=sub[['average_price','total_volume']].resample('W-SUN').mean()
    sub[['average_price','total_volume']]=sub[['average_price','total_volume']].ffill()
    for lag in (1,2,4):
        sub[f'price_{lag}week']=sub['average_price'].shift(lag)
        sub[f'volume_{lag}week']=sub['total_volume'].shift(lag)
    week=sub.index.isocalendar().week.astype(int)
    sub['sin_year']=np.sin(2*np.pi*week/52.18)
    sub['cos_year']=np.cos(2*np.pi*week/52.18)
    sub['year']=sub.index.year
    return sub.dropna(subset=FEATURES+['average_price'])

def scaled(estimator):
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler()),('model',estimator)])

def models():
    return {
      'Linear Regression':scaled(LinearRegression()),
      'KNN':scaled(KNeighborsRegressor(n_neighbors=5)),
      'SVR':scaled(SVR(C=10,epsilon=0.1)),
      'Random Forest':RandomForestRegressor(n_estimators=200,min_samples_leaf=2,random_state=42,n_jobs=-1),
      'Neural Network':scaled(MLPRegressor(hidden_layer_sizes=(64,32),early_stopping=True,max_iter=1500,random_state=42)),
    }

def score(y,p):
    return {'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),'R2':r2_score(y,p)}

def run(csv_path,output,geography):
    data=pd.read_csv(csv_path)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    all_results=[]
    for kind in ('conventional','organic'):
        frame=prepare(data,kind,geography)
        train=frame.loc[frame['year']<2020]
        test=frame.loc[frame['year']==2020]
        if train.empty or test.empty: raise ValueError(f'Missing pre-2020 training or 2020 test data for {kind}')
        Xtr,ytr=train[FEATURES],train['average_price'];Xte,yte=test[FEATURES],test['average_price']
        # Reserve the latest training year for model tuning, never the 2020 holdout.
        last_year=int(train['year'].max()); fold=np.where(train['year']==last_year,0,-1)
        if len(np.unique(fold))<2: raise ValueError('Need at least two training years for validation')
        cv=PredefinedSplit(fold)
        mlp=scaled(MLPRegressor(max_iter=1500,early_stopping=True,random_state=42))
        grid=GridSearchCV(mlp,{'model__hidden_layer_sizes':[(32,),(64,32),(128,64)],'model__alpha':[0.0001,0.01],'model__activation':['relu','tanh']},scoring='neg_mean_absolute_error',cv=cv,n_jobs=-1,refit=True)
        grid.fit(Xtr,ytr)
        pd.DataFrame(grid.cv_results_).to_csv(output/f'{kind}_neural_network_grid.csv',index=False)
        estimators=models();estimators['Neural Network']=grid.best_estimator_
        # Equal-weight ensemble; each component is trained on the same training partition.
        estimators['Voting Regressor']=VotingRegressor(estimators=[(f'm{i}',clone(m)) for i,m in enumerate(estimators.values())],n_jobs=-1)
        predictions=pd.DataFrame({'date':test.index,'actual':yte.to_numpy()})
        for name,model in estimators.items():
            fitted=clone(model).fit(Xtr,ytr)
            pred=fitted.predict(Xte)
            predictions[name]=pred
            all_results.append({'type':kind,'model':name,**score(yte,pred)})
        predictions.to_csv(output/f'{kind}_predictions_2020.csv',index=False)
        plt.figure(figsize=(11,5))
        plt.plot(predictions['date'],predictions['actual'],label='Actual',color='black',linewidth=2)
        for name in ('Neural Network','Voting Regressor'):
            plt.plot(predictions['date'],predictions[name],label=name,alpha=.85)
        plt.title(f'{kind.title()} avocado: 2020 holdout predictions');plt.xlabel('Date');plt.ylabel('Average price');plt.legend();plt.tight_layout()
        plt.savefig(output/f'{kind}_prediction_plot.png',dpi=160);plt.close()
        print(kind,'MLP best params:',grid.best_params_)
    results=pd.DataFrame(all_results).sort_values(['type','MAE'])
    results.to_csv(output/'model_comparison.csv',index=False)
    print(results.to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/avocado-updated-2020.csv');p.add_argument('--output',default='results');p.add_argument('--geography',default='San Francisco');a=p.parse_args()
    run(a.data,a.output,a.geography)
