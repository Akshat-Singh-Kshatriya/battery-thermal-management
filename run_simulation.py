from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from thermal_model import simulate_module

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)
runs=[]
for cooling in ["passive","air","liquid"]:
 for current in [10,20,30,40,50]:
  for ambient in [15,25,35,40]:
   d=simulate_module(current,ambient,cooling,seed=current+ambient); d["run_id"]=f"{cooling}_{current}_{ambient}"; runs.append(d)
data=pd.concat(runs,ignore_index=True); data.to_csv(OUT/"thermal_data.csv",index=False)
X=pd.get_dummies(data[["current_a","ambient_c","time_s","cooling"]],columns=["cooling"],dtype=int)
y=data.max_temp_c
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
m=RandomForestRegressor(n_estimators=200,max_depth=12,min_samples_leaf=2,random_state=42,n_jobs=-1)
m.fit(Xtr,ytr); p=m.predict(Xte)
metrics={"mae_c":float(mean_absolute_error(yte,p)),"rmse_c":float(mean_squared_error(yte,p)**.5),
         "r2":float(r2_score(yte,p)),"rows":len(data)}
(OUT/"surrogate_metrics.json").write_text(json.dumps(metrics,indent=2))
data.groupby(["cooling","current_a","ambient_c"]).agg(
    peak_temp_c=("max_temp_c","max"),final_spread_c=("temp_spread_c","last"),
    final_soc=("mean_soc","last")).to_csv(OUT/"cooling_comparison.csv")
print(json.dumps(metrics,indent=2)); print("Saved:",OUT)
