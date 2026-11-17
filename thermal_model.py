import numpy as np
import pandas as pd

def simulate_module(current_a, ambient_c, cooling, duration_s=900, dt=1, seed=1):
    rng=np.random.default_rng(seed); cells=12
    resistance=rng.normal(.0022,.00012,cells); thermal_mass=np.full(cells,900.)
    coupling=.8; h={"passive":.35,"air":2.0,"liquid":8.0}[cooling]
    temp=np.full(cells,float(ambient_c)); soc=np.full(cells,.85); rows=[]
    for t in range(int(duration_s/dt)):
        heat=current_a**2*resistance+.015*abs(current_a)
        neighbour=np.roll(temp,1)+np.roll(temp,-1)-2*temp
        temp += (heat+coupling*neighbour-h*(temp-ambient_c))/thermal_mass*dt
        soc=np.clip(soc-current_a*dt/(5*3600),0,1)
        if t%30==0: rows.append((t,current_a,ambient_c,cooling,temp.max(),temp.min(),temp.std(),soc.mean()))
    return pd.DataFrame(rows,columns=["time_s","current_a","ambient_c","cooling","max_temp_c","min_temp_c","temp_spread_c","mean_soc"])
