import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf

d_name = "brazil_data"

def get_curves(dir_name):
    
    file_names = os.listdir(dir_name)
    roasts = []
    for f_name in file_names:
        print(f_name) 
        bt_curve = extract_curve(f_name,dir_name,"'temp2':")
        et_curve = extract_curve(f_name,dir_name,"'temp1':")
        burner_curve = extract_burner(f_name,dir_name)
        np_bt_curve = np.array(bt_curve)
        np_et_curve = np.array(et_curve)
        np_burner_curve = np.array(burner_curve)
        if (len(np_bt_curve) < 900 and len(np_et_curve) < 900 and len(np_burner_curve) < 900 
            and len(np_bt_curve) == len(np_et_curve) == len(np_burner_curve)):
            roast = np.column_stack((np_bt_curve,np_et_curve,np_burner_curve))
            roasts.append(roast)
    return roasts

def extract_burner(f_name,dir_name):
    events = []
    events_types = []
    events_value = []
    burner_curve = []
    
    with open(dir_name+"/"+f_name,"r") as f:
        lines = [
        line.strip("\n").split(" ") + [index + 1]
        for index, line in enumerate(f.readlines())
        ]
        if len(lines)==0:
            return []
        lines = lines[0]
        
        charge_idx = get_charge(lines)
        drop_index = int(float(lines[lines.index("'DROP_time':")+1][:-1]))
        
        events_idx = lines.index("'specialevents':")
        event_types_idx = lines.index("'specialeventstype':")
        events_value_idx = lines.index("'specialeventsvalue':")
        n_events = event_types_idx - events_idx - 1
        for i in range(events_idx+1,event_types_idx):
            e = lines[i][:-1]
            if i == events_idx + 1:
                e = e[1:]
            elif i == event_types_idx - 1:
                e = e[:-1]

            events.append(float(e))

        for i in range(event_types_idx+1,event_types_idx+1+n_events):
            e = lines[i][:-1]
            if i == event_types_idx+1:
                e = e[1:]
            elif i == event_types_idx + n_events:
                e = e[:-1]

            events_types.append(int(e))
        
        for i in range(events_value_idx+1,events_value_idx+1+n_events):
            e = lines[i][:-1]
            if i == events_value_idx+1:
                e = e[1:]
            elif i == events_value_idx + n_events:
                e = e[:-1]
            
            events_value.append(float(e))
            
    for i,e in  enumerate(events_value):
        if(events_types[i]==3):
                if i == len(events)-1:
                    while len(burner_curve)<=drop_index:
                        burner_curve.append(e)
                else:
                    for j in range(int(events[i]-(0 if i==0 else events[i-1]))):
                        if j + (0 if i==0 else events[i-1]) >= charge_idx:
                            burner_curve.append(e)
    return burner_curve   
            
def extract_curve(f_name,dir_name,curve_name):
    curve = []
    with open(dir_name+"/"+f_name,"r") as f:
            lines = [
            line.strip("\n").split(" ") + [index + 1]
            for index, line in enumerate(f.readlines())
            ]
            if len(lines)==0:
                return []
            lines = lines[0]

            #calculate indexes of CHARGE and DROP with regard to reading points, to extract temp readings only during roast
            charge_index = get_charge(lines)
            drop_index = int(float(lines[lines.index("'DROP_time':")+1][:-1]))+charge_index
            
            #shift indexes to correspond to proper place in the file
            curve_index = lines.index(curve_name)
            charge_index += curve_index+1
            drop_index += curve_index+1
            
            current_index = charge_index
            while current_index <= drop_index:
                curve.append(float(lines[current_index][:-1]))
                current_index += 1
    
    #mapping to get rid 
    return curve
        
def get_charge(lines):
    # find TP info to derive CHARGE index
    tp_info_index = lines.index("'TP_idx':")
    TP_idx = lines[tp_info_index+1]
    TP_time = lines[tp_info_index+3]
    
    #get rid of coma and cast to int
    TP_idx = int(float(TP_idx[:-1]))
    TP_time = int(float(TP_time[:-1]))
    
    # CHARGE IS TP_time seconds before TP_idx
    # could include time_step info as we assume that each step is 1s which sometimes might not be the case
    CHARGE_idx = TP_idx - TP_time
    
    return CHARGE_idx

data = get_curves(d_name)

for roast in data:
        plt.plot(roast)
plt.show()

# find the longest curve
clean_data = []
for roast in data:
    if roast.shape[0] > 400 and roast.shape[0] < 800:
        clean_data.append(roast)

max_length = 0
for roast in clean_data:
    if roast.shape[0] > max_length:
        max_length = roast.shape[0]
        

# pad the curves with nan
for i, roast in enumerate(clean_data):
    if roast.shape[0] < max_length:
        clean_data[i] = np.pad(roast, ((0, max_length - roast.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
# stack the curves
clean_data = np.array(clean_data)
print(clean_data.shape)

# save the data
np.save('data.npy', clean_data)




