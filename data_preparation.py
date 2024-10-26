import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import json 
import ast

def get_curves(dir_name,pre_roast_length=0):
    
    file_names = os.listdir(dir_name)
    roasts = []
    meta_datas = []
    for f_name in file_names:
        # print(f_name) 
        
        success,bt_curve = extract_curve(f_name,dir_name,"'temp2':",pre_roast_length)
        if success: # if we don't have enough pre roast data we don't use this roast
            _,et_curve = extract_curve(f_name,dir_name,"'temp1':",pre_roast_length)
            burner_curve = extract_burner(f_name,dir_name,pre_roast_length)
            np_bt_curve = np.array(bt_curve)
            np_et_curve = np.array(et_curve)
            np_burner_curve = np.array(burner_curve)
            if (len(np_bt_curve) < 1000 and len(np_et_curve) < 1000 and len(np_burner_curve) < 1000 
                and len(np_bt_curve) == len(np_et_curve) == len(np_burner_curve)):
                roast = np.column_stack((np_bt_curve,np_et_curve,np_burner_curve))
                roasts.append(roast)
                meta_data = create_meta_data(f_name)
                meta_datas.append(meta_data)
    meta_datas = one_hot_encoding(meta_datas,roasts,pre_roast_length)
    return roasts,meta_datas

def one_hot_encoding(meta_datas,roasts,pre_roast_length=0):
    origin = []
    processing = []
    for meta_data in meta_datas:
        origin.append(meta_data['origin'])
        processing.append(meta_data['processing'])
    origin = pd.get_dummies(origin)
    processing = pd.get_dummies(processing)
    one_hot_encoded = pd.concat([origin,processing],axis=1)
    one_hot_encoded = np.array(one_hot_encoded,dtype=bool)

    roast_degrees = []
    for roast in roasts:
        if len(roast) > 600+pre_roast_length:
            roast_degrees.append(1)
        else:
            roast_degrees.append(0)

    roast_degrees = np.array(roast_degrees,dtype=bool)
    one_hot_encoded = np.column_stack((one_hot_encoded,roast_degrees))
    print(one_hot_encoded[:10])

    return one_hot_encoded
def create_meta_data(f_name):
    meta_data = {}
    # origin
    meta_data['origin'] = get_origin(f_name)
    meta_data['processing'] = get_processing(f_name)

    return meta_data

def get_processing(f_name):
    washed_tags = ['washed','myta','kenia','la soledad','san jose','la maravilla','indie','uganda']
    honey_tags = ['honey','costa','tajlandia']
    natural_tags = ['natural','monjolo','fronteira','rwanda','nicaragua']
    tags = {
        'washed':washed_tags,
        'honey':honey_tags,
        'natural':natural_tags
    }
    for tag in tags:
        for t in tags[tag]:
            if t.lower() in f_name.lower():
                return tag.lower()
            
    return 'other'

def get_origin(f_name):
    brazil_tags = ['Brazylia','Brazil','Monjolo','Fazenda','California','Fronteira']
    gwatemala_tags = ['Gwatemala','Guatemala','San Jose Del Lago','La Maravilla','La soledad']
    colombia_tags = ['colombia','kolumbia','Jesus','ceron']
    costarica_tags = ['Costa','Rica','Costa Rica','las lajas']
    tajlandia_tags = ['Tajlandia','Thailand','Thajlandia','chiang','mai']
    other_tags = ['uganda','honduras','indie','Myanmar','Nicaragua','Kenya','rwanda']
    tags = {
        'brazil':brazil_tags,
        'gwatemala':gwatemala_tags,
        'colombia':colombia_tags,
        'costarica':costarica_tags,
        'tajlandia':tajlandia_tags,
    }
    for tag in tags:
        for t in tags[tag]:
            if t.lower() in f_name.lower():
                return tag.lower()
    for t in other_tags:
        if t.lower() in f_name.lower():
            return t.lower()
        
    return 'other'

            
def extract_meta_data(f_name,dir_name):
    # load whole data in the file into a single string
    with open(dir_name+"/"+f_name,"r") as f:
        lines = f.readlines()
        data = "".join(lines)
        data = ast.literal_eval(data)
        print(data)


def extract_burner(f_name,dir_name,pre_roast_length=0):
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
        drop_index = int(float(lines[lines.index("'DROP_time':")+1][:-1]))+charge_idx
        start_idx = charge_idx-pre_roast_length
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
    
    burner_value = []
    burner_events = []
    for i,e in enumerate(events_types):
        if(e==3):
            burner_value.append(events_value[i])
            burner_events.append(events[i])
    i=0
    while len(burner_curve) < drop_index-start_idx+1:
        while i< len(burner_events)-1 and len(burner_curve) + start_idx >= burner_events[i+1]:
            i+=1
        if len(burner_curve) < burner_events[0]-start_idx:
            burner_curve.append(2.1)
        elif i >= len(burner_events):
            burner_curve.append(burner_value[-1])
        else:
            burner_curve.append(burner_value[i])


        # burner_curve.append(0)
    burner_curve = [(x-1.0)*10.0 for x in burner_curve]
    return burner_curve
            
        
            
def extract_curve(f_name,dir_name,curve_name,pre_roast_length=0):
    curve = []
    with open(dir_name+"/"+f_name,"r") as f:
            lines = [
            line.strip("\n").split(" ") + [index + 1]
            for index, line in enumerate(f.readlines())
            ]
            if len(lines)==0:
                return False,[]
            lines = lines[0]

            #calculate indexes of CHARGE and DROP with regard to reading points, to extract temp readings only during roast
            if not has_enough_pre_roast_data(lines,pre_roast_length):
                return False,curve
            charge_index = get_charge(lines)
            drop_index = int(float(lines[lines.index("'DROP_time':")+1][:-1]))+charge_index
            
            #shift indexes to correspond to proper place in the file
            curve_index = lines.index(curve_name)
            charge_index += curve_index+1
            drop_index += curve_index+1
            
            current_index = charge_index-pre_roast_length
            while current_index <= drop_index:
                curve.append(float(lines[current_index][:-1]))
                current_index += 1
    
    #mapping to get rid 
    return True,curve
        
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
def has_enough_pre_roast_data(lines,pre_roast_length):
    tp_info_index = lines.index("'TP_idx':")
    TP_idx = lines[tp_info_index+1]
    TP_time = lines[tp_info_index+3]
    
    #get rid of coma and cast to int
    TP_idx = int(float(TP_idx[:-1]))
    TP_time = int(float(TP_time[:-1]))

    available_length = TP_idx - TP_time
    return available_length >= pre_roast_length

if __name__ == "__main__":
    d_name = "data"

    pre_roast_length = 75
    data,meta_data = get_curves(d_name,pre_roast_length)

    data = data[:int(len(data)*0.9)]
    for roast in data:
            plt.plot(roast)
    plt.show()

    # find the longest curve
    clean_data = []
    clean_meta_data = []
    for roast,meta in zip(data,meta_data):
        if roast.shape[0] > 400 and roast.shape[0] < 800:
            clean_data.append(roast)
            clean_meta_data.append(meta)

    max_length = 0
    for roast in clean_data:
        if roast.shape[0] > max_length:
            max_length = roast.shape[0]
    print(max_length)
            

    # pad the curves with nan
    for i, roast in enumerate(clean_data):
        if roast.shape[0] < max_length:
            clean_data[i] = np.pad(roast, ((0, max_length - roast.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
    # stack the curves
    clean_data = np.array(clean_data)
    clean_meta_data = np.array(clean_meta_data)
    print(clean_data.shape)
    print(clean_meta_data.shape)

    # save the data
    np.save('data.npy', clean_data)
    np.save('meta_data.npy', clean_meta_data)




