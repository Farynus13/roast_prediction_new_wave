import numpy as np
import matplotlib.pyplot as plt
import os
import re


def get_curves(dir_name):
    
    file_names = os.listdir(dir_name)
    roasts = []
    meta_datas = []
    for f_name in file_names:
        # print(f_name) 
        
        success,bt_curve = extract_curve(f_name,dir_name,"'temp2':")
        if success: # if we don't have enough pre roast data we don't use this roast
            _,et_curve = extract_curve(f_name,dir_name,"'temp1':")
            burner_curve = extract_burner(f_name,dir_name)
            np_bt_curve = np.array(bt_curve)
            np_et_curve = np.array(et_curve)
            np_burner_curve = np.array(burner_curve)
            if (len(np_bt_curve) < 1000 and len(np_et_curve) < 1000 and len(np_burner_curve) < 1000 
                and len(np_bt_curve) == len(np_et_curve) == len(np_burner_curve)):
                roast = np.column_stack((np_bt_curve,np_et_curve,np_burner_curve))
                roasts.append(roast)
                meta_data = create_meta_data(dir_name+"/"+f_name)
                meta_datas.append(meta_data)
    return roasts,meta_datas


def create_meta_data(f_name):
    """
    Extracts moisture and density values from the file content.
    
    Args:
        f_name (str): The file name containing the data.
    
    Returns:
        np.ndarray: A NumPy array containing [moisture, density].
    """
    with open(f_name, 'r') as file:
        content = file.read()

    # Extract 'moisture_greens' value
    moisture_match = re.search(r"'moisture_greens':\s*([\d.]+)", content)
    moisture = float(moisture_match.group(1)) if moisture_match else None

    # Extract 'density' value
    density_match = re.search(r"'density':\s*\[([\d.]+),\s*'g',\s*[\d.]+,\s*'l'\]", content)
    density = float(density_match.group(1)) if density_match else None

    return np.array([moisture, density])

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
        drop_index = int(float(lines[lines.index("'DROP_time':")+1][:-1]))+charge_idx
        start_idx = charge_idx
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
            
        
            
def extract_curve(f_name,dir_name,curve_name):
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

if __name__ == "__main__":
    d_name = "roasts/annotated"

    data,meta_data = get_curves(d_name)
    print(meta_data[:5])
    data = data[:int(len(data)*0.9)]
    for roast in data:
            plt.plot(roast)
    plt.show()

    # find the longest curve
    clean_data = []
    clean_meta_data = []
    for roast,meta in zip(data,meta_data):
        if roast.shape[0] > 400 and roast.shape[0] < 800 and not meta[0] is None and not meta[1] is None:
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




