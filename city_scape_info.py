
TrainIdToName = { 11: 'Person', 12: 'Rider', 13: 'Car', 14: 'Truck', 15: 'Bus', 16: 'Train', 17: 'Motorcycle', 18: 'Bicycle' }

OriginalIdToName = { 0: 'unlabeled',  
                     1: 'ego vehicle',
                     2: 'rectification border',
                     3: 'out of roi',
                     4: 'static',
                     5: 'dynamic',
                     6: 'ground',
                     7: 'road',
                     8: 'sidewalk',
                     9: 'parking',
                     10: 'rail track',
                     11: 'building',
                     12: 'wall',
                     13: 'fence',
                     14: 'guard rail',
                     15: 'bridge',
                     16: 'tunnel',
                     17: 'pole',
                     18: 'polegroup',
                     19: 'traffic light',
                     20: 'traffic sign',
                     21: 'vegetation',
                     22: 'terrain',
                     23: 'sky',
                     24: 'person',
                     25: 'rider',
                     26: 'car',
                     27: 'truck',
                     28: 'bus',
                     29: 'caravan',
                     30: 'trailer',
                     31: 'train',
                     32: 'motorcycle',
                     33: 'bicycle' }

def ImportantLabelMapping(value):
    if value == 11:
        return 11
    elif value == 21:
        return 21
    elif value == 24:
        return 24
    elif value == 25:
        return 24
    elif value == 26:
        return 26
    elif value == 27:
        return 26
    elif value == 28:
        return 26
    elif value == 29:
        return 26
    elif value == 33:
        return 33
    else:
        return -1

def PurityLabelMapping(value):
    #if value == 11:
    #    return 11
    #elif value == 21:
    #    return 21
    if value == 24:
        return 24
    elif value == 25:
        return 24
    elif value == 26:
        return 26
    elif value == 27:
        return 26
    elif value == 28:
        return 26
    elif value == 29:
        return 26
    elif value == 33:
        return 33
    else:
        return -1

def PurityLabelMapping2(value):
    if value == 11:
        return 11
    #elif value == 21:
    #    return 21
    elif value == 24:
        return 24
    elif value == 25:
        return 24
    elif value == 26:
        return 26
    elif value == 27:
        return 26
    elif value == 28:
        return 26
    elif value == 29:
        return 26
    else:
        return -1

def PurityLabelMapping3(value):
    #if value == 11:
    #    return 11
    if value == 21:
        return 21
    elif value == 24:
        return 24
    elif value == 25:
        return 24
    elif value == 26:
        return 26
    elif value == 27:
        return 26
    elif value == 28:
        return 26
    elif value == 29:
        return 26
    else:
        return -1
