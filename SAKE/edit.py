import numpy as np
import ot
from tqdm import tqdm

def learn_mappings_counterfact(cf, indexes=(0,10), reg=1e-2):
    maps = []
    for i in tqdm(range(indexes[0], indexes[1])):
        e = cf[i]
        x = np.concatenate([e['source_embs'], e['forced_source_embs'], e['target_embs']])
        x_source = x[:2*len(e['source_embs'])]
        x_target = x[2*len(e['source_embs']):]
        ot_linear = ot.da.LinearTransport(reg=1e-2)
        ot_linear.fit(Xs=x_source, Xt=x_target)
        maps.append(ot_linear)
    
    return maps

