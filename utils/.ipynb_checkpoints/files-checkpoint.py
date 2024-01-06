import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import bloscpack as bp

def get_file(name):
    f = open(name, "r")
    content=f.read()
    lines=content.split("\n")
    
    l_points=[]
    
    for l in lines:
        vec=torch.empty(6)
        l=l.split(" ")
        cpt=0
        for number in (l):
            if len(number)>3:
                vec[cpt]=float(number)
                cpt+=1
        if cpt>2:
            l_points.append(vec)
    
    res=torch.empty((len(l_points),6))
    for idx_tensor,tensor in enumerate(l_points):
        res[idx_tensor]=tensor
    return res