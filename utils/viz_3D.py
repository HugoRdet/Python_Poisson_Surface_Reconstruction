import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


from utils import*

def count_nodes_init(dico,node_id,x,y,z):
    
    boold=(node_id+"1") in dico
    
    if not boold:
        return 1
    
    for idx_son in range(1,9):
        tmp_idx=node_id+str(idx_son)
        _,_,_,borders,l=dico[tmp_idx]
        (gx,dx,gy,dy,gz,dz)=borders
        
        if x<gx or x>dx:
            continue
        if y<gy or y>dy:
            continue
        if z<gz or z>dz:
            continue
        
            
        return 1 + count_nodes_init(dico,tmp_idx,x,y,z) 
    return 0
        
        
        
    
    
def save_quadtree_viz(points_c,dico,name="quadtree_viz.blp"):
    
    borders=get_borders(points_c)
    
    min_x=borders[0].item()
    max_x=borders[1].item()
    min_y=borders[2].item()
    max_y=borders[3].item()
    min_z=borders[4].item()
    max_z=borders[5].item()
    
    min_coord=min(min_x,min_y)
    min_coord=min(min_coord,min_z)
    min_coord=min_coord
    
    max_coord=max(max_x,max_y)
    max_coord=max(max_coord,max_z)
    max_coord=max_coord
    
    X, Y, Z = np.mgrid[min_x:max_x:64j, min_y:max_y:64j, min_z:max_z:64j]
    resolution=64
    values =np.zeros((resolution,resolution,resolution))
    
    for ax in range(0,resolution):
        print(ax)
        for ay in range(0,resolution):
            for az in range(0,resolution):
                tmp_x=X[ax,ay,az]
                tmp_y=Y[ax,ay,az]
                tmp_z=Z[ax,ay,az]
                
                score=count_nodes_init(dico,"",tmp_x,tmp_y,tmp_z)
                
                values[ax,ay,az]=score

    
    bp.pack_ndarray_to_file(values, name)
    
    
    
def viz_quadtree(name="quadtree_viz.blp"):

    
    
    min_coord=-1
    
    max_coord=1

    X, Y, Z = np.mgrid[min_coord:max_coord:64j, min_coord:max_coord:64j, min_coord:max_coord:64j]

    values = bp.unpack_ndarray_from_file(name)
    
    
    layout = go.Layout(
    autosize=False,
    width=1000,
    height=1000,
    xaxis=go.layout.XAxis(linecolor="black", linewidth=1, mirror=True),
    yaxis=go.layout.YAxis(linecolor="black", linewidth=1, mirror=True),
    margin=go.layout.Margin(l=50, r=50, b=100, t=100, pad=4),
    )
    
    fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=0,
    isomax=25,
    opacity=0.3, # needs to be small to see through all surfaces
    surface_count=21, # needs to be a large number for good volume rendering
    ),layout=layout)
    
    fig.show()
    