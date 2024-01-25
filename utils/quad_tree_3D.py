import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import bloscpack as bp

def get_borders(points):
    min_x=torch.min(points[:,0])
    max_x=torch.max(points[:,0])
    min_y=torch.min(points[:,1])
    max_y=torch.max(points[:,1])
    min_z=torch.min(points[:,2])
    max_z=torch.max(points[:,2])
    
    sep_x=torch.sqrt((max_x-min_x)**2)
    sep_y=torch.sqrt((max_y-min_y)**2)
    sep_z=torch.sqrt((max_z-min_z)**2)
    
    width=max(sep_x,sep_y)
    width=max(width,sep_z)
    
    return (min_x,min_x+width,min_y,min_y+width,min_z,min_z+width)

def ajout_node(borders,borders_father,l_p,label,tmp_depth,dico):
    gx,dx,gy,dy,gz,dz=borders
    
    tmp_oc=torch.zeros(3)
    tmp_oc[0]=(gx+dx)/2.0
    tmp_oc[1]=(gy+dy)/2.0
    tmp_oc[2]=(gz+dz)/2.0
    
    dico[label]=(tmp_depth,tmp_oc,dx-gx,borders,borders_father,l_p)
    

def create_quadtree(dico,tmp_depth,depth,points,borders,label):
    min_x=borders[0]
    max_x=borders[1]
    min_y=borders[2]
    max_y=borders[3]
    min_z=borders[4]
    max_z=borders[5]
    
    sep_x=min_x+torch.sqrt((max_x-min_x)**2)/2.0
    sep_y=min_y+torch.sqrt((max_y-min_y)**2)/2.0
    sep_z=min_z+torch.sqrt((max_z-min_z)**2)/2.0
    
     
    
    #avant sep z
    mask=(points[:,0]<= sep_x) & (points[:,1]> sep_y ) & (points[:,2]<= sep_z )
    hgb= points[mask]
    mask=(points[:,0]> sep_x) & (points[:,1]> sep_y) & (points[:,2]<= sep_z )
    hdb= points[mask]
    mask=(points[:,0]<= sep_x) & (points[:,1]<= sep_y) &  (points[:,2]<= sep_z )
    bgb= points[mask]
    mask=(points[:,0]> sep_x) & (points[:,1]<= sep_y) & (points[:,2]<= sep_z )
    bdb= points[mask]
    
    #apres sep z
    mask=(points[:,0]<= sep_x) & (points[:,1]> sep_y) & (points[:,2]> sep_z )
    hga= points[mask]
    mask=(points[:,0]> sep_x) & (points[:,1]> sep_y) & (points[:,2]> sep_z )
    hda= points[mask]
    mask=(points[:,0]<= sep_x) & (points[:,1]<= sep_y) & (points[:,2]> sep_z )
    bga= points[mask]
    mask=(points[:,0]> sep_x) & (points[:,1]<= sep_y) & (points[:,2]> sep_z )
    bda= points[mask]
    
    
    gx=min_x
    gy=sep_y
    gz=min_z
    dx=sep_x
    dy=max_y
    dz=sep_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,hgb,label+"1",tmp_depth,dico)
    
    if (hgb.shape[0]>1)and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,hgb.clone(),border,label+"1")
    
    gx=sep_x
    gy=sep_y
    gz=min_z
    dx=max_x
    dy=max_y
    dz=sep_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,hdb,label+"2",tmp_depth,dico)
    
    if (hdb.shape[0]>1)and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,hdb.clone(),border,label+"2")
    
    gx=min_x
    gy=min_y
    gz=min_z
    dx=sep_x
    dy=sep_y
    dz=sep_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,bgb,label+"3",tmp_depth,dico)
    
    
    
    
    if (bgb.shape[0]>1)and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,bgb.clone(),border,label+"3")
    
    gx=sep_x
    gy=min_y
    gz=min_z
    dx=max_x
    dy=sep_y
    dz=sep_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,bdb,label+"4",tmp_depth,dico)

    if (bdb.shape[0]>1) and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,bdb.clone(),border,label+"4")
        
    gx=min_x
    gy=sep_y
    gz=sep_z
    dx=sep_x
    dy=max_y
    dz=max_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,hga,label+"5",tmp_depth,dico)
    
    if (hga.shape[0]>1)and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,hga.clone(),border,label+"5")
    
    gx=sep_x
    gy=sep_y
    gz=sep_z
    dx=max_x
    dy=max_y
    dz=max_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,hda,label+"6",tmp_depth,dico)
    
    if (hda.shape[0]>1)and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,hda.clone(),border,label+"6")
    
    gx=min_x
    gy=min_y
    gz=sep_z
    dx=sep_x
    dy=sep_y
    dz=max_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,bga,label+"7",tmp_depth,dico)
    
    
    
    
    if (bga.shape[0]>1)and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,bga.clone(),border,label+"7")
    
    gx=sep_x
    gy=min_y
    gz=sep_z
    dx=max_x
    dy=sep_y
    dz=max_z
    
    border=(gx,dx,gy,dy,gz,dz)
    
    ajout_node(border,borders,bda,label+"8",tmp_depth,dico)

    if (bda.shape[0]>1) and depth!=tmp_depth:
        create_quadtree(dico,tmp_depth+1,depth,bda.clone(),border,label+"8")
    