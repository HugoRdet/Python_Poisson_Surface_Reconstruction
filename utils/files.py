import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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

def get_file_obj(name):
    f = open(name, "r")
    content=f.read()
    lines=content.split("\n")
   
    nb_vert=int(lines[7].split(" ")[-1])
    nb_faces=int(lines[8].split(" ")[-1])

    vert=dict()
    cpt_vert=1
    cpt_norm=1
    norm=dict()
    res=torch.empty(nb_vert,6)
    cpt_res=0
    L_fait=[]

    for l in lines:
        
        if len(l)<1 or (l[0]!="v" and l[:2]!="vn" and l[0]!="f"):
            continue
        else:
            if l[:2]=="vn":
                l=l.split(" ")
                x=float(l[1])
                y=float(l[2])
                z=float(l[3])
                norm[cpt_vert]=(x,y,z)
                cpt_norm+=1
                continue
            if l[:2]=="v ":
                l=l.split(" ")
                x=float(l[1])
                y=float(l[2])
                z=float(l[3])

                vert[cpt_vert]=(x,y,z)
                cpt_vert+=1
                continue
            if l[:2]=="f ":
                l=l.split(" ")
                l=l[1:]
                for elem in l:
                    if len(elem)<=1:
                        continue
                    else:
                        elem=elem.split("/")
                        if int(elem[0]) in L_fait:
                            continue
                        L_fait.append(int(elem[0]))
                        xv,yv,zv=vert[int(elem[0])]
                        xn,yn,zn=norm[int(elem[-1])]

                        res[cpt_res,0]=xv
                        res[cpt_res,1]=yv
                        res[cpt_res,2]=zv
                        res[cpt_res,3]=-xn
                        res[cpt_res,4]=-yn
                        res[cpt_res,5]=-zn
                        cpt_res+=1
                        

                
                    
                
                continue
    return res
            