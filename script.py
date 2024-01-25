import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import bloscpack as bp
import time

from utils import*

points_c=get_file_ply("./data/dargon.obj")

dico=dict()
bord=get_borders(points_c)

create_quadtree(dico,0,11,points_c,bord,"")

def get_min_width(dico,D):
    min_width=1e9
    for node in dico:
        if (dico[node][2]<min_width and dico[node][0]<=D ):
            min_width=dico[node][2]
    return min_width
        



def get_os(nb_s,dico,D):
    cpt_p=0
    s_p=torch.ones((nb_s,6))
    o_s=torch.ones((nb_s,8,5)) #interpolation weight , width , center
    for node in dico:
        if dico[node][0]==D or (dico[node][0]<=D and len(dico[node][-1])==1):
            depth,o_c,o_w,border,l_points=dico[node]

            for p in l_points:
                s_p[cpt_p]=p.clone()

                gx,dx,gy,dy,gz,dz=dico[node[:-1]][-2]

                tmp_o_vec=torch.empty((8,5))
                for idx_nbgr in range(1,9):
                    tmp_node=node[:-1]+str(idx_nbgr)
                    tmp_o_vec[idx_nbgr-1,1]=dico[tmp_node][2]
                    tmp_o_vec[idx_nbgr-1,2:]=dico[tmp_node][1]

                    

                    u=(s_p[cpt_p,0]-gx)/(dx-gx)
                    v=(s_p[cpt_p,1]-gy)/(dy-gy)
                    w=(s_p[cpt_p,2]-gz)/(dz-gz)
                    

                    
                    if idx_nbgr==1:
                        tmp_o_vec[0,0]=(1-u)*v*w
                    if idx_nbgr==2:
                        tmp_o_vec[1,0]=u*v*w
                    if idx_nbgr==3:
                        tmp_o_vec[2,0]=(1-u)*(1-v)*w
                    if idx_nbgr==4:
                        tmp_o_vec[3,0]=u*(1-v)*w
                    if idx_nbgr==5:
                        tmp_o_vec[4,0]=(1-u)*v*(1-w)
                    if idx_nbgr==6:
                        tmp_o_vec[5,0]=u*v*(1-w)
                    if idx_nbgr==7:
                        tmp_o_vec[6,0]=(1-u)*(1-v)*(1-w)
                    if idx_nbgr==8:
                        tmp_o_vec[7,0]=u*(1-v)*(1-w)
                        
                
                
                
                o_s[cpt_p]=tmp_o_vec.clone()
                cpt_p+=1
    return s_p,o_s

def get_V_tmp(coords_grid,l_o,l_s):
    #q [ r_x , r_y , r_z,  3 ]
    #     0    1     2    3

    q=coords_grid.clone().unsqueeze(3).unsqueeze(4).repeat(1,1,1,l_o.shape[0],8,1)
    #q [ r_x , r_y , r_z, l_o , 8 , 3 ]
    #     0     1     2    3    4   5

    q[:,:,:,:,:,0]-=l_o[:,:,2]
    q[:,:,:,:,:,0]/=l_o[:,:,1]
    q[:,:,:,:,:,1]-=l_o[:,:,3]
    q[:,:,:,:,:,1]/=l_o[:,:,1]
    q[:,:,:,:,:,2]-=l_o[:,:,4]
    q[:,:,:,:,:,2]/=l_o[:,:,1]
    
    test=torch.exp(-torch.sum((q)**2,dim=5)/(1.5**2))
    
    
    test=test*l_o[:,:,0]
    

    res=l_s[:,3:].clone().unsqueeze(1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    res=res.repeat(coords_grid.shape[0],coords_grid.shape[1],coords_grid.shape[2],1,8,1)
    



    res[:,:,:,:,0,0]*=test[:,:,:,:,0]
    res[:,:,:,:,0,1]*=test[:,:,:,:,0]
    res[:,:,:,:,0,2]*=test[:,:,:,:,0]
    res[:,:,:,:,1,0]*=test[:,:,:,:,1]
    res[:,:,:,:,1,1]*=test[:,:,:,:,1]
    res[:,:,:,:,1,2]*=test[:,:,:,:,1]
    res[:,:,:,:,2,0]*=test[:,:,:,:,2]
    res[:,:,:,:,2,1]*=test[:,:,:,:,2]
    res[:,:,:,:,2,2]*=test[:,:,:,:,2]
    res[:,:,:,:,3,0]*=test[:,:,:,:,3]
    res[:,:,:,:,3,1]*=test[:,:,:,:,3]
    res[:,:,:,:,3,2]*=test[:,:,:,:,3]
    res[:,:,:,:,4,0]*=test[:,:,:,:,4]
    res[:,:,:,:,4,1]*=test[:,:,:,:,4]
    res[:,:,:,:,4,2]*=test[:,:,:,:,4]
    res[:,:,:,:,5,0]*=test[:,:,:,:,5]
    res[:,:,:,:,5,1]*=test[:,:,:,:,5]
    res[:,:,:,:,5,2]*=test[:,:,:,:,5]
    res[:,:,:,:,6,0]*=test[:,:,:,:,6]
    res[:,:,:,:,6,1]*=test[:,:,:,:,6]
    res[:,:,:,:,6,2]*=test[:,:,:,:,6]
    res[:,:,:,:,7,0]*=test[:,:,:,:,7]
    res[:,:,:,:,7,1]*=test[:,:,:,:,7]
    res[:,:,:,:,7,2]*=test[:,:,:,:,7]

    
    
    res=torch.sum(res,dim=(3,4))
   
    
    return res

#octree depth

#grid resolution


#
#l_s,l_o=get_os(s_p.shape[0],dico,D)

def get_V_vec(points_c,dico,D,name="V.blp"):
    
    l_s,l_o=get_os(points_c.shape[0],dico,D)

    batch_size=1

    min_x=torch.min(points_c[:,0]).item()
    max_x=torch.max(points_c[:,0]).item()
    min_y=torch.min(points_c[:,1]).item()
    max_y=torch.max(points_c[:,1]).item()
    min_z=torch.min(points_c[:,2]).item()
    max_z=torch.max(points_c[:,2]).item()
    resolution=1024
    
    x_range=torch.linspace(min_x,max_x,resolution)
    y_range=torch.linspace(min_y,max_y,resolution)
    z_range=torch.linspace(min_z,max_z,resolution)

    grid_coords=torch.zeros((resolution,resolution,resolution,3))
    grid_res=torch.zeros((resolution,resolution,resolution,3))

    grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

    grid_coords[:,:,:,0]=grid_x
    grid_coords[:,:,:,1]=grid_y
    grid_coords[:,:,:,2]=grid_z

    sep_grid=4
    
    for batch_samples in range(0,l_o.shape[0],batch_size):
        for sep_grid_x in [0,256,512,768]:
            for sep_grid_y in [0,256,512,768]:
                for sep_grid_z in [0,256,512,768]:
        
        
                    grid_res[sep_grid_x:sep_grid_x+256,sep_grid_y:sep_grid_y+256,sep_grid_z:sep_grid_z+256]+=get_V_tmp(grid_coords[sep_grid_x:sep_grid_x+256,sep_grid_y:sep_grid_y+256,sep_grid_z:sep_grid_z+256],l_o[batch_samples:batch_samples+batch_size],l_s[batch_samples:batch_samples+batch_size])
        

    bp.pack_ndarray_to_file(grid_res.numpy(), name)

    
    

    
    
start_time = time.time()    
#get_V_vec(points_c,dico,D=10,name="V_dargon_10.blp")   
end_time = time.time()

print("--- %s seconds ---" % (np.around(end_time - start_time,5)))
    
    #q_vec=torch.where( (q_vec>=0)&(q_vec<100), )

values = bp.unpack_ndarray_from_file("./V_dargon_6.blp")
values=torch.from_numpy(values)

def compute_divergence(V):
    # Initialize the divergence tensor with the same spatial dimensions, but only one channel
    div_V = torch.zeros((V.shape[0], V.shape[1], V.shape[2]))
    
    # Compute the x-component of the divergence using central differences
    # For the borders, you can use forward or backward differences
    div_V[1:-1, :,:] += (V[2:, :,:, 0] - V[:-2, :,:, 0]) / 2
    
    # Compute the y-component of the divergence using central differences
    # For the borders, you can use forward or backward differences
    div_V[:, 1:-1,:] += (V[:, 2:,:, 1] - V[:, :-2,:, 1]) / 2
    div_V[:, :,1:-1] += (V[:,: ,2:, 2] - V[:, :, :-2, 2]) / 2
    
    
    # Handle the borders if necessary (here we assume a zero-gradient boundary condition)
    # This can be replaced with a more appropriate condition for your specific case
    div_V[0, :,:] += (V[1, :,:, 0] - V[0, :,:, 0])
    div_V[-1, :,:] += (V[-1, :,:, 0] - V[-2, :,:, 0])
    div_V[:, 0,:] += (V[:, 1,:, 1] - V[:, 0,:, 1])
    div_V[:, -1,:] += (V[:, -1,:, 1] - V[:, -2,:, 1])
    div_V[:,:, 0] += (V[:, :,1, 1] - V[:,:, 0, 1])
    div_V[:,:, -1] += (V[:, :,-1, 1] - V[:,:, -2, 1])

    return div_V

grad=compute_divergence(values)



def get_list_gaussians(borders,D):
    res_centers=torch.empty(8**D,4)
    borders=get_borders(points_c)

    min_x=borders[0].item()
    max_x=borders[1].item()
    min_y=borders[2].item()
    max_y=borders[3].item()
    min_z=borders[4].item()
    max_z=borders[5].item()

    Width=(max_x-min_x)/(2**D)
    cpt_idx=0

    for i in range(2**D):
        for j in range(2**D):
            for w in range(2**D):
            
                xa=min_x+(i*Width)+Width/2.0
                ya=min_y+(j*Width)+Width/2.0
                za=min_z+(w*Width)+Width/2.0
        
                res_centers[cpt_idx,0]=xa
                res_centers[cpt_idx,1]=ya
                res_centers[cpt_idx,2]=za
                res_centers[cpt_idx,3]=Width
                
                cpt_idx+=1
    return res_centers


bord=get_borders(points_c)




def get_inner_product_gradVB(V,L_gaussians,borders):
    res=torch.empty(L_gaussians.shape[0])

    min_x=torch.min(points_c[:,0]).item()
    max_x=torch.max(points_c[:,0]).item()
    min_y=torch.min(points_c[:,1]).item()
    max_y=torch.max(points_c[:,1]).item()
    min_z=torch.min(points_c[:,2]).item()
    max_z=torch.max(points_c[:,2]).item()

    resolution=128
    
    x_range=torch.linspace(min_x,max_x,resolution)
    y_range=torch.linspace(min_y,max_y,resolution)
    z_range=torch.linspace(min_z,max_z,resolution)
    
    
    grid_coords=torch.zeros((resolution,resolution,resolution,3))
    

    grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')

    grid_coords[:,:,:,0]=grid_x
    grid_coords[:,:,:,1]=grid_y
    grid_coords[:,:,:,2]=grid_z

    
    


    for idx_gaussian in range(L_gaussians.shape[0]):
        tmp_res=grid_coords.clone()
        tmp_sum_g=(tmp_res[:,:,:,0]-L_gaussians[idx_gaussian,0])**2
        tmp_sum_g+=(tmp_res[:,:,:,1]-L_gaussians[idx_gaussian,1])**2
        tmp_sum_g+=(tmp_res[:,:,:,2]-L_gaussians[idx_gaussian,2])**2
        
        tmp_sum_g=torch.exp(-(tmp_sum_g/(2*L_gaussians[idx_gaussian,3]**2)))
        
        tmp_sum_g=tmp_sum_g*V*(2*L_gaussians[idx_gaussian,3]**2)
        tmp_sum_g=torch.sum(tmp_sum_g)
        res[idx_gaussian]=tmp_sum_g
        
    
    
        
    return res

def get_inner_product_gradB_B_vec(size_grid, L_gaussians, borders):
    res = torch.zeros((L_gaussians.shape[0], L_gaussians.shape[0]))

    batch_size=min(L_gaussians.shape[0],256)

    min_x=torch.min(points_c[:,0]).item()
    max_x=torch.max(points_c[:,0]).item()
    min_y=torch.min(points_c[:,1]).item()
    max_y=torch.max(points_c[:,1]).item()
    min_z=torch.min(points_c[:,2]).item()
    max_z=torch.max(points_c[:,2]).item()

    width_x = max_x - min_x
    width_y = max_y - min_y
    width_z = max_z - min_z

    xs = torch.linspace(min_x - width_x * 0.125, max_x + width_x * 0.125, steps=size_grid)
    ys = torch.linspace(min_y - width_y * 0.125, max_y + width_y * 0.125, steps=size_grid)
    zs = torch.linspace(min_z - width_z * 0.125, max_z + width_z * 0.125, steps=size_grid)
    
    x, y, z = torch.meshgrid(xs, ys, zs, indexing='xy')

    grid = torch.empty(size_grid, size_grid, size_grid, 3)
    grid[:, :, :, 0] = x
    grid[:, :, :, 1] = y
    grid[:, :, :, 2] = z
    
    cpt_economies=0

    for idx_gaussian_1 in range(L_gaussians.shape[0]):
        if (idx_gaussian_1%1000==0):
            print(idx_gaussian_1,"\t\t",cpt_economies)
        for idx_gaussian_2 in range(idx_gaussian_1, L_gaussians.shape[0], batch_size):
            batch_end = min(idx_gaussian_2 + batch_size, L_gaussians.shape[0])
            taille_tmp = batch_end - idx_gaussian_2

            tmp_g1 = grid.clone().unsqueeze(0).repeat((taille_tmp, 1, 1, 1, 1))
            
            tmp_g1 = torch.exp(-((tmp_g1[:,:,:,:,0] - L_gaussians[idx_gaussian_1, 0])**2 + (tmp_g1[:,:,:,:,1] - L_gaussians[idx_gaussian_1, 1])**2+(tmp_g1[:,:,:,:,2] - L_gaussians[idx_gaussian_1, 2])**2) / (2 * L_gaussians[idx_gaussian_1, 3]**2))

            tmp_g2 = grid.clone().unsqueeze(0).repeat((taille_tmp, 1, 1, 1, 1))
            L_gaussian_0_vec = L_gaussians[idx_gaussian_2:batch_end, 0].view((taille_tmp, 1, 1, 1)).repeat((1,tmp_g2.shape[-2], tmp_g2.shape[-2], tmp_g2.shape[-2]))
            L_gaussian_1_vec = L_gaussians[idx_gaussian_2:batch_end, 1].view((taille_tmp, 1, 1, 1)).repeat((1,tmp_g2.shape[-2], tmp_g2.shape[-2], tmp_g2.shape[-2]))
            L_gaussian_2_vec = L_gaussians[idx_gaussian_2:batch_end, 2].view((taille_tmp, 1, 1, 1)).repeat((1,tmp_g2.shape[-2], tmp_g2.shape[-2], tmp_g2.shape[-2]))
            L_gaussian_3_vec = L_gaussians[idx_gaussian_2:batch_end, 3].view((taille_tmp, 1, 1, 1)).repeat((1,tmp_g2.shape[-2], tmp_g2.shape[-2], tmp_g2.shape[-2]))

            fac_x = ((tmp_g2[:,:,:,:,0] - L_gaussian_0_vec)**2 - (L_gaussian_3_vec**2)) / (L_gaussian_3_vec**4)
            fac_y = ((tmp_g2[:,:,:,:,1] - L_gaussian_1_vec)**2 - (L_gaussian_3_vec**2)) / (L_gaussian_3_vec**4)
            fac_z = ((tmp_g2[:,:,:,:,2] - L_gaussian_2_vec)**2 - (L_gaussian_3_vec**2)) / (L_gaussian_3_vec**4)

            map_g2 = torch.exp(-((tmp_g2[:,:,:,:,0] - L_gaussian_0_vec)**2 + (tmp_g2[:,:,:,:,1] - L_gaussian_1_vec)**2 + (tmp_g2[:,:,:,:,2] - L_gaussian_2_vec)**2) / (2 * L_gaussian_3_vec**2))
            tmp_g2[:,:,:,:,0] = fac_x * map_g2.clone()
            tmp_g2[:,:,:,:,1] = fac_y * map_g2.clone()
            tmp_g2[:,:,:,:,2] = fac_z * map_g2.clone()

            tmp_res = torch.sum(tmp_g1 * tmp_g2[:,:,:,:,0] * (L_gaussian_3_vec**2), dim=[1, 2, 3]) 
            tmp_res += torch.sum(tmp_g1 * tmp_g2[:,:,:,:,1] * (L_gaussian_3_vec**2), dim=[1, 2, 3])
            tmp_res += torch.sum(tmp_g1 * tmp_g2[:,:,:,:,2] * (L_gaussian_3_vec**2), dim=[1, 2, 3])
            if torch.sum(torch.abs(tmp_res))<1e-8:
                cpt_economies+= max(0,L_gaussians.shape[0]-(idx_gaussian_2+batch_size))
                break
            res[idx_gaussian_1, idx_gaussian_2:batch_end] = tmp_res

    res += torch.flip(res, [0, 1])

    for idx_gaussian_1 in range(L_gaussians.shape[0]):
        res[idx_gaussian_1, idx_gaussian_1] *= 0.5

    cpt_economies=cpt_economies+(((L_gaussians.shape[0])**2)/2.0)-L_gaussians.shape[0]

    print(cpt_economies,"     ",(cpt_economies/(L_gaussians.shape[0]**2))*100,"%")

    return res

for depth in [2,3,4,5,6,8,10]:
    print(depth)
    bord=get_borders(points_c)
    L_gaussians=get_list_gaussians(bord,depth)
    L_oo=get_inner_product_gradB_B_vec(32,L_gaussians,bord)
    bp.pack_ndarray_to_file(L_oo.numpy(),"L_oo_"+str(depth)+".blp")


