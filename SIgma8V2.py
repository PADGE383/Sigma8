#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import illustris_python as il
import pandas as pd
import matplotlib
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy import spatial
from scipy.spatial import cKDTree
from tqdm import tqdm

import caesar
import readgadget
from readgadget import readsnap
import MAS_library as MASL
import smoothing_library as SL


# In[ ]:


TNG3snappath   = "/Volumes/Seagate/TNG300-3/output"
TNG3grouppath   = "/Volumes/Seagate/TNG300-1/output"
TNG3num=99

TNG1snappath   = "/Volumes/Seagate/TNG100-3/output"
TNG1grouppath   = "/Volumes/Seagate/TNG100-1/output"
TNG1num=99

ILLsnappath    = "/Volumes/Seagate/Illustris-3/output"
ILLgrouppath    = "/Volumes/Seagate/Illustris-1/output"
ILLnum=135

SIMBAsnappath  = "/Volumes/Seagate/Simba-Flag/output/snapdir_001/snap_m100n1024_151.hdf5"
SIMBAgrouppath  = "/Volumes/Seagate/Simba-Flag/output/groups_001/m100n1024_151.hdf5"
SIMBAnum= 0


# In[1]:


def SimParam(grouppath,snappath,num):
    if num == 0:
        #SIMBA
        #Boxsize
        subhalos=caesar.load(grouppath)
        boxsize=np.float64(subhalos.simulation.boxsize.to('Mpc/h')) # Mpc/h

        #DM Positions
        snap_pos = np.array(readgadget.read_block(snappath, "POS ", [1])/1e3,dtype=np.float32) #positions in Mpc/h

        #SubHalo Positions and Luminosity
        subhalo_pos=np.array([i.pos.to('Mpc/h') for i in subhalos.galaxies],dtype=np.float32) # Mpc/h
        subhalo_lum=np.array([i.absmag['wfcam_k'] for i in subhalos.galaxies],dtype=np.float32)-1.87 #k_mag

        #Halo Posotion and Mass/Using 
        halo_pos=subhalo_pos # Mpc/h
        halo_mass=np.array([i.masses['total'].to('Msun/h') for i in subhalos.galaxies],dtype=np.float32) # Msun
    else:
        #IllustrisTNG
        #Boxsize
        header = il.groupcat.loadHeader(grouppath,num)
        boxsize = header["BoxSize"]/1000 # Mpc/h

        #DM Positions
        snap=il.snapshot.loadSubset(snappath,num,'dm')
        snap_pos=np.array(snap['Coordinates'],dtype=np.float32)/1000 #Mpc/h

        #SubHalo Positions and Luminosity
        subhalo_fields = ["SubhaloPos","SubhaloStellarPhotometrics","SubhaloMass"]
        subhalos = il.groupcat.loadSubhalos(grouppath,num,fields=subhalo_fields)
        subhalo_pos = subhalos["SubhaloPos"]/1000   # Mpc/h
        subhalo_lum = subhalos["SubhaloStellarPhotometrics"][:,3] #K_Mag

        #Halo Posotion and Mass/Using 
        #halo_fields = ["GroupPos","GroupMass"]
        #halos = il.groupcat.loadHalos(grouppath,num,fields=halo_fields)
        halo_pos = subhalo_pos # Mpc/h #halos["GroupPos"]/1000 
        halo_mass = subhalos["SubhaloMass"]*1e10 # Msun/h #halos["GroupMass"]*1e10 # Msun/h

    #Mag Cut
    cut = (subhalo_lum <= 20) & (subhalo_lum >= -20)
    subhalo_pos = subhalo_pos[cut]
    subhalo_lum = subhalo_lum[cut]

    #Sim volume
    vol_sim=boxsize**3

    #Spheres
    N = int(boxsize//16)
    coords = np.linspace(8, boxsize - 8, N)
    xv, yv, zv = np.meshgrid(coords, coords, coords, indexing="ij")
    centers = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T

    #Sphere Volume
    vol_sphere=(4/3) * np.pi * 8**3

    ################
    #DM S8
    #DM Tree
    dm_pos=snap_pos%boxsize
    dm_tree=cKDTree(dm_pos,boxsize=boxsize)

    #DM Sim Density
    dm_sim=len(snap_pos)
    dmrho_sim=dm_sim/vol_sim

    #Sphere Densities
    dmrho_sphere=[]
    for i in tqdm(range(len(centers))):
        mask = dm_tree.query_ball_point(centers[i],r=8)
        dm_single = len(snap_pos[mask])

        dmrho_single=dm_single/vol_sphere
        dmrho_sphere.append(dmrho_single)
    dmrho_sphere=np.array(dmrho_sphere)

    #Sphere Overdensities
    dmdelta_sphere=(dmrho_sphere-dmrho_sim)/dmrho_sim

    #Sigma8/RMS
    dms8=np.sqrt(np.mean(dmdelta_sphere**2))

    ################
    ################

    #Mass S8
    #Mass Tree
    mass_pos=halo_pos%boxsize
    mass_tree=cKDTree(mass_pos,boxsize=boxsize)

    #Mass Sim Density
    mass_sim=sum(halo_mass)
    massrho_sim=mass_sim/vol_sim

    #Sphere Densities
    massrho_sphere=[]
    for i in tqdm(range(len(centers))):
        mask = mass_tree.query_ball_point(centers[i],r=8)
        mass_single = sum(halo_mass[mask])

        massrho_single=mass_single/vol_sphere
        massrho_sphere.append(massrho_single)
    massrho_sphere=np.array(massrho_sphere)

    #Sphere Overdensities
    massdelta_sphere=(massrho_sphere-massrho_sim)/massrho_sim

    #Sigma8/RMS
    masss8=np.sqrt(np.mean(massdelta_sphere**2))

    ################
    ################

    #Lum S8
    #Lum Tree
    lum_pos=subhalo_pos%boxsize
    lum_tree=cKDTree(lum_pos,boxsize=boxsize)

    #Lum Sim Density
    lum_sim=sum(subhalo_lum)
    lumrho_sim=lum_sim/vol_sim

    #Sphere Densities
    lumrho_sphere=[]
    for i in tqdm(range(len(centers))):
        mask = lum_tree.query_ball_point(centers[i],r=8)
        lum_single = sum(subhalo_lum[mask])

        lumrho_single=lum_single/vol_sphere
        lumrho_sphere.append(lumrho_single)
    lumrho_sphere=np.array(lumrho_sphere)

    #Sphere Overdensities
    lumdelta_sphere=(lumrho_sphere-lumrho_sim)/lumrho_sim

    #Sigma8/RMS
    lums8=np.sqrt(np.mean(lumdelta_sphere**2))

    ################


    return dms8,masss8,lums8


# In[ ]:


#TNG3
tng3_dm,tng3_mass,tng3_lum=SimParam(TNG3grouppath,TNG3snappath,TNG3num)

#TNG1
tng1_dm,tng1_mass,tng1_lum=SimParam(TNG1grouppath,TNG1snappath,TNG1num)

#Illustris
ill_dm,ill_mass,ill_lum=SimParam(ILLgrouppath,ILLsnappath,ILLnum)

#SIMBA
# simba_dm,simba_mass,simba_lum=SimParam(SIMBAgrouppath,SIMBAsnappath,SIMBAnum)


# In[ ]:


print("TNG3")
print(tng3_dm)
print(tng3_mass)
print(tng3_lum)
print("TNG1")
print(tng1_dm)
print(tng1_mass)
print(tng1_lum)
print("Illustrious")
print(ill_dm)
print(ill_mass)
print(ill_lum)
# print("SIMBA")
# print(simba_dm)
# print(simba_mass)
# print(simba_lum)

