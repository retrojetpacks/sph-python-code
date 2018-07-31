'''Write Gadget initial condition file in hdf5'''

from __future__ import division
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import os
from shutil import copyfile
import Quick_Render as Q

#Code Units
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
code_G     = c.G /code_L**3*code_M*code_time**2
AU_scale   = 100


gas_col  = '#224499'
dust_col = '#880088'
graft_dir = 'graft_dir/'
init_setups = 'init_setups/'
snapprefix = ''
#Set up parameters


#====+ Initial Condition Dictionaries +====#
#Work in Progress

#Basic Star
Star_A_dict = {'name':'Mstar1','modes':['star'],'M_star':1}

#Relaxed Polytropes
poly_B_dict  = {'modes':['polytrope_Tc'],'N_poly':10000,'M_poly':0.001,'Tc_poly':200,'n_poly':1.5}
poly_C_dict  = {'modes':['polytrope_Tc'],'N_poly':100000,'M_poly':0.001,'Tc_poly':200,'n_poly':1.5}
#Clement polytrope tests
Poly_D_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.05,'n_poly':1.5}
Poly_E_dict  = {'modes':['polytrope_R'],'N_poly':1000000,'M_poly':0.005,'R_poly':0.05,'n_poly':1.5}
Poly_F_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.03,'n_poly':1.5}
Poly_G_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.01,'n_poly':1.5}
#Clump in disc polytropes
Poly_H_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.003,'R_poly':0.03,'n_poly':1.5}
Poly_I_dict  = {'modes':['polytrope_R'],'N_poly':400000,'M_poly':0.003,'R_poly':0.03,'n_poly':1.5}
Poly_J_dict  = {'modes':['polytrope_R'],'N_poly':800000,'M_poly':0.005,'R_poly':0.03,'n_poly':2.5}
Poly_K_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.03,'n_poly':2.5}
Poly_L_dict  = {'modes':['polytrope_R'],'N_poly':60000,'M_poly':0.003,'R_poly':0.02,'n_poly':2.5}
Poly_M_dict  = {'modes':['polytrope_R'],'N_poly':20000,'M_poly':0.001,'R_poly':0.01,'n_poly':2.5}



#Relaxed Discs
disc_B_dict = {'modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.1,'Rout':3.0,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
disc_C_dict = {'modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.1,'Rout':3.0,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
disc_D_dict = {'name':'Disc_N1e5_R00120_M0005','modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.005,'M_star':1,'grad':1,'Teq':20,'Roll':True}
disc_E_dict = {'name':'Disc_N1e5_R00120_M0005_hot','modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.005,'M_star':1,'grad':1,'Teq':30,'T_ind':0.75,'Roll':True}
#Clump in disc Discs
disc_F_dict = {'name':'Disc_N3e5_R00120_M0009','modes':['gas_disc','star'],'N_gas_disc':300000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.009,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_G_dict = {'name':'Disc_N12e5_R00120_M0009','modes':['gas_disc','star'],'N_gas_disc':1200000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.009,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_H_dict = {'name':'Disc_N15e5_R00120_M0075','modes':['gas_disc','star'],'N_gas_disc':1500000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_I_dict = {'name':'Disc_N15e5_R00110_M0075','modes':['gas_disc','star'],'N_gas_disc':1500000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_J_dict = {'name':'Disc_N12e6_R00110_M0075','modes':['gas_disc','star'],'N_gas_disc':12000000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_K_dict = {'name':'Disc_N12e6_R00110_M0075','modes':['gas_disc','star'],'N_gas_disc':12000000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_dia,'Roll':True}
disc_L_dict = {'name':'Disc_N15e5_R00110_M0075_g75','modes':['gas_disc','star'],'N_gas_disc':1500000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_dia,'Roll':True}

#Giovanni discs
Giovanni_dict  = {'modes':['gas_disc','star'],'N_gas_disc':2000000,'Rin':0.01,'Rout':1,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
Giovanni_dictN1e6 = {'name':'Disc_N1e6_R00210','modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.02,'Rout':1,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
Giovanni_dictN1e6_00120 = {'name':'Disc_N1e6_R00120','modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.01,'Rout':2.0,'M_gas_disc':0.02,'M_star':1,'grad':1,'Teq':20,'Roll':True}
Giovanni_dictN1e5_00120 = {'name':'Disc_N1e5_R00120','modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.01,'Rout':2.0,'M_gas_disc':0.02,'M_star':1,'grad':1,'Teq':20,'Roll':True}



#Ring test
Ring_dict  = {'name':'Ring_N5e5_R05','modes':['gas_disc','star'],'N_gas_disc':500000,'Rin':0.495,'Rout':0.505,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}







#====+ Data Generation Functions +====#

def disc(Rin,Rout,M_disc,M_star,N,grad,Teq,Ptype,IDoffset,T_ind=0.5,gamma=c.gamma_mono,Roll=False):
    '''Generate disc of particles. Ptype: PartType0=gas,PartType2=dust
    Teq = equilibrium temp at 100 AU'''
    ID    = np.arange(N)+IDoffset
    q     = 2-grad


    if Ptype == 'PartType0':
        M_sph = M_disc/N
    elif Ptype == 'PartType2':
        M_sph = M_disc/N*Z_met

    Sig0  = q*M_disc/(2*np.pi) * (Rout**q - Rin**q)**(-1)

    
    #Generate Rs including mirroring
    R          = np.zeros(N)
    R[0:2]     = np.array([Rin,Rin])
    Theta      = np.zeros(N)
    rangle = np.random.rand(int(N/2))*np.pi
    Theta[::2] = rangle
    Theta[1::2] = rangle-np.pi
    
    for i in range(int(N/2-1)):
        dR = 2*M_sph / (2*np.pi*Sig0) * R[2*i]**(grad-1)
        R[2*i+2:2*i+4] = R[2*i:2*i+2]+dR

    if Roll == True:
        for i in range(len(R)):
            if R[i] < 1.2*Rin:
                R[i] = 2*Rin-R[i]
            elif R[i] > 0.8*Rout:
                R[i] = 2*Rout-R[i]
        
    X     = R*np.cos(Theta)
    Y     = R*np.sin(Theta)

    #Enclosed mass
    M_plus = np.arange(N)*M_sph
    
    #Velocities
    V_K   =  b.v_kepler((M_star+M_plus)*code_M,R*code_L)/code_L*code_time
    VX    = -V_K*np.sin(Theta)
    VY    =  V_K*np.cos(Theta)
    VZ    =  np.zeros(N)

    T     = Teq* (1/R)**T_ind
    Hs    = R/V_K * np.sqrt(c.kb*T/(c.mu*c.mp)) *code_time/code_L
    Z     = np.random.normal(0,Hs)
    
    print 'Coords',X,Y,Z
    plt.figure(0)
    plt.scatter(X,VY)

    POS   = np.stack((X,Y,Z)).T
    MASS  = np.ones(N)*M_sph
    U     = np.zeros(N)
    
    if Ptype == 'PartType0':
        #Gas pressure
        n       = 11/4
        eta     = n * (Hs/R)**2
        gas_sub = (1-eta)**0.5
        print 'gas sub', gas_sub
        V_K_gas = V_K * gas_sub
        VX, VY  = VX*gas_sub, VY*gas_sub
        U       = T*c.kb / (c.mu*c.mp*(gamma-1))/code_L**2 * code_time**2
        plt.figure(3)
        plt.hist(U*code_L**2/code_time**2,alpha=0.5)

    VEL   = np.stack((VX,VY,VZ)).T

    print 'POS', type(POS), np.shape(POS), POS

    return POS,VEL,ID,MASS,U




def star(M_star,IDoffset):
    POS  = np.array([[0.,0.,0.]])
    VEL  = np.array([[0.,0.,0.]])
    ID   = [IDoffset]
    MASS = [float(M_star)]
    U    = [0]
    return POS,VEL,ID,MASS,U




def polytrope(N,M,R=0,Tc=0,n=1.5,IDoffset=0):
    '''Solve Polytrope. Theta = polytropic temperature. Epsilon = scaled radius.
    n=1.5 -> gamma=5/3 adiabatic solution
    N = number of particles'''
    ID    = np.arange(N)+IDoffset
    M_sph = M/N
    MASS = np.ones(N)*M_sph
    gamma = (1+n)/n

    #Solve Polytrope
    if R==0:
        Rs, rhos, rho_c, polyK_cgs, polyK_code  = b.polytrope(M=M,Tc=Tc,n=n)
    elif Tc==0:
        Rs, rhos, rho_c, polyK_cgs, polyK_code  = b.polytrope(M=M,R=R,n=n)
    
    R_outs   = np.zeros(N)
    rho_outs = np.zeros(N)
    R_outs[0] = (3*M_sph / (4*np.pi*rho_c))**(1/3)
    rho_outs[0] = rho_c
    
    for i in range(N-1):
        rho_i  = np.interp(R_outs[i],Rs,rhos)
        dri            = M_sph/(4*np.pi*R_outs[i]**2*rho_i)
        if (np.isnan(dri) == True) or (np.isinf(dri) == True):
            dri = 0
            rho_i = rho_outs[i]

        R_outs[i+1]    = R_outs[i] + dri
        rho_outs[i+1]  = rho_i
    
    theta_outs = np.arccos(2*np.random.rand(N)-1)
    phi_outs   = np.random.rand(N)*2*np.pi

    plt.figure(1)
    plt.scatter(R_outs,rho_outs)
    plt.figure(2)
    plt.hist(theta_outs,bins=200)
    plt.show()
    
    #Convert to output values
    X = R_outs * np.cos(phi_outs) * np.sin(theta_outs)
    Y = R_outs * np.sin(phi_outs) * np.sin(theta_outs)
    Z = R_outs * np.cos(theta_outs)
    Vs = np.zeros(N)
    
    POS   = np.stack((X,Y,Z)).T
    VEL   = np.stack((Vs,Vs,Vs)).T
    U     = 1/(gamma-1)*polyK_code*rho_outs**(1/n)

    plt.figure(3)
    plt.hist(U*code_L**2/code_time**2,alpha=0.5)
    
    return POS,VEL,ID,MASS,U







#====+ Write Initial Condition Files +====#

def write_hdf5(init_dict):
    '''Write a hdf5 gadget initial condition file'''

    placeholder = 'placeholder'
    try:
        os.remove(placeholder)
    except OSError:
        pass
    
    init = h5py.File(placeholder)
    num_array = np.array([0,0,0,0,0,0])
    modes = init_dict['modes']

    
    #====+ Generate Input Data +====#
    #Gas=0,dust=2,sink=5. Write in order!
   
    #--- Gas Disc ---#
    if 'gas_disc' in modes:
        Ptype      = 'PartType0'
        Type       = init.create_group(Ptype)
        Rin        = init_dict['Rin']
        Rout       = init_dict['Rout']
        M_gas_disc = init_dict['M_gas_disc']
        M_star     = init_dict['M_star']
        N_gas_disc = init_dict['N_gas_disc']
        grad       = init_dict['grad']
        Teq        = init_dict['Teq']
        T_ind      = init_dict['T_ind']
        gamma      = init_dict['gamma']
        filename   = init_dict['name']
        try:
            Roll   = init_dict['Roll']
        except:
            Roll   = False

        POS,VEL,ID,MASS,U = disc(Rin=Rin,Rout=Rout,M_disc=M_gas_disc,
                                 M_star=M_star,N=N_gas_disc,grad=grad,gamma=gamma,
                                 Teq=Teq,T_ind=T_ind,Ptype=Ptype,IDoffset=np.sum(num_array))
        num_array[0] += N_gas_disc
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        Type.create_dataset('InternalEnergy',data=U)

    #--- Polytrope ---#
    if 'polytrope_Tc' in modes:
        Ptype = 'PartType0'
        Type = init.create_group(Ptype)
        N_poly  = init_dict['N_poly']
        M_poly  = init_dict['M_poly']
        Tc_poly = init_dict['Tc_poly']
        n_poly  = init_dict['n_poly']
        filename   = 'Poly_N'+str(N_poly)+'_M'+str(M_poly)+'_T'+str(Tc_poly)+'_n'+str(n_poly)

        POS,VEL,ID,MASS,U = polytrope(N=N_poly,M=M_poly,Tc=Tc_poly,n=n_poly,
                                      IDoffset=np.sum(num_array))
        num_array[0] += N_poly
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        Type.create_dataset('InternalEnergy',data=U)

    if 'polytrope_R' in modes:
        Ptype = 'PartType0'
        Type = init.create_group(Ptype)
        N_poly  = init_dict['N_poly']
        M_poly  = init_dict['M_poly']
        R_poly = init_dict['R_poly']
        n_poly  = init_dict['n_poly']
        filename   = 'Poly_N'+str(N_poly)+'_M'+str(M_poly)+'_R'+str(R_poly)+'_n'+str(n_poly)

        POS,VEL,ID,MASS,U = polytrope(N=N_poly,M=M_poly,R=R_poly,n=n_poly,
                                      IDoffset=np.sum(num_array))
        num_array[0] += N_poly
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        Type.create_dataset('InternalEnergy',data=U)
        
    #--- Dust Disc ---#
    if 'dust_disc' in modes:
        Ptype       = 'PartType2'
        Type        = init.create_group(Ptype)
        Rin         = init_dict['Rin']
        Rout        = init_dict['Rout']
        M_dust_disc = init_dict['M_dust_disc']
        M_star      = init_dict['M_star']
        N_dust_disc = init_dict['N_dust_disc']
        grad        = init_dict['grad']
        Teq         = init_dict['Teq']

        POS,VEL,ID,MASS,U = disc(Rin=Rin,Rout=Rout,M_disc=M_disc,
                                 M_star=M_star,N=num_dust,grad=1,
                                 Teq=Teq,Ptype=Ptype,IDoffset=np.sum(num_array))
        num_array[2] += num_dust
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)


    #--- Star ---#
    if 'star' in modes:
        Ptype  = 'PartType5'
        Type   = init.create_group(Ptype)
        M_star = init_dict['M_star']
        filename   = init_dict['name']
        print 'M_star', M_star
        
        POS,VEL,ID,MASS,U = star(M_star,IDoffset=np.sum(num_array))
        print 'MASS', MASS
        num_array[5] += 1
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        

    #====+ Build Header +====#
    header = init.create_group('Header')
    header.attrs.create('NumPart_ThisFile',num_array)
    header.attrs.create('NumPart_Total',num_array)
    header.attrs.create('NumPart_Total_HighWord',np.array([0,0,0,0,0,0]))
    header.attrs.create('MassTable',np.array([0,0,0,0,0,0]))
    header.attrs.create('Time',0.0)
    header.attrs.create('Redshift',0.0)
    header.attrs.create('Boxsize',1.0)
    header.attrs.create('NumFilesPerSnapshot',1)
    header.attrs.create('Omega0',0.0)
    header.attrs.create('OmegaLambda',0.0)
    header.attrs.create('HubbleParam',1.0)
    header.attrs.create('Flag_Sfr',1)
    header.attrs.create('Flag_Cooling',1)
    header.attrs.create('Flag_StellarAge',1)
    header.attrs.create('Flag_Metals',0)
    header.attrs.create('Flag_Feedback',1)
    header.attrs.create('Flag_DoublePrecision',0)

    #Finalise File Details
    for char in '.':
        filename = filename.replace(char,'')
    filename += '.hdf5'
    try:
        os.remove(filename)
    except OSError:
        pass
    os.rename(placeholder,filename)

    init.close()
    return

    

def graft(file1,file2,graft_file,offset_pos=np.array([0,0,0]),offset_vel=np.array([0,0,0]),offset_vK=False,
          COM_correction=False,rho_correction=False,one2one_dust_disc=False):
    '''Graft together two gadget snapshots'''
    
    #+==== Establish new hdf5 file ====+#
    try:
        os.remove(graft_dir+graft_file)
    except OSError:
        pass
    init = h5py.File(graft_dir+graft_file)

    #+==== Load old hdf5 files ====+#
    load1 = h5py.File(graft_dir+file1) 
    load2 = h5py.File(graft_dir+file2)
    keys1 = load1.keys()
    keys2 = load2.keys()
    
    header1 = load1[keys1[0]]
    header2 = load2[keys2[0]]
    attrs1 = header1.attrs.items()
    attrs2 = header2.attrs.items()
    num_part = np.array([0,0,0,0,0,0])
    
    #+==== Combine particle type data====+#
    #Check datasets for each particle type sequentially
    partkeys = ['PartType0', 'PartType1', 'PartType2', 'PartType3', 'PartType4', 'PartType5']
    datasets = ['Coordinates','Velocities','ParticleIDs','Masses','InternalEnergy','Density']
    print keys1

    #====== Correct for star sink drift ======#
    try:
        M_bodies = load1['PartType5']['Masses'][:]
        star_pos = load1['PartType5']['Coordinates'][np.argmax(M_bodies)]
        star_vel = load1['PartType5']['Velocities'][np.argmax(M_bodies)]
        print 'Pos Star', star_pos
    except:
        print 'No star in load1, cannot correct position'
        star_pos = np.array([0,0,0])
        star_vel = np.array([0,0,0])

    #==== Offset V kepler ====#
    if offset_vK == True:
        theta  = np.arctan2(offset_pos[1],offset_pos[0])
        M_star = load1['PartType5']['Masses'][:]
        RP = np.sqrt(offset_pos[0]**2+offset_pos[1]**2+offset_pos[2]**2)
        vK = b.v_kepler(M_star*code_M,RP*code_L)/code_L*code_time
        offset_vel = np.array([-1*np.sin(theta)*vK[0],np.cos(theta)*vK[0],0])
        print 'pos',offset_pos
        print 'vel',offset_vel
        
    ID_offset = 0
    for i in range(len(partkeys)):
        print partkeys[i]

        #Only use internal energy for PartType0 (gas)
        if i == 0:
            u_mark = 1
        else:
            u_mark = 0


        for j in range(len(datasets)-1 + u_mark):
            #Detect whether particle types exist
            try:
                data1 = load1[partkeys[i]][datasets[j]][:]
                if j == 0:
                    data1 -= star_pos
                    print 'COM data1 partkey'+str(i)+': ', np.mean(data1,axis=0)
                if j == 1:
                    data1 -= star_vel
            except:
                data1 = None
            try:
                data2 = load2[partkeys[i]][datasets[j]][:]
                #Apply position offsets
                if j ==0:
                    print 'COM gas 2: ', np.mean(data2,axis=0)
                    if COM_correction==True:
                        data2 = data2 - np.mean(data2,axis=0)
                    if rho_correction==True:
                        rho_data = load2[partkeys[0]][datasets[5]][:]
                        rho_sort  = np.argsort(rho_data)
                        corr = np.mean(data2[rho_sort[-10:],:],axis=0)
                        data2 = data2 - corr
                        print 'Position correction complete'
                    data2 += offset_pos
                    print 'New COM gas 2: ', np.mean(data2,axis=0)

                #Apply velocity offsets
                elif j ==1:
                    if COM_correction==True:
                        data2 = data2 - np.mean(data2,axis=0)
                    if rho_correction==True:
                        corr = np.mean(data2[rho_sort[-10:],:],axis=0)
                        data2 = data2 - corr
                        print 'Velocity correction complete'
                    data2 += offset_vel
            
            except:
                data2 = None

            print 'data1', np.shape(data1)
            print 'data2', np.shape(data2)

                
            #Combine data from both files. 
            if (data1==None) & (data2==None):
                num_part[i] = 0
                break
            else:
                if j == 0:
                    TypeG = init.create_group(partkeys[i])
                    
                if (data1!=None) & (data2!=None):
                    dataG = np.concatenate((data1,data2))

                elif data1 != None:
                    dataG = data1
                elif data2 != None:
                    dataG = data2

                #Manage and combine IDs
                if j == 2:
                    TypeG.create_dataset(datasets[j],data=np.arange(len(dataG))+ID_offset)
                    ID_offset += len(dataG)
                else:
                    TypeG.create_dataset(datasets[j],data=dataG)

                #Set particle numbers
                num_part[i] = len(dataG)

        print TypeG
    print 'num_part', num_part

    #+==== Build new header ====+#

    header = init.create_group('Header')
    header.attrs.create('NumPart_ThisFile',num_part)
    header.attrs.create('NumPart_Total',num_part)
    header.attrs.create('NumPart_Total_HighWord',np.array([0,0,0,0,0,0]))
    header.attrs.create('MassTable',np.array([0,0,0,0,0,0]))
    header.attrs.create('Time',0.0)
    header.attrs.create('Redshift',0.0)
    header.attrs.create('Boxsize',1.0)
    header.attrs.create('NumFilesPerSnapshot',1)
    header.attrs.create('Omega0',0.0)
    header.attrs.create('OmegaLambda',0.0)
    header.attrs.create('HubbleParam',1.0)
    header.attrs.create('Flag_Sfr',1)
    header.attrs.create('Flag_Cooling',1)
    header.attrs.create('Flag_StellarAge',1)
    header.attrs.create('Flag_Metals',0)
    header.attrs.create('Flag_Feedback',1)
    header.attrs.create('Flag_DoublePrecision',0)


    load1.close()
    load2.close()
    init.close()
    return





def V_settle(dust_r,dust_z,r_gas,M_sph,M_star,grain_a,grain_rho):
    '''Calculate analytic dust settling velocities'''
    rbins  = np.linspace(np.min(dust_r),np.max(dust_r),501)
    rbin_mids = (rbins[1:]+rbins[:-1])/2
    drbin = rbins[1]-rbins[0]
    N_bins = np.histogram(r_gas,rbins)[0]
    binned_Sig = M_sph*N_bins /  (2*np.pi*rbin_mids*drbin)

    #Calculate analytic quantities
    T      = b.T_profile(R=rbin_mids,T0=20,R0=1,power=-0.5)
    cs     = np.sqrt(c.kb*T/(c.mu*c.mp))/code_L*code_time
    Om_K   = b.v_kepler(M_star*code_M,rbin_mids*code_L)/code_L*code_time /rbin_mids
    rho_0  = binned_Sig*Om_K / (np.sqrt(2*np.pi)*cs)         
    a      = grain_a/code_L
    rho_a  = grain_rho/code_M*code_L**3
    t_stop = a*rho_a /rho_0/cs *np.sqrt(np.pi/8)

    #Allocate analytic settling velocities for each dust particle
    VZs = np.zeros(len(dust_r))
    for i in range(len(dust_r)):
        ind = np.argmin((dust_r[i]-rbin_mids)**2)
        t_stopi = t_stop[ind]
        Om_Ki   = Om_K[ind]
        csi     = cs[ind]
        VZs[i] = -Om_Ki**2*t_stopi*dust_z[i] * np.exp(dust_z[i]**2/2 * (Om_Ki/csi)**2)

      
    return VZs
    

def Z_settled(dust_r,dust_z,r_gas,Z_gas,gas_hs,M_sph,M_star,grain_a,grain_rho,alpha_AV):
    '''Calculate analytic dust settled scale'''
    rbins  = np.linspace(np.min(dust_r),np.max(dust_r),501)
    rbin_mids = (rbins[1:]+rbins[:-1])/2
    drbin = rbins[1]-rbins[0]
    N_bins = np.histogram(r_gas,rbins)[0]
    binned_Sig = M_sph*N_bins /  (2*np.pi*rbin_mids*drbin)
    binned_hs,std_hs = b.calc_binned_data(gas_hs,r_gas,rbins)
    #binned_Hs = b.calc_binned_data(Z_gas,r_gas,rbins,H68_mode=True)
    
    #Calculate analytic quantities
    T      = b.T_profile(R=rbin_mids,T0=20,R0=1,power=-0.5)
    cs     = np.sqrt(c.kb*T/(c.mu*c.mp))/code_L*code_time
    Om_K   = b.v_kepler(M_star*code_M,rbin_mids*code_L)/code_L*code_time /rbin_mids
    rho_0  = binned_Sig*Om_K / (np.sqrt(2*np.pi)*cs)         
    a      = grain_a/code_L
    rho_a  = grain_rho/code_M*code_L**3
    t_stop_a = rho_a /rho_0/cs *np.sqrt(np.pi/8)
    print 'lah!', np.shape(t_stop_a)
    print len(dust_r)
    
    
    #Allocate analytic settling velocities for each dust particle
    Zs = np.zeros(len(dust_r))
    for i in range(len(dust_r)):
        ind = np.argmin((dust_r[i]-rbin_mids)**2)
        t_stop_ai = t_stop_a[ind]
        Om_Ki   = Om_K[ind]
        #print t_stop_ai, Om_Ki, a[ind]
        supp_fac = np.sqrt(alpha_AV/t_stop_ai/Om_Ki/a[ind])
        supp_fac = np.clip(supp_fac,-1,1)
        Zs[i] = Z_gas[i] * supp_fac
    print np.shape(Zs)
    print Zs
    return Zs



def one2one_dust(file1,M_dust_frac,Rin=0.1,Rout=1,Zsupp=1,
                 static_dust=False,settled_dust=False,alpha_AV=1.,
                 recentre_gas=False,
                 TEST_vsettle=False,TEST_dust_ring=False,Nring=1e6,
                 grain_a=1.,grain_rho=3.,dust_readin=False,logamin=-4,logamax=1,
                 name='',N_dust_frac=1.0,
                 polydust=False,poly_R=3):
    '''Match dust particles to gas particles in a given region
    Zsupp = suppression factor
    TEST_vsettle - start dust particles with analytic v settle
    TEST_dust_ring - start particles in narrow dust ring
    N_dust_frac = fraction of gas particles that recieve dust particles
    '''
    

    
    #========= Build new file name ========#
    oldfile = file1
    file1 = file1[:-5]+'_MD'+str(M_dust_frac).replace('.','')+'.hdf5'
    file1 = file1[:-5]+'_Z'+str(Zsupp)+'.hdf5'
    if TEST_vsettle == True:
        print 
        file1 = file1[:-5]+'_a'+str(grain_a).replace('.','')+'.hdf5'
    if TEST_dust_ring == True:
        file1 = file1[:-5]+'_dustring.hdf5'
    if dust_readin == True:
        file1 = file1[:-5]+'_dustreadin.hdf5'
    if settled_dust == True:
        file1 = file1[:-5]+'_Zset.hdf5'
    if N_dust_frac != 1.0:
        file1 = file1[:-5]+'_df'+str(N_dust_frac).replace('.','')+'.hdf5'
    try:
        os.remove(graft_dir+file1)
    except OSError:
        pass
    copyfile(graft_dir+oldfile,graft_dir+file1)
    print 'Dust addition started for ' + str(file1)

    
    #==== Load hdf5 datasets ====#
    load1 = h5py.File(graft_dir+file1)
    try:
        pos_bodies  = load1['PartType5']['Coordinates'][:]
        vel_bodies  = load1['PartType5']['Velocities'][:]
        M_bodies    = load1['PartType5']['Masses'][:]
        num_bodies    = int(len(M_bodies))
        ID_bodies   = load1['PartType5']['ParticleIDs'][:]
        M_star      = M_bodies[np.argmax(M_bodies)]  #code_M
        pos_star    = pos_bodies[np.argmax(M_bodies)]
        vel_star    = vel_bodies[np.argmax(M_bodies)]
        pos_bodies  = pos_bodies - pos_star
        vel_bodies  = vel_bodies - vel_star
        print 'pos star',pos_star
        print 'M star', M_star
    except:
        pos_star = [0,0,0]
        vel_star = [0,0,0]
        
    #==== Load gas data ====#
    gas_pos  = load1['PartType0']['Coordinates'][:] - pos_star
    gas_vel  = load1['PartType0']['Velocities'][:]  - vel_star
    M_gas    = load1['PartType0']['Masses'][0]
    gas_IDs  = load1['PartType0']['ParticleIDs'][:]
    gas_Us   = load1['PartType0']['InternalEnergy'][:]
    #gas_rhos  = load1['PartType0']['Density'][:]
    num_gas  = len(gas_Us)
    print 'mean gas pos',np.mean(gas_pos,axis=0)

    if recentre_gas == True:
        gas_pos = gas_pos - np.mean(gas_pos)
        gas_vel = gas_vel - np.mean(gas_vel)

    #==== Calculate dust positions ====#
    r_gas     = np.sqrt(gas_pos[:,0]**2+gas_pos[:,1]**2)
    dust_pos  = gas_pos[(r_gas<Rout)&(r_gas>Rin),:]
    '''
    if polydust == True:        
        gas_rhos  = load1['PartType0']['Density'][:]
        rho_sort  = np.argsort(gas_rhos)
        poly_core = np.mean(gas_pos[rho_sort[-10:],:],axis=0)
        gas_rel_core = gas_pos-poly_core
        dust_pos  = gas_pos[(r_gas<Rout)&(r_gas>Rin)&(gas_rel_core>poly_R/AU_scale),:]
        Cdust_pos = gas_pos[gas_rel_core<poly_R/AU_scale,:]
    '''    
    if N_dust_frac != 1.0:
        N_dust = int(len(r_gas)*N_dust_frac)
        np.random.shuffle(dust_pos)
        dust_pos = dust_pos[:N_dust]
    dust_r    = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
    theta     = np.arctan2(dust_pos[:,1],dust_pos[:,0]) 

    print 'new mean gas pos',  np.mean(gas_pos[:,0]),np.mean(gas_pos[:,1]),np.mean(gas_pos[:,2])
    print 'new mean dust pos', np.mean(dust_pos[:,0]),np.mean(dust_pos[:,1]),np.mean(dust_pos[:,2])

    num_gas   = len(r_gas)
    num_dust  = len(dust_pos[:,0])
    M_dust    = M_dust_frac * M_gas / N_dust_frac

    mean_gas_Z = np.mean(gas_pos[:,2])
    dust_pos[:,2] = (dust_pos[:,2]-mean_gas_Z)/Zsupp + mean_gas_Z
    print 'Mean gas Z', np.mean(gas_pos[:,2])
    print 'Mean dust Z', np.mean(dust_pos[:,2])
    
    if dust_readin == True:
        print 'Setting dust radii!'
        dust_as = np.logspace(logamin,logamax,num_dust)
        np.random.shuffle(dust_as)
        
        if settled_dust == True:
            gas_hs = load1['PartType0']['SmoothingLength'][:]
            dust_pos[:,2] = Z_settled(dust_r,dust_pos[:,2],r_gas,gas_pos[:,2],
                                      gas_hs,M_gas,M_star,dust_as,grain_rho,alpha_AV)
    
    #==== Dust Ring Test ====#
    if TEST_dust_ring == True:
        sortgasZ = np.sort(np.sqrt(gas_pos[:,2]**2))
        GasH     = sortgasZ[int(0.68*len(sortgasZ))]
        Rmid     = (Rout+Rin)/2
        dR       = 0.01
        dust_r   = Rmid + (np.random.rand(Nring)-0.5)*dR
        theta    = 2*np.pi*np.linspace(0,1,Nring)
        dustZs   = GasH/Zsupp* (np.random.rand(Nring)-0.5)*2  
        dust_pos = np.array([dust_r*np.cos(theta),dust_r*np.sin(theta),dustZs]).T
        num_dust = int(Nring)
        print 'DUST_POS',dust_pos
        print 'Dust R', dust_r
        print 'mean dust pos',np.mean(dust_pos,axis=0)
        
    #==== Dust Velocities ====#
    VZ        = np.zeros(len(theta))
    if TEST_vsettle == True:
        VZ = V_settle(dust_r,dust_pos[:,2],r_gas,M_sph,M_star,grain_a)
    if static_dust == False:
        vK        = b.v_kepler(M_star*code_M,dust_r*code_L)/code_L*code_time
        dust_vel  = np.array([-1*np.sin(theta)*vK,np.cos(theta)*vK,VZ]).T
    else:
        dust_vel = np.zeros((num_gas,3))
        print 'pos', np.shape(dust_pos)
    print 'vel', np.shape(dust_vel)
    
    
    
    
    #==== Reconstruct hdf5 files ===#
    load1.__delitem__('PartType0')
    new_gas_IDs = np.arange(len(gas_IDs))
    Type0 = load1.create_group('PartType0')
    Type0.create_dataset('Coordinates',data=gas_pos)
    Type0.create_dataset('Velocities',data=gas_vel)
    Type0.create_dataset('ParticleIDs',data=new_gas_IDs)
    Type0.create_dataset('Masses',data=np.ones(num_gas)*M_gas)
    Type0.create_dataset('InternalEnergy',data=gas_Us)
    '''
    try:
        Type0.create_dataset('Density',data=gas_rhos)
    except:
        print 'No density info'
    '''
    try:
        load1.__delitem__('PartType2')
    except:
        pass
    new_dust_IDs = np.arange(num_dust)+num_gas
    Type2 = load1.create_group('PartType2')
    Type2.create_dataset('Coordinates',data=dust_pos)
    Type2.create_dataset('Velocities',data=dust_vel)
    Type2.create_dataset('ParticleIDs',data=new_dust_IDs)
    Type2.create_dataset('Masses',data=np.ones(num_dust)*M_dust)
    if dust_readin == True:
        print 'reading in dust!'
        Type2.create_dataset('DustRadius',data=dust_as)
    try:
        load1.__delitem__('PartType5')
        new_body_IDs = np.arange(num_bodies)+num_dust+num_gas
        Type5 = load1.create_group('PartType5')
        Type5.create_dataset('Coordinates',data=pos_bodies)
        Type5.create_dataset('Velocities',data=vel_bodies)
        Type5.create_dataset('ParticleIDs',data=new_body_IDs)
        Type5.create_dataset('Masses',data=M_bodies)
    except:
        num_bodies = 0
        pass
    
    

    #Update Header information
    num_part = np.array([num_gas,0,num_dust,0,0,num_bodies])
    print 'New Num Particles: ', num_part
    header = load1[load1.keys()[0]]
    header.attrs.modify('NumPart_ThisFile',num_part)
    header.attrs.modify('NumPart_Total',num_part)
    load1.close()

    print 'Dust addition complete for ', oldfile
    print 'Total gas mass = ', num_part[0]*M_gas
    print 'Total dust mass = ', num_part[2]*M_dust
    print 'New filename:', file1
    print '\n'
    return







def insert_sink_planet(file1,MP,X=0,Y=0,Z=0,rhomax=False):
    '''Insert a planet. MP in MJ'''
    newname = file1[:-5]+'_MP'+str(MP).replace('.0','').replace('.','')+'.hdf5'
    MP = MP/code_M*c.MJ
    try:
        os.remove(graft_dir+newname)
    except OSError:
        pass
    copyfile(graft_dir+file1,graft_dir+newname)
    
    #==== Load hdf5 data ====#
    load1 = h5py.File(graft_dir+newname)
    pos_bodies  = load1['PartType5']['Coordinates'][:]
    vel_bodies  = load1['PartType5']['Velocities'][:]
    M_bodies    = load1['PartType5']['Masses'][:]
    ID_bodies   = load1['PartType5']['ParticleIDs'][:]
    M_star      = M_bodies[np.argmax(M_bodies)]  #code_M
    pos_star    = pos_bodies[np.argmax(M_bodies)]
    vel_star    = vel_bodies[np.argmax(M_bodies)]
    pos_bodies  = pos_bodies - pos_star
    vel_bodies  = vel_bodies - vel_star
    print 'star', M_star
    print pos_star

    #==== Load gas and dust ====#
    gas_pos  = load1['PartType0']['Coordinates'][:] - pos_star
    gas_vel  = load1['PartType0']['Velocities'][:]  - vel_star
    M_gas    = load1['PartType0']['Masses'][0]
    gas_IDs  = load1['PartType0']['ParticleIDs'][:]
    gas_Us   = load1['PartType0']['InternalEnergy'][:]
    num_gas  = len(gas_Us)
    gas_r    = np.sqrt(gas_pos[:,0]**2+gas_pos[:,1]**2)
    
    try:
        dust_pos  = load1['PartType2']['Coordinates'][:] - pos_star
        dust_vel  = load1['PartType2']['Velocities'][:]  - vel_star
        M_dust    = load1['PartType2']['Masses'][0]
        dust_IDs  = load1['PartType2']['ParticleIDs'][:]
        num_dust  = len(dust_IDs)
    except:
        num_dust = 0
    
    
    #==== Calculate Planet data ====# 
    pos_P     = np.array([X,Y,Z])
    print 'Pos P', pos_P*AU_scale, 'AU'
    if rhomax == True:
        gas_rhos  = load1['PartType0']['Density'][:]
        rho_sort  = np.argsort(gas_rhos)
        pos_P = np.mean(gas_pos[rho_sort[-10:],:],axis=0)
    rP        = np.sqrt(pos_P[0]**2+pos_P[1]**2)
    theta     = np.arctan2(pos_P[1],pos_P[0])   


    Mgas_interior = M_gas*len(gas_r[gas_r<rP])
    vK        = b.v_kepler((M_star+Mgas_interior)*code_M,rP*code_L)/code_L*code_time
    vel_P     = np.array([-1*np.sin(theta)*vK,np.cos(theta)*vK,0])

    body_pos = np.vstack((pos_bodies,pos_P))
    body_vel = np.vstack((vel_bodies,vel_P))
    body_IDs = np.hstack((ID_bodies,ID_bodies[-1]+1))
    print M_bodies
    print MP
    body_Ms  = np.hstack((M_bodies,MP))

    #Recreate hdf5 datasets
    load1.__delitem__('PartType0')
    Type0 = load1.create_group('PartType0')
    Type0.create_dataset('Coordinates',data=gas_pos)
    Type0.create_dataset('Velocities',data=gas_vel)
    Type0.create_dataset('ParticleIDs',data=gas_IDs)
    Type0.create_dataset('Masses',data=M_gas*np.ones(num_gas))
    Type0.create_dataset('InternalEnergy',data=gas_Us)

    try:
        load1.__delitem__('PartType2')
        Type2 = load1.create_group('PartType2')
        Type2.create_dataset('Coordinates',data=dust_pos)
        Type2.create_dataset('Velocities',data=dust_vel)
        Type2.create_dataset('ParticleIDs',data=dust_IDs)
        Type2.create_dataset('Masses',data=M_dust*np.ones(num_dust))
    except:
        pass
    
    load1.__delitem__('PartType5')
    Type5 = load1.create_group('PartType5')
    Type5.create_dataset('Coordinates',data=body_pos)
    Type5.create_dataset('Velocities',data=body_vel)
    Type5.create_dataset('ParticleIDs',data=body_IDs)
    Type5.create_dataset('Masses',data=body_Ms)


    #Update Header information
    header = load1[load1.keys()[0]]
    num_part = header.attrs.__getitem__('NumPart_ThisFile')
    num_part[5] += 1
    print 'New Num part', num_part
    header.attrs.modify('NumPart_ThisFile',num_part)
    header.attrs.modify('NumPart_Total',num_part)
    load1.close()
    return



def Gdust_removal(Gfile):
    '''Load Sergei initial snap. Remove dust and planet. make hdf5 for dust addition!
    Place 'snapshot_000' in graft_dir/fname/..'''
    fname = 'graft_dir/'+Gfile.rstrip('/')+'.hdf5'
    
    load_dict = Q.load_Gsnap('graft_dir/',Gfile,'000','gas')

    #==== Load sph data ====#
    headertime  = load_dict['headertime']
    N_sph       = load_dict['N_sph']       
    M_sph       = load_dict['M_sph']      
    M_star      = load_dict['M_star']
    N_planets   = load_dict['N_planets']
    M_planets   = load_dict['M_planets']   
    pos_planets = load_dict['pos_planets']
    vel_planets = load_dict['vel_planets']
    sph_pos     = load_dict['sph_pos']
    sph_vel     = load_dict['sph_vel']
    sph_U       = load_dict['sph_A']
    num_gas = len(sph_U)
    
    print M_star
    print pos_planets
    print np.mean(sph_pos,axis=0)
    print sph_pos
    
    
    #==== Build hdf5 file ====#
    try:
        os.remove(fname)
    except OSError:
        pass
    init = h5py.File(fname)
    num_array =np.array([num_gas,0,0,0,0,1])
    
    #====+ Build Header +====#
    header = init.create_group('Header')
    header.attrs.create('NumPart_ThisFile',num_array)
    header.attrs.create('NumPart_Total',num_array)
    header.attrs.create('NumPart_Total_HighWord',np.array([0,0,0,0,0,0]))
    header.attrs.create('MassTable',np.array([0,0,0,0,0,0]))
    header.attrs.create('Time',0.0)
    header.attrs.create('Redshift',0.0)
    header.attrs.create('Boxsize',1.0)
    header.attrs.create('NumFilesPerSnapshot',1)
    header.attrs.create('Omega0',0.0)
    header.attrs.create('OmegaLambda',0.0)
    header.attrs.create('HubbleParam',1.0)
    header.attrs.create('Flag_Sfr',1)
    header.attrs.create('Flag_Cooling',1)
    header.attrs.create('Flag_StellarAge',1)
    header.attrs.create('Flag_Metals',0)
    header.attrs.create('Flag_Feedback',1)
    header.attrs.create('Flag_DoublePrecision',0)

    '''
    plt.figure(4)
    plt.scatter(sph_pos[:,0],sph_pos[:,1])
    plt.show()
    '''
    
    Type0 = init.create_group('PartType0')
    Type0.create_dataset('Coordinates',data=sph_pos)
    Type0.create_dataset('Velocities',data=sph_vel)
    Type0.create_dataset('ParticleIDs',data=np.arange(num_gas))
    Type0.create_dataset('Masses',data=np.ones(num_gas)*M_sph)
    Type0.create_dataset('InternalEnergy',data=sph_U)
    
    Type5 = init.create_group('PartType5')
    Type5.create_dataset('Coordinates',data=np.array([0.,0.,0.],dtype=float))
    Type5.create_dataset('Velocities',data=np.array([0.,0.,0.],dtype=float))
    Type5.create_dataset('ParticleIDs',data=[num_gas])
    Type5.create_dataset('Masses',data=[M_star])

    init.close()

    
    return


    


    
    
def load_test(path,snap): 
    load_test = h5py.File(path+snap,'r')
    print '\n'
    print("Keys: %s" % load_test.keys())
    keys = load_test.keys()

    header = load_test[keys[0]]
    #print header.attrs.items()
    attrs = header.attrs.items()
    for i in range(len(attrs)):
        print attrs[i]

    for i in range(len(keys)-1):
        print '\n', keys[i+1]
        
        typei = load_test[keys[i+1]]
        print typei.attrs.items()
        print 'pos',  typei['Coordinates']
        print 'mean pos', np.mean(typei['Coordinates'],axis=0)
        print 'pos',  typei['Coordinates'][:]
        print 'vel',  typei['Velocities'][:]
        #print 'v_K',  np.sqrt(typei['Velocities'][:100,0]**2 + typei['Velocities'][:100,1]**2) 
        print 'ids',  typei['ParticleIDs'][:]
        print 'mass', typei['Masses'][:]
        if keys[i+1] == 'PartType0':
            print 'u',    typei['InternalEnergy'][:]

        try:
            typei['DustRadius'][:]
            print 'dust a', typei['DustRadius'][:]
        except:
            pass
    return




if __name__ == "__main__":

    #==== Tests on Dust Accretion project ====#
    #Gdust_removal('O_M2_beta10_a1_n2e6/')
    #one2one_dust('O_M2_beta10_a1_n2e6.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    #load_test('graft_dir/','O_M2_beta10_a1_n2e6_dust.hdf5')
    #Gdust_removal('O_M2_beta01_n1e6/')
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=1)
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=0.1)
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=0.01)
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=0.001)
    #load_test('graft_dir/','O_M2_beta01_n1e6_a001.hdf5')
    
    #Use Gio Disc for non-self gravity dominated discs
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.1)
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.01)
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.001)
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.0001)
    #load_test('graft_dir/','Gio_N1e6_213_a001.hdf5')

    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9)
    #load_test('graft_dir/','Gio_N1e6_213_dust.hdf5')

    #one2one_dust('Disc_N1e6_R0130_M002_beta01_S301.hdf5',M_dust_frac=1e-8,Rin=0.49,Rout=0.51,Zsupp=10,TEST_dust_ring=True)
    #load_test('graft_dir/','Disc_N1e6_R0130_M002_beta01_S301_dustring.hdf5')


    
    #==== Giovanni Project ===#
    #write_hdf5(init_dict=Giovanni_dict)
    #write_hdf5(init_dict=Giovanni_dictN1e6)
    #insert_sink_planet('Gio_N1e6_213.hdf5',0.1,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',0.5,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',1.0,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',2.0,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',4.0,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',8.0,0.5,0,Z=0)
    #one2one_dust('Gio_N1e6_213.hdf5',M_dust_frac=0.001,Rin=0.2,Rout=1,Zsupp=1)
    #one2one_dust('Gio_N1e6_aav1_S301.hdf5',M_dust_frac=0.001,Rin=0.2,Rout=1,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_S301.hdf5',M_dust_frac=0.001,Rin=0.2,Rout=1,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav1_S301.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=1,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_S301.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=1,Zsupp=10)
    #write_hdf5(init_dict=Giovanni_dictN1e6_00120)
    #write_hdf5(init_dict=Giovanni_dictN1e5_00120)
    #one2one_dust('Disc_N1e5_R00120.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,Zsupp=10,dust_readin=True)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,Zsupp=10)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD001_Z10.hdf5',0.3,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD001_Z10.hdf5',1.0,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD001_Z10.hdf5',3.0,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD01_Z10.hdf5',0.3,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD01_Z10.hdf5',1.0,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD01_Z10.hdf5',3.0,0.6,0,Z=0)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #load_test('graft_dir/','Gio_N1e6_aav01_R00120_S040_MD001_dustreadin_Zset.hdf5')
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #load_test('graft_dir/','Gio_N1e6_aav01_R00120_S040_MD01_dustreadin_Zset.hdf5')
    #load_test('/rfs/TAG/rjh73/Gio_disc/','Gio_N1e6_aav01_R00120_MD001_dustreadin_Zset/snapshot_040.hdf5')
    #load_test('graft_dir/','Disc_N1e5_R00120_dust_Z10_dustreadin.hdf5')
    #load_test('/rfs/TAG/rjh73/Gio_disc/','Gio_N1e5_aav01_dustin/snapshot_006.hdf5')



    
    #Dust Work
    #one2one_dust('Disc_N100000_M001.hdf5',M_ddisc=0.001)
    #one2one_dust('Disc_N100000_M001_relaxed.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    #one2one_dust('Disc_N100000_M001_relaxed_sink.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    #one2one_dust('Disc_N100000_Poly.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    


    
    #===== Clement disruption runs ====#
    #write_hdf5(init_dict=Poly_D_dict)
    #write_hdf5(init_dict=Poly_E_dict)
    #write_hdf5(init_dict=Poly_F_dict)
    #write_hdf5(init_dict=Poly_G_dict)


    #one2one_dust('Poly_N1e5_M5_R5_n15_S008.hdf5',M_dust_frac=1e-4,Rin=0.0,Rout=2.0,static_dust=True,recentre_gas=True)
    #load_test('graft_dir/','Poly_N1e5_M5_R5_n15_S008_dust.hdf5')
    #one2one_dust('Poly_N1e5_M5_R5_n15_S21.hdf5',M_dust_frac=1e-4,Rin=0.0,Rout=2.0,static_dust=True,recentre_gas=True)
    #one2one_dust('Poly_N1e6_M5_R5_n15_S089.hdf5',M_dust_frac=1e-4,Rin=0.0,Rout=2.0,static_dust=True,recentre_gas=True)
    #write_hdf5(init_dict=Star_A_dict)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dust.hdf5','Poly_N1e6_M5_dust_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dust.hdf5','Poly_N1e6_M5_dust_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)

    #Dust Shell
    #one2one_dust('Poly_N1e6_M5_R5_n15_S089.hdf5',M_dust_frac=1e-2,Rin=0.03,Rout=0.05,static_dust=True,recentre_gas=True,name='shell35')
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #N1e5 Runs
    #one2one_dust('Poly_N1e5_M5_R5_n15_S100.hdf5',M_dust_frac=1e-2,Rin=0.03,Rout=0.05,static_dust=True,recentre_gas=True,name='shell35')
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R5_n15_S100_dustshell35.hdf5','Poly_N1e5_M5_dustshell35_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R5_n15_S100_dustshell35.hdf5','Poly_N1e5_M5_dustshell35_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #one2one_dust('Poly_N100000_M0005_R001_n15.hdf5',M_dust_frac=1e-2,Rin=0.006,Rout=0.01,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N100000_M0005_R003_n15.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R001_n15_dustshell.hdf5','Poly_N1e5_M5_R001_dustshell_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #one2one_dust('Poly_N1e5_M5_R3_evap_rho1e12_T30_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N1e5_M5_R3_evap_rho1e12_T50_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N1e5_M5_R3_evap_rho2e12_T30_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N1e5_M5_R3_evap_rho2e12_T50_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho1e12_T30_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho1e12_T30_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho1e12_T50_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho1e12_T50_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho2e12_T30_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho2e12_T30_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho2e12_T50_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho2e12_T50_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #one2one_dust('Poly_N1e5_M5_R3_S050.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_S050_dustshell.hdf5','Poly_N1e5_M5_R3_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #write_hdf5(init_dict=disc_D_dict)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)


    #Why is shell off centre?
    #one2one_dust('Poly_N100000_M0005_R003_n15.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_MD001.hdf5','Poly_N1e5_M5_R003_MD001_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=disc_E_dict)
    #graft('Disc_N1e5_R00120_M0005.hdf5','Poly_N100000_M0005_R003_n15_MD001.hdf5','Polydisc_N1e5_M5_R3_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N1e5_M0005_R01200_aav1_beta1_T30_Ti34_S060.hdf5','Poly_N100000_M0005_R003_n15_MD001.hdf5','Polydisc_N1e5_M5_R3_dust_r60_b1_T30_Ti34.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)




    #==== Clump in disc runs ====#
    #write_hdf5(init_dict=disc_F_dict)
    #write_hdf5(init_dict=Poly_H_dict)
    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD001.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_a1_r60_b5.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #write_hdf5(init_dict=disc_G_dict)
    #write_hdf5(init_dict=Poly_I_dict)

    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=100)
    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-1,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD001_Z100.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_r60_b5_MD001_Z100.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD001_Z10.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_r60_b5_MD001_Z10.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD01_Z10.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_r60_b5_MD01_Z10.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #Higher res
    #one2one_dust('Disc_N12e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #graft('Disc_N12e5_R00120_M0009_b5_S075_MD001_Z10.hdf5','Poly_N400000_M0003_R003_n15_S075.hdf5','Polydisc_N16e5_M3_R3_r60_b5_MD001_Z10.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #Higher mass disc
    #write_hdf5(init_dict=disc_H_dict)
    #write_hdf5(init_dict=Poly_F_dict)
    #one2one_dust('Disc_N15e5_R00120_M0075_b5_S080.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=100)
    #one2one_dust('Disc_N15e5_R00120_M0075_b5_S080.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #one2one_dust('Disc_N15e5_R00120_M0075_b5_S080.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=1)
    #graft('Disc_N15e5_R00120_M0075_b5_S080_MD001_Z100.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r100_b5_MD001_Z100.hdf5',offset_pos=np.array([1.0,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00120_M0075_b5_S080_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10.hdf5',offset_pos=np.array([1.0,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00120_M0075_b5_S080_MD001_Z1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r100_b5_MD001_Z1.hdf5',offset_pos=np.array([1.0,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=disc_I_dict)
    

    #Polytrope disruption tests
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #one2one_dust('Disc_N15e5_R00110_M0075_b5_S060.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10)
    #graft('Disc_N15e5_R00110_M0075_b5_S060_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)


    #CDS3 runs
    #write_hdf5(init_dict=Poly_J_dict)
    #write_hdf5(init_dict=disc_J_dict)
    #one2one_dust('Disc_N12e6_R00110_M0075_S038.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10)
    #graft('Disc_N12e6_R00110_M0075_S038_MD001_Z10.hdf5','Poly_N800000_M0005_R003_n25_S129.hdf5','Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N12e6_R00110_M0075_S038.hdf5','Poly_N800000_M0005_R003_n25_S129.hdf5','Polydisc_N13e6_M5_R3_r50_b5.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=Poly_K_dict)
    #one2one_dust('Disc_N15e5_R00110_M0075_S070.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10)
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=disc_L_dict)

    #one2one_dust('Disc_N15e5_R00110_M0075_S070.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10,N_dust_frac=0.1)
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10_df01.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df01.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=Poly_L_dict)
    #write_hdf5(init_dict=Poly_M_dict)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r75_b5_g75.hdf5',offset_pos=np.array([0.75,0,0]),offset_vK=True,COM_correction=True)

    #one2one_dust('Polydisc_N16e5_M5_R3_r50_b5_g75_S032.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M5_R3_r75_b5_g75_S032.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10,N_dust_frac=0.1)
    #insert_sink_planet('Polydisc_N16e5_M5_R3_r50_b5_g75_S032_MD001_Z10_df01.hdf5',3e-6,rhomax=True)
    #insert_sink_planet('Polydisc_N16e5_M5_R3_r75_b5_g75_S032_MD001_Z10_df01.hdf5',3e-6,rhomax=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Polydisc_N16e5_M1_R1_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Polydisc_N16e5_M1_R1_r75_b5_g75.hdf5',offset_pos=np.array([0.75,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Polydisc_N16e5_M3_R2_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Polydisc_N16e5_M3_R2_r75_b5_g75.hdf5',offset_pos=np.array([0.75,0,0]),offset_vK=True,COM_correction=True)

    
    #one2one_dust('Polydisc_N16e5_M5_R3_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M5_R3_r75_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M3_R2_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M3_R2_r75_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M1_R1_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M1_R1_r75_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)

    #insert_sink_planet('Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.75,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M3_R2_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M3_R2_r75_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.75,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M1_R1_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M1_R1_r75_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.75,Y=0,Z=0)

    #Z height fix?
    graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd_N16e5_M5_R3_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    one2one_dust('Pd_N16e5_M5_R3_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    insert_sink_planet('Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)

    graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Pd_N16e5_M3_R2_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    one2one_dust('Pd_N16e5_M3_R2_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    insert_sink_planet('Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)

    graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Pd_N16e5_M1_R1_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    one2one_dust('Pd_N16e5_M1_R1_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    insert_sink_planet('Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    
    #Grafting Runs
    #graft('Disc_N100000_M001_relaxed.hdf5','Poly_N10000_M0001.hdf5', 'Disc_N100000_Poly.hdf5',offset_pos=offset_pos,offset_vK=True,COM_correction=True)

    #insert_sink_planet('Disc_N100000_M001_relaxed.hdf5',MP=0.001,X=1.2,Y=0)

    
    #load_test('graft_dir/','Disc_N100000_M001_dust.hdf5')
    #load_test('','Disc_N2000000_M001.hdf5')
    #load_test('graft_dir/','Disc_N100000_M001_relaxed_dust.hdf5')
    #load_test('graft_dir/','Disc_N100000_Poly_dust.hdf5')
    #load_test('graft_dir/','Disc_N100000_Poly.hdf5')
    #load_test('graft_dir/','Disc_N100000_M001_relaxed_sink_dust.hdf5')
    #load_test('graft_dir/','O_M2_beta10_a1_n2e6.hdf5')
    #load_test('graft_dir/','O_M2_beta10_a1_n2e6_dust.hdf5')
    #load_test('/rfs/TAG/rjh73/','O_M2_beta10_a1_n2e6_dust_INIT.hdf5')
    #load_test('/rfs/TAG/rjh73/O_M2_beta10_a1_n2e6_dust/','snapshot_000.hdf5')
    plt.show()
