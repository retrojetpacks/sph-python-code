#!/usr/bin/env python

'''Basic functions for astrophysics'''
from __future__ import division
import numpy as np
import astrophysical_constants_cgs as c
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import norm
from scipy import spatial
import glob
from pygadgetreader import *


#==== Code units ====#
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
code_G     = c.G /code_L**3*code_M*code_time**2
AU_scale   = 100


#==== Font styles, sizes and weights ====#
def set_rcparams(fsize=12):
    font = {'weight' : 'normal',
            'size'   : fsize,
            'serif' : 'Minion Pro'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    plt.rc('lines',linewidth = 2)
    
    return
set_rcparams()

#==== File directory paths ====#
savedir = '/rfs/TAG/rjh73/save_data/'
run_cols = ['#2EC4C6','#CD6622','#440044','#FFC82E','#FF1493','#6a5acd']
linestyles = [':','-.','--','-',':','--']
linewidths = [1,1,1,1,2,2]



def calc_binned_data(data,bin_dim_coord,bins,H68_mode=False):
    '''Bin data according to some coordinate. 
    Calculate the mean and std of a different stat based on those binnings'''
    
    #Find bin number index for each value
    bin_inds = np.digitize(bin_dim_coord,bins)-1
    
    #Calculate 68% value for each bin
    if H68_mode == True:
        H68s = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            H_sort  = np.sort(abs(data[bin_inds==i]))
            try:
                H68s[i] = H_sort[int(0.68*len(H_sort))]
            except:
                print 'Bin empty'
                H68s[i] = 0
        return H68s

    #Calculate means and stdevs for each bin
    else:
        means = np.zeros(len(bins)-1)
        stds  = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            means[i],stds[i] = norm.fit(data[bin_inds==i])
        return means,stds



def star_planet_angle(pos):
    '''Calculate the angle of an object at pos wrt the star, star must be at (0,0). Returns in degrees'''
    try:
        theta = np.arctan2(pos[:,1],pos[:,0])
    except:
        theta = np.arctan2(pos[1],pos[0])
    return theta*180/np.pi



def v_r_azi(pos,vel):
    '''relative to 0,0,0,0,0,0'''
    vel2 = vel[:,0]**2+vel[:,1]**2
    v_dot_r   = np.zeros(len(pos))
    v_cross_r = np.zeros(len(pos))
    for i in range(len(pos)):
        v_dot_r[i]   = np.dot(pos[i,:],vel[i,:])
        v_c = np.cross(pos[i,:],vel[i,:])
        v_cross_r[i] = np.sqrt(np.sum(v_c**2))
    
    r = np.sqrt(pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)
 
    v_azi = v_cross_r/r
    v_r = v_dot_r /r
    
    return v_r, v_azi




def hill_radius(R,Mp,M_star):
    '''Hill radius: R,Mp,M_star'''
    RH = R * (Mp / (3*M_star))**(1/3)
    return RH
    
def v_kepler(M0,R):
    '''Keplerian velocity: M0,R'''
    v_kep = np.sqrt(M0*c.G/R)
    return v_kep
    

def disk_profile(R,M_disk,R_out,dH_dR):
    '''Calculate a simple disk profile'''
    Sigma_0 = M_disk / (2*np.pi*R_out**2)
    
    '''
    Sigma = Sigma_0 * R_out / R
    H = R * dH_dR
    rho_gas = Sigma/H
    T = 10*np.sqrt(100*c.AU) * R**(-0.5)
    '''
    T = 20 * np.sqrt(100*c.AU/R)
    H = np.sqrt(c.kb*T/(c.mu*c.mp)) * R / v_kepler(c.Msol,R)
    #Sigma = Sigma_0 * R_out / R
    Sigma = 10 * (1 + 1/R)
    rho_gas = Sigma/H
    
    return Sigma, H, rho_gas, T

def T_profile(R,T0=20,R0=1,power=-0.5):
    T = T0 * (R/R0)**power
    return T
    
def mean_free_path(rho_gas):
    mfp = 40 / (10**10*rho_gas)
    return mfp  
          
          
def reynolds(s,v,mfp,cs):
    Re = s*v/mfp/cs
    return Re
    
def thermal_vel(T0):
    vth0 = np.sqrt(8*c.kb*T0/(np.pi*c.mu*c.mp))   
    return vth0
    
def sound_speed(T0):
    cs = np.sqrt(c.gamma_dia*c.kb*T0/c.mp/c.mu)
    return cs
    


    
#Migration Regimes
def type_one(R,Mp,M_star,Sigma,H,f_acc):
    '''Migrate inside disk. RH<H  -  (Bate 2003)'''
    om_kep = np.sqrt(c.G*M_star/R**3)
    v_r = -f_acc * Mp / M_star**2 * Sigma / H**2 * R**5 * om_kep
    return v_r
    
    
def dimensionalise_mass(M_star,mu_c,R0,H):
    '''For a given dimensionless mass and radius, outputs a dimensional mass. See Lambrechts & Johansen 2012'''
    om_K = v_kepler(M_star,R0)/R0
    M_c = om_K**2 * H**3 * mu_c / c.G
    return M_c 


#Polytropes
def polytrope(M,R=0,Tc=0,n=1.5):
    '''Return polytrope. Numerically calculated so indeterminant number of output
    M, R, Tc, n. PolyK returned in cgs'''
    if R==0:
        mode = 'Tc'
    elif Tc==0:
        mode = 'R'
    
    gamma = (1+n)/n
    M     = M*code_M #g
    R     = R*code_L #cm
    
    #Boundary Conditions at epsilon = 0
    theta    = 1
    dth_dep  = 0
    epsilon  = 0.00001
    thetas   = [theta]

    dth_deps = [dth_dep]
    epsilons = [epsilon]
    dep      = 0.001
    
    #Solve Lane-Emden equation numerically
    while theta > 0:
        dth_dep = dth_dep - (2/epsilon*dth_dep + theta**n)*dep
        theta   = theta   + dth_dep*dep
        thetas.append(theta)
        dth_deps.append(dth_dep)
        epsilons.append(epsilon)
        epsilon = epsilon + dep

    print 'Epsilon -1', epsilons[-1]
    
    #Undo parameterisation
    if mode == 'R':
        alpha = R/epsilons[-1]
    elif mode == 'Tc':        
        alpha = -1*M * c.mu*c.mp*c.G/(c.kb*Tc *(n+1)) /epsilons[-1]**2 /dth_deps[-1]
        R = alpha*epsilons[-1]

    rho_c = -1*M / (4*np.pi*alpha**3) / (epsilons[-1]**2 * dth_deps[-1])
    #polyK_cgs = 4*np.pi*c.G*alpha**2 / (n+1) * rho_c**((n-1)/n)
    temp      = c.G * (4*np.pi)**(1/n)/(n+1) * epsilons[-1]**((n-3)/n) * (-1*epsilons[-1]**2*dth_deps[-1])**((1-n)/n)
    

    polyK_cgs = R**((3-n)/n)* M**((n-1)/n) *temp
    print 'PolyK: ', '{0:.10f}'.format(polyK_cgs), ' [cgs]'
    
    P_c       = polyK_cgs * rho_c**((n+1)/n)
    T = c.mp*c.mu*P_c /c.kb / rho_c
    
    print 'R, M, rho_c, T', R,M, rho_c, T
    
    thetas    = np.asarray(thetas)
    epsilons  = np.asarray(epsilons)
    

    #Convert to code units. PolyK returned in cgs
    Rs    = alpha * epsilons  /code_L
    rhos  = rho_c * thetas**n /code_M*code_L**3
    rho_c = rho_c /code_M*code_L**3
    #polyK_code = polyK_cgs *code_M**(-1*n) *code_L**(n/(2*n+3)) * code_time**(-0.5)
    polyK_code = polyK_cgs *code_M**(1/n) *code_L**(-2-3/n) * code_time**(2)

    #plt.figure(0)
    #plt.scatter(epsilons,dth_deps)
    #plt.figure(1)
    #plt.scatter(Rs,rhos,color='g')
    
    return Rs, rhos, rho_c, polyK_cgs, polyK_code





def Gadget_kernel(r_hs,hs):
    '''Evaluate Gadget kernel Wk'''
    r_hsA = r_hs[r_hs<0.5]
    r_hsB = r_hs[r_hs>0.5]
    
    Wks = np.zeros(np.shape(r_hs))
    Wks[r_hs<0.5] = 1 - 6*r_hsA**2 + 6*r_hsA**3
    Wks[r_hs>0.5] = 2 * (1 - r_hsB)**3
    Wks *= 8/(np.pi*hs[:,None]**3)

    return Wks
    
    
def Gadget_smooth(xyz_SPH, A_SPH=1, M_SPH=1, xyz_output=[0,0,0], rho_output=False,compute_tree=True,tree=1):
    '''Using sph info, calculate the kernel weighted quantity A at the coordinate xyz_output.
    Has an option to import a tree.'''

    Nngb = 40
    if compute_tree == True:
        tree = spatial.KDTree(xyz_SPH)
        print 'Tree is built'
    
    print 'Querying Tree'
    #Compute kernel SPH value at particle coords
    dists, inds = tree.query(xyz_output,k=Nngb)
    hs = dists[:,-1]
    r_hs = dists/hs[:,None]
    print 'Calculating Gadget kernel'
    Wks = Gadget_kernel(r_hs,hs)
    
    if rho_output == True:
        return M_SPH * np.sum(Wks,axis=1)

    else:
        #==== Find SPH rho for normalising ====#
        SPH_dists,SPH_inds = tree.query(xyz_SPH,k=Nngb)
        print 'Finished SPH query'
        SPH_hs = SPH_dists[:,-1]
        SPH_r_hs = SPH_dists/ SPH_hs[:,None]
        Wks_SPH = Gadget_kernel(SPH_r_hs,SPH_hs)
        rho_SPH = M_SPH * np.sum(Wks_SPH,axis=1)
        
        A_ngbs =  A_SPH[inds]
        A_output = M_SPH * np.sum(A_ngbs * Wks /rho_SPH[inds],axis=1)
        return A_output


    

def find_RH_2(M_star,M_gas,M_dust,gas_pos,dust_pos,frag_pos,M_sinks=0,sink_pos=[0,0,0]):
    '''Find the half Hill radius and enclosed gas and dust mass for a clump'''
    gas_pos  -= frag_pos
    dust_pos -= frag_pos
    sink_pos -= frag_pos
    gas_R    = np.sort(np.sqrt(gas_pos[:,0]**2+gas_pos[:,1]**2+gas_pos[:,2]**2))
    dust_R   = np.sort(np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2+dust_pos[:,2]**2))
    sink_R   = np.sqrt(sink_pos[:,0]**2+sink_pos[:,1]**2+sink_pos[:,2]**2)
    a        = np.sqrt(frag_pos[0]**2+frag_pos[1]**2+frag_pos[2]**2)

    i = 10000
    Ri = gas_R[i]
    Rh_2 = a * (M_gas*(i+1)/3*M_star)**(1/3)

    Macc_dust,Mint_sinks = 0,0
    while Ri < Rh_2:
        i += 10
        Macc_dust = len(dust_R[dust_R<Rh_2])*M_dust
        Mint_sinks = np.sum(M_sinks[sink_R<Rh_2])
        Rh_2 = a * ((M_gas*(i+1)+Macc_dust+Mint_sinks)/3*M_star)**(1/3) /2
        Ri = gas_R[i]
    print 'Macc', Macc_dust
        
    M_frag = M_gas * i
    return a, Rh_2, M_frag, Macc_dust, Mint_sinks




    
class Load_Snap:
    '''New snap reader. Uses classes and methods. All positions normalised to star'''

    def __init__(self,filepath,runfolder,snapid='000',snapprefix='snapshot_',):
        snapid          = str(snapid).zfill(3)        
        print filepath+runfolder+snapprefix+snapid
        self.headertime = readheader(filepath+runfolder+snapprefix+snapid,'time')* code_time /c.sec_per_year #Years
        self.N_gas      = readheader(filepath+runfolder+snapprefix+snapid,'gascount')
        self.M_gas      = readsnap(filepath+runfolder+snapprefix+snapid,'mass','gas')[0]

        #==== SPH particles ====#
        self.gas_pos  = readsnap(filepath+runfolder+snapprefix+snapid,'pos','gas')
        self.gas_vel  = readsnap(filepath+runfolder+snapprefix+snapid,'vel','gas')
        self.gas_ID   = readsnap(filepath+runfolder+snapprefix+snapid,'pid','gas')
        self.gas_u    = readsnap(filepath+runfolder+snapprefix+snapid,'u','gas')
        self.gas_rho  = readsnap(filepath+runfolder+snapprefix+snapid,'rho','gas')
        try:
            self.gas_h = readsnap(filepath+runfolder+snapprefix+snapid,'hsml','gas')
        except:
            print 'No smoothing length data. Is this Phantom? :)'
            self.gas_h = np.zeros(N_sph) 


        #==== Dust particles ====#
        try:
            self.M_dust   = readsnap(filepath+runfolder+snapprefix+snapid,'mass','disk')[0]
            self.N_dust   = readheader(filepath+runfolder+snapprefix+snapid,'diskcount')
            self.dust_pos = readsnap(filepath+runfolder+snapprefix+snapid,'pos','disk')
            self.dust_vel  = readsnap(filepath+runfolder+snapprefix+snapid,'vel','disk')

            try:
                self.dust_a = readsnap(filepath+runfolder+snapprefix+snapid,'DustRadius','disk')
                if len(self.dust_a) != np.shape(self.dust_pos[:,0])[0]:
                    1/0
            except:
                print 'Error loading dust size. Radius set to 1 cm'
                self.dust_a = np.ones(np.shape(self.dust_pos[:,0]))
            
           
 
        except:
            print 'No dust!'
            self.M_dust = 0
            self.dust_pos = [[0,0,0],[0,0,0]]

        #==== Stars and Planets ====#
        try:
            #if 1 == 1:    
            body_type        = 'bndry' #'disk'#
            M_bodies         = readsnap(filepath+runfolder+snapprefix+snapid,'mass',body_type)
            bodies_pos       = readsnap(filepath+runfolder+snapprefix+snapid,'pos',body_type)
            bodies_vel       = readsnap(filepath+runfolder+snapprefix+snapid,'vel',body_type)
            self.M_star      = M_bodies[np.argmax(M_bodies)] #code_M
            self.M_planets   = M_bodies[np.arange(len(M_bodies))!=np.argmax(M_bodies)]
            self.N_planets   = len(self.M_planets)
            print 'N_planets: ', self.N_planets, 'M [ME]: ',  np.flipud(np.sort(self.M_planets))*c.Msol/c.ME
            self.star_pos         = bodies_pos[np.argmax(M_bodies)]
            self.star_vel         = bodies_pos[np.argmax(M_bodies)]
            self.planets_pos = bodies_pos[np.arange(len(bodies_pos))!=np.argmax(M_bodies)] - self.star_pos
            self.planets_vel = bodies_vel[np.arange(len(bodies_vel))!=np.argmax(M_bodies)] - self.star_vel

            self.gas_pos = self.gas_pos - self.star_pos
            self.gas_vel = self.gas_vel - self.star_vel
            try:
                self.dust_pos = self.dust_pos - self.star_pos
                self.dust_vel = self.dust_vel - self.star_vel

            except:
                print 'Still no dust!'
        except:
            print 'No Star or planets present in simulation'
            self.M_star      = 0
            self.M_planets   = []
            self.N_planets   = 0
            self.planets_pos = np.array([[0,0,0]])
            self.planets_vel = np.array([[0,0,0]])    

        
            
    def zoom(self,zoom_pos,zoom_vel):
        '''Correct positions and velocities for a given zoom coordinate'''
        
        #==== Correct positions ====#
        self.gas_pos     = self.gas_pos - zoom_pos
        self.dust_pos    = self.dust_pos - zoom_pos
        self.planets_pos = self.planets_pos - zoom_pos
        self.star_pos    = self.star_pos - zoom_pos


        #==== Correct velocities ====#
        self.gas_vel     = self.gas_vel - zoom_vel
        self.dust_vel    = self.dust_vel - zoom_vel
        self.planets_vel = self.planets_vel - zoom_vel
        self.star_vel    = self.star_vel - zoom_vel
        
        return
    

    def rotate(self,inc=0,azi=0,inc2=0):
        '''Rotate 3d coordinate matrix. array=[:,3],inc,azi,inc2'''
        inc,azi,inc2 = np.radians(inc),np.radians(azi),np.radians(inc2)
    
        c_inc, s_inc = np.cos(inc), np.sin(inc)
        c_azi, s_azi = np.cos(azi), np.sin(azi)
        c_inc2, s_inc2 = np.cos(inc2), np.sin(inc2)
        R_inc = np.matrix('{} {} {}; {} {} {}; {} {} {}'
                      .format(1,0,0,0,c_inc,-s_inc,0,s_inc,c_inc))
        R_azi = np.matrix('{} {} {}; {} {} {}; {} {} {}'
                      .format(c_azi,-s_azi,0,s_azi,c_azi,0,0,0,1))
        R_inc2 = np.matrix('{} {} {}; {} {} {}; {} {} {}'
                       .format(1,0,0,0,c_inc2,-s_inc2,0,s_inc2,c_inc2))

        coords = [self.gas_pos,self.dust_pos,self.star_pos,self.planets_pos]
        vels   = [self.gas_vel,self.dust_vel,self.star_vel,self.planets_vel]

        for i in coords:
            i = np.dot(np.dot(np.dot(i,R_inc),R_azi),R_inc2)
        for j in vels:
            j = np.dot(np.dot(np.dot(j,R_inc),R_azi),R_inc2)

        return

    
    def subsample(self,box_lim):
        '''Remove all particles from beyond the box_lim'''
        sub = 1.1 * box_lim

        #==== Subsample Gas ====#
        gas_r2 = self.gas_pos[:,0]**2+self.gas_pos[:,1]**2+self.gas_pos[:,2]**2
        inds   = np.where(gas_r2<2*sub**2)[0]
        
        self.gas_pos = self.gas_pos[inds,:]
        self.gas_vel = self.gas_vel[inds,:]
        self.gas_ID  = self.gas_ID[inds]
        self.gas_h   = self.gas_h[inds]
        self.gas_u   = self.gas_u[inds]
        self.gas_rho = self.gas_rho[inds]

        #==== Subsample dust ====#
        dust_r2 = self.dust_pos[:,0]**2+self.dust_pos[:,1]**2+self.dust_pos[:,2]**2
        inds   = np.where(dust_r2<2*sub**2)[0]
        self.dust_pos = self.dust_pos[inds,:]
        self.dust_vel = self.dust_vel[inds,:]
        self.dust_a   = self.dust_a[inds]

        return


    def max_rho(self):
        '''Find the positin and velocity of the maximum gas density. Useful for polytrope locating'''
        rho_sort = np.argsort(self.gas_rho)
        zoom_pos = np.mean(self.gas_pos[rho_sort[-10:],:],axis=0)
        zoom_vel = np.mean(self.gas_vel[rho_sort[-10:],:],axis=0)
        return zoom_pos,zoom_vel
    


    







    
def bin_data_save(filepath,runfolder,Rin,Rout,vr_mode=False,zoom='',Poly_data=False):
    '''Save: 
    Time, M_star, M_gas, M_dust, N_Rbins, Rin, Rout, N_abins, amin, amax, N_planets...
    gas_count, gas_u, (gas_vr), dust_count[all_sizes]'''

    #==== Radial binning ====#
    N_Rbins  = 200
    Rbins = np.linspace(Rin,Rout,N_Rbins+1)
    Rbin_mids = (Rbins[1:]+Rbins[:-1])/2

    #==== Grain size binning ====#
    N_abins = 6
    amin,amax = -3.5,2.5
    abins     = np.logspace(amin,amax,N_abins+1)
    da_2 = (amax-amin)/2/N_abins
    abin_mids = np.logspace(amin+da_2,amax-da_2,N_abins)

    #==== Load snapshots ====#
    N_snaps = len(glob.glob(filepath+runfolder+'snap*'))-1
    print 'Load file: ', filepath+runfolder
    print 'N snaps: ', N_snaps
    save_array = np.zeros((N_snaps+1,6+N_abins,N_Rbins))

    #==== Fill out header info ====#
    snap0 = Load_Snap(filepath,runfolder,0)
    snapf = Load_Snap(filepath,runfolder,N_snaps)
    save_array[0,0,0]  = snap0.headertime 
    save_array[0,0,1]  = snap0.M_star
    save_array[0,0,2]  = snap0.M_gas
    save_array[0,0,3]  = snap0.M_dust
    save_array[0,0,4]  = N_Rbins
    save_array[0,0,5]  = Rin
    save_array[0,0,6]  = Rout
    save_array[0,0,7]  = N_abins
    save_array[0,0,8]  = amin
    save_array[0,0,9]  = amax
    save_array[0,0,20] = snapf.N_planets
    print '\n'

    
    #======== Save all info for each snap =======#
    for snapid in range(N_snaps):
        print 'Load snap', snapid
        S = Load_Snap(filepath,runfolder,snapid)
        save_array[snapid+1,0,0]  = S.headertime
        save_array[snapid+1,0,1]  = np.mean(S.gas_pos[:,2])
        save_array[snapid+1,0,2]  = np.mean(S.dust_pos[:,2])
        print 'Snap time', S.headertime

        #==== Zoom modes ====#
        if zoom == 'Zrho':
            rho_sort  = np.argsort(S.gas_rho)
            zoom_pos = np.mean(S.gas_pos[rho_sort[-10:],:],axis=0)
            #zoom_pos = S.gas_pos[np.argmax(S.gas_rho),:]
        elif zoom == 'ZP':
            zoom_pos = S.pos_planets[0]
        if (zoom == 'Zrho') or (zoom == 'ZP'):
            S.gas_pos  = S.gas_pos - zoom_pos
            try:
                S.dust_pos = S.dust_pos - zoom_pos
            except:
                pass
            try:
                S.pos_planets = S.pos_planets - zoom_pos
            except:
                pass
            
        #==== Bin gas particles ====#
        r_gas  = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
        gas_count = np.histogram(r_gas,Rbins)[0]
        gas_u     = calc_binned_data(S.gas_u,r_gas,Rbins)[0]
        gas_h     = calc_binned_data(S.gas_h,r_gas,Rbins)[0]
        save_array[snapid+1,1,:] = gas_count
        save_array[snapid+1,2,:] = gas_u
        save_array[snapid+1,3,:] = gas_h

        if vr_mode == True:
                    v_r, v_azi = v_r_azi(S.gas_pos,S.gas_vel)
                    gas_vr     = calc_binned_data(v_r,r_gas,Rbins)[0]
                    save_array[snapid+1,4,:] = gas_vr

                    
        #==== Bin dust species ====#
        try:
            r_dust   = np.sqrt(S.dust_pos[:,0]**2+S.dust_pos[:,1]**2+S.dust_pos[:,2]**2)
            a_ids = np.digitize(S.dust_a,abins)-1
            for j in range(N_abins):
                dust_count = np.histogram(r_dust[a_ids==j],Rbins)[0]
                save_array[snapid+1,5+j,:] = dust_count
        except:
            pass
        try:
            save_array[snapid+1,11,:] = S.dust_Vcoll
        except:
            pass
        
        #==== Save planet positions ====#
        if S.N_planets != 0:
            R_planets = np.sqrt(S.planets_pos[:,0]**2 + S.planets_pos[:,1]**2 + S.planets_pos[:,2]**2)

            #Sort by planet mass. Not ideal but useful
            massinds = np.argsort(-S.M_planets)
            MP_sort = S.M_planets[massinds]
            RP_sort = R_planets[massinds]

            for i in range(S.N_planets):
                save_array[snapid+1,0,20] = S.N_planets
                try:
                    save_array[snapid+1,0,25+2*i] = MP_sort[i]
                    save_array[snapid+1,0,26+2*i] = RP_sort[i]
                except:
                    print 'Too many planets for save array!'
                    save_array[snapid+1,0,20] = int((N_Rbins-25)/2)
                    
        if Poly_data == True:
            rho_sort  = np.argsort(S.gas_rho)
            frag_pos  = np.mean(S.gas_pos[rho_sort[-10:],:],axis=0)
            a_frag, RH_2, M_frag, Macc_dust,Mint_sinks = find_RH_2(S.M_star,S.M_gas,S.M_dust,S.gas_pos,S.dust_pos,
                                                                   frag_pos,M_sinks=S.M_planets,sink_pos=S.planets_pos)
            save_array[snapid+1,0,21] = M_frag
            save_array[snapid+1,0,22] = a_frag
            save_array[snapid+1,0,23] = RH_2
            save_array[snapid+1,0,24] = Macc_dust
            
            
        print '\n'

    
    #==== Save output ====#
    print 'Saving output: '
    print savedir+runfolder.rstrip('//')+zoom+'_store'
    np.save(savedir+runfolder.rstrip('//')+zoom+'_store',save_array)
    print '#==== Binning routine complete ====#'
    
    return save_array














def animate_1d(filepath,runfolders,var1='Sigma',var2='T',rerun=False,Rin=0.1,Rout=2.0,norm_y=False,zoom='',write=False):
    '''New function to generalise animation code
    ZP = zoom around planet
    Zrho = zoom on max rho SPH particle'''
    
    plot_dict,anim_dict,snap_list = {},{},[]
    plot_vars = [var1,var2]
    dust_scale = 1

    #======== Set up Figure ========#
    fig1 = plt.figure(1,facecolor='white',figsize=(10,6))#(6,10))
    ax1  = fig1.add_axes([0.15,0.45,0.83,0.5])
    ax2  = fig1.add_axes([0.15,0.09,0.83,0.3],sharex=ax1)
    axes = [ax1,ax2]
    ax1.semilogy()
    ax2.semilogy()
    ax2.set_xlabel('R [AU]')

    for runid in range(len(runfolders)):
        runfolder = runfolders[runid]
        try:
            if rerun == True:
                1/0
            save_array = np.load(savedir+runfolder.rstrip('//')+zoom+'_store.npy')
        except:
            print '#==== Need to run data binning routine! ====#'
            save_array = bin_data_save(filepath,runfolders[runid],Rin,Rout,zoom=zoom)
            
        #==== Load bin information ====#
        snap_list.append(len(save_array[1:,0,0]))
        plot_dict[str(runid)+'time'] = save_array[1:,0,0]
        N_Rbins,Rin,Rout = save_array[0,0,4],save_array[0,0,5],save_array[0,0,6]
        Rbins = np.linspace(Rin,Rout,N_Rbins+1)
        Rbin_mids = (Rbins[1:]+Rbins[:-1])/2
        dRbin = Rbins[1]-Rbins[0]
        Rbin_areas = Rbin_mids*2*np.pi*dRbin
        Rbin_volumes = 4*np.pi/3 * (Rbins[1:]**3-Rbins[:-1]**3)
        N_abins,amin,amax = int(save_array[0,0,7]),save_array[0,0,8],save_array[0,0,9]
        abins     = np.logspace(amin,amax,N_abins+1)
        M_gas,M_dust = save_array[0,0,2],save_array[0,0,3]
        
        #============ Construct plotting dictionaries for appropriate variables ============#
        #===================================================================================#
        for i in range(len(plot_vars)):
            
            #============= Plot Sigma =============#
            if plot_vars[i] == 'Sigma':
                gas_sig = save_array[1:,1,:]*M_gas/Rbin_areas*code_M/code_L**2
                scale_fac = Rbin_mids*code_L/c.AU
                
                plot_dict[str(runid)+'_'+str(i)] = gas_sig*scale_fac
                axes[i].set_ylabel(r'$\Sigma$ R [gcm$^{-2}$ AU]')
                if norm_y == True:
                    plot_dict[str(runid)+'_'+str(i)] = gas_sig/gas_sig[0,:]
                    axes[i].set_ylabel(r'Normalised $\Sigma$')

                #-------- Dust Sigma --------#
                scale_fac *= dust_scale
                for j in range(N_abins):
                    dust_sig = save_array[1:,5+j,:]*M_dust/Rbin_areas*code_M/code_L**2
                    plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_sig*scale_fac
                    if norm_y == True:
                        plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_sig/dust_sig[0,:]


                        
            #============= Plot Rho =============#
            if plot_vars[i] == 'rho':
                gas_rho = save_array[1:,1,:]*M_gas/Rbin_volumes*code_M/code_L**3
                
                plot_dict[str(runid)+'_'+str(i)] = gas_rho
                axes[i].set_ylabel(r'$\rho$ [gcm$^{-3}$]')
                axes[i].set_ylim(np.min(gas_rho)/1.5,np.max(gas_rho)*1.5)
                if norm_y == True:
                    plot_dict[str(runid)+'_'+str(i)] = gas_rho/gas_rho[0,:]
                    axes[i].set_ylabel(r'Normalised $\rho$')

                #-------- Dust rho --------#
                scale_fac = dust_scale
                for j in range(N_abins):
                    dust_rho = save_array[1:,5+j,:]*M_dust/Rbin_volumes*code_M/code_L**3
                    plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_rho*scale_fac
                    if norm_y == True:
                        plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_rho/dust_rho[0,:]



                        
            #========= Plot Temperature =========#
            if plot_vars[i] == 'T':
                gas_T = save_array[1:,2,:]*(c.gamma_mono-1)*c.mu*c.mp/c.kb*code_L**2/code_time**2
                plot_dict[str(runid)+'_'+str(i)] = gas_T
                axes[i].set_ylabel(r'Temperature [K]')

            #======== Plot smoothing length ========#
            if plot_vars[i] == 'h':
                gas_h = save_array[1:,3,:]*AU_scale #[AU]
                plot_dict[str(runid)+'_'+str(i)] = gas_h
                axes[i].set_ylabel(r'Smoothing Length [AU]')
                
            #========= Plot Radial Velocity ===========#
            if plot_vars[i] == 'vr':
                gas_vr = save_array[1:,3,:]*code_L/code_time
                plot_dict[str(runid)+'_'+str(i)] = gas_vr
                axes[i].set_ylabel(r'$Gas V_R [cms^{-1}]$')

            #========= Plot Dust to gas ratio =========#
            if plot_vars[i] == 'dust_gas':
                dust_gas = np.sum(save_array[1:,4:4+N_abins,:],axis=1) /save_array[1:,1,:] *M_dust/M_gas
                plot_dict[str(runid)+'_'+str(i)] = dust_gas 
                axes[i].set_ylabel('Dust to Gas Ratio')
                dg_0 = axes[i].axhline(M_dust/M_gas,color=run_cols[2],ls='--')
                if norm_y == True:
                    plot_dict[str(runid)+'_'+str(i)] *= M_gas/M_dust 
                    axes[i].set_ylabel('Normalised Dust to Gas Ratio')
                    dg_0.set_ydata(1)

            #========= Plot enclosed mass =========#
            if plot_vars[i] == 'M_enc':
                M_enc = np.cumsum(save_array[1:,1,:],axis=1)*M_gas
                print np.shape(M_enc)
                print np.shape(save_array[1:,1,:])
                plot_dict[str(runid)+'_'+str(i)] = M_enc
                axes[i].set_ylabel(r'Enclosed Mass [$M_{\odot}$]')
                for j in range(N_abins):
                    M_enc_dust = np.cumsum(save_array[1:,4+j,:],axis=1)*M_dust
                    plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = M_enc_dust

            #======== Plot Collison Velocity =======#
            if plot_vars[i] == 'Vcoll':
                Vcoll = save_array[1:,11,:]*code_L/code_time
                plot_dict[str(runid)+'_'+str(i)] = Vcoll
                axes[i].set_ylabel(r'$Dust Collision Velocity [cms^{-1}]$')

                    
            #===------------= Establish animation objects =------------===#
            anim_dict[str(runid)+'_'+str(i)] = axes[i].plot(Rbin_mids*AU_scale,
                    plot_dict[str(runid)+'_'+str(i)][0],color=run_cols[runid],
                    label=str(runfolders[runid]))
            
            #-------- Dust lines --------#
            if (plot_vars[i]=='Sigma') or (plot_vars[i]=='M_enc') or (plot_vars[i]=='rho'):
                for j in range(N_abins):
                    anim_dict[str(runid)+'_'+str(i)+'_'+str(j)] = axes[i].plot(
                        Rbin_mids*AU_scale,plot_dict[str(runid)+'_'+str(i)+'_'+str(j)][0],
                        color=run_cols[runid],ls=linestyles[j],lw=linewidths[j])
                    if runid == len(runfolders)-1:
                        anim_dict[str(runid)+'_'+str(i)+'_'+str(j)][0].set_label(
                            'a = {:.3f}'.format(abins[j])+'-{:.3f}'.format(abins[j+1])+'cm')

            #====== Planet lines ======#
            N_planets = int(np.max(save_array[1:,0,20]))
            print 'N_planets ani', N_planets
            plot_dict[str(runid)+'NP'] = N_planets
            for p in range(N_planets):
                MPs = save_array[1:,0,25+2*p]*code_M/c.MJ
                RPs = save_array[1:,0,26+2*p]*AU_scale
                plot_dict[str(runid)+'p'+str(p)+'_MPs'] = MPs
                plot_dict[str(runid)+'p'+str(p)+'_RPs'] = RPs
                anim_dict[str(runid)+'_RP_'+str(i)+'_'+str(p)] = axes[i].axvline(
                    RPs[0],lw=1,color=run_cols[runid])
                if i == 1:
                    anim_dict[str(runid)+'_MPtext_'+str(i)+'_'+str(p)] = ax2.text(
                        0.8,0.8+runid*0.05,r'M$_P$ {:.2f} M$_J$'.format(MPs[1]),transform=ax2.transAxes)

                    
    #======= Check that time arrays are self-consistent =======#
    print 'Need to write time array self-consistent check'


    #=========================== Plotting code ===========================#
    #=====================================================================#
    timetext = ax1.text(0.06,0.91,'Time: {:.2f}'.format(plot_dict[str(runid)+'time'][0])
                        + ' Years',transform=ax1.transAxes)
    N_frames = np.min(snap_list)
    print 'N frames', N_frames
    ax1.legend(frameon=False)

    def animate(anim_i):
        output = []
        for runid in range(len(runfolders)):
            for i in range(len(plot_vars)):
                anim_dict[str(runid)+'_'+str(i)][0].set_ydata(plot_dict[str(runid)+'_'+str(i)][anim_i])
                N_planets = plot_dict[str(runid)+'NP']
                for p in range(N_planets):
                    anim_dict[str(runid)+'_RP_'+str(i)+'_'+str(p)].set_xdata(
                        plot_dict[str(runid)+'p'+str(p)+'_RPs'][anim_i])
                    if i == 1:
                        anim_dict[str(runid)+'_MPtext_'+str(i)+'_'+str(p)].set_text(
                            r'M$_P$ {:.2f} M$_J$'.format(plot_dict[str(runid)+'p'+str(p)+'_MPs'][anim_i]))

                for j in range(N_abins):
                    try:
                        anim_dict[str(runid)+'_'+str(i)+'_'+str(j)][0].set_ydata(
                            plot_dict[str(runid)+'_'+str(i)+'_'+str(j)][anim_i])
                    except:
                        pass

        
        timetext.set_text('Time: {:.2f}'.format(plot_dict[str(runid)+'time'][anim_i]) + ' Years')
        output.append(timetext)
        output.append(anim_dict)
        return output

    ani = animation.FuncAnimation(fig1, animate, interval=80, frames=N_frames, blit=False, repeat=True)
    plt.show()

    if write == True:
            print 'Writing savefile'
            writer = animation.writers['ffmpeg'](fps=5)
            print runfolder.strip('//')+'_'+var1+'_'+var2+'.mp4'
            ani.save(runfolder.strip('//')+'_'+var1+'_'+var2+'.mp4',writer=writer)
    
    return





    
def subsample_dust_size(pos,a,amin,amax):
    '''Subsamples dust data to select only grains of a certain grain size a'''
    inds = []
    for i in range(len(a)):
        if (a[i]>amin) & (a[i]<amax):
            inds.append(i)
            
    pos = pos[inds,:]
    a   = a[inds]
    
    return pos,a

    
def plot_thermalvel():
    Ts = np.logspace(1,5,100)
    T_vels = thermal_vel(Ts)
    plt.plot(Ts,T_vels)
    plt.semilogx()
    plt.semilogy()
    plt.show()


def temp_floors():
    Rs = np.arange(200)
    
    Gadget_T = 20*(Rs/100)**-0.5
    Seren_T  = 250*Rs**-0.75

    plt.figure(1,facecolor='w')
    plt.plot(Rs,Gadget_T,label='Gadget 3 (Humphries 2018)')
    plt.plot(Rs,Seren_T,label = 'Seren (Stamatellos 2018)')
    plt.legend(frameon=False)
    return



def find_f_peb(M_largedust,amin,amax,apow,amid=0):
    '''For coupled and uncoupled dust mix, find f_peb
    amin < coupled dust < amid < uncoupled dust < amax'''
    #Find normalisation factor from large dust mass (known)
    if apow == 4:
        fac = np.log10(amax)-np.log10(amid)
    else:
        fac = (amax**(4-apow)-amid**(4-apow)) / (4-apow)
    A = M_largedust / fac

    if apow == 4:
        facB = np.log10(amax)-np.log10(amin)
    else:
        facB = (amax**(4-apow)-amin**(4-apow)) / (4-apow)
    Mtot_dust = A * facB

    f_peb = M_largedust/Mtot_dust
    return Mtot_dust, f_peb



def plot_dust_mass_frac(Mtot_dust,amin,amax,apow):
    '''Plot the fraction of mass in each grain size'''
    a = np.logspace(np.log10(amin),np.log10(amax),100)
    a_mids = 10**((np.log10(a[1:])+np.log10(a[:-1]))/2)
    
    if apow == 4:
        fac = np.log10(amax)-np.log10(amin)
        A = Mtot_dust/fac
        M_cum = A * (np.log10(a)-np.log10(amin))
        M_bin = A * (np.log10(a[1:])-np.log10(a[:-1]))
        
    else:
        fac = (amax**(4-apow)-amin**(4-apow)) / (4-apow)
        A = Mtot_dust/fac
        M_cum = A * (a**(4-apow)-amin**(4-apow)) / (4-apow)
        M_bin = A * (a[1:]**(4-apow)-a[:-1]**(4-apow)) / (4-apow)
        
    fig0 = plt.figure(0,facecolor='w')
    ax1 = fig0.add_axes([0.1,0.55,0.8,0.4])
    ax2 = fig0.add_axes([0.1,0.1,0.8,0.4],sharex=ax1)

    ax1.set_ylabel('M_frac')
    ax1.plot(a,M_cum)
    ax1.semilogy()
    ax1.semilogx()

    ax2.bar(a[:-1],M_bin,log=True)
    ax2.set_xlabel('Grain size [cm]')
    ax2.set_ylabel('Mass per bin')
    return

    
    


if __name__ == "__main__":
    filepath='/rfs/TAG/rjh73/Gio_disc/'
    filepath = '/rfs/TAG/rjh73/Clump_project/'
    runfolders = ['Gio_N1e6_aav01_R00120_MD01_Z10_MP03/','Gio_N1e6_aav01_R00120_MD01_Z10_MP3/']
    runfolders = ['P1e5_M5_R3_b5_rho2e11_r60_T30_Ti34/','P1e5_M5_R3_b5_rho2e11_r60_T30_Ti12/']
    #temp_floors()
    #animate_1d(filepath=filepath,runfolders=runfolders,rerun=False,var2='M_enc',zoom='Zrho',Rin=0.001,Rout=0.04)
    plot_dust_mass_frac(1,1e-6,1,3.5)

    plt.show()
