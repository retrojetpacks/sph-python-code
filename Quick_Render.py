'''Implementation of Robs SPH Renderer

#====== Main Functions ======#
load_Gsnap - function to load a gadget snapshot and store output in easilly readable dictionaries
render     - does the actual smoothing work using pyjack c code
render_SPH - produces final rendered images and movies
'''


from __future__ import division
import numpy as np
import os
import scipy.spatial
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from matplotlib import _cntr as cntr
from pygadgetreader import *
from matplotlib import animation
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import pyjack
import time
import h5py
import glob
import cProfile

from matplotlib import rcParams

def set_rcparams(fsize=14):
    font = {'weight' : 'normal',
            'size'   : fsize}#,
            #'family' : 'serif'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    return
set_rcparams()


#Code Units
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
AU_scale   = 100



#==== Frag in Disc ====#
filepath = '/rfs/TAG/rjh73/Frag_in_disc/'
runfolders = [#'Poly_N100000_M0003_R003_n15/',
              #'Disc_N3e5_R00120_M0009_b5/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01/'
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_shires/'
    #'Polydisc_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf01/'


]

savedir = '/scratch/r/rjh73/save_data/'


#======== General plotting parameters ========#
ngrid    = int(1024) #use powers of two!
XYdim    = 0.6#2.#1#.3.2#1.5
XYdim_zoom = 0.04#0.1#0.04
dust_Neighbours = 40
rho_Neighbours = 40
floor    = 1e-15#-20
roof     = 1e10
j_blues  = colors.LinearSegmentedColormap.from_list('j_blues',['#001111','#004488','#66eeff'])
cols  = ['#990099','#dd3300','#cc6600','#2EC4C6']


#Args: name, particle type, render colour, lims, zoom lims, field mode
gas_args  = ['Gas','gas','magma',[5e-3,15000],[5e-3,200000],'Sigma']
dust_args = ['Dust','dust',j_blues,[1e-4,10],[1e-2,10000],'Sigma']
tau_args  = [r'$\tau$','gas', 'viridis', [0.05,5],[5e-3,5],'tau']
Q_args    = ['Toomre Q','gas', 'magma', [0.01,50],[0.01,50],'Q']
temp_args = ['Temp','gas', 'magma', [10,1500],[10,1500],'temp']
rho_args  = ['Gas','gas','magma',[1e-14,1e-12],[1e-12,1e-8],'rho']
test_args = ['Gas','gas','magma',[1e-3,1e1],[1e-2,1e6],'Sigma']


grain_size = 1 #cm
grain_rho = 3 #g/cm^-3




#======== Miscellaneous functions ========#

def find_snap_num(runfolder):
    '''Return the number of snapshots in a runfolder'''
    globs = glob.glob(filepath+runfolder+'snap*')
    print 'Number of snapshots:', len(globs)
    return len(globs)#-1


def beta_stringify(run_name):
    '''Return a string of the correct beta value'''
    try:
        a = run_name.split('beta')[1].split('_')[0]
        if a[0] == '0':
            a = '0.'+ a[1:]
    except:
        try:
            a = run_name.split('b')[1].split('_')[0]
            if a[0] == '0':
                a = '0.'+ a[1:]
        except:
            a = '-'
    return a

def grain_stringify(run_name):
    try:
        a = run_name.split('_')[3][1:]
        if a[0] == '0':
            a = '0.'+ a[1:]
    except:
        a = '-'
    return a

def res_hists(xs,ys,sph_h,sph_u,M_star,MP,planet_p,it,time):
    '''Produce histograms of smoothing length ratios'''
    xs,ys = xs*code_L,ys*code_L                    #cm
    Rs = np.sqrt(xs**2 + ys**2)                    #cm
    hs = sph_h*code_L                              #cm
    us = sph_u*code_L**2/code_time**2              #cm^2/s^-2
    Om_Ks = b.v_kepler(M_star*code_M,Rs)/(Rs)      #/s
    Hs = np.sqrt((c.gamma_mono-1)*us) / Om_Ks      #cm
    h_Hs = hs/Hs
    
    planet_R = np.sqrt(planet_p[0]**2+planet_p[1]**2) * code_L #cm
    RH = b.hill_radius(planet_R,MP,M_star)                     #cm
    dR = np.sqrt((xs-planet_p[0])**2+(ys-planet_p[1])**2)      #cm
    h_RH = hs[dR<RH]/RH
    print 'h_RH', h_RH
    
    #=== Plotting ===#
    txt_ys = [0.9,0.85,0.8,0.75,0.7]
    bins = np.linspace(0,2,100)
    fig14 = plt.figure(14,facecolor='white')
    ax1 = fig14.add_axes([0.15,0.1,0.37,0.85])
    ax2 = fig14.add_axes([0.6,0.1,0.37,0.85])
    
    ax1.hist(h_Hs,bins=bins,edgecolor=cols[it],lw=3,facecolor='none',histtype='step')
    ax1.set_xlabel(r'$h / H$')
    ax1.set_ylabel('Particle Number')

    ax2.hist(h_RH,bins=bins,edgecolor=cols[it],lw=3,facecolor='none',histtype='step')
    ax2.set_xlabel(r'$h / R_H$')
    ax2.text(0.6,txt_ys[it],str(int(time))+' Years',color=cols[it],transform=ax2.transAxes)
        
    return





#==================== Rendering Code ===================#
#=======================================================#


def grid_render(R_pos,R_vel,R_A,R_rho,ngrid,box_lim):
    '''Python routine for Martin's slow grid render. For density'''
    
    grid_slices = np.zeros((5,ngrid**2))
        
    grid  = np.linspace(-1*box_lim,box_lim,ngrid)
    Xs,Ys = np.meshgrid(grid,grid)
    grid_points = np.dstack((Xs.ravel(),Ys.ravel()))
    grid_points = np.dstack((grid_points,np.zeros(ngrid**2)))[0]
    
    tree = scipy.spatial.cKDTree(R_pos)
    Ndist, Nind = tree.query(list(grid_points), k=rho_Neighbours)
    hsml = np.amax(Ndist,axis=1)

    for i in range(ngrid**2):
        Nindi = Nind[i]
        R_rhotemp = R_rho[Nindi]
        R_Atemp   = R_A[Nind[i]]
        R_veltemp = R_vel[Nind[i],:]
        
        ur    = Ndist[i]/hsml[i]
        rhos  = 3.52/hsml[i]**3 * (1-ur)**4 * (1+4*ur)
        norms = rhos/R_rhotemp
        As    = norms*R_Atemp
        vels  = norms[:,None]*R_veltemp
        
        grid_slices[0,i] = np.sum(norms)
        grid_slices[1,i] = np.sum(rhos)
        grid_slices[2,i] = np.sum(As)
        grid_slices[3,i] = np.sum(vels[:,0])
        grid_slices[4,i] = np.sum(vels[:,1])
        
    #Normalise. M_sph cancels.
    grid_slices = grid_slices[1:]/grid_slices[0] 
     
    return grid_slices



def render(runfolder,snapprefix,snapid=0,p_type='gas',inc=0,azi=0,inc2=0,
           render_mode='Sigma',zoom=False,polyzoom=False,overlay=False,
           track_IDs=[],hist2D=False,h_hist=False,it=0):
    '''Load SPH info and call C rendering routine.
    p_type = gas, dust'''
    
    #==== Initial setup ====#
    time0 = time.time()
    S = b.Load_Snap(filepath,runfolder,snapid=snapid)
    print 'Time: ', S.headertime, ' [Years]'
    box_lim = XYdim
    
    
    #==== Zoom on planet/fragment ====#
    if zoom == True:
        box_lim = XYdim_zoom
        if polyzoom == True:
            zoom_pos,zoom_vel = S.max_rho()
        else:
            zoom_pos,zoom_vel = S.planets_pos[0],S.planets_vel[0]
        
        #=== Rotate frame to planet ===#
        azizoom = b.star_planet_angle(zoom_pos)
        S.zoom(zoom_pos,zoom_vel)
        S.rotate(azi=azizoom)

    #==== Viewing rotations ====#
    S.rotate(inc,azi,inc2)

    #=== Subsample to reduce cost ====#
    print 'boxlim', box_lim*AU_scale, 'AU'
    S.subsample(box_lim)

    print 'ISIN mask', np.isin(S.gas_ID,track_IDs)
    #track_pos =

    
    #####=============== SPH Rendering =============#####
    if hist2D == False:
        
        #========== Choose gas or dust for render ========#
        print 'P type', p_type
        if p_type == 'gas':
            R_pos,R_vel,R_h,R_A,R_rho = S.gas_pos,S.gas_vel,S.gas_h,S.gas_u,S.gas_rho
            M_R = S.M_gas
        elif p_type == 'dust':
            #Infer render smoothing lengths for dust particles
            tree = scipy.spatial.cKDTree(S.dust_pos)
            Ndist, Nind = tree.query(S.dust_pos, k=dust_Neighbours)
            R_pos,R_vel = S.dust_pos,S.dust_vel
            R_h = np.max(Ndist,axis=1)
            R_A = S.dust_a
            M_R = S.M_gas
        

        #========= Run render code ==========#
        print '#==== Begin Render! ====#'       
        if render_mode == 'Sigma':            
            render_output = pyjack.smoother(R_pos[:,0],R_pos[:,1],R_h,R_A,R_vel[:,0],R_vel[:,1],
                                            ngrid,-1*box_lim,box_lim,4,M_R)

            rendered  = render_output[:ngrid**2].reshape((ngrid,ngrid))
            rendered  = np.nan_to_num(rendered)

            extra     = render_output[ngrid**2:2*ngrid**2].reshape((ngrid,ngrid))
            vx        = render_output[2*ngrid**2:3*ngrid**2].reshape((ngrid,ngrid))
            vy        = render_output[3*ngrid**2:4*ngrid**2].reshape((ngrid,ngrid))
            A_output  = extra/rendered
            vx_output = vx/rendered
            vy_output = vy/rendered
            
        elif render_mode == 'rho_grid':
            #Rho grid plot. A la Martin Bourne.
            render_output = grid_render(R_pos,R_vel,R_A,R_rho,ngrid,box_lim)
            rendered  = render_output[0].reshape((ngrid,ngrid))
            A_output  = render_output[1].reshape((ngrid,ngrid))
            vx_output = render_output[2].reshape((ngrid,ngrid))
            vy_output = render_output[3].reshape((ngrid,ngrid))
        print '\n#==== End Render! ====#'
        
            
    #======== Hist 2D result ========#  
    else:
        rendered,xedges,yedges = np.histogram2d(R_pos[:,0],R_pos[:,1],ngrid,range=[[-1*box_lim,box_lim],[-1*box_lim,box_lim]])
        rendered = rendered * M_R /(2*XYdim/ngrid)**2
        A_output = rendered*0

    #============ Store render output ========#
    store = np.zeros((4,ngrid+1,ngrid))   
    store[0,1:,:] = rendered    
    store[1,1:,:] = A_output
    store[2,1:,:] = vx_output
    store[3,1:,:] = vy_output

    
    #============ Store planet information =========#
    print 'Planets: ', S.planets_pos, np.shape(S.planets_pos)
    store[0,0,0] = S.M_star
    store[0,0,1] = S.N_planets
    try:
        for i in range(S.N_planets):
            store[0,0,2+i*4] = S.M_planets[i]
            store[0,0,3+i*4] = S.planets_pos[i,0]
            store[0,0,4+i*4] = S.planets_pos[i,1]
            store[0,0,5+i*4] = S.planets_pos[i,2]
    except:
        print 'Might be too many planets!'
        store[0,0,1] = int((ngrid-2)/4)


    print 'Frame time: ', (time.time()-time0)
    return store, S.dust_pos, track_pos
        


#============================= Plotting functions =====================================#    
#======================================================================================#

def array_dust_overlay(render,dust_pos,box_lim):
    '''Update rendered pictures to include dust pixels'''    
    dust_indices = np.rint((dust_pos[:,0:2]+box_lim) *ngrid/ (box_lim*2))    
    for i in range(len(dust_indices[:,0])):
        if (dust_indices[i,0]>0) & (dust_indices[i,0]<ngrid) & (dust_indices[i,1]>0) & (dust_indices[i,1]<ngrid): 
            render[dust_indices[i,1],dust_indices[i,0]] = -1#0
    return render


def calc_Om_K_field(M_star,zoom=False,p_pos=[0,0]):
    if zoom == False:
        grid_cells = np.linspace(-1,1,ngrid)*XYdim #code_L
        Xs,Ys = np.meshgrid(grid_cells,grid_cells)
        Rs = np.sqrt(Xs**2 + Ys**2) * code_L 
    else:
        grid_cells = np.linspace(-1,1,ngrid)*XYdim_zoom #code_L
        Xs_zoom,Ys_zoom = np.meshgrid(grid_cells,grid_cells)
        print np.shape(p_pos)
        print np.shape(Xs_zoom)
        Rs = np.sqrt((Xs_zoom)**2 + (Ys_zoom)**2) * code_L
    Om_K_field  = b.v_kepler(M_star[:,None,None]*code_M,Rs[None,:]) / (Rs[None,:])   #1/seconds
    print 'shape',  np.shape(Om_K_field)
    return Om_K_field, Rs


def calc_tau(u,Om_K,Sigma,render_mode):
    '''Calculate dimensionless stopping time'''
    #tau_field = grain_rho*grain_size /Sigma * np.sqrt(8/np.pi)
    if render_mode == 'Sigma':
        tau_field = grain_rho*grain_size*np.pi / (2*Sigma) 
    elif render_mode == 'rho_grid':
        Sigma = Sigma/code_L #rho
        tau_field = grain_rho*grain_size*Om_K/Sigma / np.sqrt(8*u*(c.gamma_mono-1))
    return tau_field

def calc_Q(u,Om_K,Sigma,Rs):
    '''Calculate Toomre Stability criteria Q field '''
    Q_field = np.sqrt((c.gamma_mono-1)*u)* Om_K / (Sigma * np.pi * c.G) #[Q]
    #Q_field = Sigma / np.sqrt(2*np.pi*(c.gamma_mono-1)*u) *Om_K #rhoc
    return Q_field

def calc_temp(u,Om_K,Sigma,render_mode):
    T_field = (c.gamma_mono-1)*u*c.mp*c.mu/c.kb
    return T_field




def ID_selection(runfolder,snapprefix,zoom='planet',track_R=3):
    '''Track gas particles that are initially close to the planet'''  
    S = b.Load_Snap(filepath,runfolder,snapid=0)

    
    if zoom == 'poly':
        zoom_pos = S.max_rho()
    elif zoom == 'planet':
        zoom_pos = S.planets_pos[0]

    S.gas_pos -= zoom_pos
    gas_R = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
    track_IDs = S.gas_ID[gas_R<track_R/AU_scale]

    print 'TRACK IDS', track_IDs

    return track_IDs


    


def render_SPH(runfolder,args=gas_args,snapprefix='snapshot_',snap=-1,
               inc=0,azi=0,inc2=0,render_mode='Sigma',vel_field=False,
               upto=0,zoom=False,polyzoom=False,overlay=False,write=False,
               dpi=1000,h_hist=False,it=0,rerun=False,amin=0,amax=0,tracking=True,track_R=3):
    
    '''Make rendered animation or single image of an SPH plot. Option to save.
    field_mode: Sigma - Surface Density [gcm^{-2}]
                tau   - Dimensionless stopping time []
                Q     - Toomre Q parameter []
    set snap to find snapshot. snap=-1 -> movie
    If inc != 0 or 180, plot rho instead of Sigma
    overlay   - pixellated overlay of dust particles. Scatter too expensive.
    vel_field - overlay of render data velocity structure
    zoom      - zooms on first planet
    polyzoom  - zooms on max rho SPH particles
    upto      - limits movie frames
    tracking  - track gas particles that are initially close to the planet
    '''
    
    #==== Load Arguments ====#
    mode_name,p_type,map_col = args[0],args[1],args[2]
    vmin,vmax,field_mode     = args[3][0],args[3][1],args[5]
    zoom_str,overlay_str,angle_str = '','',''
    movie       = False
    box_lim     = XYdim*AU_scale
    
    if polyzoom==True:
        zoom = True
    if zoom == True:
        zoom_str = '_zoom'
        box_lim = XYdim_zoom*AU_scale
        vmin,vmax = args[4][0],args[4][1]
    if overlay == True:
        overlay_str = '_dustgrid'
    if inc != 0:
        angle_str += '_i'+str(inc)
    if azi != 0:
        angle_str += '_a'+str(azi) 
    if field_mode == 'rho':
        render_mode='rho_grid'

    #==== Build grids ====#
    bins        = np.linspace(-box_lim,box_lim,ngrid+1)
    bin_mids    = (bins[1:]+bins[:-1])/2
    Bin_Xs,Bin_Ys = np.meshgrid(bin_mids,bin_mids)

    
    #================ Find Time data and num snaps ==================#
    time1 = time.time()
    try:
        time_zero = readheader(filepath+runfolder+snapprefix+'001','time')
        try:
            time_one = readheader(filepath+runfolder+snapprefix+'002','time')
            time_dt = time_one-time_zero
        except:
            'Only one snapshot!'
            time_dt = 0

    except:
        print 'No snapshots found!'
        time_dt = 0
        
    snap_dt = time_dt * code_time /c.sec_per_year
    if snap == -1:
        movie = True
        print savedir+runfolder
        if upto != 0:
            num_snaps = upto
        else:
            num_snaps = find_snap_num(runfolder)
        snapids = np.arange(num_snaps)
    else:
        num_snaps = 1
        snapids = [snap]


    if tracking == True:
        track_IDs = ID_selection(runfolder,snapprefix=snapprefix,track_R=track_R,zoom='planet')
    else:
        track_IDs = []

    #----------- Load saved files or rerun rendering routine ----------#
    try:
        if (rerun==True) or (movie==False):    
            1/0
        else:
            print savedir+runfolder.rstrip('//')+'_'+p_type+zoom_str+overlay_str+angle_str+'_movie.npy'
            save_array = np.load(savedir+runfolder.rstrip('//')+'_'+p_type+zoom_str+overlay_str+angle_str+'_movie.npy')

    except:
        print 'No file currently exists, must compute simulation render!'
        save_array = np.zeros((num_snaps,5,ngrid+1,ngrid))

        for i in range(len(snapids)):
            idn = snapids[i]
            print 'i',idn
            store,dust_store   = render(runfolder,snapprefix=snapprefix,snapid=idn,p_type=p_type,
                                        inc=inc,azi=azi,inc2=inc2,
                                        render_mode=render_mode,zoom=zoom,track_IDs=track_IDs,
                                        polyzoom=polyzoom,overlay=overlay,h_hist=h_hist,it=it)
            save_array[i,0,0,:]  = store[0,0,:]                      #Header info
            save_array[i,0,1:,:] = store[0,1:,:] * code_M / code_L**2        #Sigma [g/cm^2]
            save_array[i,1,1:,:] = store[1,1:,:] * code_L**2 / code_time**2  #u [cm^2/s^2]
            save_array[i,2,1:,:] = store[2,1:,:] * code_L / code_time        #vx [cm/s]
            save_array[i,3,1:,:] = store[3,1:,:] * code_L / code_time        #vy [cm/s]

            #==== Add image floor to remove low density errors ====#
            save_array[i,0,1:,:][save_array[i,0,1:,:] > roof] = floor
            save_array[i,0,1:,:][save_array[i,0,1:,:] < floor] = floor
            print 'Min/Max: ', np.min(save_array[i,0,1:,:]), np.max(save_array[i,0,1:,:])
            print '\n'


            
            #==== Add dust_overlay info ====#
            if overlay == True:
                save_array[i,0,1:,:] = array_dust_overlay(save_array[i,0,1:,:],dust_store*AU_scale,box_lim) 
                save_array[i,1,1:,:] = array_dust_overlay(save_array[i,1,1:,:],dust_store*AU_scale,box_lim)
                
        #==== Save output ====#
        if snap == -1:
            try:
                print 'try removing file'
                print savedir+runfolder.rstrip('//')+'_'+p_type+zoom_str+overlay_str+angle_str+'_movie.npy'
                os.remove(savedir+runfolder.rstrip('//')+'_'+p_type+zoom_str+overlay_str+angle_str+'_movie.npy')
                print 'file removed'
            except:
                print 'pass'
            print 'Starting save!'
            np.save(savedir+runfolder.rstrip('//')+'_'+p_type+zoom_str+overlay_str+angle_str+'_movie',save_array)
    print 'Loading Complete'


    

    #------------------======== Load Header info =========----------------#
    M_star      = save_array[:,0,0,0]               #code_M
    MP          = save_array[:,0,0,2]               #code_M
    NP          = save_array[:,0,0,1]               #Number of planets
    p_pos       = save_array[:,0,0,3:6] * AU_scale  #AU
    p_sep       = np.sqrt(p_pos[:,0]**2+p_pos[:,1]**2+p_pos[:,2]**2)
    RH          = b.hill_radius(p_sep,MP,M_star)  #AU
    Sigma       = save_array[:,0,1:,:] + floor
    bonus_field = save_array[:,1,1:,:] 
    vx_field    = save_array[:,2,1:,:]
    vy_field    = save_array[:,3,1:,:]


    NP_max = np.max(NP)
    print 'Max Number of planets: ', NP_max
    p_poss = save_array[:,0,0,2:6+4*(NP_max-1)]*AU_scale
    p_poss = np.reshape(p_poss,(num_snaps,NP_max,4))
    p_poss[p_poss==0] = 1e5

    #====================== Determine plot field ========================#
    if field_mode == 'Sigma':
        plot_field = Sigma
        cax_label = r'$\Sigma \quad \rm{[g cm^{-2}}]$'
    if field_mode == 'rho':
        plot_field = Sigma/code_L
        cax_label= r'$\rho \quad \rm{[g cm^{-3}}]$'
    else:
        Om_K,Rs = calc_Om_K_field(M_star,zoom=zoom,p_pos=p_pos)
        if field_mode == 'tau':
            plot_field = calc_tau(u=bonus_field,Om_K=Om_K,Sigma=Sigma,render_mode=render_mode)
            cax_label = r'$\tau$'
        elif field_mode == 'Q':
            if zoom == True:
                Om_K,Rs = calc_Om_K_field(MP,zoom=zoom,p_pos=[0,0],render_mode=render_mode)
            plot_field = calc_Q(u=bonus_field,Om_K=Om_K,Sigma=Sigma,Rs=Rs)
            cax_label = 'Toomre Q'
        elif field_mode == 'temp':
            plot_field = calc_temp(u=bonus_field,Sigma=Sigma,Om_K=Om_K,render_mode=render_mode)
            cax_label = 'T [K]'

            
    #========================= Plotting Code ========================#
    #-=========================-------------========================-#
    fig1 = plt.figure(it,facecolor='white',figsize=(8,8))
    ax1  = fig1.add_axes([0.15,0.2,0.76,0.76])
    cax  = fig1.add_axes([0.15,0.08,0.76,0.02])

    smoothed = ax1.imshow(plot_field[0],interpolation='none',norm=LogNorm(vmin=vmin,vmax=vmax),
                              cmap=map_col,extent=[-box_lim,box_lim,-box_lim,box_lim],origin='lower')

    
    plt.colorbar(smoothed,cax=cax,orientation='horizontal')    
    ax1.set_xlabel('x [AU]')
    ax1.set_ylabel('y [AU]')
    ax1.set_xlim(-box_lim,box_lim)
    ax1.set_ylim(-box_lim,box_lim)

    if inc == 90:
        ax1.set_ylabel('z [AU]')

    #==== Plot sinks ====#
    print np.shape(p_poss)
    planets = ax1.scatter(p_poss[0,:,1],p_poss[0,:,2])
        
        
    #Get parameter plotting strings
    beta_string = beta_stringify(runfolder)
    grain_string = grain_stringify(runfolder) + ' cm'
    tbox = dict(facecolor='white')
    ax1.set_rasterized(True)
    #ax1.text(0.04,1.015,mode_name + r',  $\beta$ = ' + beta_string, transform=ax1.transAxes)
    ax1.text(0.06,0.9,r'$\beta$ = ' + beta_string + '\n' + mode_name, transform=ax1.transAxes, bbox=tbox)

    timetext = ax1.text(0.75,1.015,'Time: {:.0f} '.format(snapids[0]*snap_dt) + ' Yrs', transform=ax1.transAxes)
    cax.set_xlabel(cax_label)

    if vel_field == True:
        n_v = 32#int(ngrid/50)
        dv = int(ngrid/n_v)
        vel_grid = np.linspace(-box_lim,box_lim,n_v)
        Quivers = ax1.quiver(vel_grid,vel_grid,vx_field[0,::dv,::dv],vy_field[0,::dv,::dv], angles='xy')
    
    if zoom == True:
        Hill_radius = plt.Circle((0, 0), RH[0], fc='none', ec='#ffffff')
        Half_Hill_radius = plt.Circle((0, 0), RH[0]/2, fc='none', ec='#ffffff', ls='--')
        ax1.add_patch(Hill_radius)
        ax1.add_patch(Half_Hill_radius)

   
    #==== Movie making code ====#
    if movie == True:
        def animate(i):
            #print '2222', contours

            returns = []
            smoothed.set_array(plot_field[i])
            timetext.set_text('Time: {:.0f} '.format(int(i*snap_dt)) + ' Yrs')
            returns.append([smoothed,timetext])

            planets.set_offsets(p_poss[i,:,1:3])
            returns.append(planets)
            
            if vel_field == True:
                Quivers.set_UVC(vx_field[i,::dv,::dv],vy_field[i,::dv,::dv])
            
            if zoom == True:
                Hill_radius.set_radius(RH[i])
                Half_Hill_radius.set_radius(RH[i]/2)
                returns.append([Hill_radius,Half_Hill_radius])

            return returns

        print 'reached animation'
        ani = animation.FuncAnimation(fig1, animate, interval=200, frames=num_snaps, blit=False, repeat=True)

        if write == True:
            print 'Writing savefile'
            writer = animation.writers['ffmpeg'](fps=5)
            ani.save(runfolder.strip('//')+'_'+p_type+zoom_str+overlay_str+angle_str+field_mode+'.mp4',writer=writer,dpi=dpi)
        plt.show()

            
    #==== If no movie, save image ====#
    else:
        if write == True:
            fig1.savefig(runfolder.strip('//')+'_'+p_type+zoom_str+overlay_str+angle_str+field_mode+'_'+str(snap).zfill(3)+'.pdf',dpi=dpi)
            
    print 'Runtime: ', time.time()-time1
    return




if __name__ == "__main__":
    
    fol = 0

    #Movies
    #render_SPH(runfolders[fol],gas_args,snap=-1,inc=0,write=False,overlay=True,polyzoom=False,it=1,rerun=False,vel_field=True)
    #render_SPH(runfolders[fol],gas_args,snap=-1,inc=0,write=False,overlay=True,zoom=True,it=1,rerun=True,vel_field=True)    
    #render_SPH(runfolders[fol],temp_args,snap=-1,inc=0,write=False,overlay=True,zoom=True,it=1,rerun=True,vel_field=True)

    #render_SPH(runfolders[fol],dust_args,snap=-1,inc=0,write=False,polyzoom=True,it=1,rerun=True,vel_field=True,upto=5)

    render_SPH(runfolders[fol],gas_args,snap=-1,inc=0,write=False,overlay=True,zoom=False,it=1,rerun=True,vel_field=True)    


    #Rendered Images   
    #render_SPH(runfolders[fol],gas_args,snap=2,inc=0,polyzoom=True,overlay=True,write=False,it=2,vel_field=True)
   


    plt.show()


    
