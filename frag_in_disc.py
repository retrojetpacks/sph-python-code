'''Investigating disc fragments'''

from __future__ import division
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import os
from pygadgetreader import *
import glob


#Font styles, sizes and weights
def set_rcparams(fsize=14):
    font = {'weight' : 'normal',
            'size'   : fsize,
            'serif' : 'Minion Pro'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    plt.rc('lines',linewidth = 2)
    
    return
set_rcparams()


#Code Units
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
code_G     = c.G /code_L**3*code_M*code_time**2
AU_scale   = 100




snapprefix = 'snapshot_'
savedir = '/rfs/TAG/rjh73/save_data/'
#run_cols = ['#cd6622','#ffc82e','#440044','#0055ff','#666666']
run_cols = ['#2EC4C6','#CD6622','#440044','#FFC82E','#FF1493','#6a5acd','#cd6622','#ffc82e','#0055ff']


filepath = '/rfs/TAG/rjh73/Frag_in_disc/'
'''
runfolders = [#'Poly_N100000_M0003_R003_n15/',
              #'Disc_N3e5_R00120_M0009_b5/',
    #'Polydisc_N4e5_M3_R3_a1_r60_b5_np8/', #Z1 MD001
    #'Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z100/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a01_MD001_Z10/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a10_MD001_Z10/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a1_MD01_Z10/'
    #'post_vfrag/Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10_s1e4/',
    #'post_vfrag/Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10_s5e4/',
    #'post_vfrag/Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10/'
]
''''''
runfolders = [#'Disc_N15e5_R00120_M0075_b5/',
              #'Poly_N100000_M0005_R003_n15/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a001/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a01/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a10/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a001_1/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a001_1_sink1e11/'
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a10_vf10/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a1_vf10/',
    'Disc_N15e5_R00110_M0075_b5_quick/'
]
''''''
runfolders = ['Poly_N1e5_M5_r30/',
              'Poly_N1e5_M5_r40/',
              'Poly_N1e5_M5_r50/',
              'Poly_N1e5_M5_r60/',
]'''


runfolders = [#'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a01/',
              #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1/',
              #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a10/',
              #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a001_1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e10/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e11/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e12/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e10_res/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e11_res/',
    #Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e12_res/'
    #'Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10_a1/',
    #'Polydisc_N13e6_M5_R3_r50_b5/',
]

runfolders = [#'Poly_N800000_M0005_R003_n25/',
              #'Poly_N100000_M0005_R003_n25/',
              #'Disc_N12e6_R00110_M0075/',
              #'Disc_N15e5_R00110_M0075/',
              #'Disc_N15e5_R00110_M0075_g75/',
    #'Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10_a1_Rin5/',
    #'Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_soft2e3/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_soft2e3/',
    
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01/',
    
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD01_Z10_a01_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD01_Z10_a1_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD01_Z10_a10_df01/',
    
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e11/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e9/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e11_lowres/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_sink1e10/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_sink1e11/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_sink1e10/',

    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_lowres/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_hires/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_shires/',
    ]             

'''
runfolders = [#'Poly_N20000_M0001_R001_n25/',
              #'Poly_N60000_M0003_R002_n25/',
              #'Poly_N100000_M0005_R003_n25/',
              'Polydisc_N16e5_M5_R3_r50_b5_g75/',
              'Polydisc_N16e5_M5_R3_r75_b5_g75/',
    #'Polydisc_N16e5_M3_R2_r50_b5_g75/',
    #'Polydisc_N16e5_M3_R2_r75_b5_g75/',
    'Polydisc_N16e5_M1_R1_r50_b5_g75/',
    'Polydisc_N16e5_M1_R1_r75_b5_g75/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_shires/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M3_R2_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M1_R1_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
]'''


runfolders = [#'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
              #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M3_R2_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M1_R1_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1/'
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_0718/',

    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf01/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf1/'
]
'''
runfolders = ['Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf01/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf1/',
              #'Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_bf01/',
              #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_bf1/',
]'''


def frag_plot(rerun=False,Z_frac=False):
    '''Plot frag a, frag mass and accreted dust mass'''

    fig11 = plt.figure(11,facecolor='white',figsize=(3.5,8))
    ax1 = fig11.add_axes([0.2,0.66,0.75,0.29])
    ax2 = fig11.add_axes([0.2,0.37,0.75,0.29],sharex=ax1)
    ax3 = fig11.add_axes([0.2,0.08,0.75,0.29],sharex=ax1)
    top, mid, bot = [ax1], [ax2], [ax3]
    for i in range(len(runfolders)):
        print 'Runfolder', runfolders[i]
        try:
            if rerun == True:
                1/0
            load = np.load(savedir+runfolders[i].rstrip('//')+'_store.npy')
            print 'Loaded data'
        except:
            print 'No file, need to compute save array'
            load = b.bin_data_save(filepath,runfolders[i],
                                   Rin=0.1,Rout=2,Poly_data=True)
            
        #=== Read Gadget output from saved np array ===#
        time      = load[:,0,0] #Years
        M_star    = load[:,0,1] #Msol        
        MH_2      = load[:,0,21]*c.Msol/c.MJ #MJ inside RH_2
        rP        = load[:,0,22]*AU_scale #AU
        RH_2      = load[:,0,23] #AU
        Macc_dust = load[:,0,24]*c.Msol/c.MJ #MJ inside RH_2
        Z_gas     = load[:,0,1]*AU_scale
        Z_dust    = load[:,0,2]*AU_scale
        
        ax1.plot(time,rP,color=run_cols[i],label=runfolders[i])
        ax2.plot(time,MH_2,color=run_cols[i],label='M$_{RH/2}$')
        ax2.plot(time,Z_gas,color=run_cols[i],label='Z$_{gas}$',ls='--')
        ax2.plot(time,Z_dust,color=run_cols[i],label='Z$_{dust}$',ls=':')

        #==== Find sink masses ====#
        N_planets = load[0,0,20]
        print 'N Planets', N_planets
        M_bigsink    = load[:,0,25]*c.Msol/c.ME
        M_othersinks = np.sum(load[:,0,27::2],axis=1)*c.Msol/c.ME
        
        if Z_frac == True:
            Z_disc = runfolders[i].split('MD')[1].split('_')[0]
            Z_disc = float('0.'+Z_disc[1:])
            Z = (Macc_dust+(M_bigsink+M_othersinks)*c.ME/c.MJ) / MH_2
            ax3.plot(time,Z,color=run_cols[i])
            ax3.set_ylabel(r'Z composition of clump')
            #ax3.set_ylabel(r'% of Total Dust Accreted')

        else:
            M_dust = Macc_dust*c.MJ/c.ME
            ax3.plot(time,M_dust,ls='--',color=run_cols[i],label='Dust mass')
            ax3.set_ylabel(r'Dust Mass [M$_\oplus$]')

            #==== Plot sink masses ====#
            ax3.plot(time,M_bigsink,label='Big sink',color=run_cols[i],ls='-.')
            ax3.plot(time,M_othersinks,label='other sink masses',color=run_cols[i],ls=':')
            ax3.plot(time,M_bigsink+M_othersinks+M_dust,label='Total Z mass',color=run_cols[i])

            
        if i == 0:
            ax2.legend(frameon=False,loc=2)
            ax3.legend(frameon=False,loc=2)

    ax1.set_ylabel(r'Orbital Sep [AU]')
    ax2.set_ylabel(r'Mass inside RH/2 [M$_J$]')
    ax3.set_xlabel('Time [Years]')
    #if betaval != 'nan':
    #    ax1.text(0.75,0.9,r'$\beta$ = '+str(betaval), transform=ax1.transAxes)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.legend(frameon=False,loc=3)
    return



def feedback_plot():
    '''Calculate the fraction of feedback energy on a planet'''

    xi = 1
    mdot_t = np.logspace(26,30,100) #g
    rho_c = 5
    M_f = 5*c.MJ
    R_f1 = 0.5*c.AU
    R_f2 = 3*c.AU

    tmp = xi * (rho_c*4*np.pi/3)**(1/3)  * (mdot_t)**(5/3) /M_f**2
    E_U1 = tmp * R_f1
    E_U2 = tmp * R_f2

    fig1 = plt.figure(4,facecolor='w')
    ax1 = fig1.add_axes([0.15,0.15,0.8,0.8])
    ax1.plot(mdot_t/c.ME,E_U1,label=r'$R_f$= 0.5 AU',color=run_cols[0])
    ax1.plot(mdot_t/c.ME,E_U2,label=r'$R_f$= 3 AU',color=run_cols[1])
    ax1.axhline(1,color=run_cols[2],ls='--')
    ax1.semilogx()
    ax1.semilogy()
    ax1.set_xlabel(r'M$_{core}$ [$M_\oplus$]')
    ax1.set_ylabel(r'$E_{fb}/U$')
    plt.legend(frameon=False,loc=2)
    plt.show()
    
    
def dust_sedimentation(filepath,runfolders,snapid):
    '''Check sedimentation velocities inside fragments'''
    
    #==== Radial binning ====#
    Rin=0.001
    Rout=0.05
    N_Rbins  = 50
    Rbins = np.linspace(Rin,Rout,N_Rbins+1)
    Rbin_mids = (Rbins[1:]+Rbins[:-1])/2

    plt.figure(0,facecolor='w')
    shades = []
    for i in range(len(runfolders)):
        S = b.Load_Snap(filepath,runfolders[i],snapid)
        M_gas  = S.M_gas
        M_dust = S.M_dust

        #Zoom on clump
        rho_sort  = np.argsort(S.gas_rho)
        zoom_pos = np.mean(S.gas_pos[rho_sort[-10:],:],axis=0)
        zoom_vel = np.mean(S.gas_vel[rho_sort[-10:],:],axis=0)

        S.gas_pos  = S.gas_pos  - zoom_pos
        S.dust_pos = S.dust_pos - zoom_pos
        S.dust_vel = S.dust_vel - zoom_vel
        S.gas_vel  = S.gas_vel  - zoom_vel

        r_gas  = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
        r_dust = np.sqrt(S.dust_pos[:,0]**2+S.dust_pos[:,1]**2+S.dust_pos[:,2]**2)

        gas_count    = np.histogram(r_gas,Rbins)[0]
        dust_count   = np.histogram(r_dust,Rbins)[0]
        M_enc        = (np.cumsum(gas_count)*M_gas + np.cumsum(dust_count)*M_dust )
        gas_u        = (b.calc_binned_data(S.gas_u,r_gas,Rbins)[0])
        gas_cs       = np.sqrt((c.gamma_dia-1)*gas_u)
        v_th         = np.sqrt(8/np.pi)*gas_cs
        
        Rbin_volumes = 4*np.pi/3 * (Rbins[1:]**3-Rbins[:-1]**3)
        gas_rho      = gas_count*M_gas/Rbin_volumes

        a     = np.mean(S.dust_a) /code_L
        rho_a = 3 /code_M*code_L**3
        v_sed = a*rho_a * code_G * M_enc / (v_th * (Rbin_mids)**2 *gas_rho)


        #dust velocities
        v_r, v_azi = b.v_r_azi(S.dust_pos,S.dust_vel)
        tmp     = b.calc_binned_data(v_r,r_dust,Rbins)
        dust_vr,sig_dust_vr = tmp[0],tmp[1]
        pvr   = -dust_vr
        pvr_p = (-dust_vr + sig_dust_vr)
        pvr_m = (-dust_vr - sig_dust_vr)

        #gas velocities
        v_r, v_azi = b.v_r_azi(S.gas_pos,S.gas_vel)
        tmp     = b.calc_binned_data(v_r,r_gas,Rbins)
        gas_vr,sig_gas_vr = -tmp[0], tmp[1]


        #==== Plotting code ====#
        plot_units = code_time/c.sec_per_year #code_L/code_time #cm/s
        #plot_units = 100/code_time*c.sec_per_year #AU/year
        
        #plt.scatter(r_dust,v_r)
        #plt.scatter(r_dust,S.dust_vel[:,1],s=1)
        #fill_min = np.maximum(pvr_m*plot_units,np.ones(len(pvr_m))*1e-5)
        #plt.fill_between(Rbin_mids*100,pvr_p*plot_units,fill_min,alpha=0.2,color=run_cols[i],label = runfolders[i])
        plt.plot(Rbin_mids*100,Rbin_mids/v_sed*plot_units,color=run_cols[i],ls='--')#r'an $V_{set}$'
        plt.plot(Rbin_mids*100,Rbin_mids/pvr*plot_units,label=runfolders[i],color=run_cols[i])#r'Mean dust $V_r$'
        plt.plot(Rbin_mids*100,Rbin_mids/gas_vr*plot_units,color=run_cols[i],ls=':')#label=r'Gas $V_r$'
        plt.plot(Rbin_mids*100,Rbin_mids/(pvr-gas_vr)*plot_units,color=run_cols[i],ls='-.')#r'Dust-Gas $V_r$'
        #plt.plot(Rbin_mids*100,gas_cs*plot_units,label=r'Gas c_s',ls='-.',color=run_cols[i])

    plt.plot([],[],color=run_cols[0],label=r'Mean dust $t_{set}$')
    plt.plot([],[],color=run_cols[0],label=r'an $t_{set}$',ls='--')
    plt.plot([],[],color=run_cols[i],label=r'Dust-Gas $t_{set}$',ls='-.')
    plt.plot([],[],color=run_cols[i],label=r'Gas $t_{set}$',ls=':')

    plt.legend(loc=2,frameon=False)
    
    plt.yscale('symlog', linthreshy=10)
    plt.xlabel('Fragment radius [AU]')
    plt.ylabel(r'$t_{set}$ [Years]')#'cm s$^{-1}$')
    #plt.semilogy()
    return

    


if __name__ == "__main__":
    
    #feedback_plot()
    frag_plot(rerun=False)#,Z_frac=True)
    #b.animate_1d(filepath,runfolders,var2='T',Rin=0.1,Rout=1.2,rerun=False,norm_y=False,write=False)

    #b.animate_1d(filepath,runfolders,var1='rho',var2='dust_gas',Rin=0.001,Rout=0.05,rerun=False,norm_y=False,zoom='Zrho',write=False)
    
    #dust_sedimentation(filepath,runfolders,30)


    plt.show()



    
