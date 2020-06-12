# -*- coding: utf-8 -*-
"""
Created on Mon Oct. 09, 2017

@author: Zhibo Zhang
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from PDA_Lib import PDA_Util as plib
import os
import matplotlib as mpl
from matplotlib import rc
import netCDF4
import PSD

def scat_ang(Theta0,Thetav,Phi0,Phiv):
	'''
	Inputs:
	Theta0: solar zenith angle > 90 [degree]
	Thetav: viewing zenith angle < 90
	Phi0,Phiv: solar and viewing azimuth [degree]
	'''
	Mu0 = np.cos(np.radians(Theta0))
	Muv = np.cos(np.radians(Thetav))
	Sin0 = np.sin(np.radians(Theta0))
	Sinv = np.sin(np.radians(Thetav))
	dPhi =  Phi0 - Phiv
	CdPhi = np.cos(np.radians(dPhi))
	SdPhi = np.sin(np.radians(dPhi))

	Mus=Mu0*Muv + Sin0*Sinv*CdPhi      # compute scattering angle

	return np.degrees(np.arccos(Mus))

def read_mie_file():
    #%%
    ncf = netCDF4.Dataset('/Users/zhibo/Polarization_Wiley/Book_content/Programs/aerosol_mie.nc','r')
    Diameter=np.array(ncf.variables['Diameter']) #actually size parameter
    N_r     =np.array(ncf.variables['Refr_real']) #real refractive indexes
    N_i     =np.array(ncf.variables['Refr_img']) #imaginary refractive indexes
    Angle   =np.array(ncf.variables['PhaseFunctionAngle']) #Scattering angle
    SSA     =np.array(ncf.variables['SingleScatteringAlbedo']) #single scattering albedo
    EE      =np.array(ncf.variables['ExtinctionEfficiency']) #Extinction effciency
    G       =np.array(ncf.variables['AsymmetryFactor']) #Asymmetry factor
    P11     =np.array(ncf.variables['P11']) #phase functions
    P12     =np.array(ncf.variables['P12'])
    P33     =np.array(ncf.variables['P33'])
    P34     =np.array(ncf.variables['P34'])

    P12 = P12*P11
    P33 = P33*P11
    P34 = P34*P11
    N_r = N_r.reshape(20,30)[:,0]
    N_i = N_i.reshape(20,30)[0,:]
    SSA = SSA.reshape(1000,20,30)
    EE  = EE.reshape(1000,20,30)
    G   = G.reshape(1000,20,30)
    P11 = P11.reshape(1000,20,30,181)
    P12 = P12.reshape(1000,20,30,181)
    P33 = P33.reshape(1000,20,30,181)
    P34 = P34.reshape(1000,20,30,181)

    #%%
    return np.array(Diameter), np.array(N_r), np.array(N_i), \
           np.array(Angle),    np.array(SSA), np.array(EE),  \
           np.array(G),        np.array(P11), np.array(P12), \
           np.array(P33),      np.array(P34)

class Aerosol(object):
    def __init__(self, R,  N_f,     Rg_f,     S_f, \
                       N_c=0.0, Rg_c=1.5, S_c=0.3,\
                       Density=1.0, PSD_base = np.e):

        '''
        Inputs:
        N_f, Rg_f, S_f: fine   mode number conentration [#/cm^3], median radius [um] and standard deviation [um]
        N_c, Rg_c, S_c: coarse mode number conentration [#/cm^3], median radius [um] and standard deviation [um]
        Density: Density of the aerosol [g / cm^3]
        PSD_base = base of logarithm for lognormal PSD computation
        rmin, rmax: min and max radius of aerosol
        '''

        self.R    = R
        self.N_f  = N_f
        self.Rg_f = Rg_f
        self.S_f  = S_f

        self.N_c  = N_c
        self.Rg_c = Rg_c
        self.S_c  = S_c
        self.Density = Density
        self.PSD_base = PSD_base

        dNdlnrf = PSD.dNdlnr_Lognormal(self.R, np.log(self.Rg_f)/np.log(self.PSD_base), self.S_f, \
                                       N0=self.N_f, normalize=True, base=self.PSD_base)

        dNdlnrc = PSD.dNdlnr_Lognormal(self.R, np.log(self.Rg_c)/np.log(self.PSD_base), self.S_c, \
                                       N0=self.N_c, normalize=True, base=self.PSD_base)
        self.dNdlnr = dNdlnrf +  dNdlnrc
        self.dNdr   = 1.0/self.R * self.dNdlnr
        self.dVdlnr = 4.0/3.0*np.pi * (self.R **3) * self.dNdlnr
        self.dAdlnr =         np.pi * (self.R **2) * self.dNdlnr
        self.dMdlnr = self.dVdlnr * self.Density

    def get_ref_index(self, Wavelength, Refr_real, Refr_img):
        """
        Inputs:
        Wavelength: incident wavelength [um]
        Refr_real, Refr_img: real and imaginary part of the aerosol w.r.t. the incident wavelengths

         """
        self.Ref_wvl   = np.array(Wavelength)
        self.Refr_real = np.array(Refr_real)
        self.Refr_imag  = np.array(Refr_img)

    def get_scat_prop(self, sizep, nr, ni, ang, alb, qe, gf, P11, P12, P33, P34):

        """
        Inputs: Scattering property database
        sizep: size parameter
        nr:    real part of refractice index
        ni:    imaginary of refractive index
        G :    projected area
        ang:   scattering angle
        alb:   single scattering albedoe
        qe:    exintciton efficiency
        gf:    asymmetry factor
        P11, P12: phase functions

        """
        G_all   = []
        qe_all  = []
        alb_all = []
        P11_all = []
        P22_all = []
        P33_all = []
        P44_all = []
        P12_all = []
        P34_all = []

        for iwvl in range(self.Ref_wvl.size):
            r=sizep*self.Ref_wvl[iwvl]/2.0/np.pi
            dNdr = np.interp(r, self.R, self.dNdr)
            nr_index = np.where(nr >=self.Refr_real[iwvl])[0][0]
            ni_index = np.where(ni >=self.Refr_imag[iwvl])[0][0]
            qe_sub  = np.squeeze( qe[:,nr_index,ni_index])
            alb_sub = np.squeeze(alb[:,nr_index,ni_index])

            P11_sub = np.squeeze(P11[:,nr_index,ni_index,:])
            P12_sub = np.squeeze(P12[:,nr_index,ni_index,:])
            P33_sub = np.squeeze(P33[:,nr_index,ni_index,:])
            P34_sub = np.squeeze(P34[:,nr_index,ni_index,:])

            G_avg, qe_avg, alb_avg, P11_avg, P12_avg, P33_avg, P34_avg = \
            PSD.PSD_avg_sphere(r, dNdr, ang*np.pi/180.0,\
                               qe_sub,alb_sub,P11_sub,P12_sub,P33_sub,P34_sub)
            G_all   = np.append(G_all,G_avg)
            qe_all  = np.append(qe_all,qe_avg)
            alb_all = np.append(alb_all,alb_avg)
            P11_all = np.append(P11_all,P11_avg)
            P22_all = np.append(P22_all,P11_avg)
            P33_all = np.append(P33_all,P33_avg)
            P44_all = np.append(P44_all,P33_avg)

            P12_all = np.append(P12_all,P12_avg)
            P34_all = np.append(P34_all,P34_avg)

        P11_all=P11_all.reshape(self.Ref_wvl.size,ang.size)
        P22_all=P22_all.reshape(self.Ref_wvl.size,ang.size)
        P33_all=P33_all.reshape(self.Ref_wvl.size,ang.size)
        P44_all=P44_all.reshape(self.Ref_wvl.size,ang.size)
        P12_all=P12_all.reshape(self.Ref_wvl.size,ang.size)
        P34_all=P34_all.reshape(self.Ref_wvl.size,ang.size)

        self.G = G_all
        self.ang = ang
        self.Qe  = qe_all
        self.Alb = alb_all
        self.P11 = P11_all
        self.P22 = P22_all
        self.P33 = P33_all
        self.P44 = P44_all

        self.P12 = P12_all
        self.P34 = P34_all

def Rayleigh_Case():
    par   = plib.PDA_Parameters(90.0, 0.866)
    wvl   = plib.PDA_Wavelength(1,[0.46])
    atmos = plib.PDA_Atmos(1, [1013.0], wvl)
    surf  = plib.PDA_Surface(wvl, [0.00], ['Black_Surf'], ['Rayleigh_0.46'])
    aer   = plib.PDA_Aerosol(1, [1], [10.0], [0.03], [1], [60.0])
    aer.get_Ref_Index(wvl,[[1.33]],[[0.0]],[['benchmark_aerosol.PDA']])
    aer.get_Loading(atmos,[[0.0]])
    plib.Write_Drive_File('Rayleigh_0.46.info', par, wvl,atmos, surf, aer)

    os.system("PDA_new ./ Rayleigh_0.46.info ./")

    maxview,phis,xmus,nview,thetav,\
		rv11,rv21,rv31,rsrf11,rsrf21,rsrf31=plib.read_rsp_ref('Rayleigh_0.46.rsp')

    fig,ax=plt.subplots()
    ax.plot(thetav,rv31)

    os.system("PDA_interp_new ./ Rayleigh_0.46 ./ 30 30 5 0 360 10")

    theta0 = 30
    phi0 = np.arange(360,-10,-10)

    RI = []
    RQ = []
    RU = []

    RI_inv = []
    RQ_inv = []
    RU_inv = []

    VZA =[]
    RAA =[]
    for p in phi0:
        fn = 'Rayleigh_0.46_F{:03d}M{:03d}.azi'.format(p,theta0)
        maxview,phis,xmus,nview,thetav,\
        rv11,rv21,rv31,rsrf11,rsrf21,rsrf31=plib.read_rsp_ref(fn)
        nview_half = np.round(nview/2).astype(int)
        VZA = np.array(thetav[0:nview_half])
        #print(thetav)
        #print(VZA)
        #print(thetav[nview/2::-1])
        RAA = np.append(RAA,p)
        RI  = np.append(RI,rv11[0:nview_half])
        RQ  = np.append(RQ,rv21[0:nview_half])
        RU  = np.append(RU,rv31[0:nview_half])
        RI_inv = np.append(RI_inv,rv11[nview_half:][::-1])
        RQ_inv = np.append(RQ_inv,rv21[nview_half:][::-1])
        RU_inv = np.append(RU_inv,rv31[nview_half:][::-1])
        print(RI.shape, RI_inv.shape)

    RI = RI.reshape(phi0.size,nview_half)
    RQ = RQ.reshape(phi0.size,nview_half)
    RU = RU.reshape(phi0.size,nview_half)

    RI_inv = RI_inv.reshape(phi0.size,nview_half)
    RQ_inv = RQ_inv.reshape(phi0.size,nview_half)
    RU_inv = RU_inv.reshape(phi0.size,nview_half)
    os.system("rm Rayleigh_0.46*.azi")

    cmap=plt.cm.rainbow
    r, t = np.meshgrid(VZA, np.radians(RAA))
    fig, ax = plt.subplots(2,2, figsize=[10,10],\
		                   subplot_kw=dict(projection='polar'))
    plt.rc('text', usetex=False)

    Nlevel=100
    #ax[0,0]= plt.subplots(subplot_kw=dict(projection='polar'))
    ax[0,0].set_theta_zero_location("E")
    cax = ax[0,0].contourf(t,r, RI_inv, Nlevel,cmap=cmap)
    cb = fig.colorbar(cax,ax=ax[0,0])
    cb.set_label(r"$R_I$")

    ax[0,1].set_theta_zero_location("E")
    cax = ax[0,1].contourf(t,r, np.sqrt(RU_inv**2 + RQ_inv**2)/RI_inv, Nlevel,cmap=cmap)
    cb = fig.colorbar(cax,ax=ax[0,1])
    cb.set_label(r"DoLP $\sqrt{Q^2 + U^2} / I$")

    ax[1,0].set_theta_zero_location("E")
    cax = ax[1,0].contourf(t,r, RQ_inv, Nlevel,cmap=cmap)
    cb = fig.colorbar(cax,ax=ax[1,0])
    cb.set_label(r"$R_Q$")

    ax[1,1].set_theta_zero_location("E")
    cax = ax[1,1].contourf(t,r, RU_inv, Nlevel,cmap=cmap)
    cb = fig.colorbar(cax,ax=ax[1,1])
    cb.set_label(r"$R_U$")

    #plt.savefig('Rayleigh_Case_wvl_0.46_thetha0_45_phi0_0.0.png',dpi=500)
    plt.show()

def Water_Case():

	par   = plib.PDA_Parameters(0.0, 0.5)
	wvl   = plib.PDA_Wavelength(3,[0.46, 0.64, 0.86])
	atmos = plib.PDA_Atmos(3, [200.0,1.0,800.0], wvl)
	surf  = plib.PDA_Surface(wvl, [0.00,0.0,0.0], \
		                          ['Black_Surf1','Black_Surf2','Black_Surf3'], \
		                          ['water_0.46','water_0.64','water_0.86'])
	aer   = plib.PDA_Aerosol(1, [1], [10.0], [0.02], [1], [60.0])
	aer.get_Ref_Index(wvl,[[1.342],[1.332],[1.324]],[[0.0],[0.0],[0.0]],\
		                   [['a1'],['a2'],['a3']])
	aer.get_Loading(atmos,[[0.0,0.01,0.0]])
	print(aer.NDZ1.shape)

	plib.Write_Drive_File('water_3wvls.info', par, wvl,atmos, surf, aer)

	# os.system("PDA_new ./ water_3wvls.info ./ >& PDA_water_case.out")

	os.system("PDA_interp_new ./ water_0.46 ./ 30 30 5 0 360 2")
	os.system("PDA_interp_new ./ water_0.64 ./ 30 30 5 0 360 2")
	os.system("PDA_interp_new ./ water_0.86 ./ 30 30 5 0 360 2")

	theta0 = 30
	phi0 = np.arange(360,-2,-2)
	#phi0 = np.arange(0,360,5)

	RI = []
	RQ = []
	RU = []
	VZA =[]
	RAA =[]
	for p in phi0:
		fn = 'water_0.46_F{:03d}M{:03d}.azi'.format(p,theta0)
		maxview,phis,xmus,nview,thetav,\
		rv11,rv21,rv31,rsrf11,rsrf21,rsrf31=plib.read_rsp_ref(fn)
		VZA = np.array(thetav[0:nview/2])
		RAA = np.append(RAA,p)
		RI  = np.append(RI,rv11[0:nview/2])
		RQ  = np.append(RQ,rv21[0:nview/2])
		RU  = np.append(RU,rv31[0:nview/2])

	RI46 = RI.reshape(phi0.size,np.round(nview/2.0))
	RQ46 = RQ.reshape(phi0.size,np.round(nview/2.0))
	RU46 = RU.reshape(phi0.size,np.round(nview/2.0))

	RI = []
	RQ = []
	RU = []
	VZA =[]
	RAA =[]
	for p in phi0:
		fn = 'water_0.64_F{:03d}M{:03d}.azi'.format(p,theta0)
		maxview,phis,xmus,nview,thetav,\
		rv11,rv21,rv31,rsrf11,rsrf21,rsrf31=plib.read_rsp_ref(fn)
		VZA = np.array(thetav[0:nview/2])
		RAA = np.append(RAA,p)
		RI  = np.append(RI,rv11[0:nview/2])
		RQ  = np.append(RQ,rv21[0:nview/2])
		RU  = np.append(RU,rv31[0:nview/2])

	RI64 = RI.reshape(phi0.size,np.round(nview/2.0))
	RQ64 = RQ.reshape(phi0.size,np.round(nview/2.0))
	RU64 = RU.reshape(phi0.size,np.round(nview/2.0))

	RI = []
	RQ = []
	RU = []
	VZA =[]
	RAA =[]
	for p in phi0:
		fn = 'water_0.86_F{:03d}M{:03d}.azi'.format(p,theta0)
		maxview,phis,xmus,nview,thetav,\
		rv11,rv21,rv31,rsrf11,rsrf21,rsrf31=plib.read_rsp_ref(fn)
		VZA = np.array(thetav[0:nview/2])
		RAA = np.append(RAA,p)
		RI  = np.append(RI,rv11[0:nview/2])
		RQ  = np.append(RQ,rv21[0:nview/2])
		RU  = np.append(RU,rv31[0:nview/2])

	RI86 = RI.reshape(phi0.size,np.round(nview/2.0))
	RQ86 = RQ.reshape(phi0.size,np.round(nview/2.0))
	RU86 = RU.reshape(phi0.size,np.round(nview/2.0))

	cmap=plt.cm.rainbow
	RI_color_norm = mpl.colors.Normalize(vmin=0.2,vmax=0.75)
	DoLP_color_norm = mpl.colors.Normalize(vmin=0.,vmax=0.25)

	r, t = np.meshgrid(VZA, np.radians(RAA))
	fig, ax = plt.subplots(3,2, figsize=[10,15],\
		                   subplot_kw=dict(projection='polar'))
	plt.rc('text', usetex=False)

	Nlevel=50
	#ax[0,0]= plt.subplots(subplot_kw=dict(projection='polar'))
	ax[0,0].set_theta_zero_location("N")
	cax = ax[0,0].contourf(t,r, RI46, Nlevel, norm=RI_color_norm,  cmap=cmap)
	cb = fig.colorbar(cax,ax=ax[0,0])
	cb.set_label(r"$R_I$")
	ax[0,0].set_title(r'0.46 $\mu m$')

	ax[0,1].set_theta_zero_location("N")
	cax = ax[0,1].contourf(t,r, np.sqrt(RU46**2 + RQ46**2)/RI46, Nlevel, norm= DoLP_color_norm, cmap=cmap)
	cb = fig.colorbar(cax,ax=ax[0,1])
	cb.set_label(r"DoLP $\sqrt{Q^2 + U^2} / I$")
	ax[0,1].set_title(r'0.46 $\mu m$')

	ax[1,0].set_theta_zero_location("N")
	cax = ax[1,0].contourf(t,r, RI64, Nlevel, norm=RI_color_norm, cmap=cmap)
	cb = fig.colorbar(cax,ax=ax[1,0],norm=RI_color_norm)
	cb.set_label(r"$R_I$")
	ax[1,0].set_title(r'0.64 $\mu m$')

	ax[1,1].set_theta_zero_location("N")
	cax = ax[1,1].contourf(t,r, np.sqrt(RU64**2 + RQ64**2)/RI64,  Nlevel,norm= DoLP_color_norm, cmap=cmap)
	cb = fig.colorbar(cax,ax=ax[1,1])
	cb.set_label(r"DoLP $\sqrt{Q^2 + U^2} / I$")
	ax[1,1].set_title(r'0.64 $\mu m$')

	ax[2,0].set_theta_zero_location("N")
	cax = ax[2,0].contourf(t,r, RI86, Nlevel, norm=RI_color_norm, cmap=cmap)
	cb = fig.colorbar(cax,ax=ax[2,0],norm=RI_color_norm)
	cb.set_label(r"$R_I$")
	ax[2,0].set_title(r'0.86 $\mu m$')

	ax[2,1].set_theta_zero_location("N")
	cax = ax[2,1].contourf(t,r, np.sqrt(RU86**2 + RQ86**2)/RI86, Nlevel,norm= DoLP_color_norm, cmap=cmap)
	cb = fig.colorbar(cax,ax=ax[2,1])
	cb.set_label(r"DoLP $\sqrt{Q^2 + U^2} / I$")
	ax[2,1].set_title(r'0.86 $\mu m$')

	# plt.savefig('Water_Case_thetha0_45_phi0_0.0.png',dpi=500)

	# fig,ax=plt.subplots()
	# #plt.rcParams["font.family"] = "Times New Roman"
	# #csfont = {'fontname':'Times New Roman'}
	# l1,=ax.plot(VZA,-RQ46[0,:],c='k',label=r'0.46 $\mu m$',ls='dashed');ax.plot(-VZA,-RQ46[180,:],c='k',ls='dashed')
	# l2,=ax.plot(VZA,-RQ64[0,:],c='k',label=r'0.64 $\mu m$',ls='solid'); ax.plot(-VZA,-RQ64[180,:],c='k',ls='solid')
	# l3,=ax.plot(VZA,-RQ86[0,:],c='k',label=r'0.86 $\mu m$',ls='dotted');ax.plot(-VZA,-RQ86[180,:],c='k',ls='dotted')
	# ax.legend(handles=[l1,l2,l3])
	# ax.set_xlabel(r'Scattering Angle $[ ^o ]$',fontsize='large')
	# ax.set_ylabel(r'$R_Q$',fontsize='large')
	# plt.tight_layout()
	# plt.savefig('Water_Case_thetha0_45_phi0_0.0_principle_plane.png',dpi=500)

	#fig,ax=plt.subplots()
	#ax.plot(RQ46,RU46,ls=' ',marker='.')

	fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
	plt.rc('text', usetex=False)
	ax.set_theta_zero_location("E")
	cax = ax.contourf(t,r, RU86, Nlevel,  cmap=plt.cm.RdBu_r)
	cb = fig.colorbar(cax)
	cb.set_label(r"$R_U$")

	plt.show()
	#os.system("rm *.azi")

def Aerosol_Case():
	DZ  = 3.0 # [km]
	nphys = 4
	Nf  = np.array([300.0,  300.0, 300.0, 300.0]) * (DZ*1e-3)
	Nc  = np.array([0.1,    1.0,   0.1,   1.0])   * (DZ*1e-3)
	Rf  = np.array([0.1,    0.1,   0.2,   0.2])
	Rc  = np.array([2.5,    2.5,   2.5,   2.5])   #  [um]
	Sf  = np.array([0.5,    0.5,   0.5,   0.5])
	Sc  = np.array([0.3,    0.3,   0.3,   0.3])

	nwl   = 4
	wl  = np.array([0.47,   0.47,  0.47,  0.47])
	#wl  = np.array([0.86,   0.86,  0.86,  0.86])
	mr = np.array([1.50,    1.50,  1.30,  1.30])
	mi = np.array([0.004,   0.04,  0.004, 0.04])

	Theta0 = 180.0 - 45.0
	Phi0   = 0.0
	par   = plib.PDA_Parameters(Phi0, np.cos(np.deg2rad(180.0-Theta0)))

	rv11_all = []
	rv21_all = []

	run_PDA = False

	data_path = 'Aerosol_Case_with_Rayleigh/'

	for iphys in range(nphys):
		for iwvl in range(nwl):
			print('processing physics',iphys,'wvl',iwvl)
			output_filename = 'Rf{:3.1f}_Nc{:04.1e}_wl{:4.2f}_mr{:3.1f}_mi{:05.1e}'.format(Rf[iphys],Nc[iphys],wl[iwvl],mr[iwvl],mi[iwvl])
			info_filename =   'Rf{:3.1f}_Nc{:04.1e}_wl{:4.2f}_mr{:3.1f}_mi{:05.1e}.info'.format(Rf[iphys],Nc[iphys],wl[iwvl],mr[iwvl],mi[iwvl])

			if run_PDA:
				wvl   = plib.PDA_Wavelength(1,[wl[iwvl]])
				atmos = plib.PDA_Atmos(2, [300.0, 700.0], wvl)

				aer   = plib.PDA_Aerosol(2, [3, 3], [Rf[iphys],Rc[iphys]], \
					                                [Sf[iphys],Sc[iphys]], \
					                                [0.001,0.001], [90.0,90.0])
				aer.get_Ref_Index(wvl,[[mr[iwvl],mr[iwvl]]],[[mi[iwvl], mi[iwvl]]],[['finemode','coarsemode']])
				aer.get_Loading(atmos, [[Nf[iphys], 0.0],\
			                    	    [Nc[iphys], 0.0] ])

				surf  = plib.PDA_Surface(wvl, [0.0], \
			                          ['Black_Surf1'], \
			                          [output_filename])

				plib.Write_Drive_File(info_filename, par, wvl,atmos, surf, aer)

				os.system("PDA_new ./ " +info_filename+ " ./ >& "+output_filename+".out")

			fn = output_filename+'.rsp'
			maxview,phis,xmus,nview,thetav,rv11,rv21,rv31,rsrf11,rsrf21,rsrf31=plib.read_rsp_ref(data_path+fn)
			rv11_all.append(rv11)
			rv21_all.append(rv21)

	RI = np.array(rv11_all).reshape(nphys,nwl,nview)
	RQ = np.array(rv21_all).reshape(nphys,nwl,nview)

	Phiv = np.zeros_like(thetav)
	Phiv[thetav<0.0] = 180.0
	print(Phiv)
	Thetav = np.abs(thetav)

	Sca = scat_ang(Theta0,Thetav,Phi0,Phiv)
	print(Sca)

	bw    = ['0.0', '0.0','0.5','0.5']
	syb   = ['s','^','s','^']
	lsy=['solid','dashed','dotted','dashdot']

	fig,ax=plt.subplots(1,2,figsize=[18,9])
	for j in range(nwl):
		for i in range(nphys):
			ax[0].plot(thetav,RI[i,j,:],c=bw[j],marker=syb[j],ls=lsy[i],fillstyle='none',markevery = 20,\
				label=r'$n_r$={:3.1f} $n_i$={:4.3f} $N_c$ = {:3.1f} $r_{{m,f}}$={:3.1f} '.format(mr[j],mi[j], Nc[i]/(DZ*1e-3),Rf[i]))
			ax[1].plot(Sca,-RQ[i,j,:],c=bw[j],marker=syb[j],ls=lsy[i],fillstyle='none',markevery = 20)
	ax[0].legend(loc=0)
	ax[0].set_xlim([-85,85])
	ax[0].set_xlabel(r'Viewing Zenigh Angle [$ ^o $]',fontsize='large')
	ax[0].set_ylabel(r'$R_I$',fontsize='x-large')
	ax[1].set_xlim([60,180])
	ax[1].set_xlabel(r'Scattering Angle [$ ^o $]',fontsize='large')
	ax[1].set_ylabel(r'$-R_Q$',fontsize='x-large')
	ax[0].legend(loc=0)
	#plt.savefig('Aerosol_Case_wo_Rayleigh_wvl_{:4.2f}.png'.format(wl[0]),dpi=500)
	# plt.figure()
	# for i in range(nphys):
	# 	for j in range(nwl):
	# 		plt.plot(Sca,RI[i,j,:])
	# plt.figure()
	# plt.scatter(thetav,Sca)
	plt.show()

if __name__ == '__main__':
	Rayleigh_Case()
	# Water_Case()
	#Aerosol_Case()
