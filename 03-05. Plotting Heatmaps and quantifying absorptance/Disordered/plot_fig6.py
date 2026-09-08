#!/usr/bin/python

"""
plot_fig6_N_v129_V213.py -- Fig 6 of the mode-coupling rebuttal (cleaned 2026-08-28).

Provenance:
    plot_fig6_V213.py (670 lines, original)
      -> plot_fig6_weeded_V213.py   (dead functions removed; verified pixel-perfect vs original)
      -> plot_fig6_N_v129_V213.py   (this file: grouped imports, settings section, comments,
                                     dead vars and commented-out blocks stripped)

Fig 6 layout (3-row GridSpec: heatmaps + PSD on top, power-fraction panels below):
    [0,0] / [0,2] far-field power vs (photon energy, in-plane momentum), viridis scatter,
               Si light-line background, white light line, colorbar per panel
    [0,1]     PSD of the Poisson-random and hyperuniform structures (narrow column)
    [2,0] / [2,2] power fraction below/beyond/within critical angle + Lambertian limit

Usage:
    python plot_fig6_N_v129_V213.py      (VER parsed from the filename -> "213")

Outputs (written to the current directory):
    Fig_6.pdf, Fig_6_V213.pdf, Fig_6_V213.png   (dpi from general_settings)

Dependencies: numpy, scipy, matplotlib (TeX/Arial via general_settings), PIL.
general_settings.py must sit next to this script.

Clean-pass notes (vs weeded):
    - removed dead imports (matplotlib, matplotlib.ticker, scipy.spatial.Delaunay)
    - removed never-read vars: k_target, radius, height, dics
    - removed the post-def k1,k2=[10,20] redefinition (dead: plotfigure's defaults
      already bound k1,k2=[9.3,18.7] at def time)
    - stripped commented-out code blocks; added section banners + docstrings
"""

# ────────────────────────── 1. IMPORTS ──────────────────────────
import sys
from glob import glob as gl

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, AutoMinorLocator
from scipy.interpolate import griddata
import scipy.constants as cnt
import scipy.interpolate as scinter
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # raise the decompression-bomb limit

# Shared settings: figsize/dpi/fontsize, C0/e0/Pi/h0, shade_colors, custom_formatter, TeX/Arial.
from general_settings import *

# ────────────────────── 2. MATERIAL DATA ──────────────────────
# Silicon refractive index vs wavelength (um); Refractiveindex.info set covering
# 0.25 < lambda(um) < 1.2 (2008). Fallback: generic constant n below.
try:
    nSiR=np.loadtxt("siliconR.txt", skiprows=1)
    nSi=scinter.interp1d(nSiR[:,0],nSiR[:,1], fill_value='extrapolate')
except:
    print('siliconR not found, using generic n')

x=np.arange(.1,10,1)
ng=np.asarray([ [xi, 3.5] for xi in x ])
ng=scinter.interp1d(ng[:,0],ng[:,1], fill_value='extrapolate')
nair=np.asarray([ [xi, 1] for xi in x ])
nair=scinter.interp1d(nair[:,0],nair[:,1], fill_value='extrapolate')

# Si complex permittivity -> (n, k) vs wavelength (nm), then interpolants.
sieps=np.loadtxt('Si.dat', skiprows=1, delimiter='\t')
sink=[]
for line in sieps:
    lamb=line[0]
    er=line[1]
    ei=line[2]
    n=np.real(np.sqrt(er + 1j*ei))
    k=np.imag(np.sqrt(er + 1j*ei))
    sink.append([lamb,n,k])
sink=np.asarray(sink)

#Interpolate Data
siN=scinter.interp1d(sink[:,0],sink[:,1])
siK=scinter.interp1d(sink[:,0],sink[:,2])

# ────────────────────────── 3. SETTINGS ──────────────────────────
# Version suffix parsed from the filename (plot_fig6_N_v129_V213.py -> "213").
VER=(sys.argv[0][-6:-3])
print('Version', VER)

# --- Data files (Lumerical FSP far-field outputs, tab-separated) ---
fnames=gl('500*Random*H175*V22*full*129.dat')
fnames.append(gl('500*Pillar*H175R1*BS*V23*.5um*full*.dat')[0])
labels=["Poisson random", "Hyperuniform disordered"]

# --- Panel parameters ---
k1,k2=[9.3,18.7]   # HUD range; plotfigure() binds these as call defaults at def time
hud_color,rand_color=['#9BC1C1','#7585A8']   # PSD panel colors (random / hyperuniform)
vmaxs=[3,3]        # colorbar max per heatmap [%]
t='T'              # normalize power to transmission ('T') or (1-R) ('R')

# --- Figure & style ---
f=1.2
figsize=(f*figsize[1]*.8,f*figsize[0]*.7)   # portrait layout (swap + scale A6 from general_settings)

# --- Material (refractive index used for the light line / background) ---
n=nSi      # real silicon; nair/ng for other media

# --- Sample thickness for the Lambertian limit ---
thick=500  # nm

# ──────────────────────── 4. HELPERS ────────────────────────
def nm_to_eV(lam): #Returns energy in eV from wavelength in nm
    return C0/lam*1e9/e0*h0

def eV_to_nm(e): #Returns wavelength in nm from energy in eV
    return C0/e*1e9/e0*h0


# Function to convert energy (eV) to wavelength (nm)
def energy_to_wavelength(energy_eV):
    energy_J = energy_eV * cnt.e  # Convert eV to Joules
    wavelength_m = cnt.h * C0 / energy_J  # Wavelength in meters
    wavelength_nm = wavelength_m * 1e9  # Convert meters to nm
    return wavelength_nm

def normalize(array):            #Normalizes an array to its max values
    mx=max(array)
    mn=min(array)
    newar=[]
    for j in array:
        newar.append( (j-mn)/(mx-mn) )
    return np.asarray(newar)

def beerl(energy,thick):
    """Beer-Lambert absorption fraction of a slab `thick` [nm] thick at `energy` [eV]."""
    lamb=energy_to_wavelength(energy)
    alpha=4*Pi/lamb*siK(lamb)
    f=4*siN(lamb)**2
    exp=np.exp(-alpha*thick*f)
    return (1-exp)


def import_full_data(filename, k_min = 0, norm_power='T'): # [k_min] = per micron
    """Load one FSP far-field file.

    Columns: lambda (nm), source power (W), trans power (W), 1-R power (W),
    FF total power (W), then k (um^-1) [first half of line],
    power in cone up to k (W) [second half of line].
    Returns rows of [energy (eV), k (um^-1), normalized far-field power].
    """
    data_raw=np.loadtxt(filename, delimiter='\t', skiprows=1)

    data=[]
    for line in data_raw:
        lamda=line[0]
        ff_tot=line[4] # Total far field is found from the solid angle of 90 deg, which is at the final position of the line.
        k_ffpower=line[5:] # Contains the power in the far field for solid angles 1 - 90
        kf_split=int(len(k_ffpower)/2)
        k_values=k_ffpower[:kf_split]
        ff_power=k_ffpower[kf_split:]/ff_tot
        if norm_power == 'T':
            power_norm=float(line[2]/line[1]) # This equals transmittance, since col 1 is sourcepower and col 2 is transmitted power
        elif norm_power == 'R':
            power_norm=float(line[3]/line[1])  # This equals transmittance, since col 1 is sourcepower and col 4 (1-R) power
        else:
            power_norm=1
        lam_data=[]
        last_k=0
        for k,ff in zip(k_values,ff_power*power_norm):
            if k == 0:
                continue
            lam_data.append([nm_to_eV(lamda),k, ff])
            last_k=k

        if len(data) < 1:
            data = lam_data
        else:
            data = np.vstack([data, lam_data])
    print(ff_power[0:10])
    return data


def extract_k_array(data,crit_angle=True,k1=2,k2=1e4):
    """Sum far-field power per energy below (crit_angle=False) or beyond (True) k_c.

    Returns (power fractions, energies), sorted by energy.
    """
    # Sum the far field power within momentum intervals
    datak = []
    last_e=0
    ff_sum=0
    for line in data:
        energy, k, ff = line
        if energy != last_e:
            if ff_sum > 0:
                datak.append([energy,ff_sum])
            last_e=energy
            # Max k in air
            k0=2000*Pi/eV_to_nm(energy)
            ff_sum=0
        if crit_angle:
            k1=k0
            k2=1e4
        else:
            k1=0
            k2=k0
        if k < k1:
            continue
        if k >= k2:
            continue
        else:
            ff_sum+=ff

    datak=np.asarray(datak)
    # Sort by energies
    datak = datak[datak[:, 0].argsort()]
    return datak[:,1], datak[:,0]

# Define custom formatter function (overrides the one from general_settings)
def custom_formatter(x, pos):
    if int(x) == x:
        return f'{x:.0f}'
    else:
        return f'{x:.1f}'

# Generate cusomized colorbar
def colorbar(fig,ax,im,vmax=1):
    """Horizontal colorbar in the upper-left corner of `ax`, scaled 0..vmax."""
    pos = ax.get_position()

    # Position color bar in upper left corner of the axis: # X-pos. Y-pos, X-widht, Y-height
    cbar_pos = [pos.x0+.05*pos.width, pos.y0 + 0.95*pos.height, 0.3*pos.width, 0.05*pos.height]
    # Add a colorbar
    cbar = fig.colorbar(im, cax=fig.add_axes(cbar_pos), orientation='horizontal')

    # Customize the colorbar ticks
    tick_values = [0, vmax]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(['0', '$>${}'.format(vmax)])
    cbar.set_label('Power (\%)', labelpad=-14)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.label.set_rotation(0)  # Rotate label by XX degrees
    return fig,ax,cbar

# ──────────────────────────── 5. PLOTTING ────────────────────────────
def heatmap(ax, data, vmax=0.5, lightline=True, log=False, axv=False, scatter=True):
    """Far-field power vs (photon energy, in-plane momentum): sparse scatter or
    grid-interpolated imshow, Si light-line background, white light line."""
    data = np.asarray(data)
    # Extract x, y, and z values from the data array
    x = data[:, 0] # photon energies for x - axis
    y = data[:, 1] # In-plane momenta in per mu
    z = data[:, 2]*100 #Display power as percent

    # Shade background (Si light line region)
    energies=np.arange((min(x)),max(x)+.01,0.01)
    kn=2*Pi*energies*e0/cnt.h/C0/1e6*n(eV_to_nm(energies)/1000)
    ax.fill_between(energies, 0, kn, edgecolor=None, facecolor='#000000', interpolate=True, color='#000000')

    ## Scatter plot check of the grid
    if scatter:
        # Calculate step size for equidistant sampling
        x_unique = np.sort(np.unique(x))
        closest_values=[]
        lval=min(x_unique)
        i=0
        for val in x_unique[:]:
            i+=1
            if abs(val-lval) < 0.006:
                continue
            if abs(val-lval) < -0.012:
                continue
            else:
                lval=val
                closest_values.append(val)

        x_sparse,y_sparse,z_sparse=[],[],[]
        for idx in range(len(x)):
            xx=x[idx]
            yy=y[idx]
            zz=z[idx]
            if xx in closest_values:
                x_sparse.append(xx)
                y_sparse.append(yy)
                z_sparse.append(zz)
        x_sparse=np.asarray(x_sparse)
        y_sparse=np.asarray(y_sparse)
        z_sparse=np.asarray(z_sparse)
        im=ax.scatter(x_sparse, y_sparse, c=z_sparse, cmap='viridis',s=0.2,zorder=1, edgecolors=None, vmax=vmax)

    # Plot using imshow (grid interpolation + unphysical-k mask)
    if not scatter:
        # Create a regular grid over the range of x and y
        pts=len(np.unique(x))*4
        print('points',pts)

        xi = np.linspace(min(x), max(x), pts)  # Adjust the number of points as needed
        yi = np.linspace(min(y), max(y), pts)  # Adjust the number of points as needed
        X, Y = np.meshgrid(xi, yi)

        # Interpolate z values onto the grid
        Z = griddata((x, y), z, (X, Y), method='linear')

        # **Step 1: Compute Material-Dependent k_max**
        k_max = 2*cnt.pi/energy_to_wavelength(X)*n(energy_to_wavelength(X)/1000)*1000

        # **Step 2: Mask Out Unphysical Data**
        mask = Y > k_max  # True where in-plane momentum exceeds wavevector max
        Z[mask] = np.nan  # Remove unphysical data
        im=ax.imshow(Z, extent=(min(x), max(x), min(y), max(y)),
               origin='lower', aspect='auto', interpolation='nearest', \
               vmin=np.nanmin(Z), vmax=vmax, cmap='viridis', zorder=10)

    # Draw light line
    if lightline:
        energies=np.arange((min(x)),max(x)+.01,0.01)
        k=2*Pi*energies*e0/cnt.h/C0/1e6
        ax.plot(energies, k, linewidth=0.5, color='w')
        kn=2*Pi*energies*e0/cnt.h/C0/1e6*n(eV_to_nm(energies)/1000)

    if axv:
        for c in [k1,k2]:
            ax.axhline(x=c,linewidth=0.33, color='w', linestyle='--')

    # Set the axis labels
    ax.set_ylabel(r'$k_\parallel$ ($\text{\textmu}$m$^{-1}$)', labelpad=10)
    ax.xaxis.set_ticks_position('both')
    ax.set_yticks(range(0,80,10))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
    ax.set_ylim([0,60])

    return ax, im


def power_integrals(axes, datas, k1=0, k2=100, name='plot_zero_k1k2', crit_angle=True, ARS=False): #lamda in nm
    """Shaded power-fraction panels: backward / beyond kc / within kc, full-frame ticks."""
    for data1,ax1 in zip(datas,axes):
        # Below light line
        x,y=extract_k_array(data1,False)
        # BEYOND Light Line
        x2,y2=extract_k_array(data1)
        for xc,yc,xb in zip(x2,y2,x):
            if xc+xb > 0.999:
                pass
        # Backward scattering
        ax1.fill_between(y, 1, x+x2, edgecolor=None, facecolor=shade_colors[0], interpolate=True, color=shade_colors[0])
        # Forward
        ax1.fill_between(y, x2+x, x2, edgecolor=None, facecolor=shade_colors[2], interpolate=True, color=shade_colors[2])
        # BEYOND Light Line
        ax1.fill_between(y2, x2, 0, edgecolor=None, facecolor=shade_colors[1], interpolate=True, color=shade_colors[1])

    for a in axes:
        a.set_ylim([0,1])
        # Apply custom formatter to x-axis
        a.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        a.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

        # Create ticks all around the image
        a.tick_params(axis='x', direction='in')
        a.tick_params(axis='x', which='minor', direction='in')
        a.tick_params(axis='y', direction='inout')
        a.tick_params(axis='y', which='minor', direction='in')
        a.tick_params(top=True, right=True)
        a.tick_params(which='minor', top=True, right=True)
        a.yaxis.set_minor_locator(AutoMinorLocator())
        a.xaxis.set_minor_locator(AutoMinorLocator())
        a.yaxis.set_tick_params(labelleft=True, pad=2)

        a.set_aspect('auto')

        # Add labels
        texts=['backward', 'beyond $k_\mathrm{c}$', 'within $k_\mathrm{c}$']
        lcol=[ '#A41717', 'black', '#3C3319']
        x_pos=2.55 #a.get_xlim()[1]*.85
        lpos=[ (x_pos, .92), (x_pos, .1), (x_pos, .78)]
        rotations=[0, 0, 0, 10]
        for i in range(len(texts)):
            a.annotate(texts[i], xy=(1.5, .8), xytext=lpos[i], arrowprops=None, color=lcol[i], horizontalalignment='left', verticalalignment='center', rotation=rotations[i])

    ax1=axes[0]
    ax1.set_ylabel("Power fraction", labelpad=10)
    ax1.yaxis.set_tick_params(labelleft=True)

    return axes


def tick_params(ax):
    """Inward ticks on all sides + minor locators for the energy/k panels."""
    ax.set_xticks(np.arange(1,3.2,.2))
    ax.set_xlim([1.1,3])
    # Create ticks all around the image
    ax.tick_params(axis='x', direction='inout')
    ax.tick_params(axis='x', which='minor', direction='inout')
    ax.tick_params(axis='y', direction='inout')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(top=True, right=True)
    ax.tick_params(which='minor', top=True, right=True)
    ax.xaxis.set_minor_locator(FixedLocator(np.arange(1.,3.05,.05)))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

    return ax


def plot_psd(ax, files):
    """PSD panel: normalized power spectra of the random (mirrored about 0) and hyperuniform structures."""
    hud, rand = files
    data=np.loadtxt(rand, skiprows=1, delimiter=',')
    ax.plot(-(normalize(data[:,1])),data[:,0],color=rand_color)
    ax.fill_between(-(normalize(data[:,1])),data[:,0], edgecolor=None, facecolor=rand_color, interpolate=True, color=rand_color)
    data=np.loadtxt(hud, skiprows=1, delimiter=',')
    ax.plot((normalize(data[:,1])),data[:,0],color=hud_color)
    ax.fill_between((normalize(data[:,1])),data[:,0], edgecolor=None, facecolor=hud_color, interpolate=True, color=hud_color)
    ax.axvline(0, color='black', linewidth=.2)

    ax.set_xticks([-1,0,1])
    ax.set_xticklabels([1,0,1])
    ax.tick_params(axis='y', which='major', direction='inout', color='black',length=4)
    ax.spines['left'].set_position(('data', 0))
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_yticks([10,20,30,40,50])
    ax.tick_params(axis='y', which='minor', direction='inout', size=0)
    ax.set_xlabel('PSD (a.u.)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax

def plotfigure(datas, labels, k1=k1, k2=k2, log=False, name='fig3d'):
    """Assemble Fig 6: heatmaps + PSD (top row), power-fraction panels (bottom row)."""
    number=len(datas)
    figx,figy=figsize

    # BEGIN Plot
    fig = plt.figure(figsize=(number*figx,figy*.85), constrained_layout=False)
    gs = fig.add_gridspec(3, number+1, width_ratios=[1, 0.2, 1], height_ratios=[1,0.05,1])  # narrow middle column for the PSD

    # Containers for heatmap and power fraction axes
    haxes, paxes=[], []
    for i in range(number):
        if i == 0:
            paxes.append(fig.add_subplot(gs[2, i]))
        else:
            paxes.append(fig.add_subplot(gs[2, i+1]))

    ### Add Power fraction Plot
    paxes = power_integrals(paxes,datas, k1,k2,'test',crit_angle=True, ARS=False)

    ### Add heatmaps
    for i in range(number):
        if i == 0:
            haxes.append(fig.add_subplot(gs[0, i]))
        else:
            haxes.append(fig.add_subplot(gs[0, i+1], sharex=paxes[0], sharey=haxes[0]))

    # Adjust the distance between subplots
    plt.subplots_adjust(wspace=0.085) #Incompatible with constr layout
    plt.subplots_adjust(hspace=0.03) #Incompatible with constr layout

    ### Add PSD
    psd_files=gl('*_out.dat')
    psd_ax=fig.add_subplot(gs[0, 1], sharey=haxes[0])
    psd_ax=plot_psd(psd_ax,psd_files)

    ims,cbars=[],[]
    for i in range(number):
        haxes[i], im = heatmap(haxes[i],datas[i],vmax=vmaxs[i])
        fig, haxes[i], cbar = colorbar(fig,haxes[i],im,vmaxs[i])

    # Adjust plot axes
    ct=0
    for pax,hax,l in zip(paxes,haxes,labels):
    # Heatmap axis
        hax=tick_params(hax)
        hax.set_xlim([1.1,3])
        hax.xaxis.set_tick_params(labelbottom=False)
        hax.set_title(l, pad=15, color='#649797')
        if 'random' in l:
            hax.set_title(l, pad=15, color='#4F628E')
    # Power axis
        pax=tick_params(pax)
        pax.set_xlabel('Photon energy (eV)')
        # Remove Y labels from 2nd col onward
        if ct > 0:
            hax.yaxis.set_tick_params(labelleft=False)
            pax.yaxis.set_tick_params(labelleft=False)
            hax.yaxis.set_tick_params(pad=10)
            hax.set_ylabel('')
        ct += 1
    ### Adjust size ratios
        pax.set_aspect('auto')
        hax.set_aspect('auto')

    # Wavelength axis (secondary x-axis on top of the heatmaps)
    for ax3 in haxes:
        ax = ax3.twiny()
        ax.set_xlabel('Wavelength (nm)', labelpad=7)
        #Align the secondary y-axis with the energy values
        ax.set_xlim(ax3.get_xlim())
        ax.set_xticks(ax3.get_xticks())
        ax.set_xticklabels([f'${energy_to_wavelength(xtick):.0f}$' for xtick in ax3.get_xticks()], fontsize=fontsize)
        ax.tick_params(axis='x', direction='inout')
        ax.tick_params(axis='x', which='minor', direction='in')

    ### Plot lambertian limit
    axx=paxes
    xx=np.arange(1,4,0.01)
    yy=beerl(xx,thick)
    for a in axx:
        a.plot(xx,yy,'--', color='green')

    ### SAVING
    plt.savefig(name+'.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(name+'_V'+VER+'.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(name+'_V'+VER+'.png', dpi=dpi, bbox_inches='tight')
    plt.close()


# ───────────────────────────── 6. MAIN ─────────────────────────────
# Import files
datas = []
for file in fnames[:]:
    print(file)
    data=import_full_data(file,norm_power=t)
    datas.append(data)

# Plot
plotfigure(datas,labels,name='Fig_6')
