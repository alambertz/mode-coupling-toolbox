#!/bin/python3
"""
Figure 4 - mode coupling rebuttal (cleaned-up copy of plot_fig4_v125_V161.py).

Renders one figure column per far-field data file, with 4 rows:
  row 0: far-field power vs photon energy and in-plane momentum k
         (material light-line shading + mode curves)
  row 1: power integrals (backward / beyond light line / within light line)
         + Beer-Lambert (Lambertian) limit curve
  row 2: integrated absorptance per optical mode
  twin x-axis on row 0: wavelength (nm)

Inputs (this directory):
  500*H75*full*v127*.dat      far-field output, tab-separated
  500nmSi-highP               mode library (MSclass19 Modedata format)
  siliconR.txt / siliconI.txt Si refractive index (wl in um, n / k)
  AM15G.dat                   AM1.5G solar spectrum

Outputs: fig_4.pdf, fig_4_V61.png, fig_4_V61.pdf
         (version tag = filename[-5:-3])

Run: venv/bin/python plot_fig4_clean_V161.py   (needs TeX Live + cm-super)
"""

# =====================================================================
# 1. IMPORTS
# =====================================================================
import os
import sys
import math
import colorsys
from glob import glob as gl

import numpy as np
import scipy.constants as cnt
import scipy.interpolate as scinter
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from PIL import Image
Image.MAX_IMAGE_PIXELS = None   # no decompression-bomb guard (large sim images)

from MSclass19 import *          # Modedata and friends

# general_settings.py lives two levels above this project folder; it provides
# figsize, dpi, fontsize, shade_colors, modeax_color, C0/e0/Pi/h0, custom_formatter
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../../")))
from general_settings import *


# =====================================================================
# 2. MATERIAL DATA (loaded from files, interpolated)
# =====================================================================
# Silicon refractive index: columns are wl [um], n / k
n1r = np.loadtxt("siliconR.txt", skiprows=1)
n1i = np.loadtxt("siliconI.txt", skiprows=1)
n1ri = scinter.interp1d(n1r[:,0], n1r[:,1], fill_value='extrapolate')  # n(wl in um)
n1ii = scinter.interp1d(n1i[:,0], n1i[:,1], fill_value='extrapolate')  # k(wl in um)
siN = scinter.interp1d(n1r[:,0]*1000, n1r[:,1])   # n(wl in nm)
siK = scinter.interp1d(n1i[:,0]*1000, n1i[:,1])   # k(wl in nm)

# Generic constant-index materials (monitor-in-air or placeholder cases)
x = np.arange(.1, 10, 1)
ng = np.asarray([[xi, 3.55] for xi in x])
ng = scinter.interp1d(ng[:,0], ng[:,1], fill_value='extrapolate')    # generic n=3.55
nair = np.asarray([[xi, 1] for xi in x])
nair = scinter.interp1d(nair[:,0], nair[:,1], fill_value='extrapolate')  # air

# AM1.5G solar spectrum: columns are lambda [nm], intensity
am15 = np.loadtxt("AM15G.dat", skiprows=1)
am15flux = []
amdict = {}   # photon flux per integer nm wavelength
for i in am15:
    L = i[0]/1e9               # wavelength [m]
    Lmu = round(i[0]/1000, 4)  # wavelength [um]
    Lint = int(i[0])           # wavelength [nm], as int
    E = cnt.h*cnt.c/L          # photon energy at lambda
    N = i[1]/E                 # photon flux per lambda
    am15flux.append([Lmu, 4, N])
    amdict[Lint] = N
am15flux = np.asarray(am15flux)
am15fluxi = scinter.interp1d(am15flux[:,0], am15flux[:,2], fill_value='extrapolate')


# =====================================================================
# 3. SETTINGS - all tunables in one place
# =====================================================================

# version tag: last two digits of the script filename ("..._V161.py" -> "61")
VER = sys.argv[0][-5:-3]
print('Version', VER)

# far-field data files (one figure column per file) and their labels
fnames = gl('500*H75*full*v127*.dat')
labels = []
for f in fnames:
    labels.append("R" + f.split("R")[1].split('nm')[0] + "nm")

# colorbar maximum, percent of source power (one entry per column)
vmaxs = [20 for i in labels]

# figure size: shrink the imported A6 size by ~95% and swap to portrait-ish
f = .95
figsize = (f*figsize[1]*.9, f*figsize[0])

# material for the light line / k-space cutoff (alternatives: nair, ng)
n = n1ri          # silicon

# device geometry
thick = 500       # Si slab thickness [nm] (Beer-Lambert limit curve)
height = 75       # bump height [nm]

# mode library file (MSclass19 Modedata format)
mode_file = '500nmSi-highP'


# =====================================================================
# 4. UNIT CONVERSIONS
# =====================================================================

def nm_to_eV(lam):   # photon energy [eV] from wavelength [nm]
    return C0/lam*1e9/e0*h0

def eV_to_nm(e):     # wavelength [nm] from photon energy [eV]
    return C0/e*1e9/e0*h0


# =====================================================================
# 5. PHYSICS HELPERS
# =====================================================================

def beerl(energy, thick):
    """Beer-Lambert absorptance of a Si slab: the Lambertian limit curve."""
    lamb = eV_to_nm(energy)
    alpha = 4*Pi/lamb*siK(lamb)
    f = 4*siN(lamb)**2
    exp = np.exp(-alpha*thick*f)
    return (1-exp)


def generate_distinct_colors(num_colors):
    """num_colors distinct RGB tuples via golden-ratio HSL spread."""
    colors = []
    golden_ratio = 0.618033988749895   # golden ratio conjugate
    for i in range(num_colors):
        hue = (i * golden_ratio) % 1.0
        saturation = 0.7 + 0.3 * (i % 2)          # alternate 0.7 / 1.0
        lightness = 0.5 + 0.2 * (abs(math.sin(i * 2)))
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def average_ext_mode(xyz):
    """Sum column 1 for each unique value of column 0."""
    x_unique = np.unique(xyz[:,0])
    xz = []
    for xx in x_unique:
        summ = 0
        for line in xyz:
            if line[0] == xx:
                summ += line[1]
        xz.append([xx, summ])
    return np.asarray(xz)


def extract_k_array(data, crit_angle=True, k1=2, k2=1e4):
    """
    Per-energy sums of far-field power below (crit_angle=False) or above
    (crit_angle=True) the light-line cutoff k0. Returns (power, energy).
    """
    datak = []
    last_e = 0
    ff_sum = 0
    for line in data:
        energy, k, ff = line
        if energy != last_e:
            if ff_sum > 0:
                datak.append([energy, ff_sum])
            last_e = energy
            k0 = 2000*Pi/eV_to_nm(energy)   # max k in air (light line)
            ff_sum = 0
        if crit_angle:
            k1 = k0
            k2 = 1e4
        else:
            k1 = 0
            k2 = k0
        if k < k1:
            continue
        if k >= k2:
            continue
        ff_sum += ff

    datak = np.asarray(datak)
    datak = datak[datak[:, 0].argsort()]    # sort by energy
    return datak[:,1], datak[:,0]


# =====================================================================
# 6. DATA IMPORTERS
# =====================================================================

def import_modedata(filename='500nmSi-highP'):
    """
    Load the mode library via MSclass19 and build per-mode colors.
    Note: reads the global mode_file (SETTINGS), not the argument.
    Returns (M, mxmode, colors).
    """
    M = Modedata()
    M.Import(mode_file)
    mxmode = len(M.mindex)-1
    colors = generate_distinct_colors(mxmode)
    return M, mxmode, colors


def import_full_data(filename, k_min=0, norm_power='T'):
    """
    Parse a far-field .dat file into rows [energy(eV), k(um-1), power].

    File columns: lambda(nm), source W, trans W, 1-R W, FF total W,
    then k values (first half of the remainder) and the cumulative
    cone power up to each k (second half). Rows with lambda < 400 nm
    are skipped. norm_power: 'T' = transmittance, 'R' = 1-R fraction.
    """
    data_raw = np.loadtxt(filename, delimiter='\t', skiprows=1)

    data = []
    for line in data_raw:
        lamda = line[0]
        if lamda < 400:
            continue
        ff_tot = line[4]                     # total far-field power (90 deg solid angle)
        k_ffpower = line[5:]
        k_values = k_ffpower[:int(len(k_ffpower)/2)]
        ff_power = k_ffpower[int(len(k_ffpower)/2):]/ff_tot
        if norm_power == 'T':
            power_norm = float(line[2]/line[1])   # transmittance
        elif norm_power == 'R':
            power_norm = float(line[3]/line[1])   # 1-R fraction
        else:
            power_norm = 1
        lam_data = []
        last_k = 0
        for k, ff in zip(k_values, ff_power*power_norm):
            if k == 0:
                continue
            lam_data.append([nm_to_eV(lamda), k, ff])
            last_k = k

        if len(data) < 1:
            data = lam_data
        else:
            data = np.vstack([data, lam_data])
    return data


# =====================================================================
# 7. PLOT BUILDING BLOCKS
# =====================================================================

def colorbar(fig, ax, im, vmax=1):
    """Horizontal colorbar in the top-left corner of ax, ticks 0 .. vmax."""
    pos = ax.get_position()
    # x-pos, y-pos, x-width, y-height (fraction of axis)
    cbar_pos = [pos.x0+.015*pos.width, pos.y0 + 0.95*pos.height, 0.28*pos.width, 0.05*pos.height]
    cbar = fig.colorbar(im, cax=fig.add_axes(cbar_pos), orientation='horizontal')

    tick_values = [0, vmax]
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels(['0', '$>${}'.format(vmax)])
    cbar.set_label('Power (\%)', labelpad=-14)
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.label.set_rotation(0)
    return fig, ax, cbar


def plMode_subfig(ax, dic):
    """Plot the integrated absorptance curve of each mode (uses global colors)."""
    for m in dic.keys():
        xyz = dic[m]
        xz = average_ext_mode(xyz)
        ax.plot(xz[:,0], xz[:,1], '-', color=colors[m], linewidth=1.3/.8*.5)
    return ax


def modeabsplot(ax, data, lmode=-1, dispmodes=True, getmodeabs=True, krad=.1, shifty=0.0):
    """
    For each mode (in order), interpolate its k(E) curve from the mode library
    and sum the far-field power of 'data' rows falling in this mode's band.
    Matched rows are consumed so later modes only see what is left.
    Returns {mode_index: array([energy, power])}. Uses global M and colors.
    """
    p_energies = np.unique(data[:,0])
    if lmode == -1:
        lmode = 2*M.mxmo()
    d = {}
    skip = 0
    for l in range(skip, lmode-skip):
        result = []
        xyz = M.extractModes(l, l+1)      # mode library rows for this mode: [lambda(nm), k]
        if len(xyz) > 1:
            xm = nm_to_eV(xyz[:,0]*1000)  # mode curve in eV / um-1
            ym = xyz[:,1]
            mode_y = scinter.interp1d(xm, ym+shifty)
            if dispmodes:
                ax.plot(xm, mode_y(xm), '--', color=colors[l], linewidth=0.5)
            if getmodeabs:
                for xval in p_energies:
                    if xval < xm.min() or xval > xm.max():
                        continue
                    mask_energy = (data[:, 0] == xval)
                    if not np.any(mask_energy):
                        continue
                    yval_interp = mode_y(xval)
                    y_slice = data[mask_energy, 1]

                    # band of k values belonging to this mode
                    lower_bound = yval_interp
                    if l == skip:
                        mask_close = np.isclose(y_slice, yval_interp+krad/2, atol=krad)
                    else:
                        upper_bound = last_mode(xval)
                        mask_close = (y_slice >= lower_bound) & (y_slice <= upper_bound)

                    final_mask = mask_energy.copy()
                    final_mask[mask_energy] = mask_close
                    z_indices = np.nonzero(final_mask)[0]
                    matching_rows = data[final_mask, :]

                    if len(z_indices) > 0:
                        zz_sum = np.sum(matching_rows[:,2])
                        data = data[~final_mask]      # consume matched rows
                        if zz_sum > 100:
                            print("above 1:", l, xval, yval_interp, zz_sum)
                        result.append([xval, zz_sum])

                result = np.asarray(result)
                last_mode = mode_y

        if len(result) > 1:
            d[l] = result
    return d


def power_integrals(axes, datas, k1=0, k2=100, name='plot_zero_k1k2', crit_angle=True, ARS=False):
    """
    Shaded power-fraction plot per column: backward (below light line),
    beyond light line, within light line; plus the legend annotations.
    """
    for data1, ax1 in zip(datas, axes):
        x, y = extract_k_array(data1, False)      # below light line (backward)
        x2, y2 = extract_k_array(data1)           # beyond light line
        for xc, yc, xb in zip(x2, y2, x):
            if xc+xb > 0.999:
                print(yc, xc+xb)
        ax1.fill_between(y, 1, x+x2, edgecolor=None, facecolor=shade_colors[0], interpolate=True, color=shade_colors[0])   # backward
        ax1.fill_between(y, x2+x, x2, edgecolor=None, facecolor=shade_colors[2], interpolate=True, color=shade_colors[2])   # forward (within)
        ax1.fill_between(y2, x2, 0, edgecolor=None, facecolor=shade_colors[1], interpolate=True, color=shade_colors[1])     # beyond light line

    for a in axes:
        a.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        a.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

        # ticks all around the image
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

        # legend annotations
        texts = ['backward', 'beyond $k_\mathrm{c}$', 'within $k_\mathrm{c}$']
        lcol = ['#A41717', 'black', '#3C3319']
        x_pos = 2.55
        lpos = [(x_pos, .92), (x_pos, .1), (x_pos, .74)]
        rotations = [0, 0, 0, 10]
        for i in range(len(texts)):
            a.annotate(texts[i], xy=(1.5, .8), xytext=lpos[i], arrowprops=None, color=lcol[i], horizontalalignment='left', verticalalignment='center', rotation=rotations[i])

    ax1 = axes[0]
    ax1.set_ylabel("Power fraction")
    ax1.yaxis.set_tick_params(labelleft=True)
    return axes


def heatmap(ax, data, vmax=0.5, lightline=True, log=False, axv=False, scatter=True, dispmodes=True, getmodeabs=True, lmode=-1, krad=1., check_pe=0):
    """
    Far-field power vs (energy, k) as a scatter plot (scatter=True) or an
    interpolated image (scatter=False), with the material light-line shading,
    the mode curves and the air light line. Returns (ax, im, d).
    """
    data = np.asarray(data)
    x = data[:, 0]              # photon energy [eV]
    y = data[:, 1]              # in-plane momentum [um-1]
    z = data[:, 2]*100          # power as percent

    if check_pe > 0:
        prep_singleWL_plot(x, y, z)   # optional single-energy inspection (writes testPE.dat)

    # shade the k-space region beyond the material light line
    energies = np.arange((min(x)), max(x)+.01, 0.01)
    kn = 2*Pi*energies*e0/cnt.h/C0/1e6*n(eV_to_nm(energies)/1000)
    ax.fill_between(energies, 0, kn, edgecolor=None, facecolor='#000000', interpolate=True, color='#000000')

    if scatter:
        # equidistant-sampling bookkeeping (kept for debugging the energy grid)
        mask = z < 0
        z[mask] = 0
        x_unique = np.sort(np.unique(x))
        closest_values = []
        lval = 0
        for val in x_unique[:]:
            if abs(val-lval) < 0.002:
                continue
            elif abs(val-lval) < -0.012:
                continue
            else:
                if val < 3:
                    print(val, val-lval)
                lval = val
                closest_values.append(val)
        print(len(x_unique), len(closest_values))
        x_sparse, y_sparse, z_sparse = [], [], []
        for idx in range(len(x)):
            xx, yy, zz = x[idx], y[idx], z[idx]
            if xx in closest_values:
                x_sparse.append(xx)
                y_sparse.append(yy)
                z_sparse.append(zz)
        x_sparse = np.asarray(x_sparse)
        y_sparse = np.asarray(y_sparse)
        z_sparse = np.asarray(z_sparse)
        mask = z_sparse < 0
        z_sparse[mask] = 0
        im = ax.scatter(x, y, c=z, cmap='viridis', s=1e-6, zorder=1, marker='.', edgecolors=None, vmin=0, vmax=vmax)

    if not scatter:
        # regular grid + linear interpolation of the far field
        xi = np.linspace(min(x), max(x), 2000)
        yi = np.linspace(min(y), max(y), 2000)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), z, (X, Y), method='linear')
        # mask out the unphysical region beyond the material light line
        k_max = 2*cnt.pi/eV_to_nm(X)*n(eV_to_nm(X)/1000)*1000
        mask = Y > k_max
        Z[mask] = np.nan
        im = ax.imshow(Z, extent=(min(x), max(x), min(y), max(y)), origin='lower', zorder=1, aspect='auto', interpolation='bilinear', cmap='viridis', vmax=vmax)

    if axv:
        for c in [k1, k2]:
            ax.axhline(x=c, linewidth=0.33, color='w', linestyle='--')

    # mode coupling info from the heatmap (mode curves + per-mode power sums)
    d = modeabsplot(ax, data, lmode, dispmodes, getmodeabs, krad)

    # air light line
    if lightline:
        k = 2*Pi*energies*e0/cnt.h/C0/1e6
        ax.plot(energies, k, linewidth=0.5, color='w')
    return ax, im, d


# =====================================================================
# 8. OPTIONAL SINGLE-ENERGY INSPECTION (only used when check_pe > 0)
# =====================================================================

def prep_singleWL_plot(x, y, z):
    """
    Extract the far-field rows at the global check_pe energy and write them
    to testPE.dat for plAbsorption. Uses the global check_pe and n.
    """
    # irregular grid of (x, y, z) rows
    unique_x = np.unique(x)
    irregular_grid = []
    X, Y, Z = [], [], []
    for x_val in unique_x:
        corresponding_y = y[x == x_val]
        corresponding_z = z[x == x_val]
        for y_val, z_val in zip(corresponding_y, corresponding_z):
            irregular_grid.append((x_val, y_val, z_val))
            X.append(x_val)
            Y.append(y_val)
            Z.append(z_val)

    irregular_grid = np.array(irregular_grid)
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)

    sum_z_per_x = {x_val: np.sum(Z[X == x_val]) for x_val in unique_x}

    # rows at check_pe with k below the material light line
    cmask = (X > check_pe*0.999) & (X < check_pe*1.001) & (Y < 2*cnt.pi/eV_to_nm(check_pe)*n(eV_to_nm(check_pe)/1000)*1000)
    indices = np.where(cmask)
    x_at_check_pe = X[indices]
    y_at_check_pe = Y[indices]
    z_at_check_pe = Z[indices]
    x_flat = x_at_check_pe.flatten()
    y_flat = y_at_check_pe.flatten()
    z_flat = z_at_check_pe.flatten()

    # same extraction directly on the raw arrays (this is what gets saved)
    mask_original = (x > check_pe * 0.999) & (x < check_pe * 1.001)
    x_at_check_pe = x[mask_original]
    y_at_check_pe = y[mask_original]
    z_at_check_pe = z[mask_original]
    x_flat = x_at_check_pe.flatten()
    y_flat = y_at_check_pe.flatten()
    z_flat = z_at_check_pe.flatten()

    out_check = [[yyy, zzz] for yyy, zzz in zip(y_flat, z_flat)]
    np.savetxt('testPE.dat', out_check)


def plAbsorption(energy):
    """
    Absorptance vs k at one photon energy from testPE.dat, with the light
    line marked and the below/above sums in the legend. Saves a PDF.
    """
    lightline = 2*np.pi/eV_to_nm(energy)*1000
    ka = np.loadtxt('testPE.dat')
    k, a = ka[:,0], ka[:,1]
    esc, gmd = 0, 0
    for kk, aa in zip(k, a):
        if kk < lightline:
            esc += aa
        else:
            gmd += aa
    fig, ax = plt.subplots(1, 1)
    ax.plot(k,a, label='E: '+str(round(energy,2))+'eV, sum Esc: '+str(round((esc)))+', sum beyond: '+str(round((gmd))))
    ax.axvline(lightline, color='grey')
    ax.set_xlabel('Momentum / $\mu m^{-1}$')
    ax.set_ylabel('Absorption')
    ax.grid()
    ax.legend()
    fig.savefig("AbsorptionVsK_{}eV_V{}.pdf".format(round(energy,2), VER), bbox_inches='tight')


# =====================================================================
# 9. FIGURE ASSEMBLY
# =====================================================================

def plotfigure(datas, labels, lmode=-1, k1=0, k2=100, log=False, name='fig3d', check_pe=0, krad=1.):
    """
    Assemble and save the 4-row figure: far-field heatmap / power integrals /
    per-mode absorptance, with a wavelength twin axis. Saves <name>.pdf,
    <name>_V<VER>.pdf and <name>_V<VER>.png. Uses global figsize, vmaxs, thick, dpi.
    """
    xlim = [1.1, 3]
    number = len(datas)
    figx, figy = figsize

    fig = plt.figure(figsize=(number*figx, figy), constrained_layout=False)
    gs = fig.add_gridspec(4, number, width_ratios=[1 for i in datas], height_ratios=[1, 1, 1, 1])

    # row 0: heatmap, row 1: power integrals, row 2: mode absorptance
    haxes, paxes, maxes = [], [], []
    for i in range(number):
        paxes.append(fig.add_subplot(gs[1, i]))
        if i == 0:
            haxes.append(fig.add_subplot(gs[0, i]))
            maxes.append(fig.add_subplot(gs[2, i]))
        else:
            haxes.append(fig.add_subplot(gs[0, i], sharey=haxes[0]))
            maxes.append(fig.add_subplot(gs[2, i]))

    plt.subplots_adjust(wspace=0.07)
    plt.subplots_adjust(hspace=.11)

    ### power integrals row
    paxes = power_integrals(paxes, datas, k1, k2, 'test', crit_angle=True, ARS=False)

    ### heatmap + mode rows
    ims, cbars = [], []
    for i in range(number):
        haxes[i], im, mdic = heatmap(haxes[i], datas[i], vmax=vmaxs[i], lmode=lmode, scatter=False, krad=krad)
        maxes[i] = plMode_subfig(maxes[i], mdic)
        maxes[i].axvline(check_pe, color='b', alpha=0.5)
        haxes[i].axvline(check_pe, color='b', alpha=0.5)
        maxes[i].fill_between([0, 5], 0, 1, edgecolor=None, facecolor=modeax_color, interpolate=True, color=modeax_color)
        fig, haxes[i], cbar = colorbar(fig, haxes[i], im, vmaxs[i])

    ### axis styling
    ct = 0
    for pax, hax, maxe, l in zip(paxes, haxes, maxes, labels):
        # mode absorptance axis
        maxe.set_xlabel('Photon energy (eV)', labelpad=2)
        maxe.set_xlim(xlim)
        maxe.xaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        maxe.set_ylim([0, 1])
        maxe.set_ylabel('Mode absorptance', labelpad=2)
        # heatmap axis
        hax.set_xlim(xlim)
        hax.set_ylim([0, 40])
        hax.xaxis.set_tick_params(labelbottom=False)
        if ct == 0:
            hax.set_ylabel(r'$k_\parallel$ ($\text{\textmu}$m$^{-1}$)', labelpad=7)
        hax.xaxis.set_ticks_position('both')

        for ax in [hax, maxe]:
            # ticks all around the image
            ax.tick_params(axis='x', direction='inout')
            ax.tick_params(axis='x', which='minor', direction='inout')
            ax.tick_params(axis='y', direction='inout')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(top=True, right=True)
            ax.tick_params(which='minor', top=True, right=True)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
        # power integrals axis
        pax.set_xlim(xlim)
        pax.xaxis.set_tick_params(labelbottom=False)
        pax.xaxis.set_minor_locator(AutoMinorLocator())
        pax.set_ylim([0, 1])
        # hide y labels from the second column onward
        if ct > 0:
            hax.yaxis.set_tick_params(pad=10)
            hax.set_ylabel('')
            maxe.set_ylabel('')
        ct += 1
        pax.set_aspect('auto')
        hax.set_aspect('auto')
        maxe.set_aspect('auto')

    # wavelength twin axis on the heatmap row
    for ax3 in haxes:
        ax = ax3.twiny()
        ax.set_xlabel('Wavelength (nm)', labelpad=8)
        ax.set_xlim(xlim)
        ax.set_xticks(ax3.get_xticks())
        ax.set_xticklabels([f'${eV_to_nm(xtick):.0f}$' for xtick in ax3.get_xticks()], fontsize=fontsize)
        ax.tick_params(axis='x', direction='inout')
        ax.tick_params(axis='x', which='minor', direction='in')

    ### Beer-Lambert (Lambertian) limit on the power row
    axx = paxes
    xx = np.arange(1, 4, 0.01)
    yy = beerl(xx, thick)
    for a in axx:
        a.plot(xx, yy, '--', color='green')

    ### save
    plt.savefig(name+'_V'+VER+'.pdf', dpi=dpi, bbox_inches='tight')
    plt.savefig(name+'_V'+VER+'.png', dpi=dpi, bbox_inches='tight')
    plt.savefig(name+'.pdf', dpi=dpi, bbox_inches='tight')
    plt.close()

    if check_pe > 0:
        plAbsorption(check_pe)


# =====================================================================
# 10. MAIN
# =====================================================================

# far-field data (one figure column per file)
datas = []
t = 'R'
for file in fnames[:1]:
    print(file)
    data = import_full_data(file, norm_power=t)
    datas.append(data)

# mode library + per-mode colors
M, mxmode, colors = import_modedata(mode_file)

# render (optional single-energy inspection: pass check_pe=<eV>)
plotfigure(datas, labels, lmode=mxmode, name='fig_4', krad=0.25)
