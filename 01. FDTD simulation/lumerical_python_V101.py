#!/usr/bin/env python




import numpy as np
# import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.constants as cnt
from collections import OrderedDict
import scipy.interpolate as scinter
import importlib.util
import sys, os
from inspect import getsourcefile
from os.path import abspath
#from scipy import optimize
from scipy.optimize import minimize


#### VERSION
VER=str(sys.argv[0].split('\\')[-1][-5:-3])
print("Version",VER)

## Robust way to get current working directory
path=(abspath(getsourcefile(lambda:0))).split('/')
pth=''
for st in path[0:len(path)-1]:
    pth+='/'+st 
sys.path.append("C:\\Program Files\\ANSYS Inc\\v251\\Lumerical\\api\\python\\") #Default windows lumapi path
sys.path.append("/opt/lumerical/v222/api/python/lumapi.py") #Default linux lumapi path
sys.path.append(pth) #Current directory

#The default paths for windows and linux
try: #Windows
    spec = importlib.util.spec_from_file_location('lumapi', 'C:\\Program Files\\ANSYS Inc\\v251\\Lumerical\\api\\python\\lumapi.py')
    lumapi = importlib.util.module_from_spec(spec) #windows
    spec.loader.exec_module(lumapi)
    supercomp=False
except: #accept Linux
    print("detected linux")
    spec_lin = importlib.util.spec_from_file_location('lumapi', "/gpfs/admin/_hpc/sw/arch/AMD-ZEN2/Centos8/EB_production/2021/software/Lumerical/2021-R2.3-2834-e18f3c9-OpenMPI-4.1.1/api/python/lumapi.py")
#Functions that perform the actual loading
#lumapi = importlib.util.module_from_spec(spec) #windows
    lumapi = importlib.util.module_from_spec(spec_lin) #linux
    spec_lin.loader.exec_module(lumapi)
    supercomp=True


## Some definitions and constants:
nm=1e-9
um=1e-6
C0=cnt.c
Pi=cnt.pi

# ## Define settings

# ### HUD DESIGN ###
# Spinodal: 1
# Honeycomb: 2
# HUD Holes: 3
# Periodic: 4 


#### FILENAME 
def getFileName(NAME="",setpam={},VER=""):
    if "boxsize" in setpam.keys():
        boxsize=round(setpam["boxsize"]*1e9)
        if boxsize > 1000:
            NAME+="-BS"+str(boxsize/1000)+"um"
        else:
            NAME+="-BS"+str(boxsize)+"nm"
    NAME+='-Acc'+str(round(setpam["meshacc"]))
    NAME+='-'+str(round(setpam["minmesh"]*1e9))+"M"
    if "lambda_min" in setpam.keys():
        NAME+= '-lambda'+str(round(setpam["lambda_min"]*1e9))+'-'+str(round(setpam["lambda_max"]*1e9))
    NAME=NAME+"_V"+VER 
    return NAME

# GDS helper function: Imports a gds design file and gives to it the properties defined in setupFDTD
def addGDS(sim, props,x=0,y=0):
    sim.gdsimport(props["name"]+".gds", props["cell"], props["layer"], \
                  props["material"], props["z min"], props["z max"] )
    sim.select("GDS_LAYER_"+str(props["layer"])+":1000")
    sim.set("name", props["name"]+"-pattern")
    if "index" in props.keys():
        sim.set("index", props["index"])
    if x!=0 or y!=0:
        sim.set("x",x)
        sim.set("y",y)
    
# GDS multiple times insertion
def multiGDS(sim, props, unit_size=(1325e-9,765e-9)):
    gds_props=props['gds']
    x_size=props['FDTD']['x span']
    y_size=props['FDTD']['y span']
    x_start=-x_size/2+unit_size[0]/2
    y_start=-y_size/2+unit_size[1]/2
    for xx in range(int(x_size/unit_size[0])):
        for yy in range(int(y_size/unit_size[1])):
            x = x_start + xx * unit_size[0]
            y = y_start + yy * unit_size[1] 
            # print("XY",x,y)
            addGDS(sim, gds_props, x,y)
            
            
def set_material_fit(fdtd,dic):
    material=dic['name']
    for prop in dic.keys():
        if prop == 'name':
            continue
        fdtd.setmaterial(material,prop,dic[prop])
        


#### Check Mem requirements
def check_memory(sim,NAME='sim'):
    dic=sim.runsystemcheck()
    max_mem=round(max(list(dic["Approximate_Memory_Requirements"].values())[1:])*nm,1)
    mem_data_coll=round(dic["Approximate_Memory_Requirements"]["Data_Collection_Bytes"]*nm,1)
    mem_fsp=round(dic["Approximate_Memory_Requirements"]["FSP_Saved_Monitor_Data_Bytes"]*nm,1)
    print('Maximum memory',max_mem,' GB, data collection:',mem_data_coll,' GB, FSP size: ',mem_fsp,' GB')
    with open(NAME+'_memcheck.txt', 'w') as f:
        print('Maximum memory',max_mem,' GB',file=f)
        print('Data collection:',mem_data_coll,' GB', file=f)
        print('FSP size: ',mem_fsp,' GB', file=f)
    return max_mem,mem_data_coll,mem_fsp


# ## Run Simulation
def runFDTD(fdtd,fname="trials-run.fsp"):
    ### Pre run preparations ###


    ### Run Simulations and save them ###    
    fdtd.run()
    fdtd.save(fname)

    ### Return relevant results ###
    # result= {}
    # result["lambda"]=(1e6*C0/np.asarray(fdtd.getresult("Transmission2", "T")['f'])).flatten()
    # # Transmitted powers from monitors
    # for pm in props["mon"]:
    #     # if pm["name"]!="Transmission1":
    #     result[pm["name"]]=fdtd.getresult(pm["name"], "T")['T']
        
    return fdtd
        

#### Run Simulation 
# Return relevant results
def getResults(fdtd,props):
    result= {}
    result["lambda"]=(np.asarray(fdtd.getresult(props["mon"][0]["name"], "T")['lambda'])/um).flatten()
    # Transmitted powers from monitors
    for pm in props["mon"]:
        result[pm["name"]]=fdtd.getresult(pm["name"], "T")['T']
    return result
        
        
# Run Simulation on supercomputer
def srunFDTD(fdtd,props,itr):
    ### Run Simulations and save them ###  
    #shelper.sh is actually the runfile maker and run script
    #Here we need to make sure the run is successful
    status=os.system('bash shelper.sh {} >>shlog.txt'.format(itr))/256
    
    #Status receives the exit code from the bash script, multiplied by 256.
    print("STATUS",status)
    fil=open("log.txt", "a")
    if status == 2: #Exit code 0 means successful, exit code 1 means diverged, code 2 means other error
        print('Something went wrong in iteration',itr,', trying again once',file=fil)
        status=os.system('bash shelper.sh {} >>shlog.txt'.format(itr))/256
        print("STATUS2",status)
    elif status == 1:
        print('Simulation diverged in iteration',itr,file=fil)
        quit()
    elif status == 3:
        print('Simulation exceeded time limit in iteration',itr,file=fil)
        quit()
    if status != 0:
        print('Something went wrong twice in iteration',itr,file=fil)
        quit()
    fil.close()

    # Opening the simulation after run
    fdtd=lumapi.FDTD(filename="run{}/sim{}.fsp".format(itr,itr),hide=True)

    return fdtd,getResults(fdtd,props)
        
#### Logistics functions
#Quickly plot anything
def plot(x,y,name='plot',xl='Wavelength / nm',yl="Absorptance"): 
    fig, ax = plt.subplots(1,1)
    ax.plot(x,y, '-',  color='red', label=name)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.grid()
    ax.legend()
    fig.savefig("{}.png".format(name), bbox_inches='tight')
    plt.close()
        

#Export to File
def exportxt(result):
    global setpam,optpam,NAME
    abs_Si=abs(result["Transmission1"])-abs(result["Transmission3"])
    try:
        out=np.asarray([ [result["lambda"][i],abs_Si[i]] for i in range(len(abs_Si))])
        np.savetxt(NAME+"Thickness-{}.dat".format(round(setpam[optpam]/nm)) , out, delimiter=',')
    except:
        print("Saving Failed")

# FILE LOGGING #
def logit(x,fun,itr=0):
    if itr == 1000:
        fil=open("log.txt", "w")
        print("Iteration,   X,   FOM",file=fil)
        fil.close()
    else:
        fil=open("log.txt", "a")
        if itr != 0:
            print("{},   {},   {}".format(itr,round(float(x),3),round(fun,3)),file=fil)
        else:
            print("{},   {}".format(round(float(x),3),round(fun,3)),file=fil)
    fil.close()

#### Optimize FCT 
#Optimize on server or supercomp
def optimize(optpam,x0,bnd): # parameter to optimize optpam needs to be in the setpam dict.
    global setpam
    itr=1000 #Iteration will keep track of the current run number, it starts at 1000
    logit(0, 0, 1000)
    def f(x):
        global itr
        x=x*nm
        X=str(round(x/nm))  # The sweep variable in nm for printing
        # print('Current',X,'nm, IA:', end='')
        setpam[optpam]=x[0]
        p=setupFDTD(**setpam)
        sim,p=buildFDTD(p,"sim"+str(itr))
        if supercomp:
            sim,result=srunFDTD(sim, p, itr)    
        else:
            sim,result=runFDTD(sim, p)
        r=anaResult(result,optpam)
        itr+=1
        logit(X,(1-r)*100,itr)
        return r
    #resu = minimize(f,  x0=200, method='Powell', options={'xtol' : 1e-10,'ftol':0.001},bounds=Bounds(50,1000))
    resu = minimize(f,  x0=x0, method='trust-constr', tol=0.0001,bounds=bnd)
    # print('plotting optimum',f(resu.x))
    logit(resu.x,resu.fun,itr)
    return resu

#### SWEEP FCT 
#Sweep on server or supercomputer
def sweep(optpam,strt=80,end=100,stp=10):
    global setpam,supercomp
    logit(0, 0, 1000) # Start a log file for the results and progress
    xx=np.arange(strt,end+stp,stp)
    res=[]
    for x in xx:
        x=x*nm #The sweep variable in SI
        X=str(round(x/nm))  # The sweep variable in nm for printing
        print('Current',X,'nm, IA:', end='')
        setpam[optpam]=x
        setpam["ARthick"]=x+40*nm
        p=setupFDTD(**setpam)
        sim,p=buildFDTD(p,'sim'+X)
        if supercomp:
            sim,result=srunFDTD(sim, p, X)    
        else:
            sim,result=runFDTD(sim, p)
        r=anaResult(result,optpam)
        r=(1-anaResult(result,optpam))
        res.append([float(X),r])
        logit(X, r)
        print(r*100)
    res=np.asarray(res)
    np.savetxt(NAME+'sweepOut.dat', res, delimiter=',')
    plot(res[:,0],res[:,1],NAME+"Thickness-sweep",xl='Thickness AR / nm',yl="AM1.5G Absorptance")

#### FAR FIELD ANALYSIS METHODS
# Import data for Refractive Index against wavelength in um
try:
    nSiR=np.loadtxt("siliconR.txt", skiprows=1) 
    nSi=scinter.interp1d(nSiR[:,0],nSiR[:,1], fill_value='extrapolate')
except:
    print('siliconR not found, using generic n')
nG=scinter.interp1d(np.linspace(1,1000,100),np.ones(100)*4, fill_value='extrapolate')

# Calculate our FOM and plot the result, return the FOM
def anaResult(result,optpam):
    global setpam
    abs_Si=abs(result["Transmission1"])-abs(result["Transmission3"])
    ia=IA(result["lambda"],abs_Si)
    plot(result["lambda"],abs_Si,NAME+optpam+"-{} nm. IA {}".format(round(setpam[optpam]/nm),round(ia*100,1)))
    exportxt(result)
    return float(1-ia)

def getLamidx(result, lam): # Everything in Microns!
    lambdas=result["lambda"]
    idx=0
    while lambdas[idx]<lam:
        idx+=1
        if idx == len(lambdas):
            print("Lambda ",lam," out of range! Max ",lambdas[-1])
            break
    return idx


def fftrans(fdtd,result,monitor,lam=0,n=3.5,resolution=600,period=1):
    fname=NAME[:-1]+"."
    lambdas=result["lambda"]
    if lam==0:
        rmi,rma=0,len(lambdas)-1
    elif type(lam) != type([]):
        idx=getLamidx(result, lam)
        rmi,rma=idx,idx+1
    else:
        rmi=getLamidx(result, lam[0])
        rma=getLamidx(result, lam[1])+1
    K,E=[],[]
    for r in range(rmi,rma):
        fname=fname+str(r)+".dat"
        L=lambdas[r]
        ux = fdtd.farfieldux(monitor,r,resolution,resolution) 
        uy = fdtd.farfielduy(monitor,r,resolution,resolution)
        kx = ux*2*Pi/L*n(L)
        ky = uy*2*Pi/L*n(L)
        K.append( [ kx, ky ] )
        E.append(np.asarray(fdtd.farfield3d(monitor,r,resolution,resolution,1,period,period)))
    return np.asarray(K),np.asarray(E)

def mon_to_img(fdtd,result,mname,imname,lam,fieldtype='P'):
    # Exports a matrix of E,H,P field values at wavelength lam to an image file.
    # Wants a monitor label mname, output img file name imname, and which fields to 
    # export, fieldtype = 'E' or 'H' pr 'P', default 'P'.
   array=fdtd.getresult(mname, fieldtype)[fieldtype][:,:,0,getLamidx(result, lam),2]
   # array=255*normaliz2d((np.abs(np.real(array))))
   array=((np.abs(array)))
   # imageio.imwrite(imname,array)
   array=cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
   cv2.imwrite(imname,array)
   # plt.imshow(array)
   # plt.colorbar()
   # plt.savefig('esimm.png')

def array_to_img(array,imname):
    # Exports a matrix of E,H,P field values at wavelength lam to an image file.
    # Wants a monitor label mname, output img file name imname, and which fields to 
   # array=255*normaliz2d((np.abs(np.real(array))))
   # array=(20*np.log(np.abs(array)))
   # imageio.imwrite(imname,array)
   array=cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
   cv2.imwrite(imname,array)
   print("I exported")
   fig, ax = plt.subplots(1,1)
   fig.imshow(array)
   fig.colorbar()
   fig.savefig('test_mpl.png')

   
def calc_psd(mname,imname, fieldtype='P'):
    # Creates PSD matrix of E,H,P field values from a monitor.
    # Wants a monitor label mname, output img file name imname, and which fields to 
    # export, fieldtype = 'e' or 'h' pr 'p', default 'p'.
    mon_to_img(mname, imname, fieldtype)
    ftimg=do_fft(imname)
    arr=get_radial_avg(imname)
    return arr

def extract_vector(fdtd):
    pass


def eV_to_nm(energy):
    return cnt.c/(energy*cnt.e/cnt.h)*1e9

