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


libpath=""
sys.path.append(libpath) #Library Path
from lumerical_python_V101 import *


#### VERSION
VER=str(sys.argv[0].split('\\')[-1][-5:-3])
print("Version",VER)

## Robust way to get current working directory
path=(abspath(getsourcefile(lambda:0))).split('/')
pth=''
for st in path[0:len(path)-1]:
    pth+='/'+st 
sys.path.append("C:\\Program Files\\Lumerical\\v242\\api\\python\\") #Default windows lumapi path
sys.path.append("/opt/lumerical/v222/api/python/lumapi.py") #Default linux lumapi path
sys.path.append(pth) #Current directory



# SETUP FUNCTION
def setupFDTD(properties={}):
    # Assign variables for better readability
    DESIGN = properties.get("DESIGN", 3)
    array_size = properties.get("array_size", 1)
    boxsize = properties.get("boxsize", 0)
    subthick = properties.get("subthick", um)
    seedthick = properties.get("seedthick", 0)
    pradius = properties.get("pradius", 50 * nm)
    pheight = properties.get("pheight", 5e-8)
    ARthick = properties.get("ARthick", 2e-7)
    ARindex = properties.get("ARindex", 1.81)
    refin = properties.get("refin", 2.2)
    meshvariant = properties.get("meshvariant", 2)
    minmesh = properties.get("minmesh", 10)
    meshacc = properties.get("meshacc", 4)
    dmesh = properties.get("dmesh", 10 * nm)
    lambda_min = properties.get("lambda_min", 300 * nm)
    lambda_max = properties.get("lambda_max", 1200 * nm)
    frequency_points = properties.get("freq_points", 800)
    t_pos = properties.get("t_pos", -1 * nm)
    headspace = properties.get("headspace", 1 * um)
    fields = properties.get("fields", True)
    
    
    
# ### MESH Settings
# 
# There are two mesh size variables, one for the refined mesh region "dmesh", <br>
# and another for the mesh everywhere else "meshacc".<br>
# "mesh" defines the meshing algorithm (Conformal Variant 0, etc.)
# Conformal Variant 0: Metals and PEC get non-conformal meshing (stable)
# Conformal Variant 1: Metals and PEC get conformal meshing (may diverge easier)
# Conformal variant 2: The Yu-Mittra method 1    
# Explained: https://optics.ansys.com/hc/en-us/articles/360034382614-Selecting-the-best-mesh-refinement-option-in-the-FDTD-simulation-object

    meshdict={
        1: "staircase", 
        2: "conformal variant 0",
        3: "conformal variant 1", 
        4: "conformal variant 2"
    }
    mesh=meshdict[2] #Conformal variant 1, for metal

# ## Define Constructors for Objects
    properties={}
    
    #### Material Fit Settings
    properties["materials"]=[]
    
    #Silicon small steps
    props_material= OrderedDict([
        ('name', "Silicon-small-steps"),
        ("max coefficients", 16),
        ("imaginary weight", 7),
        ("tolerance", 1e-4),
        ("improve numerical stability", 1),
        ("make fit passive", 1),
        ("specify fit range", 1),
        ("wavelength max", 1200*nm), 
        ("wavelength min", 400*nm),
        
        ])
    
    properties["materials"].append(props_material)
        
        
    
    
    #### FDTD      
    props_fdtd=OrderedDict([
    # Geometry
        ("dimension" , "3D"),
        ("x" ,0),
        ("y" ,0),
        ("z max" , pheight+ARthick+headspace + 6e-7),
        ("z min" , -subthick),
        ("x span" ,boxsize),
        ("y span" ,boxsize),
    # Meshing and Boundary Conditions
        ("min mesh step" ,minmesh),
        ("allow symmetry on all boundaries", 1),
        ("x min bc" , "symmetric"),
        ("y min bc" , "anti-symmetric"),
        ("x max bc" , "symmetric"),
        ("y max bc" , "anti-symmetric"),
        # ("x min bc" , "periodic"),
        # ("y min bc" , "periodic"),
        ("z min bc" , "metal"),
        ("z max bc" , "PML"),        
        ("mesh refinement" ,mesh),
        ("mesh accuracy", meshacc),
        ("index" ,1),
        ("dt stability factor" ,.99),
        ("simulation time" ,1e-10),
        ("auto shutoff min" , 1e-6),
        ("auto shutoff max" , 1e1),
    #PML settings
    #PML profile: 1: Standard, 2: Stabilized, 3: Steep Angle, 4: Custom
        ("pml profile", 3),
        ("pml layers", 64),
        # ("pml kappa", 2),
        # ("pml alpha", 5),
        # ("pml sigma", 1) ,
    #PML settings EXTREME
    #PML profile: 1: Standard, 2: Stabilized, 3: Steep Angle, 4: Custom
        # ("pml profile", 4),
        # ("pml layers", 64),
        # ("pml kappa", 2),
        # ("pml alpha", 5),
        # ("pml sigma", 1) ,
    #Source Settings
        ('global source set wavelength', 1),
        ('global source wavelength start', lambda_min),
        ('global source wavelength stop', lambda_max),
    #Monitor Settings
        ('global monitor frequency points', frequency_points),
        ('global monitor use source limits', 1),
        ("global monitor use wavelength spacing", 0)

    ])
    properties["FDTD"]=props_fdtd
    
    #### Patterned from GDS   
    designname={ #Filename of the GDS design without extension .gds
    1:"Spinodal",
    2:"Honeycomb",
    # 2: "2umHONE",
    3:"HUD-Holes",
    4:"Periodic",
    5:"Honeycomb1",
    }
    cellname={ # Cell name of the GDS design cell
    1:"main-centered",
    2:"main",
    3:"main",
    4:"main",
    5:"main-centered",
    }
    #Layer must be = 1 and database type = 1000 for now!
    layername={
        1: 1,
        2: 0,
        3: 1,
        4: 1,
        5: 1
        }
    designsize={ # Lateral exact sizes of the designs, only used if boxsize is unspecified
        1: 10e-6,
        2: 15019e-9,
        3: 15177e-9,
        4: 1325e-9,
        5: 15020e-9
        }
    # If the boxsize was not overridden, the lateral dimensions are design specific.
    if boxsize == 0 or DESIGN == 4:
        boxsize=designsize[DESIGN]
        if DESIGN != 4:
            props_fdtd["x span"]=designsize[DESIGN]
            props_fdtd["y span"]=designsize[DESIGN]
        else:
            props_fdtd["x span"]=designsize[DESIGN]*array_size
            props_fdtd["y span"]=765e-9*array_size
            boxsize=max(props_fdtd["x span"],props_fdtd["y span"])
        properties["FDTD"]=props_fdtd
        
    # GDS Layer    
    props_gds =OrderedDict([
        ("name",     designname[DESIGN]),
        ("cell",     cellname[DESIGN]),
        ("layer",    layername[DESIGN]),
        # ("material", "Si4N3-beliaev"),
        # ("material", "<Object defined dielectric>"),
        # ("index", 3.5),
        ("material", "SiO2 (Glass) - Palik Copy 1"),
        ("z min",    -pheight),
        ("z max",    0),
        # ("override mesh order from material database", True), This does not WORK!!
        # ("mesh order", 99)
    ])
    properties["gds"]=props_gds
    
        
   
#### Add pillar/hole object
    props_PL=OrderedDict([
    ("name", "pillar"),
    ("x", 0),
    ("radius", pradius),
    ("y", 0),
    ("z max", pheight),
    ("z min", 0),
    # ("material", "etch"),
    # ("material", "Si (Silicon) - Palik Copy 1"),
    # ("material", "<Object defined dielectric>"),
    ("index", refin),
    ("material", "Silicon-small-steps"),
    ("override mesh order from material database", True),
    ("mesh order", 1),
    ])
    properties["PL"]=props_PL
    
#### Add pillar/hole object AR coating
    props_PLAR=OrderedDict([
    ("name", "pillar-AR"),
    ("x", 0),
    ("radius", pradius+ARthick),
    ("y", 0),
    ("z max", pheight+ARthick),
    ("z min", 0),
    # ("material", "etch"),
    # ("material", "Si (Silicon) - Palik Copy 1"),
    ("material", "<Object defined dielectric>"),
    ("index", ARindex),
    ("override mesh order from material database", True),
    ("mesh order", 10),
    ])
    properties["PLAR"]=props_PLAR

    
    #### AR Flat
    props_AR=OrderedDict([
    ("name", "AR"),
    ("x span", 1.2*boxsize),
    ("x", 0),
    ("y span", 1.2*boxsize),
    ("y", 0),
    ("z max", ARthick),
    ("z min", 0),
    # ("material", "Si4N3-beliaev"),
    # ("material", "<Object defined dielectric>"),
    ("index", ARindex),
    # ("material", "SiO2 (Glass) - Palik Copy 1"),
    ("override mesh order from material database", True),
    ("mesh order", 8)
    ])
    properties["AR"]=props_AR
   
    #### Substrate
    props_Si=OrderedDict([
    ("name", "substrate"),
    ("x span", 1.2*boxsize),
    ("x", 0),
    ("y span", 1.2*boxsize),
    ("y", 0),
    ("z max", 0),
    ("z min", -2*subthick),
    ("material", "Silicon-small-steps"),
    # ("material", "<Object defined dielectric>"),
    # ("index", refin),    
    
    ("override mesh order from material database", True),
    ("mesh order", 10)
    ])
    properties["Si"]=props_Si
    
    #### Back Reflector
    props_BR=OrderedDict([
    ("name", "BR"),
    ("x span", 1.2*boxsize),
    ("x", 0),
    ("y span", 1.2*boxsize),
    ("y", 0),
    ("z max", -subthick),
    ("z min", -subthick-10e-7),
    # ("material", "<Object defined dielectric>"),
    # ("index", 2.12),    
    # ("material", "SiO2 (Glass) - Palik Copy 1")    
    # ("material", "Ag (Silver) - Palik (0-2um)")    
    ("material", "Ag (Silver) - CRC Copy 1")    
    ])
    properties["BR"]=props_BR 
       
       
    #### Fine Mesh around Pattern
    properties["mesh"]=[]
    props_mesh=OrderedDict([
        ("name", designname[DESIGN]+"-mesh"),
        ("x", 0),
        ("x span", boxsize*1.2),
        ("y", 0),
        ("y span", boxsize*1.2),
        ("z max", pheight+200*nm),
        ("z min", -100*nm),
        ("override x mesh",1),
        ("override y mesh",1),
        ("override z mesh",1),
        ("set maximum mesh step",1),
        ("dx",dmesh),
        ("dy",dmesh),
        ("dz",dmesh)
    ])
    properties["mesh"].append(props_mesh)
    
    #### Fine Mesh around Si-BR interface
    props_mesh2=OrderedDict([
        ("name", "Si-BR"),
        ("x", 0),
        ("x span", boxsize*1.2),
        ("y", 0),
        ("y span", boxsize*1.2),
        ("z max", -subthick+2*meshacc),
        ("z min", -subthick-2*meshacc),
        ("override x mesh",1),
        ("override y mesh",1),
        ("override z mesh",1),
        ("set maximum mesh step",1),
        ("dx",dmesh),
        ("dy",dmesh),
        ("dz",dmesh)
    ])
    # properties["mesh"].append(props_mesh2)
    
    #### Source (Plane Wave)
    props_source=OrderedDict([
    ("name", "source"),
    ("direction", "backward"),
    ("injection axis", "z-axis"),
    ("x", 0),
    ("x span", 1.2*boxsize),
    ("y", 0),
    ("y span", 1.2*boxsize),
    ("z", pheight+ARthick+headspace),
    ("plane wave type", "Bloch/periodic"),
    ("override global source settings", 0),
    # ("wavelength start", lambda_min),
    # ("wavelength stop", lambda_max),
    ("polarization angle", 90)
    ])
    properties["source"]=props_source
    
# #### Source (Total Field Scattered Field)    
#     props_source=OrderedDict([
#     ("name", "tfsf-source"),
#     ("direction", "backward"),
#     ("injection axis", "z-axis"),
#     ("x", 0),
#     ("x span", boxsize-25*nm),
#     ("y", 0),
#     ("y span", boxsize-25*nm),
#     ("z max", pheight + headspace),
#     ("z min", -300*nm),
#     ("wavelength start", 300*1e-9),
#     ("wavelength stop", 1200*1e-9),
#     ("polarization angle", 90)
#     ])
#     properties["source"]=props_source
    
    #### Monitors
    # Define names of monitors and their z-values.
    Mnames={    # Commenting out the name here disables the monitor entirely
        1: "Reflection",
        # 2: "Transmission1",
        3: "Transmission2",
        # 4: "Transmission3",
        # 5: "Transmission4",
           }
    Mzees={    
        1: pheight+ARthick+headspace+3e-7,
        2: -pheight/2,
        3: min(-minmesh, -10*nm),
        4: -pheight -100*nm,
        5: -subthick-seedthick-5.01e-7
           }
    # Should the field vectors be saved?
    fields=True

    props_mon = [ OrderedDict([ # For each monitor we make a properties dictionary.
        ("name",Mnames[i]),
        ("x", 0.),
        ("y", 0.),
        ("monitor type",7),
        ("x span",boxsize*1.2),
        ("y span",boxsize*1.2),
        ("z",Mzees[i]),
        #Disable all fields except power
        ("output Ex",fields),
        ("output Ey",fields),
        ("output Ez",fields),
        ("output Hx",fields),
        ("output Hy",fields),
        ("output Hz",fields),
        ("output Px",fields),
        ("output Py",fields),
        ("output Pz",fields),
    ])
    for i in Mnames.keys() ] #The only differences are names and z-values.
    properties["mon"]=props_mon
    
   
#### Absorbed Power Boxes
 # Define names of 3D monitor boxes and their geometry.
    pabsName={ # names of the boxes   
        1: "Pabs",
#        2: "Pabs SF",
        }
    pabsDim={ # Geometry of the boxes    
        1: { "xy" : boxsize, "z" : [ -subthick+minmesh, -minmesh ] },
 #       2: { "xy" : boxsize,  "z" : [ pheight + headspace + 100*nm, -500*nm ] }
        }
    
    
    props_pow= [OrderedDict([ 
    #The name has to be scriptID in the Object tree, so make a tuple with the actual name in the second entry,
    #while in the first entry the ordered dict is stored.                           
        ("name", pabsName[i]),
        ("x", 0),
        ("x span", pabsDim[i]["xy"]),
        ("y", 0),
        ("y span", pabsDim[i]["xy"]),
        ("z", pabsDim[i]["z"][1] - (abs(pabsDim[i]["z"][0])+abs(pabsDim[i]["z"][1]))/2),
        ("z span",   abs(pabsDim[i]["z"][0])+abs(pabsDim[i]["z"][1])),
        # #Disable all fields except power
        # ("output Ex",fields),
        # ("output Ey",fields),
        # ("output Ez",fields),
        # ("output Hx",fields),
        # ("output Hy",fields),
        # ("output Hz",fields),
        # ("output Px",fields),
        # ("output Py",fields),
        # ("output Pz",fields),           
        ]) for i in ((pabsName.keys())) ]
    
    properties["pow"]=props_pow
    
    #Return all properties
    return properties

#### Build Simulation
def buildFDTD(props,fname='sim',hide=True):
    global setpam
    #Create simulation object and load an FSP that has optimized material fits!
    fdtd=lumapi.FDTD(filename="material-setup.fsp",hide=hide)
    #Clear All Objects
    fdtd.switchtolayout()
    fdtd.deleteall()
    
    #Add FDTD region
    fdtd.addfdtd(properties=props["FDTD"])
    #Add Flat AR
    # fdtd.addrect(properties=props["AR"])
    #Add substrate
    fdtd.addrect(properties=props["Si"])
    #Add Backreflector
#    fdtd.addrect(properties=props["BR"])

    #Add pattern AR
    # fdtd.addcircle(properties=props["PLAR"])
    #Add pattern Obj
    fdtd.addcircle(properties=props["PL"])
    # multiGDS(fdtd, props)   # For creating real arrays
    # addGDS(fdtd, props["gds"]) # For importing the gds only once
    
    #Add FINE meshes
    for msh in props["mesh"]:
        fdtd.addmesh(properties=msh)
    #fdtd.addmesh(properties=props["mesh"])
    
    #Add 3D absorption monitors
    # for pp in props["pow"]:
    #     fdtd.addobject("pabs",properties=pp)
    
    #Add source
    fdtd.addplane(properties=props["source"])
    # fdtd.addtfsf(properties=props["source"])
    
    #add monitors
    #[fdtd.addpower(properties=props_mon[i]) for i in range(len(props_mon))]
    for pm in props["mon"]:
        # fdtd.adddftmonitor(**pm)
        fdtd.addpower(properties=pm)
        
    # Set material fiitings
    if "materials" in props.keys():
        for dic in props["materials"]:
            set_material_fit(fdtd, dic)

    
    #Save and return
    fdtd.save(fname+".fsp")
    return fdtd,props


def complete_run_analysis(properties):
    props=setupFDTD(properties)
    fdtd,props=buildFDTD(props, fname=NAME,hide=True)
    pass

#### MAIN PART START

#### SETUP PARAMETERS
#Some very general options are exposed to the parameters of the FDTD SETUP function.
#These can be conveniently set in the dictionary below and changed for repeated calls to the setup.
setpam={
    'subthick': 2.  *um, # Substrate thickness
    'pheight': 175 * nm,
    'pradius': 75 * nm,
    'ARthick':  55  *nm,     # Total AR thickness including pattern
    'ARindex': 1.81,
    'meshacc':   4,    # Global mesh minimum step (in nm in Lumerical)
    'minmesh':   4  *nm,
    # 'dmesh':     4  *nm,     # Refined mesh region step (in m in lumerical)
    # 'DESIGN':   2,
    'refin':    3.55,          # Refractive index of <Object defined material>
    'boxsize':  .300   *um,       # Override lateral size of simulation
    'headspace': 500*nm,
    'lambda_min': 400 *nm,
    'lambda_max': 1100 *nm,
    # 'array_size': 1,
}


#### FILENAME 
def getFileName():
    global setpam
    NAME=str(round(setpam["subthick"]/nm))
    NAME+="nmSi-PHOTOwAbsNoAR"+"-withSiBumpH"+str(round(setpam["pheight"]*1e9)) \
        +"R"+str(round(setpam["pradius"]*1e9))+'nm'
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
    NAME=NAME+'_400nmfit'
    NAME=NAME+"_V"+VER 
    return NAME



# #### Single Run
# NAME=getFileName()
# props=setupFDTD(setpam)
# F,props=buildFDTD(props, fname=NAME,hide=True)

#### loop
thicknesses=[500,1000,2000]
heights=[75,175]
radii=[75,150]
periodicities=[350,600,900]

for t in thicknesses[:1]:
    for h in heights[1:]:
        for r in radii[:1]:
            for p in periodicities[:1]:
                if 2*r >= p:
                    continue
                setpam={
                    'subthick': t  *nm, # Substrate thickness
                    'pheight':  h  *nm,
                    'pradius':  r  *nm,
                    'boxsize':  p  *nm,       # Override lateral size of simulation
                    'ARthick':  72  *nm,     # Total AR thickness including pattern
                    'ARindex': 1.81,
                    'meshacc':   6,    # Global mesh minimum step (in nm in Lumerical)
                    'minmesh':   4 *nm,
                    'dmesh':     4 *nm,     # Refined mesh region step (in m in lumerical)
                    # 'DESIGN':   2,
                    'refin':    3.55,          # Refractive index of <Object defined material>
                    'headspace': .5*um,
                    'lambda_min': eV_to_nm(3)*nm,
                    'lambda_max': eV_to_nm(1.)*nm,
                    'freq_points': 1600,
                    # 'array_size': 1,
                }
                              
                
                NAME=getFileName()
                print(NAME)

                props=setupFDTD(setpam)
                F,props=buildFDTD(props, fname=NAME,hide=True)
                # F=runFDTD(F)
                # result=getResults(F,p)
