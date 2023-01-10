from __future__ import division, print_function
import sys
import re
import os
import shutil
import subprocess
import numpy as np
from numpy.random import permutation
from numpy import sqrt
import math
from sklearn.cluster import DBSCAN

#     stiffrange =[300,500,700]
def main():
    stiffrange = np.arange(300,501,100)

    cwd = os.getcwd()
    # restartfile  = 'restart_attach2.dat'
    radii = np.arange(1.5,2.1,0.25)
    vcurvs = np.arange(3000,10001,1000)
#     restartfile = 'restart_attach2.dat'
    
    for Rval in radii:
        for stiff in stiffrange:
            for vcurv in vcurvs:
                fulldirname = os.path.join(cwd, 
                'NPHrand_R_'+str(Rval)+'_S_'+str(stiff)+'_Tcurv_'+str(vcurv))
                if not os.path.exists(fulldirname):
                    os.makedirs(fulldirname)                  
    #             if not os.path.exists(os.path.join(fulldirname, spherecoords)):
    #                 shutil.copy(spherecoords, fulldirname)  
#                 if not os.path.exists(os.path.join(fulldirname, restartfile)):
#                     shutil.copy(restartfile, fulldirname)  
                os.chdir(fulldirname)
                in_name = change_inlocal(Rval,stiff,vcurv)
                submit_generator_myriad(fulldirname,Rval,stiff,vcurv)
                subprocess.run(["qsub", "submit_myriad.pbs"])
                os.chdir(cwd)


def submit_generator_myriad(fulldirname,Rval,stiff,vcurv):
    submit_name = 'submit_myriad.pbs'
    pbs_job_name = 'NPHrnadD_R_'+str(Rval)+'_S_'+str(stiff)+'_Tcurv_'+str(vcurv)
    job_script = open(os.path.join(fulldirname,submit_name),'w')
    job_script.write('#!/bin/bash -l \n')
    job_script.write('#$ -S /bin/bash \n')
    job_script.write('#$ -l h_rt=31:00:0 \n')
    job_script.write('#$ -l mem=2G \n')
    job_script.write('#$ -l tmpfs=15G \n')
    job_script.write('#$ -N '+str(pbs_job_name) +'\n')
    job_script.write('#$ -pe mpi 1 \n')
    job_script.write('#$ -wd ' + str(fulldirname) +  '\n')
#     job_script.write('#Local2Scratch \n')
#     job_script.write('origwd=$PWD \n')
#     job_script.write('cp * $TMPDIR \n')
#     job_script.write('cd $TMPDIR \n')
    job_script.write('gerun /home/ucbtljk/bin/lmp_mpi < in.local > my.out \n')
#     job_script.write('cp * $origwd \n')
    job_script.close()
    os.chmod(os.path.join(fulldirname,submit_name),0o744)



def change_inlocal(Rval,stiff,vcurv):
#     stiff = 300
    stiff2 = 400
    tilt = 90
#     rfrac = 4
    gap = 5
    epsB2 =3
    epsB1 =3
    restartfile = 'restart_equi2.dat'
    
    # nsteps = 100
################## Spherical Membrane Coordinates #########################
    mem = 'tube'
    if mem == 'tube':
   
        bead_spacing = 0.9
        sigma_mem = 1
        Rcyl = 27
        lbox=math.fabs(200)
        boxlength = lbox*2
        lattice = TubeLattice(Rcyl, boxlength, sigma_mem, bead_spacing)
        cell_radius = Rcyl-sigma_mem
    

     # Simulation parameters
    seed = np.random.seed(425) # random seed for permutations
    
    
    # Model parameters
    factor = 1
    a0 = 0.37*factor  #lower bead shell distance between proteins if the angle is alpha0 
    
    radius = 0.5*factor # Radius of a spiral particle
    dr = 0.2*factor     # overlap length
    radius_o = radius-(dr/2)
    
    dz = 0
    
    xstepB1l =  0.03*factor
    xstepB1r = 0.03*factor
    xstepB2 = 0.03*factor

    thermo_initial = 1000	
    timestep = 0.01
    run_attach = 20000
    run_equilibrate = 1000000
    thermo_disassemble = 1000	
    thermo_attach = 1000
    thermo_equilibrate = 1000
    
    run_disassemble = 100
    thermo_final = 1000						# thermo for "run: final"
    run_final = 100000	
    run_B2 = vcurv
    run_B1 = 100


    bondstrength2 = 0
    timestep = 0.01
    outputstep = 10000

    ############ Generate spiral positions ######################
    cell_circ= 2*np.pi*cell_radius
    anglestep = 2*np.arcsin((2*radius+a0)/(2*cell_radius))
    bead_per_circ = int(np.floor(2*np.pi/anglestep))
    NcirclesB1l = 3
    NproteinB1l = int(NcirclesB1l*bead_per_circ)
    

    NcirclesB2 = 1
    NproteinB2 = int(NcirclesB2*bead_per_circ)
    
    NcirclesB1r = NcirclesB1l
    NproteinB1r = int(NcirclesB1r*bead_per_circ)
    
    xstart = xstepB1l*(NproteinB2+NproteinB1l+NproteinB1r)/2


    spiral = []
# Glue filament
    for n in range(NproteinB1l+NproteinB2+NproteinB1r):
        P1,P2,P3 = proteingenerator(radius_o,xstart,xstepB1l,n,cell_radius,0)
        protein = [P1,P2,P3]
        protein_rot = Proteinrotator(P1,P2,P3,n*anglestep)
        spiral.extend(protein_rot[:3])
    spiralB1l = spiral[:3*NproteinB1l]
    spiralB2 = spiral[3*NproteinB1l:3*NproteinB1l+3*NproteinB2]
    spiralB1r = spiral[3*NproteinB1l+3*NproteinB2:]




    ########## Bond distances #####################################
#     Define 3 filament parameters 

# Glue Filament left
    bondstrengthB1l = stiff2
    RB1l = cell_radius*0.6
    alpha0B1l = 2*np.arcsin((radius+a0/2)/(RB1l))
    tauB1l = -1*tilt*np.pi/180
    epsilonB1l = epsB1

    
# Contracting Filament
    bondstrengthB2 = stiff
    RB2 = Rval
    alpha0B2 = 2*np.arcsin((radius+a0/2)/(RB2))
    tauB2 = 0*np.pi/180
    epsilonB2 = epsB2


# Glue Filament right   
    bondstrengthB1r = stiff2
    RB1r = cell_radius*0.6
    alpha0B1r = alpha0B1l
    tauB1r = tilt*np.pi/180
#     tauBr = 60*np.pi/180
    epsilonB1r = epsilonB1l

#     Glue interaction strength
#     epsilonI= 0
#     r_cutoff = 1.122462*1.3*factor
    
    # Filament A1
    [P1,P2,P3] = proteingenerator(radius_o,0,0,0,RB1l,tauB1l)
    protein = [P1,P2,P3]
    protein_rot = Proteinrotator(P1,P2,P3,alpha0B1l)
    res_B1l = bondcalculator(protein,protein_rot)
    
    # Filament C
    [P1,P2,P3] = proteingenerator(radius_o,0,0,0,RB2,tauB2)
    protein = [P1,P2,P3]
    protein_rot = Proteinrotator(P1,P2,P3,alpha0B2)
    res_B2 = bondcalculator(protein,protein_rot)
    
    # Filament B1
    [P1,P2,P3] = proteingenerator(radius_o,0,0,0,RB1r, tauB1r)
    protein = [P1,P2,P3]
    protein_rot = Proteinrotator(P1,P2,P3,alpha0B1r)
    res_B1r = bondcalculator(protein,protein_rot)


#     Initial state
    [P1, P2, P3] = proteingenerator(radius_o, 0, 0, 0, cell_radius, 0)
    alpha0 = 2*np.arcsin((radius+a0/2)/(cell_radius))
    protein = [P1, P2, P3]
    protein_rot = Proteinrotator(P1, P2, P3, alpha0)
    res_sB1r = bondcalculator(protein, protein_rot)
    res_sB1l = res_sB1r
    res_sB2 = res_sB1r
    
#################################################################
    if mem == 'sphere':
        Nmembrane = len(x_mem)
    elif mem == 'tube':
        Nmembrane = len(lattice)
    Nspiral = NproteinB1l+NproteinB2+NproteinB1r
    Nbonds = (NproteinB1l-1)*9+(NproteinB2-1)*9+(NproteinB1r-1)*9
    Natoms=Nmembrane+Nspiral*3 # total number of 'atoms'
    Bondtypes = (NproteinB1l-1)*9+(NproteinB2-1)*9+(NproteinB1r-1)*9 #*5 instead of 3 for 5 filaments
    Atomtypes = 3*Nspiral+1
    

    
    
  
    
    #################################################################
    #########################  WRITE FILE ###########################
    outfile='tripplehelix.in'
    f=open(outfile,'w')
    f.write('LAMMPS data file generated by tc387 with a python script for hex membrane\n')
    f.write('\n')
    f.write(str(Natoms) + '   atoms\n')
    f.write(str(Nbonds) + '   bonds\n')
    f.write('\n')
    f.write(str(Atomtypes) + ' atom types\n')
    f.write(str(Bondtypes)+' bond types\n')
    f.write('\n')

    f.write(str(-lbox)+' '+str(lbox)+' xlo xhi \n')
    f.write(str(-lbox)+' '+str(lbox)+' ylo yhi \n')
    f.write(str(-lbox)+' '+str(lbox)+' zlo zhi \n')

    f.write('\n')
    f.write('Masses\n')
    f.write('\n')
    for ii in range(Atomtypes):
        f.write(str(ii+1)+' 1\n')
    f.write('\n')

    f.write('Atoms # hybrid\n')
    f.write('\n')

    for ii in range(Nmembrane):
        if mem == 'sphere':
            r = np.sqrt(x_mem[ii]**2+y_mem[ii]**2+z_mem[ii]**2)
            f.write(str(ii+1)+' '+ '1' +' ' + str(x_mem[ii]) + ' ' + str(y_mem[ii]) + ' ' + str(z_mem[ii]) +' 1 1 0  ' +   str(x_mem[ii]/r) + ' ' +str(y_mem[ii]/r) + ' ' + str(z_mem[ii]/r) + ' ' + str(ii+1) + ' \n')
        elif mem == 'tube':
            r = np.sqrt(lattice[ii][1]**2+lattice[ii][2]**2)
            f.write(str(ii+1)+' '+ '1' +' ' + str(lattice[ii][0]) + ' ' + str(lattice[ii][1]) + ' ' + str(lattice[ii][2]) +' 1 1 0  ' +   str(0) + ' ' +str(lattice[ii][1]/r) + ' ' + str(lattice[ii][2]/r) + ' ' + str(ii+1) + ' \n')


# FILAMENT B1 left
    for ii in range(NproteinB1l):
        # inner particles
        f.write(str(1+Nmembrane +ii*3) + '  ' + str(3*ii+2) + '  ' + str(spiralB1l[3*ii][0]-gap) + ' ' + str(spiralB1l[3*ii][1]) + ' ' +  str(spiralB1l[3*ii][2]) + '   1 1 0   0 0 0   '+ str(ii+1) + '\n')
        # outer particles
        f.write(str(2+Nmembrane +ii*3) + '  ' + str(3*ii+3) + '  ' + str(spiralB1l[3*ii+1][0]-gap) + ' ' + str(spiralB1l[3*ii+1][1]) + ' ' +  str(spiralB1l[3*ii+1][2]) + '   1 1 0   0 0 0   '+ str(ii+1) + '\n')
        # upper particles
        f.write(str(3+Nmembrane +ii*3) + '  ' + str(3*ii+4) + '  ' + str(spiralB1l[3*ii+2][0]-gap) + ' ' + str(spiralB1l[3*ii+2][1]) + ' ' +  str(spiralB1l[3*ii+2][2]) + '   1 1 0   0 0 0   '+ str(ii+1) + '\n')

# FILAMENT B2
    for ii in range(NproteinB2):
        # inner particles
        f.write(str(3*NproteinB1l+1+Nmembrane +ii*3) + '  ' + str(3*NproteinB1l+3*ii+2) + '  ' + str(spiralB2[3*ii][0]) + ' ' + str(spiralB2[3*ii][1]) + ' ' +  str(spiralB2[3*ii][2]) + '   1 1 0   0 0 0   '+ str(NproteinB1l+ii+1) + '\n')
        # outer particles
        f.write(str(3*NproteinB1l+2+Nmembrane +ii*3) + '  ' + str(3*NproteinB1l+3*ii+3) + '  ' +  str(spiralB2[3*ii+1][0]) + ' ' + str(spiralB2[3*ii+1][1]) + ' ' +  str(spiralB2[3*ii+1][2]) + '   1 1 0   0 0 0   '+ str(NproteinB1l+ii+1) + '\n')
        # upper particles
        f.write(str(3*NproteinB1l+3+Nmembrane +ii*3) + '  ' + str(3*NproteinB1l+3*ii+4) + '  ' + str(spiralB2[3*ii+2][0]) + ' ' + str(spiralB2[3*ii+2][1]) + ' ' +  str(spiralB2[3*ii+2][2]) + '   1 1 0   0 0 0   '+ str(NproteinB1l+ii+1) + '\n')

# FILAMENT B1 right
    for ii in range(NproteinB1r):
        # inner particles
        f.write(str(3*(NproteinB1l+NproteinB2)+1+Nmembrane +ii*3)+ '  ' + str(3*(NproteinB1l+NproteinB2)+3*ii+2) + '  ' + str(spiralB1r[3*ii][0]+gap) + ' ' + str(spiralB1r[3*ii][1]) + ' ' +  str(spiralB1r[3*ii][2]) + '   1 1 0   0 0 0   '+ str(NproteinB2+NproteinB1l+ii+1) + '\n')
        # outer particles
        f.write(str(3*(NproteinB1l+NproteinB2)+2+Nmembrane +ii*3)+ '  ' + str(3*(NproteinB1l+NproteinB2)+3*ii+3) + '  ' +str(spiralB1r[3*ii+1][0]+gap) + ' ' + str(spiralB1r[3*ii+1][1]) + ' ' +  str(spiralB1r[3*ii+1][2]) + '   1 1 0   0 0 0   '+ str(NproteinB2+NproteinB1l+ii+1) + '\n')
        # upper particles
        f.write(str(3*(NproteinB1l+NproteinB2)+3+Nmembrane +ii*3)+ '  ' + str(3*(NproteinB1l+NproteinB2)+3*ii+4) + '  ' +str(spiralB1r[3*ii+2][0]+gap) + ' ' + str(spiralB1r[3*ii+2][1]) + ' ' +  str(spiralB1r[3*ii+2][2]) + '   1 1 0   0 0 0   '+ str(NproteinB2+NproteinB1l+ii+1) + '\n')

#FILAMENT A1 BONDS
    f.write('\n')
    f.write('Bonds \n')
    f.write('\n')
#FILAMENT A1 BONDS
    for ii in range(NproteinB1l-1):
        # bond inner inner
        f.write(str(9*ii+1) + '  ' + str(9*ii+1) + '  ' +   str(1+Nmembrane +ii*3) + ' ' + str(1+Nmembrane +(ii+1)*3) + '\n')
        # bond upper outer
        f.write(str(9*ii+2)+ '  ' + str(9*ii+2) + '  ' +  str(1+Nmembrane +ii*3) + ' ' + str(2+Nmembrane +(ii+1)*3)  + '\n')        
         # bond upper inner
        f.write(str(9*ii+3)+ '  ' + str(9*ii+3) + '  ' +  str(1+Nmembrane +ii*3)  + ' ' + str(3+Nmembrane +(ii+1)*3)  + '\n')       
        
        # bond outer upper
        f.write(str(9*ii+4)+ '  ' + str(9*ii+4) + '  ' +  str(2+Nmembrane +ii*3) + ' ' + str(1+Nmembrane +(ii+1)*3) + '\n')
        # bond outer outer
        f.write(str(9*ii+5)+ '  ' + str(9*ii+5) + '  ' +  str(2+Nmembrane +ii*3) + ' ' + str(2+Nmembrane +(ii+1)*3) + '\n')
        # bond outer inner
        f.write(str(9*ii+6)+ '  ' + str(9*ii+6) + '  ' +  str(2+Nmembrane +ii*3) + ' ' + str(3+Nmembrane +(ii+1)*3) + '\n')
        
        # bond inner upper
        f.write(str(9*ii+7)+ '  ' + str(9*ii+7) + '  ' +  str(3+Nmembrane +ii*3) + ' ' + str(1+Nmembrane +(ii+1)*3) +'\n')
        # bond inner outer
        f.write(str(9*ii+8)+ '  ' + str(9*ii+8) + '  ' +  str(3+Nmembrane +ii*3) + ' ' + str(2+Nmembrane +(ii+1)*3) +'\n')
        # bond inner inner
        f.write(str(9*ii+9)+ '  ' + str(9*ii+9) + '  ' +  str(3+Nmembrane +ii*3) + ' ' + str(3+Nmembrane +(ii+1)*3) +'\n')

    
#FILAMENT B1 BONDS
    for ii in range(NproteinB2-1):
        # bond inner inner        
        f.write(str((NproteinB1l-1)*9+9*ii+1)+ '  ' + str((NproteinB1l-1)*9+9*ii+1) + '  ' +  str(3*NproteinB1l+1+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+1+Nmembrane +(ii+1)*3) + '\n')
        # bond upper outer
        f.write(str((NproteinB1l-1)*9+9*ii+2)+ '  ' + str((NproteinB1l-1)*9+9*ii+2) + '  ' +   str(3*NproteinB1l+1+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+2+Nmembrane +(ii+1)*3)  + '\n')        
         # bond upper inner
        f.write(str((NproteinB1l-1)*9+9*ii+3)+ '  ' + str((NproteinB1l-1)*9+9*ii+3) + '  ' +  str(3*NproteinB1l+1+Nmembrane +ii*3)  + ' ' + str(3*NproteinB1l+3+Nmembrane +(ii+1)*3)  + '\n')       
        
        # bond outer upper
        f.write(str((NproteinB1l-1)*9+9*ii+4)+ '  ' + str((NproteinB1l-1)*9+9*ii+4) + '  ' +   str(3*NproteinB1l+2+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+1+Nmembrane +(ii+1)*3) + '\n')
        # bond outer outer
        f.write(str((NproteinB1l-1)*9+9*ii+5) + '  ' + str((NproteinB1l-1)*9+9*ii+5) + '  ' +  str(3*NproteinB1l+2+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+2+Nmembrane +(ii+1)*3) + '\n')
        # bond outer inner
        f.write(str((NproteinB1l-1)*9+9*ii+6)+ '  ' + str((NproteinB1l-1)*9+9*ii+6) + '  ' +  str(3*NproteinB1l+2+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+3+Nmembrane +(ii+1)*3) + '\n')
        
        # bond inner upper
        f.write(str((NproteinB1l-1)*9+9*ii+7) + '  ' + str((NproteinB1l-1)*9+9*ii+7) + '  ' +  str(3*NproteinB1l+3+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+1+Nmembrane +(ii+1)*3) +'\n')
        # bond inner outer
        f.write(str((NproteinB1l-1)*9+9*ii+8)+ '  ' + str((NproteinB1l-1)*9+9*ii+8) + '  ' +  str(3*NproteinB1l+3+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+2+Nmembrane +(ii+1)*3) +'\n')
        # bond inner inner
        f.write(str((NproteinB1l-1)*9+9*ii+9)+ '  ' + str((NproteinB1l-1)*9+9*ii+9) + '  ' +  str(3*NproteinB1l+3+Nmembrane +ii*3) + ' ' + str(3*NproteinB1l+3+Nmembrane +(ii+1)*3) +'\n')

 #FILAMENT C BONDS
    for ii in range(NproteinB1r-1):
        # bond inner inner
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+1) + '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+1) + '  ' +  str(3*(NproteinB1l+NproteinB2)+1+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+1+Nmembrane +(ii+1)*3) + '\n')
        # bond upper outer
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+2)+ '  ' +str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+2) + '  ' +str(3*(NproteinB1l+NproteinB2)+1+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+2+Nmembrane +(ii+1)*3) + '\n')        
         # bond upper inner
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+3)+ '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+3) + '  ' +str(3*(NproteinB1l+NproteinB2)+1+Nmembrane +ii*3)  + ' ' + str(3*(NproteinB1l+NproteinB2)+3+Nmembrane +(ii+1)*3)  + '\n')       
        
        # bond outer upper
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+4)+ '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+4) + '  ' +str(3*(NproteinB1l+NproteinB2)+2+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+1+Nmembrane +(ii+1)*3) + '\n')
        # bond outer outer
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+5) + '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+5) + '  ' +str(3*(NproteinB1l+NproteinB2)+2+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+2+Nmembrane +(ii+1)*3) + '\n')
        # bond outer inner
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+6)+ '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+6) + '  ' +str(3*(NproteinB1l+NproteinB2)+2+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+3+Nmembrane +(ii+1)*3) + '\n')
        
        # bond inner upper
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+7)+ '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+7) + '  ' +str(3*(NproteinB1l+NproteinB2)+3+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+1+Nmembrane +(ii+1)*3) +'\n')
        # bond inner outer
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+8) + '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+8) + '  ' +str(3*(NproteinB1l+NproteinB2)+3+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+2+Nmembrane +(ii+1)*3) +'\n')
        # bond inner inner
        f.write(str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+9) + '  ' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+9) + '  ' +str(3*(NproteinB1l+NproteinB2)+3+Nmembrane +ii*3) + ' ' + str(3*(NproteinB1l+NproteinB2)+3+Nmembrane +(ii+1)*3) +'\n')

    f.close()
    
    
    
    
################## Generate in.local file #############################

    file = 'in.local'
    fo=open(file,'w')
    
    fo.write('units           lj \n')
    fo.write('atom_style      hybrid sphere dipole molecular \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('dimension       3  \n')
    fo.write('boundary        p p p \n')
    fo.write('processors 	* * 1   # set the number of MPI processors in Lz direction to 1 \n')
    fo.write(' \n')
    fo.write(' \n')
    # fo.write('read_data       "tripplehelix.in" \n')
    fo.write('read_restart '+str(restartfile)+' \n')
    fo.write(' \n')
    fo.write('group	     	mem	  type 1  # membrane \n')
    fo.write('group\t\tspiral\t\ttype 2:' + str(Atomtypes) + '\t\t# filament \n')
    fo.write(' \n')
    fo.write('set group all mass 1.0 \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\t\t\t## Initial Bonds\n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\n')
    fo.write(' \n')
    fo.write('bond_style			harmonic \n')
    fo.write('#bonds_initialA1 \n')
    for ii in range(NproteinB1l-1): 
        fo.write('bond_coeff\t' + str(9*ii+1) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[0,0]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+2) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[0,1]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+3) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[0,2]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+4) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[1,0]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+5) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[1,1]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+6) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[1,2]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+7) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[2,0]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+8) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[2,1]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+9) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[2,2]) + '\n' )
    fo.write(' \n')
    fo.write('#bonds_initialB1 \n')
    for ii in range(NproteinB2-1): 
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+1) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[0,0]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+2) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[0,1]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+3) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[0,2]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+4) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[1,0]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+5) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[1,1]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+6) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[1,2]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+7) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[2,0]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+8) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[2,1]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+9) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[2,2]) + '\n' )
    fo.write(' \n')
    fo.write('#bonds_initialC \n')
    for ii in range(NproteinB1r-1): 
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+1) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[0,0]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+2) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[0,1]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+3) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[0,2]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+4) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[1,0]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+5) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[1,1]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+6) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[1,2]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+7) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[2,0]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+8) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[2,1]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*ii+9) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[2,2]) + '\n' )
    fo.write(' \n')

    fo.write('\t\t\t##======================================##\n')
    fo.write('\t\t\t## Variables\n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\n')
    fo.write('# membrane parameters \n')
    fo.write('variable        rc_global    equal    2.6 \n')
    fo.write('variable        rc           equal    2.6 \n')
    fo.write('variable        rmin         equal    1.12 \n')
    fo.write('variable        mu           equal    3 \n')
    fo.write('variable        zeta         equal    4 \n')
    fo.write('variable        eps          equal    4.34 \n')
    fo.write('variable        sigma        equal    1.00 \n')
    fo.write('variable        theta0_11    equal    0 \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('# interaction parameters \n')
    fo.write('# membrane 1 with membrane spiral bead 4 (upper): \n')
    fo.write('variable	sigma_spiral	equal	'+str(1.00*factor)+'  \n')
    fo.write('variable	sigma_mu	  equal    (${sigma}+${sigma_spiral})/2 \n')
    fo.write('variable	eps_mu	   	  equal    2 \n')
    fo.write('variable	rc_mu	 	  equal	 (1.122462)*(${sigma_spiral}+${sigma})/2 \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('# membrane 1 with membrane spiral bead 2,3 (inner, outer): \n')
    fo.write('variable	sigma_spiral	equal	'+str(1.00*factor)+'  \n')
    fo.write('variable	sigma_ms	  equal    (${sigma}+${sigma_spiral})/2 \n')
    fo.write('variable	eps_ms	   	  equal    '+str(epsilonB1l)+' \n')
    fo.write('variable	rc_ms	 	  equal	 (1.122462)*(${sigma_spiral}+${sigma})/2*1.3 \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('variable	sigma_fil	  equal    (${sigma_spiral}+${sigma_spiral})/2 \n')
    fo.write('variable	rc_fil	 	  equal	 (1.122462)*(${sigma_spiral}+${sigma_spiral})/2*1.3 \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('# volume exclusion between spiral beads 2,3,4: \n')
    fo.write('variable	eps_ss	   	  equal    2 \n')
    fo.write('variable	sigma_ss	  equal   ${sigma_spiral} \n')
    fo.write('variable	rc_ss	 	  equal	 (1.122462)*${sigma_spiral} \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\t\t\t## Interactions \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\n')
    fo.write(' # Use hybrid overlay for pair_style membrane and lj/cut\n')
    fo.write('pair_style      hybrid/overlay  membrane     ${rc_global}  lj/cut ${rc_global} \n')
    fo.write('#pw359: Initialise LJ/expand to zero for all possible combinations  \n')
    fo.write('pair_coeff \t * \t * \t lj/cut \t 0 \t 0 \n')
    fo.write(' \n')
    fo.write('pair_coeff \t 2* \t 2* \t lj/cut \t ${eps_ss} \t ${sigma_ss} \t ${rc_ss} \n')	
    fo.write(' \n')
    fo.write(' \n')
#     fo.write('# Lateral Interaction between filaments \n')
#     for ii in range(NproteinB1l):
#         for jj in range(NproteinB2):
#             fo.write('pair_coeff \t '  + str(3*ii+2) +    '\t'  +  str(3*NproteinB1l+3*jj+3) +  '\t lj/cut \t'+str(epsilonI)+'\t${sigma_fil}\t${rc_fil}\n')		
# #             fo.write('pair_coeff \t '  +  str(3*(Nprotein+NproteinB)+3*jj+3) + '\t'   +  str(3*(Nprotein+2*NproteinB+NproteinC)+3*ii+2)  +'\t lj/cut \t'+str(epsilonI)+'\t${sigma_fil}\t${rc_fil}\n')
#     for ii in range(NproteinB1r):
#         for jj in range(NproteinB2):
#             fo.write('pair_coeff \t '  + str(3*NproteinB1l+3*jj+2) +    '\t'  +  str(3*(NproteinB1l+NproteinB2)+3*ii+3) +  '\t lj/cut \t'+str(epsilonI)+'\t${sigma_fil}\t${rc_fil}\n')		
# #             fo.write('pair_coeff \t '  + str(3*(Nprotein+NproteinB)+3*ii+3)  + '\t'   +  str(3*(Nprotein+NproteinB+NproteinC)+3*jj+2)  +'\t lj/cut \t'+str(epsilonI)+'\t${sigma_fil}\t${rc_fil}\n')
#     fo.write(' \n')
    fo.write(' \n')
    fo.write('# Interaction membrane filament \n')   
    fo.write('# membrane attraction and volume exclusion with spiral \n')
    for ii in range(NproteinB1l):
        fo.write('pair_coeff\t1\t' + str(3*ii+2) + '\tlj/cut\t'+str(epsilonB1l)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 2
        fo.write('pair_coeff\t1\t' + str(3*ii+3) + '\tlj/cut\t'+str(epsilonB1l)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 3
        fo.write('pair_coeff\t1\t' + str(3*ii+4) + '\tlj/cut\t${eps_mu}\t${sigma_mu}\t${rc_mu}\n')
    for ii in range(NproteinB2):
        fo.write('pair_coeff\t1\t' + str(3*NproteinB1l+3*ii+2) + '\tlj/cut\t'+str(epsilonB2)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 2
        fo.write('pair_coeff\t1\t' + str(3*NproteinB1l+3*ii+3) + '\tlj/cut\t'+str(epsilonB2)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 3
        fo.write('pair_coeff\t1\t' + str(3*NproteinB1l+3*ii+4) + '\tlj/cut\t${eps_mu}\t${sigma_mu}\t${rc_mu}\n')
    for ii in range(NproteinB1r):
        fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*ii+2) + '\tlj/cut\t'+str(epsilonB1r)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 2
        fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*ii+3) + '\tlj/cut\t'+str(epsilonB1r)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 3
        fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*ii+4) + '\tlj/cut\t${eps_mu}\t${sigma_mu}\t${rc_mu}\n')

#     for ii in range(NproteinB):
#         fo.write('pair_coeff\t1\t' + str(3*(Nprotein+NproteinB+NproteinC)+3*ii+2) + '\tlj/cut\t'+str(epsilonB)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 2
#         fo.write('pair_coeff\t1\t' + str(3*(Nprotein+NproteinB+NproteinC)+3*ii+3) + '\tlj/cut\t'+str(epsilonB)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 3
#         fo.write('pair_coeff\t1\t' + str(3*(Nprotein+NproteinB+NproteinC)+3*ii+4) + '\tlj/cut\t${eps_mu}\t${sigma_mu}\t${rc_mu}\n')
#     for ii in range(Nprotein):
#         fo.write('pair_coeff\t1\t' + str(3*(Nprotein+2*NproteinB+NproteinC)+3*ii+2) + '\tlj/cut\t'+str(epsilonA)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 2
#         fo.write('pair_coeff\t1\t' + str(3*(Nprotein+2*NproteinB+NproteinC)+3*ii+3) + '\tlj/cut\t'+str(epsilonA)+'\t${sigma_ms}\t${rc_ms}\n')		# attraction membrane and filament bead 3
#         fo.write('pair_coeff\t1\t' + str(3*(Nprotein+2*NproteinB+NproteinC)+3*ii+4) + '\tlj/cut\t${eps_mu}\t${sigma_mu}\t${rc_mu}\n')
    fo.write(' \n')
    fo.write('pair_coeff \t 1 \t 1 \t membrane \t ${eps} \t ${sigma} \t ${rmin} \t ${rc} \t ${zeta} \t ${mu} \t ${theta0_11} \n') 	# volume exclusion among membrane beads
    fo.write(' \n')
    fo.write(' \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\t\t\t## Computes \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\n')
    fo.write('special_bonds lj 1 1 1 angle yes \n')
    fo.write('pair_modify			shift yes \n')
    fo.write('# Reduce the delay from default 10 to 2 to get rid of dangeours builds \n')
    fo.write('#neigh_modify 	exclude molecule/intra spiral \n')
#     fo.write('neighbor	0.3 bin')
    fo.write('neigh_modify	exclude molecule spiral \n')
    fo.write('neigh_modify    every 1 delay 1 \n')
    fo.write('neigh_modify    page 200000 one 20000 \n')
    fo.write('comm_modify     cutoff 10 \n')
    fo.write(' \n')
    fo.write('variable       dofsub      equal "count(mem)" \n')
    fo.write('compute	       cT          all    temp/sphere \n')
    fo.write('compute_modify cT          extra  ${dofsub} \n')
        
    fo.write('compute         cKe     all ke \n')
    fo.write('compute         cPe     all pe \n')

    fo.write('variable          E       equal c_cKe+c_cPe+c_cErot \n')
    fo.write('compute         cErot   all erotate/sphere \n')
    fo.write('variable	 K      equal c_cKe \n')
    fo.write('variable	 P      equal c_cPe \n')
    fo.write('variable	 Rot    equal c_cErot \n')
    fo.write('compute emem all pair membrane \n')
    fo.write('compute elj all pair lj/cut \n')
    fo.write(' \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\t\t\t## Integrator \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('velocity	all create 1.0 12342 \n') 

#     fo.write('fix		fNVE	mem	nve/sphere update dipole \n')
#     if mem == 'tube':
    fo.write('fix		fNPH  	    mem     nph/sphere x 0.0 0.0 5 update dipole dilate all \n')
    fo.write('fix_modify	fNPH        temp cT press thermo_press  \n')
#     if mem == 'sphere':
#     fo.write('fix		fNVE	mem	nve/sphere update dipole \n')
    fo.write('fix		fLANG	    all    langevin 1.0 1.0 1 12341 zero yes omega yes\n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write(' \n')
    fo.write('dump			coords all custom '+str(outputstep) +' output2.xyz id mol type x y z mux muy muz \n')
    fo.write('dump_modify     coords sort id \n')
    fo.write('thermo_style    custom step v_E v_P v_K v_Rot temp c_emem c_elj xhi yhi zhi \n')
    fo.write('restart 1000       restart_dis1.dat     restart_dis2.dat \n')
    fo.write(' \n')
    # fo.write('\t\t\t##======================================##\n')
    # fo.write('\t\t\t## Run Attach \n')
    # fo.write('\t\t\t##======================================##\n')
    # fo.write(' \n')
    # fo.write('#run_attach\n')
    # fo.write('timestep\t' + str(timestep)+'\n')
    # fo.write('thermo\t\t' + str(thermo_attach) + '\n')
    # fo.write('run\t\t' + str(run_attach) + '\n')
    # fo.write(' \n')
    # fo.write('write_data "lastconfig_attached.out" \n')
    fo.write('fix filament spiral rigid/nve/small molecule \n')
    fo.write(' \n')
    # fo.write(' \n')
    # fo.write('\t\t\t##======================================##\n')
    # fo.write('\t\t\t## Run Randomised Equilibration \n')
    # fo.write('\t\t\t##======================================##\n')
    # fo.write(' \n')
    # permlist = permutation(NproteinB1r - 1)
    # for ii in permutation(NproteinB1l - 1):
    #     jj = permlist[ii]
    #     fo.write('bond_coeff\t' + str(9 * ii + 1) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[0, 0]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 2) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[0, 1]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 3) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[0, 2]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 4) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[1, 0]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 5) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[1, 1]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 6) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[1, 2]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 7) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[2, 0]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 8) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[2, 1]) + '\n')
    #     fo.write('bond_coeff\t' + str(9 * ii + 9) + '\t' + str(bondstrengthB1l) + '\t' + str(res_B1l[2, 2]) + '\n')
    #     fo.write(' \n')
    #     fo.write(' \n')
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+1) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[0,0]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+2) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[0,1]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+3) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[0,2]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+4) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[1,0]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+5) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[1,1]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+6) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[1,2]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+7) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[2,0]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+8) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[2,1]) + '\n' )
    #     fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+9) + '\t' + str(bondstrengthB1r) + '\t' + str(res_B1r[2,2]) + '\n' )
    #     fo.write(' \n')
    #     fo.write(' \n')
    #     fo.write('#run_\n')
    #     fo.write('timestep\t' + str(timestep) + '\n')
    #     fo.write('thermo\t\t' + str(thermo_equilibrate) + '\n')
    #     fo.write('run\t\t' + str(run_B1) + '\n')
    #     fo.write(' \n')
    #     fo.write(' \n')
    #     fo.write(' \n')
    # for ii in permutation(NproteinB2-1):
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 1) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[0, 0]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 2) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[0, 1]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 3) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[0, 2]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 4) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[1, 0]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 5) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[1, 1]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 6) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[1, 2]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 7) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[2, 0]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 8) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[2, 1]) + '\n')
    #     fo.write('bond_coeff\t' + str((NproteinB1l - 1) * 9 + 9 * ii + 9) + '\t' + str(bondstrengthB2) + '\t' + str(res_B2[2, 2]) + '\n')
    #     fo.write(' \n')
    #     fo.write('#run_\n')
    #     fo.write('timestep\t' + str(timestep)+'\n')
    #     fo.write('thermo\t\t' + str(thermo_equilibrate) + '\n')
    #     fo.write('run\t\t' + str(run_B2) + '\n')
    #     fo.write(' \n')
    #
    # fo.write('write_data "lastconfig_equilibrated.out" \n')

    fo.write('\t\t\t##======================================##\n')
    fo.write('\t\t\t## Run Disassembly B1\n')
    fo.write('\t\t\t##======================================##\n')
    fo.write(' \n')
    fo.write('#disassemble_B1\n')
    # Randomised disassembly B1
    bondstrength0 = 0
    permlist = permutation(NproteinB1r-1)
    for ii in permutation(NproteinB1l-1):
        jj = permlist[ii]
        fo.write('bond_coeff\t' + str(9*ii+1) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[0,0]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+2) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[0,1]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+3) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[0,2]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+4) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[1,0]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+5) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[1,1]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+6) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[1,2]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+7) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[2,0]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+8) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[2,1]) + '\n' )
        fo.write('bond_coeff\t' + str(9*ii+9) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[2,2]) + '\n' )
        fo.write('\n')
        fo.write('pair_coeff\t1\t' + str(3*ii+2) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
        fo.write('pair_coeff\t1\t' + str(3*ii+3) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
        fo.write('pair_coeff\t1\t' + str(3*ii+4) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
#         for kk in range(NproteinB1l):
#             fo.write('pair_coeff\t'   + str(3*ii+2) +  ' \t  ' + str(3*kk+3) + '\tlj/cut\t 2\t ${sigma_ss}\t ${rc_ss}\n')
#             fo.write('pair_coeff\t'   + str(3*ii+3) +  ' \t  ' + str(3*kk+2) + '\tlj/cut\t 2\t ${sigma_ss}\t ${rc_ss}\n')
        fo.write('\n')

        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+1) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,0]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+2) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,1]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+3) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,2]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+4) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,0]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+5) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,1]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+6) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,2]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+7) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,0]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+8) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,1]) + '\n' )
        fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+9) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,2]) + '\n' )
        fo.write('\n')
        fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*jj+2) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
        fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*jj+3) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
        fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*jj+4) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
#         for kk in range(NproteinB1r):
#             fo.write('pair_coeff \t '  +  str(3*(NproteinB1l+NproteinB2)+3*jj+2) +    '\t'  +   str(3*(NproteinB1l+NproteinB2)+3*kk+3) + '\tlj/cut\t 2\t ${sigma_ss}\t ${rc_ss}\n')
#             fo.write('pair_coeff \t '  +  str(3*(NproteinB1l+NproteinB2)+3*jj+3) +    '\t'  +   str(3*(NproteinB1l+NproteinB2)+3*kk+2) + '\tlj/cut\t 2\t ${sigma_ss}\t ${rc_ss}\n')
        fo.write('\n')
        fo.write('timestep\t' + str(timestep)+'\n')
        fo.write('thermo\t\t' + str(thermo_disassemble) + '\n')
        fo.write('run\t\t' + str(run_disassemble) + '\n')
        fo.write('\n')
    fo.write('write_data "lastconfig_B1disassembled.out" \n')
    fo.write('\t\t\t##======================================##\n')
    fo.write('\t\t\t## Run Disassembly B2\n')
    fo.write('\t\t\t##======================================##\n')
    fo.write(' \n')
    fo.write('#disassemble_B2\n')
    bondstrength0 = 0
    for ii in permutation(NproteinB2-1):
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+1) + '\t' + str(bondstrength0) + '\t' + str(res_B2[0,0]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+2) + '\t' + str(bondstrength0) + '\t' + str(res_B2[0,1]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+3) + '\t' + str(bondstrength0) + '\t' + str(res_B2[0,2]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+4) + '\t' + str(bondstrength0) + '\t' + str(res_B2[1,0]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+5) + '\t' + str(bondstrength0) + '\t' + str(res_B2[1,1]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+6) + '\t' + str(bondstrength0) + '\t' + str(res_B2[1,2]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+7) + '\t' + str(bondstrength0) + '\t' + str(res_B2[2,0]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+8) + '\t' + str(bondstrength0) + '\t' + str(res_B2[2,1]) + '\n' )
        fo.write('bond_coeff\t' + str((NproteinB1l-1)*9+9*ii+9) + '\t' + str(bondstrength0) + '\t' + str(res_B2[2,2]) + '\n' )
        fo.write('\n')
        fo.write('pair_coeff\t1\t' + str(3*NproteinB1l+3*ii+2) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
        fo.write('pair_coeff\t1\t' + str(3*NproteinB1l+3*ii+3) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
        fo.write('pair_coeff\t1\t' + str(3*NproteinB1l+3*ii+4) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )

        fo.write('\n')
        fo.write('timestep\t' + str(timestep)+'\n')
        fo.write('thermo\t\t' + str(thermo_disassemble) + '\n')
        fo.write('run\t\t' + str(run_disassemble) + '\n')
    fo.write('write_data "lastconfig_B2disassembled.out" \n')
#         
#         #     Sequential Disassembly
#     final = 0
#     bondstrength0 = 0
#     fil_lengthB1 = (NproteinB1l-1)
# #     fil_lengthB2 = (NproteinB2-1)/2
# #     B1 is longer than B2
# 
# #     if fil_lengthB1%1==0:
#     for ii in range(int(fil_lengthB1)):
#         jj = ii
#         fo.write('bond_coeff\t' + str(9*ii+1) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[0,0]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+2) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[0,1]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+3) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[0,2]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+4) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[1,0]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+5) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[1,1]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+6) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[1,2]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+7) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[2,0]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+8) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[2,1]) + '\n' )
#         fo.write('bond_coeff\t' + str(9*ii+9) + '\t' + str(bondstrength0) + '\t' + str(res_B1l[2,2]) + '\n' )
#         fo.write('\n')
#         fo.write('pair_coeff\t1\t' + str(3*ii+2) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
#         fo.write('pair_coeff\t1\t' + str(3*ii+3) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
#         fo.write('pair_coeff\t1\t' + str(3*ii+4) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
# 
# #         fo.write('\n')
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+1) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,0]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+2) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,1]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+3) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,2]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+4) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,0]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+5) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,1]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+6) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,2]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+7) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,0]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+8) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,1]) + '\n' )
# #         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*jj+9) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,2]) + '\n' )
# #         fo.write('\n')
# #         fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*jj+2) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
# #         fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*jj+2) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
# #         fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*jj+2) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
# #         fo.write('\n')
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+0)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,0]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+1)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,1]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+2)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[0,2]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+3)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,0]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+4)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,1]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+5)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[1,2]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+6)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,0]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+7)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,1]) + '\n' )
#         fo.write('bond_coeff\t' + str(((NproteinB2-1)+(NproteinB1l-1))*9+9*(NproteinB1r-1)-(9*jj+8)) + '\t' + str(bondstrength0) + '\t' + str(res_B1r[2,2]) + '\n' )
#         fo.write('\n')
#         fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*(NproteinB1r-1)+2-3*jj) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
#         fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*(NproteinB1r-1)+3-3*jj) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
#         fo.write('pair_coeff\t1\t' + str(3*(NproteinB1l+NproteinB2)+3*(NproteinB1r-1)+4-3*jj) + '\t lj/cut \t ${eps_mu} \t ${sigma_mu} \t ${rc_mu} \n' )
#         fo.write('\n')
# 
#             
#         fo.write('timestep\t0.01\n')
#         fo.write('thermo\t\t' + str(thermo_disassemble) + '\n')
#         fo.write('run\t\t' + str(Tdis) + '\n')
#         fo.write('\n') 


#     fo.write('\t\t\t##======================================##\n')
#     fo.write('\t\t\t## Run Final \n')
#     fo.write('\t\t\t##======================================##\n')
#     
#     fo.write('#run_final\n')
#     fo.write('timestep\t' + str(timestep) +'\n')
#     fo.write('thermo\t\t' + str(thermo_final) + '\n')
#     fo.write('run\t\t' + str(run_final) + '\n')
#    
#     fo.write('write_data "lastconfig_final.out" \n')
    fo.write('#end \n')

    fo.close()    

def proteingenerator(radius_o,x_start,x_step,n,cell_radius,tau):
    
    # Generate unrotated protein positions
    P1 = [radius_o*np.cos(tau)-x_start+n*x_step, 0 ,-cell_radius-np.sin(tau)*radius_o]
    P2 = [-radius_o*np.cos(tau)-x_start+n*x_step, 0 ,-cell_radius+np.sin(tau)*radius_o]
    P3 = [-x_start+n*x_step+np.sin(tau)*radius_o*np.sqrt(3), 0 ,-cell_radius+np.sqrt(3)*radius_o*np.cos(tau)]

    return P1,P2,P3
    
    
def plane_rotation(theta,coordinates):
    x = coordinates[0]
    y = np.cos(theta)*coordinates[1]-np.sin(theta)*coordinates[2]
    z = np.sin(theta)*coordinates[1]+np.cos(theta)*coordinates[2]

    return x,y,z

def Proteinrotator(P1,P2,P3,angle):
    P1_r = plane_rotation(angle,P1)
    P2_r = plane_rotation(angle,P2)
    P3_r = plane_rotation(angle,P3)

    protein = [P1_r,P2_r,P3_r]
    return protein

 
def bondcalculator(protein,protein_relaxed):   
    res = np.zeros([3,3])
    for i, p in enumerate(protein):
        for j, pr in enumerate(protein_relaxed):
            res[i,j] = np.sqrt(np.sum((np.array(p)-pr)**2))

    return res


def TubeLattice(r, lboxX, sigma_mem, bead_spacing):
	L = lboxX # - 2.5*sigma_mem
	in_r = r - 3*bead_spacing - 0.5*bead_spacing
	out_r = r + 3*bead_spacing + 0.5*bead_spacing
	n_2lipids_per_turn = int(np.floor((2*np.pi*r/(bead_spacing*sigma_mem))))
	n_2lipids_length = int(np.rint(L/(bead_spacing*sigma_mem)))
	bs_turn = 2*np.pi/n_2lipids_per_turn
	bs_length = L/n_2lipids_length
	# bs_turn = sigma_mem*bead_spacing/r
	# bs_length = bead_spacing*sigma_mem
	# n_2lipids_per_turn = int(np.rint(2*np.pi/bs_turn))
	# n_2lipids_length = int(np.rint(L/bs_length))
	# print(bs_length)
	lattice = []
	x0 = -L*0.5
	for i in range(n_2lipids_length):
		for j in range(n_2lipids_per_turn):
			x = x0 + i*bs_length + 0.5*bs_length
			y = r*np.sin(j*bs_turn)
			z = r*np.cos(j*bs_turn)
			lattice.append([x,y,z])
	return lattice   
if __name__ == '__main__':
    main()