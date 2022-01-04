import os 
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import glob
import re
import rays
from threading import Thread
from multiprocessing import cpu_count
from queue import *


def rayvec_worker(A):
    q, pbar, kwargs = A
    point_maker = RunRayVec(**kwargs)
    while True:
        item = q.get()
        if item is None:
            break
        else:
            point_maker.new_trace(**item)
        pbar.update()
        q.task_done()


class RunRayVec(object):
    def __init__(self, SID, SAD, epidEdgeX, xstep, epidEdgeZ, zstep, image, CTinfo,
                 rotsource, voxDim, voxSize, rayvec, gantryangle, origin, nz):
        self.image = image
        self.CTinfo = CTinfo
        self.voxDim = voxDim
        self.voxSize = voxSize
        self.SID = SID
        self.SAD = SAD
        self.epidEdgeX = epidEdgeX
        self.epidEdgeZ = epidEdgeZ
        self.xstep, self.zstep = xstep, zstep
        self.rayvec = rayvec
        self.gantryangle, self.origin, self.rotsource = gantryangle, origin, rotsource
        self.nz = nz

    def new_trace(self, kk, ll):
        PointOnEPID = np.array([(self.epidEdgeX + kk * self.xstep), self.SID - self.SAD,
                                (self.epidEdgeZ + ll * self.zstep)])
        ray = rays.EPID_rotate(self.gantryangle, self.origin, PointOnEPID) - self.rotsource
        # Double check the indexing
        self.rayvec[self.nz - ll, kk - 1] = rays.new_trace(self.image, self.CTinfo,
                                                           self.rotsource, ray, self.voxDim,
                                                           self.voxSize)


def MakeCBCTProjection(RIpath,CBCTpath):
    
    #npfile = glob.glob(str(CBCTpath) + '\cbctproj*.npz')
    #if(npfile):
    #    print("Projection files exists skip making new ones")
    #    return
    
    # Set parameters for ray tracing 
    SID = 1540 #source to imager distance
    SAD = 1000 # source to isocenter
    print("Ray Tracing ")

    # Size of the Panel 
    nx = 256
    nz = 256

    #rayvec = np.zeros((1280,1280))
    rayvec = np.zeros((nz,nx))
    zstep = 430/nz  # Why is this 430?
    xstep = 430/nx

    epidEdgeX = -nx/2*xstep
    epidEdgeZ = -nz/2*zstep

    print('path ' + CBCTpath)

    #Search for a numpy file 
    RIfiles = glob.glob(str(RIpath) + '\RI*.npz')
    angles = np.zeros(len(RIfiles))
    dates = np.zeros(len(RIfiles))
    fxs =  np.zeros(len(RIfiles))
    i=0
    for file in RIfiles:
        tt = re.search('_G\d+_',file)
        gang = tt[0][2:len(tt[0])-1]
        angles[i] = gang
        rr = re.search('_\d+\.',file)
        date = rr[0][1:len(rr[0])-1]
        dates[i] = date
        ff = re.search('RI\d+_',file)
        fxnum = ff[0][2:len(ff[0])-1]
        fxs[i] = fxnum
        print("Gantry Angle " + str(gang) + " Date " + str(date) + " Fraction " + str(fxnum))
        i=i+1

    gangs = np.unique(angles)
    dates = np.unique(dates)
    fxs = np.unique(fxs)
        
    #Search for a numpy file 

    for j in range(0,len(dates)):
        cbctfile = str(CBCTpath) + '\cbct' + str(int(dates[j])) +'.npz'
        if(os.path.exists(cbctfile)):
            print("Load File")
        else:
            print(str(cbctfile) + " CBCT Does not Exists")
            continue
        
        print("Load file " + cbctfile)
        npcbct = np.load(cbctfile)
        image = npcbct['cbct']
        #The origin of the CBCT is used for the bounding box in the function rays, should rename.
        CTinfo = npcbct['origin']
        voxDim = npcbct['voxDim']
        voxSize = npcbct['voxSize']
        #Set the origin to be used for the source to zero 
        origin = [0.0 , 0.0, 0.0]
        
        print("Make projection for Fraction " + str(int(fxs[j])) + " On " + str(int(dates[j])) )
        for gantryang in gangs:
            print("Gantry angle for projection " + str(int(gantryang)))
            rotsource = rays.source_rotate(gantryang,origin)

            Projex = CBCTpath + "\cbctprojection" + str(int(fxs[j])) + '_G' + str(int(gantryang)) + "_" + str(int(dates[j])) + ".npz"
            print("Projection file " +str(Projex))
            if(os.path.exists(Projex)):
                print("Projection File exists")
                continue
            # # Scan over EPID panel range
            thread_count = int(cpu_count()*.8)
            q = Queue(maxsize=thread_count)
            kwargs = {'SID': SID, 'SAD': SAD, 'epidEdgeX': epidEdgeX,
                      'epidEdgeZ': epidEdgeZ, 'xstep': xstep, 'zstep': zstep,
                      'image': image, 'CTinfo': CTinfo, 'rotsource': rotsource,
                      'voxDim': voxDim, 'voxSize': voxSize, 'rayvec': rayvec,
                      'gantryangle': gantryang, 'origin': origin, 'nz':nz}
            pbar = tqdm(total=nx*nz, desc='Making ray trace')
            A = (q, pbar, kwargs)
            threads = []
            for worker in range(thread_count):
                t = Thread(target=rayvec_worker, args=(A,))
                t.start()
                threads.append(t)
            for kk in range(1,nx):
                for ll in range(1,nz):
                    item = {'kk': kk, 'll': ll}
                    q.put(item)
            for i in range(thread_count):
                q.put(None)
            for t in threads:
                t.join()
            cbctproj = np.float32(rayvec)
            projfileout = "cbctprojection" + str(int(fxs[j])) + '_G' + str(int(gantryang)) + "_" + str(int(dates[j])) + '_new'
            print("Save npz    " + projfileout)
            arrout = os.path.join(CBCTpath, projfileout)
            #print("Saving Projection "+ str(arrout))
            np.savez_compressed(arrout,cbctproj)
            return None

def main():
    # Main loop
    Basepath = 'P:\Image_Prediction\Marginal'
    MRNs = os.listdir(Basepath)

    for i in range(0,len(MRNs)):
        RTIpath = os.path.join(Basepath,MRNs[i],'RTIMAGE')
        CBCTpath = os.path.join(Basepath,MRNs[i],'CT')
        print(RTIpath)
        MakeCBCTProjection(RTIpath,CBCTpath)


if __name__ == '__main__':
    pass
"""
# Single patient
fid = open(os.path.join('.', 'MRN.txt'))
MRN = fid.readline().strip('\n')
fid.close()
Basepath = 'P:\\Image_Prediction\\Marginal\\' + MRN
MRNs = os.listdir(Basepath)


RTIpath = os.path.join(Basepath,'RTIMAGE')
CBCTpath = os.path.join(Basepath,'CT')
print(RTIpath)
MakeCBCTProjection(RTIpath,CBCTpath)
"""