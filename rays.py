# Algorithm taken from Siddon Med Phys 1985

# New trace is based on Lemahien 1998 


import numpy as np
# OLD DEPCREATED FCN, much slower that new_trace. 
def ray_trace(imagearr,origin,sourceCT,ray,voxDim,voxSize):

    #value to return
    matrixsum = 0

    voxDimX = voxDim[0]
    voxDimY = voxDim[1]
    voxDimZ = voxDim[2]

    voxSizeX = voxSize[0]
    voxSizeY = voxSize[1]
    voxSizeZ = voxSize[2]

    #find the outerEdge of the CT dimentions  
    CTEdge = origin-voxSize*voxDim/2.0
    
    #Define Edges in X,Y,Z
    Xmin = CTEdge[0]
    Xmax = CTEdge[0]+voxDimX*(voxSizeX-1)
    
    # CBCT scan is a right handed coordinate system with +Y pointing down 
    #change the ordering in Y. More negative is larger ?? 
    Ymin = CTEdge[1]
    Ymax = CTEdge[1]+voxDimY*(voxSizeY-1)
    #Ymax = CTEdge[1]+voxDimY*voxSizeY

    Zmin = CTEdge[2]
    Zmax = CTEdge[2]+voxDimZ*(voxSizeZ-1)
    #Zmax = CTEdge[2]+voxDimZ*voxSizeZ

    sourceCTX = sourceCT[0]
    sourceCTY = sourceCT[1]
    sourceCTZ = sourceCT[2]

    rayX = ray[0]
    rayY = ray[1]
    rayZ = ray[2]

    #Find the parameteric intersection values
    
    alphaXmin = 0.0
    alphaXmax = 1.0

    alphaYmin = 0.0
    alphaYmax = 1.0

    alphaZmin = 0.0
    alphaZmax = 1.0

    if(rayX != 0.0):
        alphaXmin = (Xmin-sourceCTX)/rayX
        alphaXmax = (Xmax-sourceCTX)/rayX
     
    if(rayY != 0.0):
        alphaYmin = (Ymin-sourceCTY)/rayY
        alphaYmax = (Ymax-sourceCTY)/rayY
    
    if(rayZ != 0.0):
        alphaZmin = (Zmin-sourceCTZ)/rayZ
        alphaZmax = (Zmax-sourceCTZ)/rayZ
        

    alphaMIN = max(0,min(alphaXmin,alphaXmax), min(alphaYmin,alphaYmax),min(alphaZmin,alphaZmax))
    alphaMAX = min(1,max(alphaXmin,alphaXmax), max(alphaYmin,alphaYmax),max(alphaZmin,alphaZmax))
    #print(" Alpha MIN " + str(alphaMIN) + " alphaMAX " + str(alphaMAX))
    #print("Alpha min intersections")
    #print(" x " + str(sourceCTX+alphaMIN*rayX) + " y " + str(sourceCTY+alphaMIN*rayY) + " z " + str(sourceCTZ+alphaMIN*rayZ))
    
    #print("Alpha max intersections")
    #print(" x " + str(sourceCTX+alphaMAX*rayX) + " y " + str(sourceCTY+alphaMAX*rayY) + " z " + str(sourceCTZ+alphaMAX*rayZ))

    if(alphaMAX < alphaMIN ):
        #print("Ray does not intersect CT")
        return matrixsum
    
    # find the range of indicices for intersections 
    #Setting the floor and ceiling seems to be nessecary to keep indicies within lim
    
    if (rayX > 0):
        ixmin = int(np.ceil(voxSizeX-(Xmax -alphaMIN*rayX- sourceCTX)/voxDimX))
        ixmax = int(np.floor(1.0 + ( sourceCTX + alphaMAX*rayX-Xmin )/voxDimX)) 
        #print(" X imin arg " + str((Xmax - alphaMIN*rayX-sourceCTX)/voxDimX))
        #print(" X imax arg " + str( (sourceCTX + alphaMAX*rayX-Xmin)/voxDimX))
    else:
        ixmin = int(np.ceil(voxSizeX-(Xmax -alphaMAX*rayX- sourceCTX)/voxDimX))
        ixmax = int(np.floor(1.0 + ( sourceCTX + alphaMIN*rayX-Xmin )/voxDimX))
        #print(" X imin arg " + str(voxSizeX-(Xmax -alphaMAX*rayX- sourceCTX)/voxDimX))
        #print(" X imax arg " + str( 1.0 + ( sourceCTX + alphaMIN*rayX-Xmin )/voxDimX))
    if(rayY > 0):
        iymin = int(np.ceil(voxSizeY-(Ymax - alphaMIN*rayY-sourceCTY)/voxDimY))
        iymax = int(np.floor(1.0 + ( sourceCTY + alphaMAX*rayY-Ymin )/voxDimY))
        #print("imin arg " + str((Ymax - alphaMIN*rayY-sourceCTY)/voxDimY))
        #print("imax arg " + str( (sourceCTY + alphaMAX*rayY-Ymin)/voxDimY))
    else:
        iymin = int(np.ceil(voxSizeY-(Ymax - alphaMAX*rayY-sourceCTY)/voxDimY))
        iymax = int(np.floor(1.0 + ( sourceCTY + alphaMIN*rayY-Ymin )/voxDimY) )
    if (rayZ > 0):
        izmin = int(np.ceil(voxSizeZ-(Zmax -alphaMIN*rayZ- sourceCTZ)/voxDimZ))   
        izmax = int(np.floor(1.0 + ( sourceCTZ + alphaMAX*rayZ-Zmin )/voxDimZ))
    else:
        izmin = int(np.ceil(voxSizeZ-(Zmax -alphaMAX*rayZ- sourceCTZ)/voxDimZ))   
        izmax = int(np.floor(1.0 + ( sourceCTZ + alphaMIN*rayZ-Zmin )/voxDimZ) )

     # This can happen when ray is close to parrallel and rounded with ceiling and floor   
    if(ixmax < ixmin):
        ixmax = ixmin
 
    if(izmax < izmin):
        izmax = izmin

    if(iymax < iymin):
        iymax = iymin


    #print("ixmin " + str(ixmin) + " ixmax " + str(ixmax))
    #print("iymin " + str(iymin) + " iymax " + str(iymax))
    #print("izmin " + str(izmin) + " izmax " + str(izmax))

    #Calculate the sets of parameteric values. 
    alphaX = np.zeros(ixmax-ixmin)
    alphaY = np.zeros(iymax-iymin)
    alphaZ = np.zeros(izmax-izmin)
        
    for i in range(ixmin,ixmax):
        if(rayX != 0.0):
            alphaX[i-ixmin] = (Xmin+voxDimX*(i-1.0) - sourceCTX)/rayX
    
    for j in range(iymin,iymax):
        if(rayY != 0.0):
            alphaY[j-iymin] = (Ymin+voxDimY*(j-1.0) - sourceCTY)/rayY
        #print("Alpha Y  " + str((Ymin+voxDimY*(j-1.0) - sourceCTY)/rayY))
    for k in range(izmin,izmax):
         if(rayZ != 0.0):
            alphaZ[k-izmin] = (Zmin+voxDimZ*(k-1.0) - sourceCTZ)/rayZ


    # Merge the alpha values together take care of Special cases where line is parralell CT planes
    if(rayX != 0.0 and rayZ != 0.0  ):
        alphaMERGE = np.concatenate((alphaX,alphaY,alphaZ),axis=None)
    elif(rayX == 0.0 and rayZ != 0.0 ):
        alphaMERGE = np.concatenate((alphaY,alphaZ),axis=None)
    elif(rayX != 0.0 and rayZ == 0.0):
        alphaMERGE = np.concatenate((alphaX,alphaY),axis=None)
    elif(rayX == 0.0 and rayZ == 0.0 ):
        alphaMERGE = alphaY 


    alphaMERGE = np.sort(alphaMERGE)

    sizealpha = alphaMERGE.shape[0]
    #print("size Alpha " + str(sizealpha))
    if(sizealpha == 0):
        #print("Glancing Blow to not Calc")
        return matrixsum
    #Containers to store indices for full matrix 
    ialpha = np.zeros(sizealpha-1)
    jalpha = np.zeros(sizealpha-1)
    kalpha = np.zeros(sizealpha-1)

    fileout = open('rayold.txt','w')
    for m in range(0,sizealpha-1):
        # Find the average alpha value 
        alphamid =  (alphaMERGE[m+1]+alphaMERGE[m])/2.0
        X1 =  alphaMERGE[m]*rayX + sourceCTX
        X2 =  alphaMERGE[m+1]*rayX + sourceCTX
        Y1 =  alphaMERGE[m]*rayY + sourceCTY
        Y2 =  alphaMERGE[m+1]*rayY + sourceCTY
        #print (" m+1 ", str(m+1) + " alphaMERGE[m+1] " + str(alphaMERGE[m+1]) + "alphaMERGE[m]" +  str(alphaMERGE[m]) )
        Z1 =  alphaMERGE[m]*rayZ + sourceCTZ
        Z2 =  alphaMERGE[m+1]*rayZ + sourceCTZ
       
        ialpha = int(np.rint((X1+alphamid*(X2-X1)-Xmin)/voxDimX))  # remove +1 from Siddon formula? 
        jalpha = int(np.rint((Y1+alphamid*(Y2-Y1)-Ymin)/voxDimY))  # remove +1 from Siddon formula? 
        kalpha = int(np.rint((Z1+alphamid*(Z2-Z1)-Zmin)/voxDimZ))  # remove +1 from Siddon formula? 
        #print( " ialpha " + str(ialpha) + " jalpha " + str(jalpha) + " kalpha " + str(kalpha))
        #matrixsum = matrixsum + 0.001*dicom_sitk_handle.GetPixel(ialpha,jalpha,kalpha)+1
        #Using simple linear function to convert HU to ED. 
        #matrixsum = matrixsum + dist*(alphaMERGE[m+1]-alphaMERGE[m])*(0.001*imagearr.GetPixel(ialpha,jalpha,kalpha)+1)
        #print("iz " + str(kalpha) + " iy "+ str(jalpha) + " ix "+ str(ialpha) + " Image Arr " + str(imagearr[kalpha,jalpha,ialpha]))
        s = "iz " + str(kalpha) + " iy "+ str(jalpha) + " ix "+ str(ialpha) + " Alpha " + str(alphaMERGE[m]) + " \n"
        fileout.write(s)
        matrixsum = matrixsum + np.sqrt(np.square(X2-X1)+np.square(Y2-Y1)+np.square(Z2-Z1))*(alphaMERGE[m+1]-alphaMERGE[m])*(0.001*imagearr[kalpha,jalpha,ialpha]+1)
        #print(" matrix sum " + str(matrixsum) ) 
    fileout.close
    return matrixsum


#Function to rotate the source based on the gantry angle. In global co-ordinates
def source_rotate(angle,origin): 

    #hardcode the vector from the origin to the source at gantry 0
    SAD = 1000 # source to isocenter
    source = np.array([0,-SAD,0])
    #input gantry angle in degrees
    # Rotate around iso. 

    theta = np.deg2rad(angle)
    # never rotated around the z axis
    rot = np.array([[np.cos(theta), -np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0, 0,1]])
    
    pt_rot = np.dot(rot,source)+origin

    return pt_rot

# Rotate the position of the EPID pixels in global co-ordinates 
def EPID_rotate(angle,origin,EPIDpt): 

    #input gantry angle in degrees
    # Rotate around iso. 

    theta = np.deg2rad(angle)
    # never rotated around the z axis
    rot = np.array([[np.cos(theta), -np.sin(theta),0], [np.sin(theta), np.cos(theta),0],[0, 0,1]])
    
    pt_rot = np.dot(rot,EPIDpt)+origin

    return pt_rot

#Main function to ray trace point source to EPID pixel.  
def new_trace(imagearr,CTinfo,sourceCT,ray,voxDim,voxSize,headfirst=True):

    matrixsum = 0
    # Epsilon value for rounding errors associated with alphaMAX 
    ep = 1.0e-12  
    voxDimX = voxDim[0]
    voxDimY = voxDim[1]
    voxDimZ = voxDim[2]

    voxSizeX = voxSize[0]
    voxSizeY = voxSize[1]
    voxSizeZ = voxSize[2]

    #find the outerEdge of the CT dimentions  
    CTEdgeX = CTinfo[0]
    CTEdgeY = CTinfo[1]
    CTEdgeZ = CTinfo[2]
        
    # CBCT scan is a right handed coordinate system with +Y pointing down 
    #change the ordering in Y. 
    
   
    Ymin = CTEdgeY
    Ymax = CTEdgeY+voxDimY*(voxSizeY-1)
   
    #swap edges for a feet first scan, Y does not change
    
    
    if(headfirst):
        Xmin = CTEdgeX
        Xmax = CTEdgeX+voxDimX*(voxSizeX-1)


        Zmin = CTEdgeZ
        Zmax = CTEdgeZ+voxDimZ*(voxSizeZ-1)
    else: 
        Xmax = CTEdgeX
        Xmin = CTEdgeX-voxDimX*(voxSizeX-1)

        Zmax = CTEdgeZ
        Zmin = CTEdgeZ-voxDimZ*(voxSizeZ-1)
   
    
    sourceCTX = sourceCT[0]
    sourceCTY = sourceCT[1]
    sourceCTZ = sourceCT[2]

    rayX = ray[0]
    rayY = ray[1]
    rayZ = ray[2]

    #Find the parameteric intersection values
    
    alphaXmin = 0.0
    alphaXmax = 1.0

    alphaYmin = 0.0
    alphaYmax = 1.0

    alphaZmin = 0.0
    alphaZmax = 1.0

    if(rayX != 0.0):
        alphaXmin = (Xmin-sourceCTX)/rayX
        alphaXmax = (Xmax-sourceCTX)/rayX
     
    if(rayY != 0.0):
        alphaYmin = (Ymin-sourceCTY)/rayY
        alphaYmax = (Ymax-sourceCTY)/rayY
    
    if(rayZ != 0.0):
        alphaZmin = (Zmin-sourceCTZ)/rayZ
        alphaZmax = (Zmax-sourceCTZ)/rayZ
        

    alphaMIN = max(0,min(alphaXmin,alphaXmax), min(alphaYmin,alphaYmax),min(alphaZmin,alphaZmax))
    alphaMAX = min(1,max(alphaXmin,alphaXmax), max(alphaYmin,alphaYmax),max(alphaZmin,alphaZmax))
    #print(" Alpha MIN " + str(alphaMIN) + " alphaMAX " + str(alphaMAX))
    #print("Alpha min intersections")
    #print(" x " + str(sourceCTX+alphaMIN*rayX) + " y " + str(sourceCTY+alphaMIN*rayY) + " z " + str(sourceCTZ+alphaMIN*rayZ))
    
    #print("Alpha max intersections")
    #print(" x " + str(sourceCTX+alphaMAX*rayX) + " y " + str(sourceCTY+alphaMAX*rayY) + " z " + str(sourceCTZ+alphaMAX*rayZ))

    if(alphaMAX < alphaMIN ):
        #print("Ray does not intersect CT")
        return matrixsum
    
    #find the range of indicices for intersections 
    #Setting the floor and ceiling seems to be nessecary to keep indicies within lim
    
    if (rayX > 0):
        ixmin = int(np.ceil(voxSizeX-(Xmax -alphaMIN*rayX- sourceCTX)/voxDimX))-1
        ixmax = int(np.floor(1.0 + ( sourceCTX + alphaMAX*rayX-Xmin )/voxDimX))-1
        #print(" X imin arg " + str((Xmax - alphaMIN*rayX-sourceCTX)/voxDimX))
        #print(" X imax arg " + str( (sourceCTX + alphaMAX*rayX-Xmin)/voxDimX))
    else:
        ixmin = int(np.ceil(voxSizeX-(Xmax -alphaMAX*rayX- sourceCTX)/voxDimX))-1
        ixmax = int(np.floor(1.0 + ( sourceCTX + alphaMIN*rayX-Xmin )/voxDimX))-1
        #print(" X imin arg " + str(voxSizeX-(Xmax -alphaMAX*rayX- sourceCTX)/voxDimX))
        #print(" X imax arg " + str( 1.0 + ( sourceCTX + alphaMIN*rayX-Xmin )/voxDimX))
    if(rayY > 0):
        iymin = int(np.ceil(voxSizeY-(Ymax - alphaMIN*rayY-sourceCTY)/voxDimY))-1
        iymax = int(np.floor(1.0 + ( sourceCTY + alphaMAX*rayY-Ymin )/voxDimY))-1
        #print("imin arg " + str((Ymax - alphaMIN*rayY-sourceCTY)/voxDimY))
        #print("imax arg " + str( (sourceCTY + alphaMAX*rayY-Ymin)/voxDimY))
    else:
        iymin = int(np.ceil(voxSizeY-(Ymax - alphaMAX*rayY-sourceCTY)/voxDimY))-1
        iymax = int(np.floor(1.0 + ( sourceCTY + alphaMIN*rayY-Ymin )/voxDimY))-1
    if (rayZ > 0):
        izmin = int(np.ceil(voxSizeZ-(Zmax -alphaMIN*rayZ- sourceCTZ)/voxDimZ))
        izmax = int(np.floor(1.0 + ( sourceCTZ + alphaMAX*rayZ-Zmin )/voxDimZ))
    else:
        izmin = int(np.ceil(voxSizeZ-(Zmax -alphaMAX*rayZ- sourceCTZ)/voxDimZ))
        izmax = int(np.floor(1.0 + ( sourceCTZ + alphaMIN*rayZ-Zmin )/voxDimZ))

     # This can happen when ray is close to parrallel and rounded with ceiling and floor   
    if(ixmax < ixmin):
        ixmax = ixmin
 
    if(iymax < iymin):
        iymax = iymin

    if(izmax < izmin):
        izmax = izmin

    if( (ixmax == voxSizeX and ixmin == voxSizeX) or (iymax == voxSizeY and iymin == voxSizeY) or (izmax == voxSizeZ and izmin == voxSizeZ)):
        #print("Glancing blow do not trace")
        return matrixsum
   
    #Nalpha = (ixmax-ixmin+1)+(iymax-iymin+1)+(izmax-izmin+1)
    #print("N alpha " + str(Nalpha))

    alphaXdel = voxDimX
    alphaYdel = voxDimY
    alphaZdel = voxDimZ
     
    if(rayX != 0.0):
        alphaXmin =(Xmin+voxDimX*(ixmin) - sourceCTX)/rayX
        alphaXmax =(Xmin+voxDimX*(ixmax) - sourceCTX)/rayX
        alphaXdel = voxDimX/np.abs(rayX)
   
    
    if(rayY != 0.0):
        alphaYmin =(Ymin+voxDimY*(iymin) - sourceCTY)/rayY
        alphaYmax =(Ymin+voxDimY*(iymax) - sourceCTY)/rayY
        alphaYdel = voxDimY/np.abs(rayY)

    if(rayZ != 0.0):   
        alphaZmin =(Zmin+voxDimZ*(izmin) - sourceCTZ)/rayZ
        alphaZmax =(Zmin+voxDimZ*(izmax) - sourceCTZ)/rayZ
        alphaZdel = voxDimZ/np.abs(rayZ)
    

    if( rayX < 0):
        alphaX = alphaXmax
        ixcnt = ixmax
    else:
        alphaX = alphaXmin
        ixcnt = ixmin
    
    if( rayY < 0):
        alphaY = alphaYmax
        iycnt = iymax
    else:
        alphaY = alphaYmin
        iycnt = iymin

    if( rayZ < 0):
        alphaZ = alphaZmax
        izcnt = izmax
    else:
        alphaZ = alphaZmin
        izcnt = izmin
    #print("iz " + str(izcnt) + " iy "+ str(iycnt) + " ix "+ str(ixcnt) )
    #alphaR = min(alphaX,alphaY,alphaZ)
    alphaR = alphaMIN 
    alphaC = alphaMIN
    idx = 0
    
    
      
    #fileout = open('raynew.txt','w')
    while(alphaR < alphaMAX-ep):
       
        dist = np.sqrt(np.square((alphaR-alphaC)*rayX)+ np.square((alphaR-alphaC)*rayY) +  np.square((alphaR-alphaC)*rayZ)) 
        
        if (dist < 10.0):
            if( imagearr[izcnt,iycnt,ixcnt] <= 0):
                matrixsum = matrixsum+dist*(0.001*imagearr[izcnt,iycnt,ixcnt]+1)/10.0   
            else:
                matrixsum = matrixsum+dist*(0.00037*imagearr[izcnt,iycnt,ixcnt]+1)/10.0
        # approximate with Bi linear function to estimte the radiolocal path length
        
        #print("Dist " + str(dist))
        #print("Alpha Diff " + str(alphaR-alphaC))
        #print("HU " + str(imagearr[izcnt,iycnt,ixcnt]) + " Electron Density " + str(0.001*imagearr[izcnt,iycnt,ixcnt]+1))
        #print("matrix sum " + str(matrixsum))
        #s = "iz " + str(izcnt) + " iy "+ str(iycnt) + " ix "+ str(ixcnt) + " AlphaR " + str(alphaR) + " \n"
        #fileout.write(s)
        #print("iz " + str(izcnt) + " iy "+ str(iycnt) + " ix "+ str(ixcnt) + " AlphaR " + str(alphaR) + " dist " + str(dist))

        alphaC = alphaR
        alphaR = min(alphaX+alphaXdel,alphaY+alphaYdel,alphaZ+alphaZdel)
        #print (" Alpha X " + str(alphaX) + " Alpha Y " + str(alphaY) + " Alpha Z " + str(alphaZ) )
        #print(" Alpha X + Step " + str(alphaX+alphaXdel) + " Alpha Y " + str(alphaY+alphaYdel) + " Alpha Z " + str(alphaZ+alphaZdel)  )
        #print(" Step Del X " + str(alphaXdel) + " del Y " + str(alphaYdel) + " del Z " + str(alphaZdel)  )
        #print (" ALpha StEP " + str(alphaR) + " Cnt " + str(idx))
        if(alphaX+alphaXdel <= alphaY+alphaYdel and  alphaX+alphaXdel <= alphaZ+alphaZdel):
            #print(" Step X " + str(alphaX+alphaXdel) )
            alphaX = alphaX+alphaXdel
            if(rayX>0):
                ixcnt = ixcnt+1
            else:
                ixcnt = ixcnt-1
        if(alphaY+alphaYdel <= alphaX+alphaXdel and  alphaY+alphaYdel <= alphaZ+alphaZdel):
            #print(" Step Y " + str(alphaY+alphaYdel) )
            alphaY=alphaY+alphaYdel
            if(rayY>0):
                iycnt = iycnt+1
            else:
                iycnt = iycnt-1
        if(alphaZ+alphaZdel <= alphaX+alphaXdel and  alphaZ+alphaZdel <= alphaY+alphaYdel):
            #print(" Step Z" + str(alphaZ+alphaZdel) )
            alphaZ = alphaZ+alphaZdel
            if(rayZ>0):
                izcnt = izcnt+1
            else:
                izcnt = izcnt-1

        
        
        
        #print( "Matrix sum : alpha diff " + str(alphaR-alphaC) + " dist " + str(dist) + " ED " + str(0.001*imagearr[izcnt,iycnt,ixcnt]+1))
        idx = idx +1
    #fileout.close()
    return matrixsum