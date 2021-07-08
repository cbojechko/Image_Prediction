# Algorithm taken from Siddon Med Phys 1985


import numpy as np

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
    print(" Alpha MIN " + str(alphaMIN) + " alphaMAX " + str(alphaMAX))
    #print("Alpha min intersections")
    #print(" x " + str(sourceCTX+alphaMIN*rayX) + " y " + str(sourceCTY+alphaMIN*rayY) + " z " + str(sourceCTZ+alphaMIN*rayZ))
    
    #print("Alpha max intersections")
    #print(" x " + str(sourceCTX+alphaMAX*rayX) + " y " + str(sourceCTY+alphaMAX*rayY) + " z " + str(sourceCTZ+alphaMAX*rayZ))

    if(alphaMAX < alphaMIN ):
        print("Ray does not intersect CT")
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
        iymin = int(np.rint(voxSizeY-(Ymax - alphaMIN*rayY-sourceCTY)/voxDimY))
        iymax = int(np.rint(1.0 + ( sourceCTY + alphaMAX*rayY-Ymin )/voxDimY))
        #print("imin arg " + str((Ymax - alphaMIN*rayY-sourceCTY)/voxDimY))
        #print("imax arg " + str( (sourceCTY + alphaMAX*rayY-Ymin)/voxDimY))
    else:
        iymin = int(np.rint(voxSizeY-(Ymax - alphaMAX*rayY-sourceCTY)/voxDimY))
        iymax = int(np.rint(1.0 + ( sourceCTY + alphaMIN*rayY-Ymin )/voxDimY) )
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

    print("ixmin " + str(ixmin) + " ixmax " + str(ixmax))
    print("iymin " + str(iymin) + " iymax " + str(iymax))
    print("izmin " + str(izmin) + " izmax " + str(izmax))

    #Calculate the sets of parameteric values. 
    alphaX = np.zeros(ixmax-ixmin)
    alphaY = np.zeros(iymax-iymin)
    alphaZ = np.zeros(izmax-izmin)
        
    for i in range(ixmin,ixmax):
        if(rayX != 0.0):
            alphaX[i-ixmin] = (Xmin+voxDimX*(i-1.0) - sourceCTX)/rayX
    
    for j in range(iymin,iymax):
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
    #Containers to store indices for full matrix 
    ialpha = np.zeros(sizealpha-1)
    jalpha = np.zeros(sizealpha-1)
    kalpha = np.zeros(sizealpha-1)

    
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
        #print( " ialpha " + str(ialpha) + " jalpha " + str(jalpha) + " kalpha " + str(kalpha) + " dist " +  str(dist)  )
        #matrixsum = matrixsum + 0.001*dicom_sitk_handle.GetPixel(ialpha,jalpha,kalpha)+1
        #Using simple linear function to convert HU to ED. 
        #matrixsum = matrixsum + dist*(alphaMERGE[m+1]-alphaMERGE[m])*(0.001*imagearr.GetPixel(ialpha,jalpha,kalpha)+1)
        matrixsum = matrixsum + np.sqrt(np.square(X2-X1)+np.square(Y2-Y1)+np.square(Z2-Z1))*(alphaMERGE[m+1]-alphaMERGE[m])*(0.001*imagearr[kalpha,jalpha,ialpha]+1)
        #print(" matrix sum " + str(matrixsum) ) 
    return matrixsum