import numba
import numpy as np
import functools

########################
#                      #
#   WORK IN PROGRESS   #
#                      #
########################

def corrfunction(shape,z,qmax):
    y, x = np.meshgrid(np.arange(shape[1], dtype=np.float64), np.arange(shape[0], dtype=np.float64))
    x -= shape[0] / 2.0
    y -= shape[1] / 2.0
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    qx, qy, qz = [(k / d * z) for k in (x, y, z)]
    qx, qy, qz = [(k - np.min(k)) for k in (qx, qy, qz)]

    def corr(input,finner):
        return np.sum(finner(input),axis=0)
    def inner(input):
        out=np.zeros((shape[0],qmax))
        for refx in numba.prange(shape[0]):
            for refy in range(shape[1]):
                qstep=min(abs(qy[refx,refy+1]-qy[refx,refy]),abs(qx[refx+1,refy]-qx[refx,refy]))
                xmin=int(max(0,-5+np.floor(refx-qmax/qstep)))
                xmax=int(min(shape[0],5+np.ceil(refx+qmax/qstep)))
                refv=input[refx,refy]
                for x in range(xmin,xmax):
                    dqy=np.sqrt((qmax/qstep)**2-(qx[x,refy]-qx[refx,refy])**2)
                    ymin=int(max(0,-5+np.floor(refy-dqy)))
                    ymax=int(min(shape[1],refy+1))
#                     ymax=int(min(shape[1],5+np.ceil(refy+dqy))) #or range(ymin,ymax)... 
                    for y in range(ymin,ymax):
                        dq=(qx[x,y]-qx[refx,refy])**2+(qy[x,y]-qy[refx,refy])**2+(qz[x,y]-qz[refx,refy])**2
                        if dq>=(qmax-.5)**2:
                            continue
                        val=refv*input[x,y]
                        qsave=int(np.rint(np.sqrt(dq)))                    
                        out[refx,qsave]+=val
        return out
    return functools.partial(corr,finner=numba.njit(inner,parallel=True).compile("float64[:,:](float64[:,:])"))