cimport mkl_dfti
cimport numpy as np
import cython

      
cdef extern from 'mkl_vml.h':
    cdef struct MKL_COMPLEX16:
        double real
        double imag       
    void vzMulByConj(int n, const MKL_COMPLEX16*,const MKL_COMPLEX16*,MKL_COMPLEX16*)
 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int autocorrelate3(np.ndarray[double, ndim=3, mode="c"] input) except -1:
    '''
    calculates NxNxN autocorrelation of (N+2)xNxN array using mkl fft. returns 0 on success.
    '''
    #strides
    cdef int N1=input.shape[0], N2=input.shape[1], N3=input.shape[2]-2 #N3: extra space needed for r2c-> do smaller fft
    cdef long N[3]
    N[:] = [N1, N2, N3]  
    cdef long rs[4]
    rs[:] = [0, N2*(N3//2+1)*2, (N3//2+1)*2, 1]
    cdef long cs[4]
    cs[:] = [0, N2*(N3//2+1), (N3//2+1), 1]
    cdef double* x = <double*> &input[0,0,0]
    cdef mkl_dfti.DFTI_DESCRIPTOR* hand = NULL;
    cdef long status = 0
       
    try:
        #r2c fft setup
        if mkl_dfti.DftiCreateDescriptor(&hand,mkl_dfti. DFTI_DOUBLE, mkl_dfti.DFTI_REAL, 3, N): raise Exception()
        if mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_CONJUGATE_EVEN_STORAGE, mkl_dfti.DFTI_COMPLEX_COMPLEX): raise Exception()
        if mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_INPUT_STRIDES, rs): raise Exception()
        if mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_OUTPUT_STRIDES, cs): raise Exception()
        if mkl_dfti.DftiCommitDescriptor(hand): raise Exception()

        #fft
        if mkl_dfti.DftiComputeForward(hand, x): raise Exception()

        ##abs (inplace)
        #xc = <MKL_COMPLEX16*>x
        #vzMulByConj(N1*N2*(N3//2+1),xc,xc,xc)

        #abs (inplace, batched to avoid bug in vml)
        for i in range(N1):
            xc = <MKL_COMPLEX16*>&x[i*(2*N2*(N3//2+1))]
            vzMulByConj(N2*(N3//2+1),xc,xc,xc)

        #c2r ifft setup
        if mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_INPUT_STRIDES, cs): raise Exception()
        if mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_OUTPUT_STRIDES, rs): raise Exception()
        if mkl_dfti.DftiCommitDescriptor(hand): raise Exception()

        #ifft
        if mkl_dfti.DftiComputeBackward(hand, x): raise Exception()
        if mkl_dfti.DftiFreeDescriptor(&hand): raise Exception()
        
    except:
        #cleanup
        mkl_dfti.DftiFreeDescriptor(&hand)
        return -1
    
    return 0
