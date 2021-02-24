cimport mkl_dfti
import cython
cdef extern from 'mkl.h' nogil:
    ctypedef struct MKL_Complex16:
        double real
        double imag   
    void vzMulByConj(int n, const MKL_Complex16*, const MKL_Complex16*,MKL_Complex16*) nogil
    int vmlGetErrStatus() nogil
    int vmlSetErrStatus(int) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef int autocorrelate3(double[:, :, ::1] input):
    '''
    calculates MxNxO autocorrelation of (M)xNx(O+2) array using mkl fft. returns 0 on success.
    '''
    #strides
    cdef long N1=input.shape[0], N2=input.shape[1], N3=input.shape[2]-2 #N3: extra space needed for r2c-> do smaller fft
    cdef long N[3]
    N[:] = [N1, N2, N3]  
    cdef long rs[4]
    rs[:] = [0, N2*(N3//2+1)*2, (N3//2+1)*2, 1]
    cdef long cs[4]
    cs[:] = [0, N2*(N3//2+1), (N3//2+1), 1]
    cdef double* x = <double*> &input[0,0,0]
    cdef mkl_dfti.DFTI_DESCRIPTOR* hand = NULL;
    cdef double scale = 1./(<double>N1*<double>N2*<double>N3)
    cdef long i = 0
    cdef bint error = False
    cdef int oldvmlerr = vmlSetErrStatus(0)

    
    #r2c fft setup
    if not error: error = mkl_dfti.DftiCreateDescriptor(&hand,mkl_dfti. DFTI_DOUBLE, mkl_dfti.DFTI_REAL, 3, N)
    if not error: error = mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_CONJUGATE_EVEN_STORAGE, mkl_dfti.DFTI_COMPLEX_COMPLEX)
    if not error: error = mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_INPUT_STRIDES, rs)
    if not error: error = mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_OUTPUT_STRIDES, cs)
    if not error: error = mkl_dfti.DftiCommitDescriptor(hand)

    #fft
    if not error: error = mkl_dfti.DftiComputeForward(hand, x)

    if not error:
        #abs (inplace)
        #xc = <MKL_Complex16*>x
        #vzMulByConj(N1*N2*(N3//2+1),<const MKL_Complex16*> xc,<const MKL_Complex16*> xc,xc)

        #abs (inplace, batched to avoid bug in vml)
        for i from 0 <= i < N1:
                xc = <MKL_Complex16*>&x[i*(2*N2*(N3//2+1))]
                vzMulByConj(N2*(N3//2+1),<const MKL_Complex16*> xc,<const MKL_Complex16*> xc,xc)

        error=vmlSetErrStatus(oldvmlerr)
        #ignore accuracy warning
        if error==1000: error =0

    #c2r ifft setup
    if not error: error = mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_INPUT_STRIDES, cs)
    if not error: error = mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_OUTPUT_STRIDES, rs)
    if not error: error = mkl_dfti.DftiSetValue(hand, mkl_dfti.DFTI_BACKWARD_SCALE, scale)
    if not error: error = mkl_dfti.DftiCommitDescriptor(hand)

    #ifft
    if not error: error = mkl_dfti.DftiComputeBackward(hand, x)

    #cleanup
    if error:
        mkl_dfti.DftiFreeDescriptor(&hand)
    else: 
        error=mkl_dfti.DftiFreeDescriptor(&hand)

    return error
