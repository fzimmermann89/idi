#Translation of mkl_dfti.h to cython

cdef extern from "mkl.h" nogil:
    ctypedef enum DFTI_CONFIG_PARAM:
        # Domain for forward transform. No default value 
        DFTI_FORWARD_DOMAIN = 0,

        # Dimensionality, or rank. No default value 
        DFTI_DIMENSION = 1,

        # Length(s) of transform. No default value 
        DFTI_LENGTHS = 2,

        # Floating point precision. No default value 
        DFTI_PRECISION = 3,

        # Scale factor for forward transform [1.0] 
        DFTI_FORWARD_SCALE  = 4,

        # Scale factor for backward transform [1.0] 
        DFTI_BACKWARD_SCALE = 5,

        # Exponent sign for forward transform [DFTI_NEGATIVE]  
        # DFTI_FORWARD_SIGN = 6, ## NOT IMPLEMENTED 

        # Number of data sets to be transformed [1] 
        DFTI_NUMBER_OF_TRANSFORMS = 7,

        # Storage of finite complex-valued sequences in complex domain [DFTI_COMPLEX_COMPLEX] 
        DFTI_COMPLEX_STORAGE = 8,

        # Storage of finite real-valued sequences in real domain [DFTI_REAL_REAL] 
        DFTI_REAL_STORAGE = 9,

        # Storage of finite complex-valued sequences in conjugate-even domain [DFTI_COMPLEX_REAL] 
        DFTI_CONJUGATE_EVEN_STORAGE = 10,

        # Placement of result [DFTI_INPLACE] 
        DFTI_PLACEMENT = 11,

        # Generalized strides for input data layout [tight, row-major for C] 
        DFTI_INPUT_STRIDES = 12,

        # Generalized strides for output data layout [tight, row-major for C] 
        DFTI_OUTPUT_STRIDES = 13,

        # Distance between first input elements for multiple transforms [0] 
        DFTI_INPUT_DISTANCE = 14,

        # Distance between first output elements for multiple transforms [0] 
        DFTI_OUTPUT_DISTANCE = 15,

        # Effort spent in initialization [DFTI_MEDIUM] 
        # DFTI_INITIALIZATION_EFFORT = 16, ## NOT IMPLEMENTED 

        # Use of workspace during computation [DFTI_ALLOW] 
        DFTI_WORKSPACE = 17,

        # Ordering of the result [DFTI_ORDERED] 
        DFTI_ORDERING = 18,

        # Possible transposition of result [DFTI_NONE] 
        DFTI_TRANSPOSE = 19,

        # User-settable descriptor name [""] 
        DFTI_DESCRIPTOR_NAME = 20, # DEPRECATED 

        # Packing format for DFTI_COMPLEX_REAL storage of finiteconjugate-even sequences [DFTI_CCS_FORMAT] 
        DFTI_PACKED_FORMAT = 21,

        # Commit status of the descriptor - R/O parameter 
        DFTI_COMMIT_STATUS = 22,

        # Version string for this DFTI implementation - R/O parameter 
        DFTI_VERSION = 23,

        # Ordering of the forward transform - R/O parameter 
        # DFTI_FORWARD_ORDERING  = 24, ## NOT IMPLEMENTED 

        # Ordering of the backward transform - R/O parameter 
        # DFTI_BACKWARD_ORDERING = 25, ## NOT IMPLEMENTED 

        # Number of user threads that share the descriptor [1] 
        DFTI_NUMBER_OF_USER_THREADS = 26,

        # Limit the number of threads used by this descriptor [0 = don't care] 
        DFTI_THREAD_LIMIT = 27,

        # Possible input data destruction [DFTI_AVOID = prevent input data]
        DFTI_DESTROY_INPUT = 28

    ctypedef enum DFTI_CONFIG_VALUE:
        # DFTI_COMMIT_STATUS 
        DFTI_COMMITTED = 30,
        DFTI_UNCOMMITTED = 31,

        # DFTI_FORWARD_DOMAIN 
        DFTI_COMPLEX = 32,
        DFTI_REAL = 33,
        # DFTI_CONJUGATE_EVEN = 34,   ## NOT IMPLEMENTED 

        # DFTI_PRECISION 
        DFTI_SINGLE = 35,
        DFTI_DOUBLE = 36,

        # DFTI_FORWARD_SIGN 
        # DFTI_NEGATIVE = 37,         ## NOT IMPLEMENTED 
        # DFTI_POSITIVE = 38,         ## NOT IMPLEMENTED 

        # DFTI_COMPLEX_STORAGE and DFTI_CONJUGATE_EVEN_STORAGE 
        DFTI_COMPLEX_COMPLEX = 39,
        DFTI_COMPLEX_REAL = 40,

        # DFTI_REAL_STORAGE 
        DFTI_REAL_COMPLEX = 41,
        DFTI_REAL_REAL = 42,

        # DFTI_PLACEMENT 
        DFTI_INPLACE = 43,          # Result overwrites input 
        DFTI_NOT_INPLACE = 44,      # Have another place for result 

        # DFTI_INITIALIZATION_EFFORT 
        # DFTI_LOW = 45,              ## NOT IMPLEMENTED 
        # DFTI_MEDIUM = 46,           ## NOT IMPLEMENTED 
        # DFTI_HIGH = 47,             ## NOT IMPLEMENTED 

        # DFTI_ORDERING 
        DFTI_ORDERED = 48,
        DFTI_BACKWARD_SCRAMBLED = 49,
        # DFTI_FORWARD_SCRAMBLED = 50, ## NOT IMPLEMENTED 

        # Allow/avoid certain usages 
        DFTI_ALLOW = 51,            # Allow transposition or workspace 
        DFTI_AVOID = 52,
        DFTI_NONE = 53,

        # DFTI_PACKED_FORMAT (for storing congugate-even finite sequencein real array) 
        DFTI_CCS_FORMAT = 54,       # Complex conjugate-symmetric 
        DFTI_PACK_FORMAT = 55,      # Pack format for real DFT 
        DFTI_PERM_FORMAT = 56,      # Perm format for real DFT 
        DFTI_CCE_FORMAT = 57        # Complex conjugate-even 


    cdef struct DFTI_DESCRIPTOR:
        int dontuse

    long DftiCreateDescriptor(DFTI_DESCRIPTOR**,
                               DFTI_CONFIG_VALUE, 
                               DFTI_CONFIG_VALUE,
                              long, ...);
    long DftiCommitDescriptor(DFTI_DESCRIPTOR*);
    long DftiComputeForward(DFTI_DESCRIPTOR*, void*, ...);
    long DftiComputeBackward(DFTI_DESCRIPTOR*, void*, ...);
    long DftiSetValue(DFTI_DESCRIPTOR*,  DFTI_CONFIG_PARAM, ...);
    long DftiGetValue(DFTI_DESCRIPTOR*,  DFTI_CONFIG_PARAM, ...);
    long DftiCopyDescriptor(DFTI_DESCRIPTOR*, DFTI_DESCRIPTOR**);
    long DftiFreeDescriptor(DFTI_DESCRIPTOR**);
    char* DftiErrorMessage(long);
    long DftiErrorClass(long, long);


