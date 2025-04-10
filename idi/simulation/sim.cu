/*
*This should be 10 lines of thrust code, but it's not.
*Partially because it started as pycuda, partially because it had to run on an old gpu.
*Can run as double or single precision with e^iqr (Fraunhofer) approximation (scattering direction constant within sample), second order in x,y (Fresnel)
*or in double precision with e^ik(R-r) for the phase (correct for incoming plane wave)
*/
#include <math.h>
#include <stdbool.h>

#define posx (threadIdx.x + blockDim.x * blockIdx.x)
#define posy (threadIdx.y + blockDim.y * blockIdx.y)

#ifndef nodist
    #define nodist 0
#endif
#ifndef secondorder
    #define secondorder 0
#endif
#ifndef usenf
    #define phaseop phaseff
#else
    #define phaseop phasenf
#endif
#ifndef nodetrot
    #define nodetrot 0
#endif
template <typename T, int cn> struct Vec;
template <> struct Vec<float, 2> {typedef float2 type; static __device__ __host__ __forceinline__ type make(float a, float b) {return make_float2(a,b);}};
template <> struct Vec<double, 2> {typedef double2 type; static __device__ __host__ __forceinline__ type make(double a, double b) {return make_double2(a,b);}};
template <> struct Vec<float, 3> {typedef float3 type; static __device__ __host__ __forceinline__ type make(float a, float b, float c) {return make_float3(a,b,c);}};
template <> struct Vec<double, 3> {typedef double3 type; static __device__ __host__ __forceinline__ type make(double a, double b, double c) {return make_double3(a,b,c);}};
template <> struct Vec<float, 4> {typedef float4 type; static __device__ __host__ __forceinline__ type make(float a, float b, float c, float d) {return make_float4(a,b,c,d);}};
template <> struct Vec<double, 4> {typedef double4 type; static __device__ __host__ __forceinline__ type make(double a, double b, double c, double d) {return make_double4(a,b,c,d);}};

__device__ __forceinline__ void sc(float x, float * sptr, float * cptr) {
    sincosf(x, sptr, cptr);
}

__device__ __forceinline__ void sc(double x, double * sptr, double * cptr) {
    sincos(x, sptr, cptr);
}

template < typename Ti, typename To >
struct phasenf {
    float detx, dety, detz, k;
    typedef typename Vec < To, 2 > ::type To2;
    typedef typename Vec < Ti, 4 > ::type Ti4;

    __device__ __forceinline__ phasenf(float _detx, float _dety, float _detz, float _k) {
        detx = _detx;
        dety = _dety;
        detz = _detz;
        k = _k;
    }
    __device__ To2 operator()(const Ti4 & atom) const {
        auto dist = norm3d(
            ((double) detx - (double) atom.x),
            ((double) dety - (double) atom.y),
            ((double) detz - (double) atom.z)
        );

        auto phase = (dist - detz + atom.z) * k + atom.w;
        double real, imag;
        sc(phase, & imag, & real);
        if (nodist) return Vec < To, 2 > ::make((To)(real), (To)(imag));
        To rdist = 1 / ((To) dist);
        return Vec < To, 2 > ::make(((To) real) * rdist, ((To) imag) * rdist);
    }
};

template < typename Ti, typename To >
struct phaseff {
    To qx, qy, qz, c, rdist;
    typedef typename Vec < To, 2 > ::type To2;
    typedef typename Vec < Ti, 4 > ::type Ti4;

    __device__ __forceinline__ phaseff(float _detx, float _dety, float _detz, float _k) {
        float rdistf = rnorm3d(_detx, _dety, _detz);
        qz = (To)(_k * (_detz * rdistf - 1));
        float kinvd = _k * rdistf;
        qx = (To)(_detx * kinvd);
        qy = (To)(_dety * kinvd);
        c = (To)(kinvd / 2);
        rdist = (To) rdistf;
    }
    __device__ __forceinline__ To2 operator()(const Ti4 & atom) const {
        To phase = -(atom.x * qx + atom.y * qy + atom.z * qz) + atom.w;
        if (secondorder) phase += c * (atom.x * atom.x + atom.y * atom.y); /* this is the second order in x and y, still first in detz*/
        To real, imag;
        sc(phase, & imag, & real);
        if (nodist) return Vec < To, 2 > ::make(real, imag);
        return Vec < To, 2 > ::make(rdist * real, rdist * imag);

    }
};

__device__ __forceinline__ double4 __ldg(const double4 * d4) {
    return *d4;
}

template < typename Ti, typename To >
struct wfkernel {
    typedef typename Vec < To, 2 > ::type To2;
    typedef typename Vec < Ti, 4 > ::type Ti4;
    __device__ __forceinline__ void operator()(To2 __restrict__ * ret, Ti4 const __restrict__ * atoms, const float * __restrict__ M, float _detx, float _dety, float _detz, float pixelsize, float k, int maxx, int maxy, int Natoms) const {
        int x = posx, y = posy;
        if ((x < maxx) && (y < maxy)) {
            float detx = (x - maxx / 2) * pixelsize + _detx;
            float dety = (y - maxy / 2) * pixelsize + _dety;
            float detz;
            if (nodetrot) {
                 detz = _detz;
            }
            else {
                 detx = M[0]*detx + M[1]*dety + M[2]*detz;
                 dety = M[3]*detx + M[4]*dety + M[5]*detz;
                 detz = M[6]*detx + M[7]*dety + M[8]*detz;
            }
            To accumx = 0, accumy = 0, cx = 0, cy = 0;
            auto op = phaseop < Ti, To > (detx, dety, detz, k);
            for (int i = 0; i < Natoms; i++) {
                /* hot loop */
                Ti4 atom = __ldg( & atoms[i]);

                auto phase = op(atom);
                /* kahan summation to give meaningful result for Natoms>1e4 in single precision */
                auto yx = phase.x - cx;
                auto yy = phase.y - cy;
                auto tx = accumx + yx;
                auto ty = accumy + yy;
                cx = (tx - accumx) - yx;
                cy = (ty - accumy) - yy;
                accumx = tx;
                accumy = ty;
            }
            ret[x * maxy + y] = Vec < To, 2 > ::make((To) accumx, (To) accumy);
        }
    };
};

extern "C"
__global__ void wfkernelf(float2 * __restrict__ ret, const float4 * __restrict__ atoms, const float * __restrict__  M, float detx, float dety, float detz, float pixelsize, float k, int maxx, int maxy, int Natoms) {
    wfkernel < float, float > ()(ret, atoms, M, detx, dety, detz, pixelsize, k, maxx, maxy, Natoms);
};

extern "C"
__global__ void wfkerneld(double2 * __restrict__ ret, const double4 * __restrict__ atoms, const float * __restrict__  M, float detx, float dety, float detz, float pixelsize, float k, int maxx, int maxy, int Natoms) {
    wfkernel < double, double > ()(ret, atoms, M, detx, dety, detz, pixelsize, k, maxx, maxy, Natoms);
};
