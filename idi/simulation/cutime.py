code = """
#include <math.h>
#include <stdio.h>
#include <cupy/cub/cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

template <typename T, int cn> struct Vec;
template <> struct Vec<float, 2> {typedef float2 type; static __device__ __host__ __forceinline__ type make(float a, float b) {return make_float2(a,b);}};
template <> struct Vec<double, 2> {typedef double2 type; static __device__ __host__ __forceinline__ type make(double a, double b) {return make_double2(a,b);}};
template <> struct Vec<float, 3> {typedef float3 type; static __device__ __host__ __forceinline__ type make(float a, float b, float c) {return make_float3(a,b,c);}};
template <> struct Vec<double, 3> {typedef double3 type; static __device__ __host__ __forceinline__ type make(double a, double b, double c) {return make_double3(a,b,c);}};
template <> struct Vec<float, 4> {typedef float4 type; static __device__ __host__ __forceinline__ type make(float a, float b, float c, float d) {return make_float4(a,b,c,d);}};
template <> struct Vec<double, 4> {typedef double4 type; static __device__ __host__ __forceinline__ type make(double a, double b, double c, double d) {return make_double4(a,b,c,d);}};

#define isFlat (blockIdx.x == 0  && blockIdx.y == 0 && threadIdx.y == 0 && blockIdx.z == 0 && threadIdx.z == 0)
#define firstThreads(N) (threadIdx.x < N && isFlat)
#define firstThread (firstThreads(1))


__device__ __forceinline__ double ab2(double2 a) {
  return a.x * a.x + a.y * a.y;
}
__device__ __forceinline__ float ab2(float2 a) {
  return a.x * a.x + a.y * a.y;
}

__device__ __forceinline__ void sc(float  x, float* sptr, float* cptr) {
  __sincosf(x, sptr, cptr);
}

__device__ __forceinline__ void sc(double  x, double* sptr, double* cptr) {
  sincos(x, sptr, cptr);
}

template <typename T,typename Tt> struct trans1 {
  typedef typename Vec<T, 2>::type T2;
  typedef typename Vec<T, 3>::type T3;
  typedef typename Vec<T, 4>::type T4;
  T3 det;
  double distoff;
  T invc;
  T k;
  __device__ __forceinline__ trans1(T3 _det, T _k, T _c) {
    det = _det;
    distoff = norm3d(_det.x, _det.y, _det.z);
    invc = 1. / _c;
    k = _k;
  }
  __device__ __forceinline__ thrust::tuple<Tt, T2>
  operator()(thrust::tuple<T4, T> tup) {
    auto atom = thrust::get<0>(tup);
    auto time = thrust::get<1>(tup);
    double d = norm3d((double)atom.x - (double) det.x, (double) atom.y - (double) det.y, (double) atom.z - (double) det.z);
    Tt t = static_cast<Tt>(d - distoff) * invc + time;
    T phase = (d - distoff) * k + atom.w;
    T real, imag;
    sc(phase, &imag, &real);
    auto amplitude = Vec<T, 2>::make((T) real / d, (T) imag / d);
    return thrust::make_tuple(t, amplitude);
  }
};

template <typename T, typename Tt> struct trans2 {
  typedef typename Vec<T, 2>::type T2;
  Tt invtauhalf;
  __device__ __forceinline__ trans2(T tau) { invtauhalf = (Tt)(2. / tau); }
  __device__ __forceinline__ T operator()(thrust::tuple<T2, Tt, Tt> tup) {
    auto a = thrust::get<0>(tup);
    auto t0 = thrust::get<1>(tup);
    auto t1 = thrust::get<2>(tup);
    return -ab2(a) * expm1(((t0 - t1) * invtauhalf));
  }
};

template <typename T, typename Tt> struct decayop {
  typedef typename Vec<T, 2>::type T2;
  Tt tau;
  __device__ __forceinline__ decayop(T t) { tau =(Tt) t; }
  CUB_RUNTIME_FUNCTION __forceinline__ thrust::tuple<Tt, T2>
  operator()(thrust::tuple<Tt, T2> &lhs, thrust::tuple<Tt, T2> &rhs) const {
    auto tl = thrust::get<0>(lhs);
    auto tr = thrust::get<0>(rhs);
    auto al = thrust::get<1>(lhs);
    auto ar = thrust::get<1>(rhs);
    T decay = exp((T) (-(tr - tl) / tau));
    return thrust::make_tuple(tr, Vec<T, 2>::make(ar.x + decay * al.x, ar.y + decay * al.y));
  }
};



template <typename Ta, typename Tt>
__forceinline__ __device__ void 
_tempsize(long long n, size_t *output) {
    typedef typename Vec<Ta, 2>::type Ta2;
  if (firstThread){
      size_t temp_storage_bytes_sort = 0;
      size_t temp_storage_bytes_scan = 0;
      void *d_temp_storage = NULL; 
      Tt *keys = NULL;
      Ta2 *values = NULL;
      cub::DoubleBuffer<Tt> b_keys(keys, keys);
      cub::DoubleBuffer<Ta2> b_values(values, values);
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes_sort, b_keys, b_values, n);
      auto in = thrust::make_zip_iterator(thrust::make_tuple(b_keys.Current(), b_values.Current()));
      auto out = thrust::make_zip_iterator(thrust::make_tuple(b_keys.Alternate(), b_values.Alternate()));
      decayop<Ta,Tt> op(1.);
      cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes_scan, in, out, op, n);
      cudaDeviceSynchronize();
      *output = max(temp_storage_bytes_sort, temp_storage_bytes_scan);
  }
}

extern "C" __global__ void 
tempsized(long long n, size_t *output){
  _tempsize<double,double>(n, output);
}
extern "C" __global__ void 
tempsizef(long long n, size_t *output){
  _tempsize<float,float>(n, output);
}
extern "C" __global__ void 
tempsizedf(long long n, size_t *output){
  _tempsize<double,float>(n, output);
}


template <typename Ta, typename Tt>
__forceinline__ __device__ void 
_simulate(const typename Vec<Ta, 4>::type *__restrict__ pos, const Ta *__restrict__ times, const typename Vec<Ta, 3>::type *__restrict__ *__restrict__ dets,
          Ta tau, Ta c, Ta k, long long n, long long threads,
          Tt * __restrict__ * __restrict__ ts, typename Vec<Ta, 2>::type * __restrict__ * __restrict__  as,
          size_t temp_storage_bytes, void *__restrict__ *__restrict__ d_temp_storages,
          Ta * __restrict__ * __restrict__ results) {
  typedef typename Vec<Ta, 2>::type Ta2;
  typedef typename Vec<Ta, 3>::type Ta3;
  typedef typename Vec<Ta, 4>::type Ta4;

  if (isFlat) { 
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int id=threadIdx.x;id<threads;id+=blockDim.x){
        void* d_temp_storage = d_temp_storages[id];
        const Ta3* det = dets[id];
        Tt* t0 = ts[2*id];
        Tt* t1 = ts[2*id+1];
        Ta2* a0 = as[2*id];
        Ta2* a1 = as[2*id+1];
        Ta* result = results[id];
        trans1<Ta,Tt> opt1(*det, k, c);
        decayop<Ta,Tt> opd(tau);
        trans2<Ta,Tt> opt2(tau);     
        const auto data = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<const Ta4>{pos}, thrust::device_ptr<const Ta>{times}));
        cub::DoubleBuffer<Tt> b_t(t0, t1);
        cub::DoubleBuffer<Ta2> b_a(a0, a1);    
        auto out1 = thrust::make_zip_iterator(thrust::make_tuple(t0, a0));
        thrust::transform(thrust::cuda::par.on(stream), data, data + n, out1, opt1);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, b_t, b_a, n, 0, 8*sizeof(Tt), stream);
        auto in = thrust::make_zip_iterator(thrust::make_tuple(b_t.Current(), b_a.Current()));
        auto out = thrust::make_zip_iterator(thrust::make_tuple(b_t.Alternate(), b_a.Alternate()));
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, in, out, opd, n, stream);
        b_t.selector = b_t.selector ^ 1;
        b_a.selector = b_a.selector ^ 1;
        auto transit = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(b_a.Current(), b_t.Current(), b_t.Current() + 1)), opt2);
        Ta res = thrust::reduce(thrust::cuda::par.on(stream), transit, transit + n - 1);
        cudaDeviceSynchronize(); 
        __syncthreads();
        *result = (res + ab2(b_a.Current()[n - 1])) * (tau / 2.);     
    }  
  }
}

/* Float inputs and calculations */
extern "C"
__global__ void 
simulatef(const float4 *__restrict__ pos, const float *__restrict__ times, const float3 *__restrict__ *__restrict__ dets,
          double tau, double c, double k, long long n, long long threads,
          float * __restrict__ * __restrict__ ts, float2 * __restrict__ * __restrict__  as,
          size_t temp_storage_bytes, void *__restrict__ *__restrict__ d_temp_storages,
          float * __restrict__ * __restrict__ results)
          {
          _simulate<float,float>(pos, times, dets, (float) tau, (float) c, (float) k, n, threads, ts, as, temp_storage_bytes, d_temp_storages, results);
          }

/* Double inputs and calculations */         
extern "C"
__global__ void 
simulated(const double4 *__restrict__ pos, const double *__restrict__ times, const double3 *__restrict__ *__restrict__ dets,
          double tau, double c, double k, long long n, long long threads,
          double * __restrict__ * __restrict__ ts, double2 * __restrict__ * __restrict__  as,
          size_t temp_storage_bytes, void *__restrict__ *__restrict__ d_temp_storages,
          double * __restrict__ * __restrict__ results)
          {
          _simulate<double,double>(pos, times, dets, tau, c, k, n, threads, ts, as, temp_storage_bytes, d_temp_storages, results);
          }

/* Double inputs and mixed precision calculations, good tradeoff between performance and precision*/                   
extern "C"
__global__ void 
simulatedf(const double4 *__restrict__ pos, const double *__restrict__ times, const double3 *__restrict__ *__restrict__ dets,
          double tau, double c, double k, long long n, long long threads,
          float * __restrict__ * __restrict__ ts, double2 * __restrict__ * __restrict__  as,
          size_t temp_storage_bytes, void *__restrict__ *__restrict__ d_temp_storages,
          double * __restrict__ * __restrict__ results)
          {
          _simulate<double,float>(pos, times, dets, tau, c, k, n, threads, ts, as, temp_storage_bytes, d_temp_storages, results);
          }
"""
import math
import cupy
import numpy as np

module=cupy.RawModule(code=code,backend='nvcc',options=('-dc','--std=c++11','--expt-relaxed-constexpr','-O3','--use_fast_math'))

def simulate(simobject, Ndet, pixelsize, detz, k, c, tau, pulsewidth, precision='mixed', threads=None): 
    cupy.random.seed(np.random.randint(2**64-1,dtype=np.uint64))
    if precision == 'mixed':
        ftempsize=module.get_function("tempsizedf")
        fsimulate=module.get_function("simulatedf")
        ttype, atype, inouttype = np.float32, np.complex128, np.float64
    elif precision == 'double':
        ftempsize=module.get_function("tempsized")
        fsimulate=module.get_function("simulated")
        ttype, atype, inouttype = np.float64, np.complex128, np.float64
    elif precision == 'single':
        ftempsize=module.get_function("tempsizef")
        fsimulate=module.get_function("simulatef") 
        ttype, atype, inouttype = np.float32, np.complex64, np.float32
    else: 
        raise ValueError
    
    N=simobject.N
    if threads is None:
        if N>1e8:
            threads=1
        elif N>1e7:
            threads=2
        elif N>1e6:
            threads=4
        else:
            threads=8
              
    if np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    
    
    det = cupy.array(np.array(np.meshgrid(
        pixelsize * (np.arange(Ndet[0]) - (Ndet[0] / 2)), 
        pixelsize * (np.arange(Ndet[1]) - (Ndet[1] / 2)), 
        detz
    )).T.reshape(-1,3), inouttype)
    pdet = cupy.array([i.data.ptr for i in det],np.uint64)
    
    
  
    tmpout=cupy.zeros(1,np.int64)
    ftempsize(grid=(1,),block=(1,),args=(N,tmpout))
    tempsize=int(tmpout.get()[0])
    temp = [cupy.zeros(tempsize//8+1, np.int64) for i in range(threads)]
    ptemp = cupy.array([i.data.ptr for i in temp], np.uint64)

    t = [cupy.zeros(N,ttype) for i in range(2*threads)]
    a = [cupy.zeros(N,atype) for i in range(2*threads)]
    pt = cupy.array([i.data.ptr for i in t], np.uint64)
    pa = cupy.array([i.data.ptr for i in a], np.uint64)
    data = cupy.array(simobject.get(),inouttype)
    times = (cupy.random.randn(N,dtype=np.float64)*((pulsewidth/2.35)-data[:,2]/c)).astype(inouttype) #cave: this uses different seed as numpy
    output = cupy.zeros(len(det),inouttype)
    poutput = cupy.array([i.data.ptr for i in output],np.uint64)
    cupy.cuda.get_current_stream().synchronize()
    for start in range(0,len(pdet),threads):
        cthreads=int(min(len(pdet)-start,threads))
        end=start+cthreads
        fsimulate(grid=(1,),block=(cthreads,),
                args=(data, times, pdet[start:end],float(tau),float(c),float(k),int(N),int(cthreads),pt,pa,tempsize,ptemp,poutput[start:end])
        )
    cupy.cuda.get_current_stream().synchronize()  
    return output.get().reshape(Ndet)