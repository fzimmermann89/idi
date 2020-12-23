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

template <typename T> struct trans1 {
  typedef typename Vec<T, 2>::type T2;
  typedef typename Vec<T, 3>::type T3;
  typedef typename Vec<T, 4>::type T4;
  T3 det;
  T distoff;
  T invc;
  T k;
  __device__ __forceinline__ trans1(T3 _det, T _k, T _c) {
    det = _det;
    distoff = norm3d(_det.x, _det.y, _det.z);
    invc = 1. / _c;
    k = _k;
  }
  __device__ __forceinline__ thrust::tuple<T, T2>
  operator()(thrust::tuple<T4, T> tup) {
    auto atom = thrust::get<0>(tup);
    auto time = thrust::get<1>(tup);
    T d = norm3d(atom.x - det.x, atom.y - det.y, atom.z - det.z);
    T t = (d - distoff) * invc + time;
    T phase = (d - distoff) * k + atom.w;
    T real, imag;
    sincos(phase, &imag, &real);
    auto amplitude = Vec<T, 2>::make(real / d, imag / d);
    return thrust::make_tuple(t, amplitude);
  }
};

template <typename T> struct trans2 {
  typedef typename Vec<T, 2>::type T2;
  T invtauhalf;
  __device__ __forceinline__ trans2(T tau) { invtauhalf = 2. / tau; }
  __device__ __forceinline__ double operator()(thrust::tuple<T2, T, T> tup) {
    auto a = thrust::get<0>(tup);
    auto t0 = thrust::get<1>(tup);
    auto t1 = thrust::get<2>(tup);
    return -ab2(a) * expm1((t0 - t1) * invtauhalf);
  }
};

template <typename T> struct decayop {
  typedef typename Vec<T, 2>::type T2;
  T tau;
  __device__ __forceinline__ decayop(T t) { tau = t; }
  CUB_RUNTIME_FUNCTION __forceinline__ thrust::tuple<T, T2>
  operator()(thrust::tuple<T, T2> &lhs, thrust::tuple<T, T2> &rhs) const {
    auto tl = thrust::get<0>(lhs);
    auto tr = thrust::get<0>(rhs);
    auto al = thrust::get<1>(lhs);
    auto ar = thrust::get<1>(rhs);
    T decay = exp(-(tr - tl) / tau);
    return thrust::make_tuple(tr, Vec<T, 2>::make(ar.x + decay * al.x, ar.y + decay * al.y));
  }
};

extern "C" __global__ void 
tempsize(long long n, size_t *output) {
  if (firstThread){
      size_t temp_storage_bytes_sort = 0;
      size_t temp_storage_bytes_scan = 0;
      void *d_temp_storage = NULL; 
      double *keys = NULL;
      double2 *values = NULL;
      cub::DoubleBuffer<double> b_keys(keys, keys);
      cub::DoubleBuffer<double2> b_values(values, values);
      cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes_sort, b_keys, b_values, n);
      auto in = thrust::make_zip_iterator(thrust::make_tuple(b_keys.Current(), b_values.Current()));
      auto out = thrust::make_zip_iterator(thrust::make_tuple(b_keys.Alternate(), b_values.Alternate()));
      decayop<double> op(1.);
      cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes_scan, in, out, op, n);
      cudaDeviceSynchronize();
      *output = max(temp_storage_bytes_sort, temp_storage_bytes_scan);
  }
}

extern "C" __global__ void 
simulate(const double4 *__restrict__ pos, const double *__restrict__ times, const double3 *__restrict__ *__restrict__ dets,
          double tau, double c, double k, long long n, long long threads,
          double * __restrict__ * __restrict__ ts, double2 * __restrict__ * __restrict__  as,
          size_t temp_storage_bytes, void *__restrict__ *__restrict__ d_temp_storages,
          double * __restrict__ * __restrict__ results) {
  if (isFlat) { 
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int id=threadIdx.x;id<threads;id+=blockDim.x){
        void* d_temp_storage = d_temp_storages[id];
        const double3* det = dets[id];
        double* t0 = ts[2*id];
        double* t1 = ts[2*id+1];
        double2* a0 = as[2*id];
        double2* a1 = as[2*id+1];
        double* result = results[id];
        trans1<double> opt1(*det, k, c);
        decayop<double> opd(tau);
        trans2<double> opt2(tau);     
        const auto data = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<const double4>{pos}, thrust::device_ptr<const double>{times}));
        cub::DoubleBuffer<double> b_t(t0, t1);
        cub::DoubleBuffer<double2> b_a(a0, a1);    
        auto out1 = thrust::make_zip_iterator(thrust::make_tuple(t0, a0));
        thrust::transform(thrust::cuda::par.on(stream), data, data + n, out1, opt1);
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, b_t, b_a, n, 0, 8*sizeof(double), stream);
        auto in = thrust::make_zip_iterator(thrust::make_tuple(b_t.Current(), b_a.Current()));
        auto out = thrust::make_zip_iterator(thrust::make_tuple(b_t.Alternate(), b_a.Alternate()));
        cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, in, out, opd, n, stream);
        b_t.selector = b_t.selector ^ 1;
        b_a.selector = b_a.selector ^ 1;
        auto transit = thrust::make_transform_iterator(thrust::make_zip_iterator(thrust::make_tuple(b_a.Current(), b_t.Current(), b_t.Current() + 1)), opt2);
        double res = thrust::reduce(thrust::cuda::par.on(stream), transit, transit + n - 1);
        cudaDeviceSynchronize(); 
        __syncthreads();
        *result = (double) (res + ab2(b_a.Current()[n - 1])) * (tau / 2.);     
    }  
  }
}
"""
import math
import cupy
import numpy as np

kernel=cupy.RawModule(code=code,backend='nvcc',options=('-dc','--std=c++11','--expt-relaxed-constexpr','-O3'))
ftempsize=kernel.get_function("tempsize")
fsimulate=kernel.get_function("simulate")


def simulate(simobject, Ndet, pixelsize, detz, k, c, tau, pulsewidth): 
  
    if np.size(Ndet) == 1:
        Ndet = [Ndet, Ndet]
    N=simobject.N
    
    if N>1e8:
      threads=1
    elif N>1e7:
      threads=2
    elif N>1e6:
      threads=4
    else:
      threads=8
      
    det = cupy.array(np.array(np.meshgrid(
        pixelsize * (np.arange(Ndet[0]) - (Ndet[0] / 2)), 
        pixelsize * (np.arange(Ndet[1]) - (Ndet[1] / 2)), 
        detz
    )).T.reshape(-1,3))
    pdet = cupy.array([i.data.ptr for i in det],np.uint64)
    
    
  
    tmpout=cupy.zeros(1,np.int64)
    ftempsize(grid=(1,),block=(1,),args=(N,tmpout))
    tempsize=int(tmpout.get()[0])
    temp = [cupy.zeros(tempsize//8+1, np.int64) for i in range(threads)]
    ptemp = cupy.array([i.data.ptr for i in temp], np.uint64)

    t = [cupy.zeros(N,np.float64) for i in range(2*threads)]
    a = [cupy.zeros(N,np.complex128) for i in range(2*threads)]
    pt = cupy.array([i.data.ptr for i in t], np.uint64)
    pa = cupy.array([i.data.ptr for i in a], np.uint64)
    data = cupy.array(simobject.get(),np.float64)
    times = cupy.random.randn(N,dtype=np.float64)*(pulsewidth/2.35)-data[:,2]/c #cave: this uses different seed as numpy
    output = cupy.zeros(len(det),np.float64)
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
      

    
    
    
