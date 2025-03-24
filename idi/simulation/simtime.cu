#include <math.h>
#include <stdio.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <stdbool.h>

#ifndef usenf
    #define trans1 trans1ff
#else
    #define trans1 trans1nf
#endif
#ifdef nodist
    const bool scale=false;
#else
    const bool scale=true;
#endif

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

/* Transformation to complex amplitue and arrival time for each emitter*/
template <typename T,typename Tt> struct trans1nf {
  typedef typename Vec<T, 2>::type T2;
  typedef typename Vec<T, 3>::type T3;
  typedef typename Vec<T, 4>::type T4;
  T3 det;
  double distoff;
  T invc;
  T k;
  __device__ __forceinline__ trans1nf(T3 _det, T _k, T _c) {
    det = _det;
    distoff = norm3d(_det.x, _det.y, _det.z);
    invc = 1 / _c;
    k = _k;
  }
  __device__ __forceinline__ thrust::tuple<Tt, T2>
  operator()(thrust::tuple<T4, T> tup) {
    auto atom = thrust::get<0>(tup);
    auto time = thrust::get<1>(tup);
    double d = norm3d((double)atom.x - (double) det.x, (double) atom.y - (double) det.y, (double) atom.z - (double) det.z);
    double s = d-distoff+atom.z;
    Tt t = static_cast<Tt>(s  * invc + time);
    T phase = s  * k + atom.w;
    T real, imag;
    sc(phase, &imag, &real);
    if (scale)
    {
        T invd=1/((T) d);
        real=real*invd;
        imag=imag*invd;
     }
    auto amplitude = Vec<T, 2>::make(real, imag);
    return thrust::make_tuple(t, amplitude);
  }
};

/* Transformation to complex amplitue and arrival time for each emitter, far field approximation */

template <typename T,typename Tt> struct trans1ff {
  typedef typename Vec<T, 2>::type T2;
  typedef typename Vec<T, 3>::type T3;
  typedef typename Vec<T, 4>::type T4;
  T invkc, rdist;
  T3 q;

  __device__ __forceinline__ trans1ff(T3 _det, T _k, T _c) {
    double rdistd = rnorm3d(_det.x, _det.y, _det.z);
    double kinvd = _k * rdistd;
    q = Vec<T,3>::make(_det.x * kinvd, _det.y * kinvd, _k * (_det.z * rdistd - 1));
    rdist =  rdistd;
    invkc =  1. / (_c*_k);
  }
  __device__ __forceinline__ thrust::tuple<Tt, T2>
  operator()(thrust::tuple<T4, T> tup) {
    auto atom = thrust::get<0>(tup);
    auto time = thrust::get<1>(tup);
    T s = atom.x*q.x+atom.y*q.y+atom.z*q.z;
    Tt t = -s*invkc+time;
    T phase = -s+atom.w;
    T real, imag;
    sc(phase, &imag, &real);
    if (scale)
    {
        imag=imag*rdist;
        real=real*rdist;
    }
    auto amplitude = Vec<T, 2>::make(real, imag);
    return thrust::make_tuple(t, amplitude);
  }
};


/* Integral from t0 to t1 */
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

/* Operator in sum */
template <typename T, typename Tt> struct decayop {
  typedef typename Vec<T, 2>::type T2;
  Tt tau;
  __device__ __forceinline__ decayop(T t) { tau =(Tt) t; }
  CUB_RUNTIME_FUNCTION __forceinline__ thrust::tuple<Tt, T2>
  operator()(const thrust::tuple<Tt, T2> &lhs, const thrust::tuple<Tt, T2> &rhs) const {
    auto tl = thrust::get<0>(lhs);
    auto tr = thrust::get<0>(rhs);
    auto al = thrust::get<1>(lhs);
    auto ar = thrust::get<1>(rhs);
    T decay = exp((T) (-(tr - tl) / tau));
    return thrust::make_tuple(tr, Vec<T, 2>::make(ar.x + decay * al.x, ar.y + decay * al.y));
  }
};


/* Get size for temp buffers */
template <typename Ta, typename Tt>
__forceinline__ __device__ void
_tempsize(long long n, size_t *output) {
    typedef typename Vec<Ta, 2>::type Ta2;
    auto temp_storage_bytes_sort = output[0];
    auto temp_storage_bytes_scan = output[1];
    void *d_temp_storage = NULL;
    Tt *keys = NULL;
    Ta2 *values = NULL;
    cub::DoubleBuffer<Tt> b_keys(keys, keys);
    cub::DoubleBuffer<Ta2> b_values(values, values);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes_sort, b_keys, b_values, n, 0, sizeof(Tt) * 8);
    auto in = thrust::make_zip_iterator(thrust::make_tuple(b_keys.Current(), b_values.Current()));
    auto out = thrust::make_zip_iterator(thrust::make_tuple(b_keys.Alternate(), b_values.Alternate()));
    decayop<Ta, Tt> op(1.);
    cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes_scan, in, out, op, n);
}

extern "C" __global__  void
tempsized(long long n, size_t *output){
  _tempsize<double,double>(n, output);
}
extern "C" __global__  void
tempsizef(long long n, size_t *output){
  _tempsize<float,float>(n, output);
}
extern "C" __global__ void
tempsizedf(long long n, size_t *output){
  _tempsize<double,float>(n, output);
}

__device__ __attribute__((used)) void (*dummy_tempsizef_ptr)(long long, size_t*) = tempsizef;
__device__ __attribute__((used)) void (*dummy_tempsized_ptr)(long long, size_t*) = tempsized;  
__device__ __attribute__((used)) void (*dummy_tempsizedf_ptr)(long long, size_t*) = tempsizedf;

template <typename Ta>
__global__ void compute_result_kernel(Ta* result, Ta res, typename Vec<Ta,2>::type last_val, Ta tau) {
  *result = (res + ab2(last_val)) * (tau / 2.);
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
        __syncthreads();
        compute_result_kernel<Ta><<<1, 1, 0, stream>>>(result, res, b_a.Current()[n - 1], tau);

    }
  }
}

/* Float inputs and calculations */
extern "C"
__global__  void
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
__global__  void
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
__global__  void
simulatedf(const double4 *__restrict__ pos, const double *__restrict__ times, const double3 *__restrict__ *__restrict__ dets,
          double tau, double c, double k, long long n, long long threads,
          float * __restrict__ * __restrict__ ts, double2 * __restrict__ * __restrict__  as,
          size_t temp_storage_bytes, void *__restrict__ *__restrict__ d_temp_storages,
          double * __restrict__ * __restrict__ results)
          {
          _simulate<double,float>(pos, times, dets, tau, c, k, n, threads, ts, as, temp_storage_bytes, d_temp_storages, results);
          }

// Global device pointers to force the kernels to be retained.
__device__ __attribute__((used))
void (*dummy_simulatef_ptr)(const float4 *__restrict__,
                            const float *__restrict__,
                            const float3 *__restrict__ *__restrict__,
                            double, double, double,
                            long long, long long,
                            float * __restrict__ * __restrict__,
                            float2 * __restrict__ * __restrict__,
                            size_t,
                            void *__restrict__ *__restrict__,
                            float * __restrict__ * __restrict__) = simulatef;

__device__ __attribute__((used))
void (*dummy_simulated_ptr)(const double4 *__restrict__,
                            const double *__restrict__,
                            const double3 *__restrict__ *__restrict__,
                            double, double, double,
                            long long, long long,
                            double * __restrict__ * __restrict__,
                            double2 * __restrict__ * __restrict__,
                            size_t,
                            void *__restrict__ *__restrict__,
                            double * __restrict__ * __restrict__) = simulated;

__device__ __attribute__((used))
void (*dummy_simulatedf_ptr)(const double4 *__restrict__,
                             const double *__restrict__,
                             const double3 *__restrict__ *__restrict__,
                             double, double, double,
                             long long, long long,
                             float * __restrict__ * __restrict__,
                             double2 * __restrict__ * __restrict__,
                             size_t,
                             void *__restrict__ *__restrict__,
                             double * __restrict__ * __restrict__) = simulatedf;