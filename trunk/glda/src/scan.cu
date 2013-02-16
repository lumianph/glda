#ifndef SCAN_CU
#define SCAN_CU

#include "cuda_util.h"

inline __device__ double scan1Inclusive(double idata, volatile double *s_Data, uint size) {
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        __syncthreads();
        double t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ double scan1Exclusive(double idata, volatile double *s_Data, uint size) {
    return scan1Inclusive(idata, s_Data, size) - idata;
}

inline __device__ double4 scan4Inclusive(double4 idata4, volatile double *s_Data, uint size) {
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    double oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}




#endif /*SACN_CU*/

