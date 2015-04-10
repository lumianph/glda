
#ifndef SAMPLE_KERNEL_CU
#define SAMPLE_KERNEL_CU

#include "sample_kernel.h"
#include "cuda_util.h"
#include "scan.cu"

using namespace CUDAUtil;

//output stored in data[0]

template<class T>
__device__ void reduce(T* data) {
    const unsigned int tid = threadIdx.x;
    for (unsigned int s = (blockDim.x) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            data[tid] += data[tid + s];
        }
        __syncthreads();
    }
}

__global__
void sampling_kernel(int* d_new, //new topic for each block
        const int* d_z, const int* d_nw, const int* d_nd, const int* d_nwsum, const int* d_ndsum,
        const int* d_word, const int* d_doc,
        const unsigned int* d_wordOffsetPB, const unsigned int* d_wordNumPB,
        unsigned int* d_wordProcessedPB,
        const int K,
        const double alpha, const double beta,
        const double Kalpha, const double Vbeta,
        const float* d_random) {

    extern __shared__ double p[];
    const unsigned int blockId = blockIdx.x;
    const unsigned int threadId = threadIdx.x;
    const unsigned int numThread = blockDim.x;

    //if there is no word, then terminate the thread
    if (d_wordProcessedPB[blockId] >= d_wordNumPB[blockId]) {
        return;
    }


    const unsigned int offset = d_wordOffsetPB[blockId] + d_wordProcessedPB[blockId];
    const unsigned int m = d_doc[offset];
    int topic = d_z[offset];
    const int w = d_word[offset];


    //mapping computation
    int nw, nd, nwsum;
    const unsigned int ndsum = d_ndsum[m] - 1;
    for (int k = threadIdx.x; k < K; k += numThread) {
        nw = d_nw[w * K + k];
        nd = d_nd[m * K + k];
        nwsum = d_nwsum[k];

        if (k == topic) {
            nw -= 1;
            nd -= 1;
            nwsum -= 1;
        }

        p[k] = (nw + beta) / (nwsum + Vbeta) *
                (nd + alpha) / (ndsum + Kalpha);
    }
    __syncthreads();


    //accumulation, prefix sum within the block, p[k] += p[k - 1]
    double4 idata4 = ((double4*) p)[threadId];
    __syncthreads();
    double4 odata4 = scan4Inclusive(idata4, p, K);
    __syncthreads();
    ((double4*) p)[threadId] = odata4;
    __syncthreads();


    //scale
    double u = d_random[offset] * p[K - 1];
    p[threadId] = (p[threadId] < u) ? 1.0 : 0.0;
    for (int k = numThread + threadIdx.x; k < K; k += numThread) {
        if (p[k] < u) {
            p[threadId] += 1.0;
        } else {
            break;
        }
    }
    __syncthreads();


    //reduction, result stored in p[0]
    reduce(p);

    //output
    if (threadId == 0) {
        d_new[blockId] = p[0];
    }
}

__global__
void update_kernel(int* d_z, int* d_nw, int* d_nd, int* d_nwsum, int* d_ndsum,
        const int* d_new,
        const int* d_word, const int* d_doc,
        const unsigned int* d_wordOffsetPB, const unsigned int* d_wordNumPB,
        unsigned int* d_wordProcessedPB,
        const int K) {

    const unsigned int blockId = blockIdx.x;

    //if there is no word, then terminate the thread
    if (d_wordProcessedPB[blockId] >= d_wordNumPB[blockId]) {
        return;
    }



    if (threadIdx.x == 0) {

        const unsigned int offset = d_wordOffsetPB[blockId] + d_wordProcessedPB[blockId];
        const int m = d_doc[offset];
        const int old_topic = d_z[offset];
        const int w = d_word[offset];
        const int new_topic = d_new[blockId];

        //decrease the old topic counts
        atomicSub(&(d_nw[w * K + old_topic]), 1);
        atomicSub(&(d_nd[m * K + old_topic]), 1);
        atomicSub(&(d_nwsum[old_topic]), 1);
        //atomicSub(&(d_ndsum[m]), 1); //actually no necessary using atomic


        //increase the new topic counts
        atomicAdd(&(d_nw[w * K + new_topic]), 1);
        atomicAdd(&(d_nd[m * K + new_topic]), 1); //no necessary using atomic
        atomicAdd(&(d_nwsum[new_topic]), 1);
        //atomicAdd(&(d_ndsum[m]), 1); //no necessary using atomic


        //update the topic in z
        d_z[offset] = new_topic;

        //update the word counters
        d_wordProcessedPB[blockId] += 1;
    }
}

void cuda_sampling(int* d_new,
        const int* d_z, const int* d_nw, const int* d_nd, const int* d_nwsum, const int* d_ndsum,
        const int* d_word, const int* d_doc,
        const unsigned int* d_wordOffsetPB, const unsigned int* d_wordNumPB,
        unsigned int* d_wordProcessedPB,
        const int K,
        const double alpha, const double beta,
        const double Kalpha, const double Vbeta,
        const float* d_random,
        const unsigned int numBlock, const unsigned int numThread, const unsigned int sharedMemSize) {

    sampling_kernel << <numBlock, numThread, sharedMemSize >> >(d_new, d_z, d_nw, d_nd, d_nwsum, d_ndsum,
            d_word, d_doc,
            d_wordOffsetPB, d_wordNumPB,
            d_wordProcessedPB,
            K,
            alpha, beta,
            Kalpha, Vbeta,
            d_random);
    getLastCudaError("sampling_kernel");
}

void cuda_update(int* d_z, int* d_nw, int* d_nd, int* d_nwsum, int* d_ndsum,
        const int* d_new,
        const int* d_word, const int* d_doc,
        const unsigned int* d_wordOffsetPB, const unsigned int* d_wordNumPB,
        unsigned int* d_wordProcessedPB,
        const int K,
        const unsigned int numBlock, const unsigned int numThread) { //numBlock and numThread should be the same as cuda_sampling now

    update_kernel << <numBlock, numThread >> >(d_z, d_nw, d_nd, d_nwsum, d_ndsum,
            d_new,
            d_word, d_doc,
            d_wordOffsetPB, d_wordNumPB,
            d_wordProcessedPB,
            K);
    getLastCudaError("update_kernel");
}

void cuda_random(float* d_random, const unsigned int numTotalWord) {
    float* h_random = new float[numTotalWord];

    for (unsigned int i = 0; i < numTotalWord; i += 1) {
        h_random[i] = random() / static_cast<double> (RAND_MAX);
    }

    checkCudaErrors(cudaMemcpy(d_random, h_random, sizeof (float) *numTotalWord, cudaMemcpyHostToDevice));

    delete h_random;
}


#endif /*SAMPLE_KERNEL_CU*/

