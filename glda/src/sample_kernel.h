/* 
 * File:   sample_kernel.h
 * Author: mianlu
 *
 * Created on February 6, 2013, 3:34 PM
 */

#ifndef SAMPLE_KERNEL_H
#define	SAMPLE_KERNEL_H

void cuda_sampling(int* d_new,
        const int* d_z, const int* d_nw, const int* d_nd, const int* d_nwsum, const int* d_ndsum,
        const int* d_word, const int* d_doc,
        const unsigned int* d_wordOffsetPB, const unsigned int* d_wordNumPB,
        unsigned int* d_wordProcessedPB,
        const int K,
        const double alpha, const double beta,
        const double Kalpha, const double Vbeta,
        const float* d_random,
        const unsigned int numBlock, const unsigned int numThread, const unsigned int sharedMemSize);


void cuda_update(int* d_z, int* d_nw, int* d_nd, int* d_nwsum, int* d_ndsum,
        const int* d_new,
        const int* d_word, const int* d_doc,
        const unsigned int* d_wordOffsetPB, const unsigned int* d_wordNumPB,
        unsigned int* d_wordProcessedPB,
        const int K,
        const unsigned int numBlock, const unsigned int numThread);


void cuda_random(float* _d_random, const unsigned int _numTotalWord);


#endif	/* SAMPLE_KERNEL_H */

