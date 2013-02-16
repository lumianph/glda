/* 
 * File:   CUDASampling.cpp
 * Author: mianlu
 * 
 * Created on February 5, 2013, 5:42 PM
 */

#include "CUDASampling.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <cmath>
#include <memory.h>
#include "cuda_util.h"
#include "sample_kernel.h"

using namespace std;
using namespace CUDAUtil;

template<class T>
T max(const T* data, const unsigned int size) {
    T v = data[0];
    for (unsigned int i = 1; i < size; i += 1) {
        if (data[i] > v) {
            v = data[i];
        }
    }
    return v;
}

template<class T>
T sum(const T* data, const unsigned int size) {
    T r = 0;
    for (unsigned int i = 0; i < size; i += 1) {
        r += data[i];
    }
    return r;
}

//--------------------------------------------------------------------------
//ParallelSampling

bool ParallelSampling::checkConflict(const int* doc, const unsigned int numTotalWord,
        const unsigned int* wordOffsetPB, const unsigned int* wordNumPB, const unsigned int numBlock) const {

#if DEBUG_LEVEL == DEBUG_ALL
    //check whether doc is sorted
    for (unsigned int i = 0; i < numTotalWord - 1; i += 1) {
        if (doc[i] > doc[i + 1]) {
            throw std::runtime_error(ERR_MSG("unordered for doc"));
            return false;
        }
    }
#endif

    unsigned int maxWordNumPB = max(wordNumPB, numBlock);


    vector<int> word;
    for (unsigned int i = 0; i < maxWordNumPB; i += 1) {
        //put every word into the vector
        word.clear();
        for (unsigned int blockId = 0; blockId < numBlock; blockId += 1) {
            if (i < wordNumPB[blockId]) {
                word.push_back(doc[wordOffsetPB[blockId] + i]);
            }
        }

        //check whether there are duplicates in the vector
        if (word.size() > 1) {
            for (unsigned int k = 0; k < word.size() - 1; k += 1) {
                if (word[k] == word[k + 1]) {
                    return false;
                }
            }
        }
    }

    return true;
}

void ParallelSampling::computeWordPB(unsigned int* wordOffsetPB, unsigned int* wordNumPB, const unsigned int numTotalWord,
        const int* doc) {

    const unsigned int numWordPB = ceil(numTotalWord / static_cast<double> (_numBlock));
    assert(numWordPB > 0);
    
    unsigned int count = numTotalWord; //#remaining elements
    for (unsigned int i = 0; i < _numBlock; i += 1) {
        wordOffsetPB[i] = i*numWordPB;
        if (count >= numWordPB) {
            wordNumPB[i] = numWordPB;
        } else if (count > 0) {
            wordNumPB[i] = count;
        } else {
            wordNumPB[i] = 0;
        }
        count -= wordNumPB[i];
    }
    assert(0 == count);


#if DEBUG_LEVEL == DEBUG_ALL
    //print the number of real working block
    unsigned int tmp = 0;
    for(unsigned int i = 0; i < _numBlock; i += 1) {
        if(wordNumPB[i] > 0) {
            tmp += 1;
        }
    }
    cout << "*** real working #block: " << tmp << " ***" << endl;
    
    //check the results
    tmp = 0;
    for (unsigned int i = 0; i < _numBlock; i += 1) {
        tmp += wordNumPB[i];
    }
    assert(tmp == numTotalWord);
#endif

    //check whether there are conflict
    //actually no need to check, conflict will be handled by the atomic operations
    /*if (!checkConflict(doc, numTotalWord, wordOffsetPB, wordNumPB, _numBlock)) {
        cout << "There are write conflicts! Will handle this issue in the next release, sorry ..." << endl;
        throw std::runtime_error(ERR_MSG("!checkConflict"));
        exit(EXIT_FAILURE);
    }*/
}

unsigned long long ParallelSampling::fingerprint(const model& lda) const {
    unsigned long long zz = 0;
    const unsigned int M = lda.M;

    unsigned int count = 0;
    for (int i = 0; i < M; i += 1) {
        const int length = ((lda.ptrndata)->docs)[i]->length;
        for (int j = 0; j < length; j += 1) {
            zz += ((lda.z)[i][j] + count);
            count += 1;
        }
    }

    return zz;
}


//-----------------------------------------------------------------------------
//CUDASampling

CUDASampling::CUDASampling(const model& lda, const unsigned int numBlock, const int device) :
ParallelSampling(lda, numBlock) {

    cout << "CUDASampling initialization ...." << endl;
    cout << "\tM: " << lda.M << endl;
    cout << "\tV: " << lda.V << endl;
    cout << "\tK: " << lda.K << endl;
    cout << "\tnumBlock: " << numBlock << endl;
    assert(numBlock == _numBlock);

    //check parameters, calculate _numThread, _sharedMemSize
    _numThread = this->computeNumThread(lda);
    _sharedMemSize = max(sizeof (double) *2 * _numThread, sizeof (double) *(lda.K));
    cout << "\tnumThread: " << _numThread << endl;
    cout << "\tsharedMemSize: " << _sharedMemSize << endl;


    //

    checkCudaErrors(cudaSetDevice(device));

    const unsigned int M = lda.M;
    const unsigned int V = lda.V;
    const unsigned int K = lda.K;

    //memory allocation
    checkCudaErrors(cudaMalloc(&_d_nw, sizeof (int) *V * K));
    checkCudaErrors(cudaMalloc(&_d_nd, sizeof (int) *M * K));
    checkCudaErrors(cudaMalloc(&_d_nwsum, sizeof (int) *K));
    checkCudaErrors(cudaMalloc(&_d_ndsum, sizeof (int) *M));

    unsigned int numTotalWord = 0;
    for (int m = 0; m < M; m += 1) {
        numTotalWord += ((lda.ptrndata)->docs)[m]->length;
    }
    _numTotalWord = numTotalWord;
    checkCudaErrors(cudaMalloc(&_d_word, sizeof (int) *numTotalWord));
    checkCudaErrors(cudaMalloc(&_d_z, sizeof (int) *numTotalWord));
    checkCudaErrors(cudaMalloc(&_d_doc, sizeof (int) *numTotalWord));
    checkCudaErrors(cudaMalloc(&_d_random, sizeof (float) *numTotalWord));

    cout << "\tnumTotalWord: " << numTotalWord << endl;

    checkCudaErrors(cudaMalloc(&_d_wordOffsetPB, sizeof (unsigned int) *numBlock));
    checkCudaErrors(cudaMalloc(&_d_wordNumPB, sizeof (unsigned int) *numBlock));
    checkCudaErrors(cudaMalloc(&_d_new, sizeof (int) *numBlock));
    checkCudaErrors(cudaMalloc(&_d_wordProcessedPB, sizeof (unsigned int) *numBlock));



    //initialize the GPU data
    //checkCudaErrors(cudaMemcpy(_d_nw, lda.nw, sizeof (int) *V*K, cudaMemcpyHostToDevice));
    for (int i = 0; i < V; i += 1) {
        checkCudaErrors(cudaMemcpy(_d_nw + i*K, (lda.nw)[i], sizeof (int) *K, cudaMemcpyHostToDevice));
    }
    //checkCudaErrors(cudaMemcpy(_d_nd, lda.nd, sizeof (int) *M*K, cudaMemcpyHostToDevice));
    for (int i = 0; i < M; i += 1) {
        checkCudaErrors(cudaMemcpy(_d_nd + i*K, (lda.nd)[i], sizeof (int) *K, cudaMemcpyHostToDevice));
    }
    checkCudaErrors(cudaMemcpy(_d_nwsum, lda.nwsum, sizeof (int) *K, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_d_ndsum, lda.ndsum, sizeof (int) *M, cudaMemcpyHostToDevice));

    unsigned int count = 0;
    int* h_doc = new int[numTotalWord];
    for (int m = 0; m < M; m += 1) {
        const int length = ((lda.ptrndata)->docs)[m]->length;

        checkCudaErrors(cudaMemcpy(_d_word + count, ((lda.ptrndata->docs)[m])->words, sizeof (int) *length, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(_d_z + count, (lda.z)[m], sizeof (int) *length, cudaMemcpyHostToDevice));

        for (int i = count; i < count + length; i += 1) {
            h_doc[i] = m;
        }

        count += length;
    }
    assert(count == numTotalWord);
    checkCudaErrors(cudaMemcpy(_d_doc, h_doc, sizeof (int) *numTotalWord, cudaMemcpyHostToDevice));


    //arrange the offset and length for each block
    unsigned int* h_wordOffsetPB = new unsigned int[numBlock];
    unsigned int* h_wordNumPB = new unsigned int[numBlock];
    computeWordPB(h_wordOffsetPB, h_wordNumPB, numTotalWord, h_doc);

    checkCudaErrors(cudaMemcpy(_d_wordOffsetPB, h_wordOffsetPB, sizeof (unsigned int) *numBlock, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(_d_wordNumPB, h_wordNumPB, sizeof (unsigned int) *numBlock, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(_d_wordProcessedPB, 0, sizeof (unsigned int) *numBlock));


    _numKernelLaunch = max(h_wordNumPB, numBlock);

    delete[] h_wordOffsetPB;
    delete[] h_wordNumPB;
    delete[] h_doc;
}

CUDASampling::~CUDASampling() {
    checkCudaErrors(cudaFree(_d_nw));
    checkCudaErrors(cudaFree(_d_nd));
    checkCudaErrors(cudaFree(_d_nwsum));
    checkCudaErrors(cudaFree(_d_ndsum));
    checkCudaErrors(cudaFree(_d_word));
    checkCudaErrors(cudaFree(_d_z));
    checkCudaErrors(cudaFree(_d_doc));
    checkCudaErrors(cudaFree(_d_random));
    checkCudaErrors(cudaFree(_d_new));
    checkCudaErrors(cudaFree(_d_wordProcessedPB));
}

void CUDASampling::run() {
    cout << "numKernelLaunch: " << _numKernelLaunch << endl;

    //generate random numbers
    cuda_random(_d_random, _numTotalWord);

    //reset d_wordProcessedPB
    checkCudaErrors(cudaMemset(_d_wordProcessedPB, 0, sizeof (unsigned int) *_numBlock));


    //process word by word for each block
    for (unsigned int i = 0; i < _numKernelLaunch; i += 1) {

        //computation kernel
        cuda_sampling(_d_new,
                _d_z, _d_nw, _d_nd, _d_nwsum, _d_ndsum,
                _d_word, _d_doc,
                _d_wordOffsetPB, _d_wordNumPB,
                _d_wordProcessedPB,
                _lda.K,
                _lda.alpha, _lda.beta,
                (_lda.K)*(_lda.alpha), (_lda.V)*(_lda.beta),
                _d_random,
                _numBlock, _numThread, _sharedMemSize);

        //result write kernel
        cuda_update(_d_z, _d_nw, _d_nd, _d_nwsum, _d_ndsum,
                _d_new,
                _d_word, _d_doc,
                _d_wordOffsetPB, _d_wordNumPB,
                _d_wordProcessedPB,
                _lda.K,
                _numBlock, _numThread);
    }
}

void CUDASampling::copyBack() {

    const model &lda = _lda;

    const unsigned int M = lda.M;
    const unsigned int V = lda.V;
    const unsigned int K = lda.K;

    for (int i = 0; i < V; i += 1) {
        checkCudaErrors(cudaMemcpy((lda.nw)[i], _d_nw + i*K, sizeof (int) *K, cudaMemcpyDeviceToHost));
    }
    for (int i = 0; i < M; i += 1) {
        checkCudaErrors(cudaMemcpy((lda.nd)[i], _d_nd + i*K, sizeof (int) *K, cudaMemcpyDeviceToHost));
    }
    checkCudaErrors(cudaMemcpy(lda.nwsum, _d_nwsum, sizeof (int) *K, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(lda.ndsum, _d_ndsum, sizeof (int) *M, cudaMemcpyDeviceToHost));



    unsigned int count = 0;
    for (int m = 0; m < M; m += 1) {
        const int length = ((lda.ptrndata)->docs)[m]->length;
        checkCudaErrors(cudaMemcpy((lda.z)[m], _d_z + count, sizeof (int) *length, cudaMemcpyDeviceToHost));

        count += length;
    }


#if DEBUG_LEVEL == DEBUG_ALL
    cout << "===> fingerprint of z: " << fingerprint(_lda) << endl;
#endif
}

unsigned int CUDASampling::computeNumThread(const model& lda) const {
    const int validK[] = {128, 256, 512, 1024, 2048};
    const unsigned int n = sizeof (validK) / sizeof (int);

    bool found = false;
    for (unsigned int i = 0; i < n; i += 1) {
        if (lda.K == validK[i]) {
            found = true;
            break;
        }
    }
    if (!found) {
        cout << "lda.K: " << lda.K << endl;
        throw std::runtime_error(ERR_MSG("invalid K"));
        exit(EXIT_FAILURE);
    }

    const unsigned int numThread = (lda.K)/4;
    
    return numThread;
}


/////////////////////////////////////////////////////////////////////////////
//GoldSampling

GoldSampling::GoldSampling(const model& lda, const unsigned int numBlock) :
ParallelSampling(lda, numBlock) {

    const unsigned int M = lda.M;
    const unsigned int V = lda.V;
    const unsigned int K = lda.K;

    //memory allocation
    _nw = new int[V * K];
    _nd = new int[M * K];
    _nwsum = new int[K];
    _ndsum = new int[M];

    unsigned int numTotalWord = 0;
    for (int m = 0; m < M; m += 1) {
        numTotalWord += ((lda.ptrndata)->docs)[m]->length;
    }
    cout << "numTotalWord: " << numTotalWord << endl;
    _word = new int[numTotalWord];
    _z = new int[numTotalWord];
    _doc = new int[numTotalWord];
    _random = new float[numTotalWord];

    _wordOffsetPB = new unsigned int[numBlock];
    _wordNumPB = new unsigned int[numBlock];
    _new = new unsigned int[numBlock];
    _wordProcessedPB = new unsigned int[numBlock];



    //memory initialization
    for (int i = 0; i < V; i += 1) {
        for (int j = 0; j < K; j += 1) {
            _nw[i * K + j] = (lda.nw)[i][j];
        }
    }
    for (int i = 0; i < M; i += 1) {
        for (int j = 0; j < K; j += 1) {
            _nd[i * K + j] = (lda.nd)[i][j];
        }
    }
    memcpy(_nwsum, lda.nwsum, sizeof (int) *K);
    memcpy(_ndsum, lda.ndsum, sizeof (int) *M);


    int count = 0;
    for (int i = 0; i < M; i += 1) {
        const int length = ((lda.ptrndata)->docs)[i]->length;
        for (int j = 0; j < length; j += 1) {
            _word[count] = (((lda.ptrndata)->docs)[i])->words[j];
            _z[count] = (lda.z)[i][j];
            _doc[count] = i;

            count += 1;
        }
    }
    computeWordPB(_wordOffsetPB, _wordNumPB, numTotalWord, _doc);
    memset(_wordProcessedPB, 0, sizeof (unsigned int) *_numBlock);

}

GoldSampling::~GoldSampling() {
    delete[] _nw;
    delete[] _nd;
    delete[] _nwsum;
    delete[] _ndsum;

    delete[] _word;
    delete[] _z;
    delete[] _doc;
    delete[] _random;

    delete[] _wordOffsetPB;
    delete[] _wordNumPB;
    delete[] _wordProcessedPB;
    delete[] _new;
}

void GoldSampling::run() {
    const unsigned int numPass = max(_wordNumPB, _numBlock);

    cout << "numBlock: " << _numBlock << endl;
    cout << "numPass: " << numPass << endl;

    const double Vbeta = (_lda.V)*(_lda.beta);
    const double Kalpha = (_lda.K)*(_lda.alpha);
    double* p = new double[_lda.K];
    const int K = _lda.K;


    //generate random numbers [0, 1]
    const unsigned int numTotalWord = sum(_wordNumPB, _numBlock);
    for (unsigned int i = 0; i < numTotalWord; i += 1) {
        _random[i] = random() / static_cast<double> (RAND_MAX);
    }


    memset(_wordProcessedPB, 0, sizeof (unsigned int) *_numBlock);

    for (unsigned int pass = 0; pass < numPass; pass += 1) {
        //computation
        for (unsigned int blockId = 0; blockId < _numBlock; blockId += 1) {
            if (_wordProcessedPB[blockId] >= _wordNumPB[blockId]) { //all finished
                continue;
            }

            const unsigned int offset = _wordOffsetPB[blockId] + _wordProcessedPB[blockId];
            int topic = _z[offset];
            int w = _word[offset];
            int m = _doc[offset];
            const int ndsum = _ndsum[m] - 1;
            //multinomial sampling
            for (int k = 0; k < K; k += 1) {
                int nw = _nw[w * K + k];
                int nd = _nd[m * K + k];
                int nwsum = _nwsum[k];
                if (k == topic) {
                    nw--;
                    nd--;
                    nwsum--;
                }
                p[k] = (nw + _lda.beta) / (nwsum + Vbeta) *
                        (nd + _lda.alpha) / (ndsum + Kalpha);
            }
            //cumulate multinomial parameters
            for (int k = 1; k < K; k += 1) {
                p[k] += p[k - 1];
            }
            //scale
            double u = _random[offset] * p[K - 1];
            for (topic = 0; topic < K; topic += 1) {
                if (p[topic] >= u) {
                    break;
                }
            }
            _new[blockId] = topic;
        }

        //update
        for (unsigned int blockId = 0; blockId < _numBlock; blockId += 1) {
            if (_wordProcessedPB[blockId] >= _wordNumPB[blockId]) { //all finished
                continue;
            }

            const unsigned int offset = _wordOffsetPB[blockId] + _wordProcessedPB[blockId];
            int old_topic = _z[offset];
            int new_topic = _new[blockId];
            int w = _word[offset];
            int m = _doc[offset];

            _nw[w * K + old_topic] -= 1;
            _nd[m * K + old_topic] -= 1;
            _nwsum[old_topic] -= 1;
            _ndsum[m] -= 1;

            _nw[w * K + new_topic] += 1;
            _nd[m * K + new_topic] += 1;
            _nwsum[new_topic] += 1;
            _ndsum[m] += 1;

            _z[offset] = new_topic;

            _wordProcessedPB[blockId] += 1;
        }
    }

    delete[] p;
}

void GoldSampling::copyBack() {

    const model &lda = _lda;

    const unsigned int M = lda.M;
    const unsigned int V = lda.V;
    const unsigned int K = lda.K;

    for (int i = 0; i < V; i += 1) {
        for (int j = 0; j < K; j += 1) {
            (lda.nw)[i][j] = _nw[i * K + j];
        }
    }
    for (int i = 0; i < M; i += 1) {
        for (int j = 0; j < K; j += 1) {
            (lda.nd)[i][j] = _nd[i * K + j];
        }
    }
    memcpy(lda.nwsum, _nwsum, sizeof (int) *K);
    memcpy(lda.ndsum, _ndsum, sizeof (int) *M);


    int count = 0;
    for (int i = 0; i < M; i += 1) {
        const int length = ((lda.ptrndata)->docs)[i]->length;
        for (int j = 0; j < length; j += 1) {
            (lda.z)[i][j] = _z[count];

            count += 1;
        }
    }


#if DEBUG_LEVEL == DEBUG_ALL
    cout << "===> fingerprint of z: " << fingerprint(_lda) << endl;
#endif
}




