/* 
 * File:   CUDASampling.h
 * Author: mianlu
 *
 * Created on February 5, 2013, 5:42 PM
 */

#ifndef CUDASAMPLING_H
#define	CUDASAMPLING_H


#include "common.h"
#include "model.h"

class ParallelSampling {
protected:
    const model &_lda;
    const unsigned int _numBlock;

public:

    ParallelSampling(const model &lda, const unsigned int numBlock) :
    _lda(lda), _numBlock(numBlock) {
    }

    virtual ~ParallelSampling() {
    }


    virtual void run() = 0;
    virtual void copyBack() = 0;

protected:
    bool checkConflict(const int* doc, const unsigned int numTotalWord,
            const unsigned int* wordOffsetPB, const unsigned int* wordNumPB,
            const unsigned int numBlock) const;

    void computeWordPB(unsigned int* h_wordOffsetPB, unsigned int *wordNumPB, const unsigned int numTotalWord, const int* doc);

    unsigned long long fingerprint(const model &lda) const;
};

class CUDASampling : public ParallelSampling {
    friend class model;

private:

    int* _d_nw; //size V x K
    int* _d_nd; //size M x K
    int* _d_nwsum; //size K
    int* _d_ndsum; //size M

    int* _d_word; //size M x doc.length
    int* _d_z; //size M x doc.length
    int* _d_doc; //size M x doc.length, document id for each word
    float* _d_random; 

    int* _d_new; //size numBlock
    unsigned int* _d_wordOffsetPB; //size numBlock
    unsigned int* _d_wordNumPB; //size numBlock, number of words processed per block
    unsigned int* _d_wordProcessedPB; //size numBlock

    unsigned int _numThread;
    unsigned int _sharedMemSize;
    
    unsigned int _numKernelLaunch; //equal to the max number of words for blocks
    unsigned int _numTotalWord;

public:
    CUDASampling(const model &lda, const unsigned int numBlock, const int device = 0);
    virtual ~CUDASampling();

    void run(); //execute one sampling

    void copyBack();

private:
    CUDASampling(const CUDASampling& orig);
    CUDASampling& operator=(const CUDASampling&);

    unsigned int computeNumThread(const model &lda) const;
};

class GoldSampling : public ParallelSampling {
    friend class model;

private:

    int* _nw; //size V x K
    int* _nd; //size M x K
    int* _nwsum; //size K
    int* _ndsum; //size M

    int* _word; //size M x doc.length
    int* _z; //size M x doc.length
    int* _doc; //size M x doc.length, document id for each word
    float* _random;

    unsigned int* _wordOffsetPB; //size numBlock
    unsigned int* _wordNumPB; //size numBlock, number of words processed per block
    unsigned int* _wordProcessedPB; //size numBlock
    unsigned int* _new;
    
public:
    GoldSampling(const model &lda, const unsigned int numBlock);
    virtual ~GoldSampling();

    void run();

    void copyBack();

private:

};


#endif	/* CUDASAMPLING_H */

