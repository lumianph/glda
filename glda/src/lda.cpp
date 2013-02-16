/*
 * Copyright (C) 2013 by
 * 
 * 	Mian Lu
 *	lumianph@gmail.com
 *
 * GLDA is a free software. It is modified based on GibbsLDA++ (http://gibbslda.sourceforge.net/)
 * GLDA accelerates GibbsLDA++ with GPU acceleration
 */

#include "model.h"

#include <cstdio>
#include <cstdlib>

using namespace std;

void show_help();

int main(int argc, char ** argv) {


    srandom(0); // initialize for random number generation


    model lda;

    if (lda.init(argc, argv)) {
        show_help();
        return 1;
    }

    if (lda.model_status == MODEL_STATUS_EST || lda.model_status == MODEL_STATUS_ESTC) {
        // parameter estimation
        //lda.estimate();
        lda.cuda_estimate();
    }

    if (lda.model_status == MODEL_STATUS_INF) {
        // do inference
        lda.inference();
    }

    return 0;
}

void show_help() {
    printf("Command line usage:\n");
    printf("\tlda -est -alpha <double> -beta <double> -ntopics <int> -niters <int> -savestep <int> -twords <int> -dfile <string>\n");
    printf("\tlda -estc -dir <string> -model <string> -niters <int> -savestep <int> -twords <int>\n");
    printf("\tlda -inf -dir <string> -model <string> -niters <int> -twords <int> -dfile <string>\n");
    // printf("\tlda -inf -dir <string> -model <string> -niters <int> -twords <int> -dfile <string> -withrawdata\n");
}

