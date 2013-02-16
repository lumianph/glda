/* 
 * File:   common.h
 * Author: mianlu
 *
 * Created on February 6, 2013, 2:58 PM
 */

#ifndef COMMON_H
#define	COMMON_H


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace std;

#define DEBUG_NONE      (0)
#define DEBUG_ALL       (9)
#define DEBUG_LEVEL     (DEBUG_ALL)

template<class T>
string toString(const T& data) {
    string result;
    ostringstream converter;
    converter << data;
    result = converter.str();

    return result;
}

#define ERR_MSG(msg) ("!!!ERROR(" + toString(__FILE__) + ", " + toString(__LINE__) + "): " + msg)

#endif	/* COMMON_H */

