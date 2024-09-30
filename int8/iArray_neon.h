#ifdef USING_NEON

#ifndef IARRAY_NEON_H
#define IARRAY_NEON_H

#include <stdlib.h>
#include <stdio.h>
#include <arm_neon.h>

#include "int8_ops.h"

iArray *add(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
        fprintf(stderr, "iArrays must have the same size\n");
        exit(1);
    }
    iArray *new_arr = copy(arr1);
    size_t i = 0;
    for(; i < arr1->size; i+=16){
        int8x16_t a = vld1q_s8(&arr1->data[i]);
        int8x16_t b = vld1q_s8(&arr2->data[i]);
        int8x16_t c = vaddq_s8(a, b);
        vst1q_s8(&new_arr->data[i], c);
    }
    for(; i < arr1->size; i++){
        new_arr->data[i] = arr1->data[i] + arr2->data[i];
    }
    return new_arr;
}

#endif
#endif