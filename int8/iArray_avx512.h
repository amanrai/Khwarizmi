//All methods are to be implemented using AVX512 instructions

#ifdef USING_AVX512

#ifndef IARRAY_AVX512_H
#define IARRAY_AVX512_H

#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include "int8_ops.h"

iArray *add(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
        fprintf(stderr, "iArrays must have the same size\n");
        exit(1);
    }
    iArray *new_arr = copy(arr1);
    size_t i = 0;
    for(; i < arr1->size; i+=64){
        __m512i a = _mm512_loadu_si512((__m512i *)&arr1->data[i]);
        __m512i b = _mm512_loadu_si512((__m512i *)&arr2->data[i]);
        __m512i c = _mm512_add_epi8(a, b);
        _mm512_storeu_si512((__m512i *)&new_arr->data[i], c);
    }
    for(; i < arr1->size; i++){
        new_arr->data[i] = arr1->data[i] + arr2->data[i];
    }
    return new_arr;
}

iArray *addScalar(iArray *arr, int8_t scalar){
    
    iArray *new_arr = copy(arr);
    size_t i = 0;
    for(; i < arr->size; i+=64){
        __m512i a = _mm512_loadu_si512((__m512i *)&arr->data[i]);
        __m512i b = _mm512_set1_epi8(scalar);
        __m512i c = _mm512_add_epi8(a, b);
        _mm512_storeu_si512((__m512i *)&new_arr->data[i], c);
    }
    for(; i < arr->size; i++){
        new_arr->data[i] = arr->data[i] + scalar;
    }
    return new_arr;
}

iArray *sub(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
        fprintf(stderr, "iArrays must have the same size\n");
        exit(1);
    }
    iArray *new_arr = copy(arr1);
    size_t i = 0;
    for(; i < arr1->size; i+=64){
        __m512i a = _mm512_loadu_si512((__m512i *)&arr1->data[i]);
        __m512i b = _mm512_loadu_si512((__m512i *)&arr2->data[i]);
        __m512i c = _mm512_sub_epi8(a, b);
        _mm512_storeu_si512((__m512i *)&new_arr->data[i], c);
    }
    for(; i < arr1->size; i++){
        new_arr->data[i] = arr1->data[i] - arr2->data[i];
    }
    return new_arr;
}

iArray *subScalar(iArray *arr, int8_t scalar){
    iArray *new_arr = copy(arr);
    size_t i = 0;
    for(; i < arr->size; i+=64){
        __m512i a = _mm512_loadu_si512((__m512i *)&arr->data[i]);
        __m512i b = _mm512_set1_epi8(scalar);
        __m512i c = _mm512_sub_epi8(a, b);
        _mm512_storeu_si512((__m512i *)&new_arr->data[i], c);
    }
    for(; i < arr->size; i++){
        new_arr->data[i] = arr->data[i] - scalar;
    }
    return new_arr;
}

iArray *mul(iArray *arr1, iArray *arr2) {
    if (arr1->size != arr2->size) {
        fprintf(stderr, "iArrays must have the same size\n");
        exit(1);
    }
    iArray *new_arr = copy(arr1);
    size_t i = 0;
    for (; i + 64 <= arr1->size; i += 64) {
        __m512i a = _mm512_loadu_si512((__m512i *)&arr1->data[i]);
        __m512i b = _mm512_loadu_si512((__m512i *)&arr2->data[i]);
        __m512i c = _mm512_mullo_epi16(a, b);
        c = _mm512_and_si512(c, _mm512_set1_epi16(0x00FF));
        _mm512_storeu_si512((__m512i *)&new_arr->data[i], c);
    }
    for (; i < arr1->size; i++) {
        new_arr->data[i] = arr1->data[i] * arr2->data[i];
    }
    return new_arr;
}

iArray *mulScalar(iArray *arr, int8_t scalar) {
    iArray *new_arr = copy(arr);
    size_t i = 0;
    for (; i + 64 <= arr->size; i += 64) {
        __m512i a = _mm512_loadu_si512((__m512i *)&arr->data[i]);
        __m512i b = _mm512_set1_epi8(scalar);
        __m512i c = _mm512_mullo_epi16(a, b);
        c = _mm512_and_si512(c, _mm512_set1_epi16(0x00FF));
        _mm512_storeu_si512((__m512i *)&new_arr->data[i], c);
    }
    for (; i < arr->size; i++) {
        new_arr->data[i] = arr->data[i] * scalar;
    }
    return new_arr;
}

#endif
#endif