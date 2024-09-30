#ifdef USING_NEON

#ifndef ITENSOR_NEON_H
#define ITENSOR_NEON_H

#include <stdlib.h>
#include <stdio.h>
#include <arm_neon.h>
#include "iArray_neon.h"

void rebaseIntrinsics(iTensor *A, float new_scale, i8 new_zero_point) {
    //return a new iTensor with data from A rebased to the new scale and zero point

    iTensor *B = (iTensor *)malloc(sizeof(iTensor));
    int32x4_t zero_point = vdupq_n_s32(new_zero_point);
    float32x4_t scale = vdupq_n_f32(new_scale);
    float32x4_t A_scale = vdupq_n_f32(A->scale);
    int32x4_t A_zero_point = vdupq_n_s32(A->zero_point);
    int32x4_t diff_zero_point = vsubq_s32(zero_point, A_zero_point);
    float32x4_t diff_scale = vdivq_f32(scale, A_scale);
    int8x16_t zero = vdupq_n_s8(0);

    iArray *arr = A->arr;
    iArray *out = create(arr->shape, arr->rank);
    size_t i = 0;
    
    #pragma omp parallel for 
    for(; i < arr->size; i+=16){
        int8x16_t a = vld1q_s8(&arr->data[i]);
        int16x8_t a_low = vsubl_s8(vget_low_s8(a), vget_low_s8(A_zero_point));
        int16x8_t a_high = vsubl_s8(vget_high_s8(a), vget_high_s8(A_zero_point));
        int16x8_t res_low = vqdmulhq_s16(a_low, vget_low_s16(diff_scale));
        int16x8_t res_high = vqdmulhq_s16(a_high, vget_high_s16(diff_scale));
        int16x8_t res_low_shifted = vqshrn_n_s16(res_low, 7);
        int16x8_t res_high_shifted = vqshrn_n_s16(res_high, 7);
        int8x16_t res = vcombine_s8(vqmovn_s16(res_low_shifted), vqmovn_s16(res_high_shifted));
        res = vqaddq_s8(res, zero);
        vst1q_s8(&out->data[i], res);
    }
    
    for(; i < arr->size; i++){
        out->data[i] = (i8)round((arr->data[i] - A->zero_point) * A->scale / new_scale) + new_zero_point;
    }
    B->scale = new_scale;
    B->zero_point = new_zero_point;
    B->arr = out;
    return B;
}


iTensor *addTensors(iTensor *A, iTensor *B){
    if (A->arr->size != B->arr->size){
        fprintf(stderr, "iTensors must have the same size\n");
        exit(1);
    }
    
    iTensor *new_tensor;
    //First identify which base is larger
    if (A->scale > B->scale){
        iArray *x = A->arr;
        iTensor *new_tensor = rebaseIntrinsics(A, B->scale, B->zero_point);
        iArray *y = new_tensor->arr;
    } else {
        iArray *x = B->arr;
        iTensor *new_tensor = rebaseIntrinsics(B, A->scale, A->zero_point);
        iArray *y = new_tensor->arr;      
    } 

    iArray *out = add(x, y);
    return new_tensor;
}

#endif
#endif