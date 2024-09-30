#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#if defined(__AVX512F__)
    #define USING_INTRINSICS
    #define USING_AVX512
    #include "iArray_avx512.h"
#elif defined(__AVX2__)
    #define USING_INTRINSICS
#elif defined(__ARM_NEON)
    #define USING_INTRINSICS
    #include "iArray_neon.h"
    #include "iTensor_neon.h"
    #include "int8_ops.h"
#else 
    #include "int8_ops.h"
#endif

i8 roundi8(float x) {
    int y = (int)round(x);
    if (y > iMAX) {
        return (i8)iMAX;
    } else if (y < iMIN) {
        return (i8)iMIN;
    } else {
        return (i8)y;
    }
}

iArray *create(size_t *shape, size_t rank){
    iArray *arr = (iArray *)malloc(sizeof(iArray));
    arr->shape = shape;
    arr->rank = rank;
    arr->size = 1;
    for(size_t i = 0; i < rank; i++){
    if (shape[i] == 0){
        fprintf(stderr, "Shape cannot be zero\n");
        free(arr);
        free(arr->shape);
        exit(1);
    }
    arr->size *= shape[i];
    }
    // arr->data = (i8 *)malloc(arr->size * sizeof(i8));
    int result = posix_memalign((void **)&arr->data, 32, arr->size * sizeof(i8));

    if (result != 0){
        fprintf(stderr, "Allocation of aligned memory failed.\n");
        free(arr);
        free(arr->shape);
        exit(1);
    }
    return arr;
}

iArray *from_random(size_t *shape, size_t rank, i8 min, i8 max){
    iArray *arr = create(shape, rank);
    for(size_t i = 0; i < arr->size; i++){
        arr->data[i] = (i8)(rand() % (max - min + 1) + min);
    }
    return arr;
}

iArray *from_data(i8 *data, size_t *shape, size_t rank){
    iArray *arr = create(shape, rank);
    for(size_t i = 0; i < arr->size; i++){
        arr->data[i] = data[i];
    }
    return arr;
}

iArray *from_zero(size_t *shape, size_t rank){
    iArray *arr = create(shape, rank);
    for(size_t i = 0; i < arr->size; i++){
        arr->data[i] = 0;
    }
    return arr;
}

iArray *copy(iArray *arr){
    iArray *new_arr = create(arr->shape, arr->rank);
    for(size_t i = 0; i < arr->size; i++){
        new_arr->data[i] = arr->data[i];
    }
    return new_arr;
}

iArray *divScalar(iArray *arr, i8 scalar){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    if (scalar == 0){
        fprintf(stderr, "Division by zero\n");
        free_iArray(new_arr);
        exit(1);
    }
    new_arr->data[i] /= scalar;
    }
    return new_arr;
}

iArray *truediv(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "iArrays must have the same size\n");
    exit(1);
    }
    iArray *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    if (arr2->data[i] == 0){
        fprintf(stderr, "Division by zero\n");
        free_iArray(new_arr);
        exit(1);
    }
    new_arr->data[i] /= arr2->data[i];
    }
    return new_arr;
}

iArray *modScalar(iArray *arr, i8 scalar){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    if (scalar == 0){
        fprintf(stderr, "Division by zero\n");
        free_iArray(new_arr);
        exit(1);
    }
    new_arr->data[i] %= scalar;
    }
    return new_arr;
}

iArray *mod(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "iArrays must have the same size\n");
    exit(1);
    }
    iArray *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    if (arr2->data[i] == 0){
        fprintf(stderr, "Division by zero\n");
        free_iArray(new_arr);
        exit(1);
    }
    new_arr->data[i] %= arr2->data[i];
    }
    return new_arr;
}

void free_iArray(iArray *arr){
    free(arr->data);
    free(arr);
}

void free_iTensor(iTensor *tensor){
    free_iArray(tensor->arr);
    free(tensor);
}

void printiArray(iArray *arr) {

    if (arr == NULL || arr->data == NULL) {
        printf("Invalid array\n");
        return;
    }
    printf("\niArray: (");
    for (size_t i = 0; i < arr->rank; i++) {
        printf("%ld", arr->shape[i]);
        if (i < arr->rank - 1) {
            printf(", ");
        }
    }
    printf(")\n");
    deep_print(arr, 0, 0);
    printf("\n");
}

float percentile(float *arr, size_t *shape, size_t rank, float percentile){
    size_t size = 1;
    for (size_t i = 0; i < rank; i++) {
        size *= shape[i];
    }
    
    float *new_arr = (float *)malloc(size * sizeof(float));

    for(size_t i = 0; i < size; i++){
        new_arr[i] = arr[i];
    }

    if (size <= 0 || percentile < 0 || percentile > 100) {
        fprintf(stderr, "Invalid input parameters\n");
        return NAN;
    }
    #pragma omp parallel for
    for (size_t i = 0; i < size - 1; i++) {
        for (size_t j = 0; j < size - i - 1; j++) {
            if (new_arr[j] > new_arr[j + 1]) {
                float temp = new_arr[j];
                new_arr[j] = new_arr[j + 1];
                new_arr[j + 1] = temp;
            }
        }
    }

    // Calculate the index
    float index = (percentile / 100) * (size - 1);
    int lower_index = (int)floor(index);
    int upper_index = (int)ceil(index);

    if (lower_index == upper_index) {
        free(new_arr);
        return new_arr[lower_index];
    } else {
        float lower_value = new_arr[lower_index];
        float upper_value = new_arr[upper_index];
        float fraction = index - lower_index;
        free(new_arr);
        return lower_value + fraction * (upper_value - lower_value);
    }
}

float clamp(i8 value, i8 min, i8 max) {
    printf("\nClamping value: %d, min: %d, max: %d\n", value, min, max);
    return fmin(fmax(value, min), max);
}

i8 clamp_int16(int16_t value) {
    return (i8)fmin(fmax(value, -127), 127);
}

iTensor *quantize_asymmetric_minmax(float *data, size_t *shape, size_t rank, float min, float max) {
    // https://www.youtube.com/watch?v=0VdNflU08yA - Asymmetric Quantization.
    iTensor *tensor = (iTensor *)malloc(sizeof(iTensor));
    tensor->arr = create(shape, rank);
    float range = max - min;
    float scale = range / 255; //2^num_bits - 1
    i8 zero_point = (i8)-1*round(min / scale);
    printf("Quantizing with min: %f, max: %f, range: %f, scale: %f, zero_point: %d\n", min, max, range, scale, zero_point);
    tensor->scale = scale;
    tensor->zero_point = zero_point;
    for(size_t i = 0; i < tensor->arr->size; i++){        
        int16_t q_val = (int16_t) roundi8(data[i] / scale) + zero_point; //if the zero_point is not 0, this has capacity to overflow, so store in an int16 instead.  
        q_val = clamp_int16(q_val);        
        tensor->arr->data[i] = q_val;
    }
    return tensor;
}

iTensor *quantize(float *data, size_t *shape, size_t rank) {
    // https://www.youtube.com/watch?v=0VdNflU08yA - Asymmetric Quantization with Percentiles.
    //Only real requirement is larger values remain larger than smaller values. 
    float q = 95;
    float max = percentile(data, shape, rank, q);
    float min = percentile(data, shape, rank, 100-q);    
    return quantize_asymmetric_minmax(data, shape, rank, min, max);
}

iTensor *quantize_symmetric(float *data, size_t *shape, size_t rank) {
    // https://www.youtube.com/watch?v=0VdNflU08yA - Symmetric Quantization.
    iTensor *tensor = (iTensor *)malloc(sizeof(iTensor));
    tensor->arr = create(shape, rank);
    float max = data[0];
    for(size_t i = 0; i < tensor->arr->size; i++){
        if (data[i] > max){
            max = data[i];
        }
    }
    float scale = max / 127;
    i8 zero_point = 0;
    tensor->scale = scale;
    tensor->zero_point = zero_point;
    for(size_t i = 0; i < tensor->arr->size; i++) {
        tensor->arr->data[i] = (i8)round(data[i] / scale);
    }
    return tensor;
}

float *dequantize(iTensor *tensor){
    //Type of Quantization shouldn't matter because the zero point is 0 for symmetric and min for asymmetric
    printf("Dequantizing with scale: %f, zero_point: %d\n", tensor->scale, tensor->zero_point);
    float *data = (float *)malloc(tensor->arr->size * sizeof(float));
    for(size_t i = 0; i < tensor->arr->size; i++){
        data[i] = (tensor->arr->data[i] - tensor->zero_point) * tensor->scale;
    }
    return data;
}

iTensor *rebase(iTensor *A, iTensor *B) {
    //in place rebase B to A
    #pragma omp parallel for
    for(size_t i = 0; i < A->arr->size; i++){
        B->arr->data[i] = (i8)round((B->arr->data[i] - B->zero_point) * B->scale / A->scale) + A->zero_point;
    }
    B->scale = A->scale;
    B->zero_point = A->zero_point;
}

void deep_print(iArray *arr, int dimension, int start_at) {
    if (dimension == arr->rank - 1) {
        printf("%*s[", dimension*2, "");        
        for (size_t i = 0; i < arr->shape[dimension]; i++) {
            printf("%d", arr->data[start_at*arr->shape[dimension] + i]);
            if (i < arr->shape[dimension] - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    } else {
        printf("%*s[\n", dimension*2, "");
        for (size_t i = 0; i < arr->shape[dimension]; i++) {
            deep_print(arr, dimension + 1, start_at*arr->shape[dimension] + i);
        }
        printf("\n%*s]", dimension*2, "");
    }
}

#ifndef USING_INTRINSICS
//This is the default implementation if no intrinsics are available
iArray *addScalar(iArray *arr, i8 scalar){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] += scalar;
    }
    return new_arr;
}

iArray *subScalar(iArray *arr, i8 scalar){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] -= scalar;
    }
    return new_arr;
}

iArray *mulScalar(iArray *arr, i8 scalar){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] *= scalar;
    }
    return new_arr;
}

iArray *add(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "iArrays must have the same size\n");
    exit(1);
    }
    iArray *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    new_arr->data[i] += arr2->data[i];
    }
    return new_arr;
}

iArray *sub(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "iArrays must have the same size\n");
    exit(1);
    }
    iArray *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    new_arr->data[i] -= arr2->data[i];
    }
    return new_arr;
}

iArray *mul(iArray *arr1, iArray *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "iArrays must have the same size\n");
    exit(1);
    }
    iArray *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    new_arr->data[i] *= arr2->data[i];
    }
    return new_arr;
}

iArray *neg(iArray *arr){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] = -new_arr->data[i];
    }
  return new_arr;
}

iArray *absiArray(iArray *arr){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] = abs(new_arr->data[i]);
    }
    return new_arr;
}

#endif

// int main(){
//     // int rank = 2;
//     // size_t shape[2] = {16,16};
//     // iArray *arr1 = from_random(shape, rank, 0, 10);
//     // printiArray(arr1);
//     // iArray *arr2 = from_random(shape, rank, -10, 10);
//     // printiArray(arr2);
//     // iArray *result = mul(arr1, arr2);
//     // printiArray(result);
//     // free_iArray(arr1);
//     // free_iArray(arr2);
//     // free_iArray(result);
//     return 0;
// }






/*
for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
*/