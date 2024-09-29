#include <stdlib.h>
#include <stdio.h>
#include "int8_ops.h"

#if defined(__AVX512F__)
    #define USING_INTRINSICS
#elif defined(__AVX2__)
    #define USING_INTRINSICS
#elif defined(__ARM_NEON)
    #define USING_INTRINSICS
#endif

Array *create(size_t *shape, size_t rank){
    Array *arr = (Array *)malloc(sizeof(Array));
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
    arr->data = (int8_t *)malloc(arr->size * sizeof(int8_t));
    //do an aligned alloc to 32 bytes
    // posix_memalign((void **)&arr->data, 32, arr->size * sizeof(int8_t));
    return arr;
}

Array *from_random(size_t *shape, size_t rank, int8_t min, int8_t max){
    Array *arr = create(shape, rank);
    for(size_t i = 0; i < arr->size; i++){
        arr->data[i] = (int8_t)(rand() % (max - min + 1) + min);
    }
    return arr;
}

Array *from_data(int8_t *data, size_t *shape, size_t rank){
    Array *arr = create(shape, rank);
    for(size_t i = 0; i < arr->size; i++){
    arr->data[i] = data[i];
    }
    return arr;
}

Array *from_zero(size_t *shape, size_t rank){
    Array *arr = create(shape, rank);
    for(size_t i = 0; i < arr->size; i++){
    arr->data[i] = 0;
    }
    return arr;
}

Array *copy(Array *arr){
    Array *new_arr = create(arr->shape, arr->rank);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] = arr->data[i];
    }
    return new_arr;
}

Array *divScalar(Array *arr, int8_t scalar){
    Array *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    if (scalar == 0){
        fprintf(stderr, "Division by zero\n");
        free_array(new_arr);
        exit(1);
    }
    new_arr->data[i] /= scalar;
    }
    return new_arr;
}

Array *truediv(Array *arr1, Array *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "Arrays must have the same size\n");
    exit(1);
    }
    Array *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    if (arr2->data[i] == 0){
        fprintf(stderr, "Division by zero\n");
        free_array(new_arr);
        exit(1);
    }
    new_arr->data[i] /= arr2->data[i];
    }
    return new_arr;
}

Array *modScalar(Array *arr, int8_t scalar){
    Array *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    if (scalar == 0){
        fprintf(stderr, "Division by zero\n");
        free_array(new_arr);
        exit(1);
    }
    new_arr->data[i] %= scalar;
    }
    return new_arr;
}

Array *mod(Array *arr1, Array *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "Arrays must have the same size\n");
    exit(1);
    }
    Array *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    if (arr2->data[i] == 0){
        fprintf(stderr, "Division by zero\n");
        free_array(new_arr);
        exit(1);
    }
    new_arr->data[i] %= arr2->data[i];
    }
    return new_arr;
}

void free_array(Array *arr){
    free(arr->data);
    free(arr);
}

void print_array(Array *arr){
    for(size_t i = 0; i < arr->size; i++){
    printf("%d ", arr->data[i]);
    }
    printf("\n");
}

#ifndef USING_INTRINSICS

Array *addScalar(Array *arr, int8_t scalar){
    Array *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] += scalar;
    }
    return new_arr;
}

Array *subScalar(Array *arr, int8_t scalar){
    Array *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] -= scalar;
    }
    return new_arr;
}

Array *mulScalar(Array *arr, int8_t scalar){
    Array *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] *= scalar;
    }
    return new_arr;
}

Array *add(Array *arr1, Array *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "Arrays must have the same size\n");
    exit(1);
    }
    Array *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    new_arr->data[i] += arr2->data[i];
    }
    return new_arr;
}

Array *sub(Array *arr1, Array *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "Arrays must have the same size\n");
    exit(1);
    }
    Array *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    new_arr->data[i] -= arr2->data[i];
    }
    return new_arr;
}

Array *mul(Array *arr1, Array *arr2){
    if (arr1->size != arr2->size){
    fprintf(stderr, "Arrays must have the same size\n");
    exit(1);
    }
    Array *new_arr = copy(arr1);
    for(size_t i = 0; i < arr1->size; i++){
    new_arr->data[i] *= arr2->data[i];
    }
    return new_arr;
}

Array *neg(Array *arr){
    Array *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] = -new_arr->data[i];
    }
  return new_arr;
}

Array *absArray(Array *arr){
    Array *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] = abs(new_arr->data[i]);
    }
    return new_arr;
}

#endif



int main(){
    size_t shape[2] = {2, 2};
    Array *arr1 = from_random(shape, 2, 0, 10);
    print_array(arr1);
    Array *arr2 = from_random(shape, 2, -10, 10);
    print_array(arr2);
    Array *result = sub(arr1, arr2);
    print_array(result);
    free_array(arr1);
    free_array(arr2);
    free_array(result);
    return 0;
}