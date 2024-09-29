#include <stdlib.h>
#include <stdio.h>

#if defined(__AVX512F__)
    #define USING_INTRINSICS
    #define USING_AVX512
    #include "iArray_avx512.h"
#elif defined(__AVX2__)
    #define USING_INTRINSICS
#elif defined(__ARM_NEON)
    #define USING_INTRINSICS
#else 
    #include "int8_ops.h"
#endif

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
    arr->data = (int8_t *)malloc(arr->size * sizeof(int8_t));
    //do an aligned alloc to 32 bytes
    // posix_memalign((void **)&arr->data, 32, arr->size * sizeof(int8_t));
    return arr;
}

iArray *from_random(size_t *shape, size_t rank, int8_t min, int8_t max){
    iArray *arr = create(shape, rank);
    for(size_t i = 0; i < arr->size; i++){
        arr->data[i] = (int8_t)(rand() % (max - min + 1) + min);
    }
    return arr;
}

iArray *from_data(int8_t *data, size_t *shape, size_t rank){
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

iArray *divScalar(iArray *arr, int8_t scalar){
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

iArray *modScalar(iArray *arr, int8_t scalar){
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

void print_iArray(iArray *arr){
    for(size_t i = 0; i < arr->size; i++){
    printf("%d ", arr->data[i]);
    }
    printf("\n");
}


#ifndef USING_INTRINSICS
//This is the default implementation if no intrinsics are available
iArray *addScalar(iArray *arr, int8_t scalar){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] += scalar;
    }
    return new_arr;
}

iArray *subScalar(iArray *arr, int8_t scalar){
    iArray *new_arr = copy(arr);
    for(size_t i = 0; i < arr->size; i++){
    new_arr->data[i] -= scalar;
    }
    return new_arr;
}

iArray *mulScalar(iArray *arr, int8_t scalar){
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

int main(){
    size_t shape[2] = {2, 2};
    iArray *arr1 = from_random(shape, 2, 0, 10);
    print_iArray(arr1);
    iArray *arr2 = from_random(shape, 2, -10, 10);
    print_iArray(arr2);
    iArray *result = add(arr1, arr2);
    print_iArray(result);
    free_iArray(arr1);
    free_iArray(arr2);
    free_iArray(result);
    return 0;
}