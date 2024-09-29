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
    // arr->data = (int8_t *)malloc(arr->size * sizeof(int8_t));
    int result = posix_memalign((void **)&arr->data, 32, arr->size * sizeof(int8_t));

    if (result != 0){
        fprintf(stderr, "Allocation of aligned memory failed.\n");
        free(arr);
        free(arr->shape);
        exit(1);
    }
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
    int rank = 2;
    size_t shape[2] = {16,16};
    iArray *arr1 = from_random(shape, rank, 0, 10);
    printiArray(arr1);
    iArray *arr2 = from_random(shape, rank, -10, 10);
    printiArray(arr2);
    iArray *result = mul(arr1, arr2);
    printiArray(result);
    free_iArray(arr1);
    free_iArray(arr2);
    free_iArray(result);
    return 0;
}