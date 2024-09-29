#include "i8_ops.h"
#include <stdlib.h>
#include <stdio.h>

I8Array* i8_create(size_t size) {
    printf("i8_create: Creating I8Array of size %zu\n", size);
    I8Array* arr = (I8Array*)malloc(sizeof(I8Array));
    if (arr == NULL) {
        printf("i8_create: Failed to allocate I8Array\n");
        return NULL;
    }
    arr->data = (int8_t*)malloc(size * sizeof(int8_t));
    if (arr->data == NULL) {
        printf("i8_create: Failed to allocate I8Array data\n");
        free(arr);
        return NULL;
    }
    arr->size = size;
    printf("i8_create: Successfully created I8Array\n");
    return arr;
}

void i8_destroy(I8Array* arr) {
    printf("i8_destroy: Destroying I8Array\n");
    if (arr) {
        free(arr->data);
        free(arr);
    }
    printf("i8_destroy: I8Array destroyed\n");
}

I8Array* i8_add(I8Array* a, I8Array* b) {
    printf("i8_add: Adding I8Arrays\n");
    if (a == NULL || b == NULL || a->size != b->size) {
        printf("i8_add: Invalid I8Arrays or sizes don't match\n");
        return NULL;
    }
    I8Array* result = i8_create(a->size);
    if (result == NULL) {
        printf("i8_add: Failed to create result array\n");
        return NULL;
    }
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    printf("i8_add: Successfully added I8Arrays\n");
    return result;
}