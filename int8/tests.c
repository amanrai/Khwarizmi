#include "int8_ops.h"
#include <stdlib.h>
#include <stdio.h>

int randArray(float *data, float min, float max, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (float)rand() / (float) RAND_MAX;
        data[i] = data[i] * (max - min) + min;
    }
    return 0;
}

void printArray(float *data, int size) {
    printf("[");
    for (size_t i = 0; i < size; i++) {
        printf("%f ", data[i]);
        if (i < size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

int main() {
    size_t size = 16;
    float *a = (float *)malloc(size * sizeof(float));    
    randArray(a, -1.0, 1.0, size);
    a[7] = 100.0;
    printArray(a, size);
    iTensor *t = quantize(a, &size, 1);
    printf("\nmodified large value: %d", t->arr->data[7]);
    printiArray(t->arr);
    float *b = dequantize(t);
    printArray(b, size);
}