#ifndef I8_OPS_H
#define I8_OPS_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    int8_t *data;
    size_t size;
} I8Array;

I8Array* i8_create(size_t size);
void i8_destroy(I8Array* arr);
I8Array* i8_add(I8Array* a, I8Array* b);

#endif // I8_OPS_H