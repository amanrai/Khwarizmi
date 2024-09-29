#include <stdint.h>
#include <stddef.h>

typedef struct Array{
  int8_t* data;
  size_t *shape;
  size_t size;
  size_t rank;
} Array;

Array *create(size_t *shape, size_t rank);
Array *copy(Array *arr);
void free_array(Array *arr);

Array *add(Array *arr1, Array *arr2);
Array *sub(Array *arr1, Array *arr2);
Array *mul(Array *arr1, Array *arr2);
Array *truediv(Array *arr1, Array *arr2);
Array *mod(Array *arr1, Array *arr2);

Array *addScalar(Array *arr, int8_t scalar);
Array *subScalar(Array *arr, int8_t scalar);
Array *mulScalar(Array *arr, int8_t scalar);
Array *divScalar(Array *arr, int8_t scalar);
Array *modScalar(Array *arr, int8_t scalar);

Array *neg(Array *arr);
Array *absArray(Array *arr);