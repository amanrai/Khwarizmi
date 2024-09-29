#include <stdint.h>
#include <stddef.h>

typedef struct iArray{
  int8_t* data;
  size_t *shape;
  size_t size;
  size_t rank;
} iArray;

iArray *create(size_t *shape, size_t rank);
iArray *copy(iArray *arr);
void free_iArray(iArray *arr);

iArray *add(iArray *arr1, iArray *arr2);
iArray *sub(iArray *arr1, iArray *arr2);
iArray *mul(iArray *arr1, iArray *arr2);
iArray *truediv(iArray *arr1, iArray *arr2);
iArray *mod(iArray *arr1, iArray *arr2);

iArray *addScalar(iArray *arr, int8_t scalar);
iArray *subScalar(iArray *arr, int8_t scalar);
iArray *mulScalar(iArray *arr, int8_t scalar);
iArray *divScalar(iArray *arr, int8_t scalar);
iArray *modScalar(iArray *arr, int8_t scalar);

iArray *neg(iArray *arr);
iArray *absiArray(iArray *arr);