#include <stdint.h>
#include <stddef.h>

typedef struct iArray{
    int8_t* data;
    size_t *shape;
    size_t size;
    size_t rank;
} iArray;

typedef struct iTensor {
    iArray *arr;
    int8_t zero_point;
    float scale;
}

void printiArray(iArray *arr);
void deep_print(iArray *arr, int dimension, int start_at);

iArray *create(size_t *shape, size_t rank);
iArray *copy(iArray *arr);
void free_iArray(iArray *arr);

iArray *add(iArray *arr1, iArray *arr2);
iArray *sub(iArray *arr1, iArray *arr2);
iArray *mul(iArray *arr1, iArray *arr2);
iArray *truediv(iArray *arr1, iArray *arr2);
iArray *mod(iArray *arr1, iArray *arr2);
// iArray *matmul(iArray *arr1, iArray *arr2)

iArray *addScalar(iArray *arr, int8_t scalar);
iArray *subScalar(iArray *arr, int8_t scalar);
iArray *mulScalar(iArray *arr, int8_t scalar);
iArray *divScalar(iArray *arr, int8_t scalar);
iArray *modScalar(iArray *arr, int8_t scalar);

iArray *neg(iArray *arr);
iArray *absiArray(iArray *arr);

iTensor *quantize(float *data);
iTensor *quantize_symmetric(float *data); 
float *dequantize(iTensor *tensor);