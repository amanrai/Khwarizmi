#include <stdint.h>
#include <stddef.h>

#define i8 int8_t
#define iMAX 127
#define iMIN -127

typedef struct iArray{
    i8* data;
    size_t *shape;
    size_t size;
    size_t rank;
} iArray;

typedef struct iTensor {
    iArray *arr;
    i8 zero_point;
    float scale;
} iTensor;

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

iArray *addScalar(iArray *arr, i8 scalar);
iArray *subScalar(iArray *arr, i8 scalar);
iArray *mulScalar(iArray *arr, i8 scalar);
iArray *divScalar(iArray *arr, i8 scalar);
iArray *modScalar(iArray *arr, i8 scalar);

iArray *neg(iArray *arr);
iArray *absiArray(iArray *arr);

iTensor *quantize(float *data, size_t *shape, size_t rank);
iTensor *quantize_symmetric(float *data, size_t *shape, size_t rank); 
iTensor *quantize_asymmetric_minmax(float *data, size_t *shape, size_t rank, float min, float max);
float *dequantize(iTensor *tensor);

iTensor *rebase(iTensor *A, iTensor *B); //in place rebase B to A
iTensor *addTensors(iTensor *A, iTensor *B);
iTensor *rebaseIntrinsics(iTensor *A, float new_scale, i8 new_zero_point);

float percentile(float *data, size_t *shape, size_t rank, float q);
i8 roundi8(float x);