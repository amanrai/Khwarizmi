#include <stdio.h>
#include <stdint.h> // For uint8_t
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h> 
#include <stdbool.h>

// void extract_mantissa(uint32_t val) {
//     int32_t mantissa = val && 0x7FFFFF;
//     print_bits(mantissa);
//     uint8_t f_mantissa = (mantissa >> 19) & 0xF;
//     print_bits(f_mantissa);
//     printf("%d\n", mantissa);
// }

// int main() {
//     float x = 15.5;
//     printf("\nOriginal Float Value: %f", x);
//     fp8 _v = from_f32(x);    
//     float y = to_f32(_v);
//     printf("\nReconverted to f32:%f", y);
//     fp8 _w = from_f32_143(x);
//     float z = to_f32_143(_w);
//     printf("\nReconverted 143 to f32:%f\n", z);
// }

void print_bits(int8_t val) {
    for (int i = 7; i >= 0; i--) {
        printf("%d", (val >> i) & 1);
    }
    printf("\n");
}


void ifma_tiled(int8_t *a, int8_t *b, int8_t *c, int size) {
    int tile_size=64;

    #pragma omp parallel for
    for (int i=0; i < size; i+=tile_size) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j+=tile_size) {
                for (int ii = i; ii < i+tile_size; ii++) {
                    for (int jj = j; jj < j+tile_size; jj++) {
                        c[ii*size + jj] += a[ii*size + k] * b[k*size + jj];
                    }
                }
            }
        }
    }
}



void ifma_tiled_strided(int8_t *a, int8_t *b, int8_t *c, int size) {
    const int tile_size = 128;
    const int inner_tile = 8;
    const int simd_size = 64;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int ii = i; ii < i + tile_size && ii < size; ii += inner_tile) {
                for (int k = 0; k < size; k+=simd_size) {
                    __m512 a_val[inner_tile];
                    for (int x=0; x < inner_tile; x++) {
                        a_val[x] = _mm512_set1_epi8(&a[ii*size + x]);
                    }
                    for (int jj=j; jj < j + tile_size && jj < size; jj+=simd_size) {
                        __m512 b_val = __m512
                    }
                }
            }
        }
    }
}


bool isEqual(const int8_t *a, const int8_t *b, int size) {
    for (int i=0; i < size * size; i++) {
        // printf("\n%d \? %d\n", a[i], b[i]);
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}



void ifma(int8_t *a, int8_t *b, int8_t *c, int size) {
    #pragma omp parallel for
    for (int i=0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            for (int j = 0; j < size; j++) {
                c[i*size + j] += a[i*size + k] * b[k*size + j];
            }
        }
    }
}


int main() {
    int size=1024;
    int8_t *a = (int8_t *)malloc(size*size*sizeof(int8_t));
    int8_t *b = (int8_t *)malloc(size*size*sizeof(int8_t));
    int8_t *c = (int8_t *)malloc(size*size*sizeof(int8_t));
    int8_t *d = (int8_t *)malloc(size*size*sizeof(int8_t));

    for (int i=0; i < size*size; i++) {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
        c[i] = 0;
        d[i] = 0;
    }

    ifma_tiled_strided(a, b, c, size);
    ifma(a, b, d, size);
    printf("Fast Matmul matches Slow Matmul: %d\n", isEqual(c, d, size));
    // double start = omp_get_wtime();
    // int num_iter = 100;
    // for (int i=0; i < num_iter; i++) ifma(a, b, c, size);
    // double end = omp_get_wtime();
    // double _t = end - start;
    // _t *= (1000/num_iter);
    // printf("\nTiled Time taken: %f ms per iteration;\n", _t);  
    // ifma(a, b, d, size);
    // printf("Fast Matmul matches Slow Matmul: %d\n", isEqual(c, d, size));
}
