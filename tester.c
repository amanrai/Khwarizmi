#include <immintrin.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

void ifma(int8_t *a, int8_t *b, int8_t *c, int size) {
    for (int i=0; i < size; i++) {
        for (int k=0; k < size; k++) {
            for (int j=0; j < size; j++) {
                c[i*size+j] += a[i*size + k] * b[k*size + j];
            }
        }
    }
}

void ifma_tiled_strided(int8_t *a, int8_t *b, int8_t *c, int size) {
    const int tile_size = 256;
    const int inner_tile = 128;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int ii = i; ii < i + tile_size && ii < size; ii += inner_tile) {
                for (int k = 0; k < size; k++) {
                    __m512i val = _mm512_loadu_si512((__m512i *)&a[ii*size + k]);
                    for (int jj = j; jj < j + tile_size && jj < size; jj += inner_tile) {
                        __m512i b_val = _mm512_loadu_si512((__m512i *)&b[k*size + jj]);
                        __m512i c_val = _mm512_loadu_si512((__m512i *)&c[ii*size + jj]);
                        
                        __m512i lower = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(val));
                        lower = _mm512_mullo_epi16(lower, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_val)));
                        __m512i upper = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(val, 1));
                        upper = _mm512_mullo_epi16(upper, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(b_val, 1)));
                        __m512i res = _mm512_packus_epi16(lower, upper);
                        
                        c_val = _mm512_add_epi8(c_val, res);
                        _mm512_storeu_si512((__m512i *)&c[ii*size + jj], c_val);
                    }
                }
            }
        }
    }
}

void ifma_tiled_strided_arbitrary(int8_t *a, int8_t *b, int8_t *c, int m, int n, int k) {
    const int tile_size = 256;
    const int inner_tile = 64;
    const int vec_size = 64;  // 512 bits / 8 bits = 64 elements

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i += tile_size) {
        for (int j = 0; j < n; j += tile_size) {
            for (int ii = i; ii < i + tile_size && ii < m; ii += inner_tile) {
                for (int kk = 0; kk < k; kk++) {
                    for (int jj = j; jj < j + tile_size && jj < n; jj += vec_size) {
                        __m512i c_val = _mm512_setzero_si512();
                        
                        if (jj + vec_size <= n) {
                            c_val = _mm512_loadu_si512((__m512i *)&c[ii*n + jj]);
                        } else {
                            // Handle edge case: load partial vector
                            c_val = _mm512_maskz_loadu_epi8((1ULL << (n - jj)) - 1, &c[ii*n + jj]);
                        }

                        for (int iii = 0; iii < inner_tile && ii + iii < m; iii++) {
                            __m512i a_val = _mm512_set1_epi8(a[(ii+iii)*k + kk]);
                            __m512i b_val;
                            
                            if (jj + vec_size <= n) {
                                b_val = _mm512_loadu_si512((__m512i *)&b[kk*n + jj]);
                            } else {
                                // Handle edge case: load partial vector
                                b_val = _mm512_maskz_loadu_epi8((1ULL << (n - jj)) - 1, &b[kk*n + jj]);
                            }

                            __m512i lower = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a_val));
                            lower = _mm512_mullo_epi16(lower, _mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_val)));
                            __m512i upper = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(a_val, 1));
                            upper = _mm512_mullo_epi16(upper, _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(b_val, 1)));
                            __m512i res = _mm512_packus_epi16(lower, upper);
                            
                            c_val = _mm512_add_epi8(c_val, res);
                        }
                        
                        if (jj + vec_size <= n) {
                            _mm512_storeu_si512((__m512i *)&c[ii*n + jj], c_val);
                        } else {
                            // Handle edge case: store partial vector
                            _mm512_mask_storeu_epi8(&c[ii*n + jj], (1ULL << (n - jj)) - 1, c_val);
                        }
                    }
                }
            }
        }
    }
}


int main() {
    const int size = 1024;
    int8_t *a = (int8_t *)malloc(size * size * sizeof(int8_t));
    int8_t *b = (int8_t *)malloc(size * size * sizeof(int8_t));
    int8_t *c = (int8_t *)malloc(size * size * sizeof(int8_t));

    for (int i = 0; i < size * size; i++) {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
    }

    int num_iter = 100; 
    double start = omp_get_wtime();
    for (int i = 0; i < num_iter; i++) ifma_tiled_strided(a, b, c, size);
    double end = omp_get_wtime();
    double _t = end - start;
    printf("Time: %f ms per iteration\n", (_t * 1000/num_iter));

    free(a);
    free(b);
    free(c);

    return 0;
}