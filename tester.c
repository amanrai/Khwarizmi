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

#if defined(__AVX512F__)
#include <immintrin.h>
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
    const int inner_tile = 128;
    const int vec_size = 128;  // 512 bits / 8 bits = 64 elements

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
                            // Handle edge case: load partial vector - fill the remaining with 0s
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
#elif defined(__AVX2__)
#include <immintrin.h>
void ifma_tiled_strided(int8_t *a, int8_t *b, int8_t *c, int m, int n, int k) {
    const int tile_size = 128;
    const int inner_tile = 32;
    const int vec_size = 32;  // 256 bits / 8 bits = 32 elements

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i += tile_size) {
        for (int j = 0; j < n; j += tile_size) {
            for (int ii = i; ii < i + tile_size && ii < m; ii += inner_tile) {
                for (int kk = 0; kk < k; kk++) {
                    for (int jj = j; jj < j + tile_size && jj < n; jj += vec_size) {
                        __m256i c_val = _mm256_setzero_si256();
                        
                        if (jj + vec_size <= n) {
                            c_val = _mm256_loadu_si256((__m256i *)&c[ii*n + jj]);
                        } else {
                            // Handle edge case: load partial vector
                            c_val = _mm256_maskz_loadu_epi8((1U << (n - jj)) - 1, &c[ii*n + jj]);
                        }

                        for (int iii = 0; iii < inner_tile && ii + iii < m; iii++) {
                            __m256i a_val = _mm256_set1_epi8(a[(ii+iii)*k + kk]);
                            __m256i b_val;
                            
                            if (jj + vec_size <= n) {
                                b_val = _mm256_loadu_si256((__m256i *)&b[kk*n + jj]);
                            } else {
                                // Handle edge case: load partial vector
                                b_val = _mm256_maskz_loadu_epi8((1U << (n - jj)) - 1, &b[kk*n + jj]);
                            }

                            __m256i even = _mm256_mullo_epi16(_mm256_and_si256(a_val, _mm256_set1_epi16(0x00ff)),
                                                              _mm256_and_si256(b_val, _mm256_set1_epi16(0x00ff)));
                            __m256i odd = _mm256_mullo_epi16(_mm256_srli_epi16(a_val, 8),
                                                             _mm256_srli_epi16(b_val, 8));
                            __m256i res = _mm256_or_si256(_mm256_and_si256(even, _mm256_set1_epi16(0x00ff)),
                                                          _mm256_slli_epi16(odd, 8));
                            
                            c_val = _mm256_adds_epi8(c_val, res);
                        }
                        
                        if (jj + vec_size <= n) {
                            _mm256_storeu_si256((__m256i *)&c[ii*n + jj], c_val);
                        } else {
                            // Handle edge case: store partial vector
                            _mm256_mask_storeu_epi8(&c[ii*n + jj], (1U << (n - jj)) - 1, c_val);
                        }
                    }
                }
            }
        }
    }
}
#elif defined (__ARM_NEON)
#include <arm_neon.h>
void ifma_tiled_strided(int8_t *a, int8_t *b, int8_t *c, int size) {
    const int tile_size = 64;
    const int vector_size = 16;  // NEON processes 16 int8_t values at a time

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int k = 0; k < size; k += tile_size) {
                for (int ii = i; ii < i + tile_size && ii < size; ii += vector_size) {
                    for (int jj = j; jj < j + tile_size && jj < size; jj += vector_size) {
                        int32x4_t c_vec = vdupq_n_s32(0);
                        
                        for (int kk = k; kk < k + tile_size && kk < size; kk++) {
                            int8x16_t a_vec = vld1q_s8(&a[ii*size + kk]);
                            int8x16_t b_vec = vld1q_s8(&b[kk*size + jj]);
                            
                            // // Subtract zero points
                            // int16x8_t a_low = vsubl_s8(vget_low_s8(a_vec), vdup_n_s8(a->zero_point));
                            // int16x8_t a_high = vsubl_s8(vget_high_s8(a_vec), vdup_n_s8(a->zero_point));
                            // int16x8_t b_low = vsubl_s8(vget_low_s8(b_vec), vdup_n_s8(b->zero_point));
                            // int16x8_t b_high = vsubl_s8(vget_high_s8(b_vec), vdup_n_s8(b->zero_point));
                            
                            // c_vec = vaddq_s32(c_vec, vmlal_s16(vmlal_s16(vdupq_n_s32(0), 
                            //                                              vget_low_s16(a_low), vget_low_s16(b_low)),
                            //                                   vget_high_s16(a_low), vget_high_s16(b_low)));
                            // c_vec = vaddq_s32(c_vec, vmlal_s16(vmlal_s16(vdupq_n_s32(0), 
                            //                                              vget_low_s16(a_high), vget_low_s16(b_high)),
                            //                                   vget_high_s16(a_high), vget_high_s16(b_high)));

                        }
                        
                        // vst1q_s32(&c[(ii/4)*size + jj], c_vec);
                    }
                }
            }
        }
    }

}

void matmul_tiled_neon(float *a, float *b, float *c, int size) {
    const int tile_size = 8;
    const int vector_size = 4;  // NEON processes 4 floats at a time
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int k = 0; k < size; k += tile_size) {
                for (int ii = i; ii < i + tile_size && ii < size; ii += vector_size) {
                    for (int jj = j; jj < j + tile_size && jj < size; jj += vector_size) {
                        float32x4_t c_vec = vld1q_f32(&c[ii*size + jj]);
                        
                        for (int kk = k; kk < k + tile_size && kk < size; kk++) {
                            float32x4_t a_vec = vld1q_f32(&a[ii*size + kk]);
                            float32x4_t b_vec = vld1q_f32(&b[kk*size + jj]);
                            c_vec = vmlaq_f32(c_vec, a_vec, b_vec);
                        }
                        
                        vst1q_f32(&c[ii*size + jj], c_vec);
                    }
                }
            }
        }
    }
}

#endif

int main() {
    const int size = 8192;
    int8_t *a = (int8_t *)malloc(size * size * sizeof(int8_t));
    int8_t *b = (int8_t *)malloc(size * size * sizeof(int8_t));
    int8_t *c = (int8_t *)malloc(size * size * sizeof(int8_t));

    for (int i = 0; i < size * size; i++) {
        a[i] = rand() % 256;
        b[i] = rand() % 256;
    }

    int num_iter = 10; 
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