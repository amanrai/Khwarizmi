#include <stdio.h>
#include <stdint.h> // For uint8_t
#include "fp8.h"


void extract_mantissa(uint32_t val) {
    int32_t mantissa = val && 0x7FFFFF;
    print_bits(mantissa);
    uint8_t f_mantissa = (mantissa >> 19) & 0xF;
    print_bits(f_mantissa);
    printf("%d\n", mantissa);
}

int main() {
    float x = 15.5;
    printf("\nOriginal Float Value: %f", x);
    fp8 _v = from_f32(x);    
    float y = to_f32(_v);
    printf("\nReconverted to f32:%f", y);
    fp8 _w = from_f32_143(x);
    float z = to_f32_143(_w);
    printf("\nReconverted 143 to f32:%f\n", z);
}
