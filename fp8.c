#include <stdint.h>
#include <stdio.h>
#include <limits.h> // For CHAR_BIT
#include <math.h>
#include <float.h>
#include "fp8.h"


void print_bits(uint8_t n) {
    int num_bits = sizeof(uint8_t) * CHAR_BIT; 

    for (int i = num_bits - 1; i >= 0; i--) {
        // Extract the bit at position i
        uint8_t bit = (n >> i) & 1u;
        printf("%u", bit);

        // Optional: Add a space every 4 bits for readability
        if (i % 4 == 0 && i != 0)
            printf(" ");
    }
    printf("\n");
}

fp8 from_f32(float input) {
    uint32_t f = *(uint32_t*)&input;
    
    uint8_t sign = (f >> 31) & 0x1;
    int32_t exponent = ((f >> 23) & 0xFF) - 127; // Unbias
    uint32_t mantissa = f & 0x7FFFFF;

    // Handle special cases
    if (isnan(input)) return 0x7F;  // NaN
    if (isinf(input)) return sign ? 0xFF : 0x7E;  // +/- Infinity

    // Adjust exponent for FP8 (1-5-2 format)
    int8_t fp8_exp = exponent + 15;
    
    // Handle subnormal numbers and underflow
    if (fp8_exp <= 0) {
        if (fp8_exp < -1) return sign << 7;  // Underflow to zero
        mantissa |= 0x800000;  // Add implicit leading 1
        mantissa >>= -fp8_exp + 1;
        fp8_exp = 0;
    } else if (fp8_exp > 31) {
        return sign ? 0xFF : 0x7E;  // Overflow to infinity
    }

    // Round to nearest, ties to even
    uint32_t round_bit = (mantissa >> 21) & 1;
    uint32_t sticky_bits = mantissa & 0x1FFFFF;
    uint8_t fp8_mant = (mantissa >> 22) & 0x3;
    
    if (round_bit && (sticky_bits || (fp8_mant & 1))) {
        if (fp8_mant == 3) {
            fp8_mant = 0;
            fp8_exp++;
            if (fp8_exp > 31) return sign ? 0xFF : 0x7E;  // Overflow
        } else {
            fp8_mant++;
        }
    }

    // Assemble FP8
    return (sign << 7) | (fp8_exp << 2) | fp8_mant;
}

float to_f32(fp8 input) {
    uint8_t sign = (input >> 7) & 0x1;
    uint8_t fp8_exp = (input >> 2) & 0x1F;
    uint8_t fp8_mant = input & 0x3;

    // Handle special cases
    if (fp8_exp == 31 && fp8_mant == 3) return sign ? -NAN : NAN;
    if (fp8_exp == 31) return sign ? -INFINITY : INFINITY;

    int32_t exponent = fp8_exp - 15;  // Unbias
    uint32_t mantissa = fp8_mant;

    if (fp8_exp == 0) {
        
        exponent = -14;
        mantissa >>= 1;
    } else {
        // Normal number, add implicit leading 1
        mantissa |= 0x4;
    }

    // Adjust to F32 format
    exponent += 127;  // Apply F32 bias
    mantissa <<= 21;  // Shift to F32 mantissa position

    uint32_t f = (sign << 31) | (exponent << 23) | mantissa;
    return *(float*)&f;
}
