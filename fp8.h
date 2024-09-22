#include <limits.h> // For CHAR_BIT

typedef uint8_t fp8;

void print_bits(uint8_t n);
fp8 from_f32(float input);
float to_f32(fp8 input);