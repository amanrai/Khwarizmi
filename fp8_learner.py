def extract_components(fp8):
    sign = (fp8 >> 7) & 1
    exponent = (fp8 >> 2) & 0x1F
    mantissa = fp8 & 0x3
    return sign, exponent, mantissa

def align_and_add(a, b):
    # Step 1: Extract components
    sign_a, exp_a, mant_a = extract_components(a)
    sign_b, exp_b, mant_b = extract_components(b)

    # Step 2: Adjust mantissa
    mant_a = (mant_a | 0x4) << 6  # Add implicit 1 and shift left
    mant_b = (mant_b | 0x4) << 6

    # Step 3: Align exponents
    if exp_a > exp_b:
        mant_b >>= (exp_a - exp_b)
        exp = exp_a
    else:
        mant_a >>= (exp_b - exp_a)
        exp = exp_b

    # Step 4: Add or subtract mantissas
    if sign_a == sign_b:
        mant = mant_a + mant_b
        sign = sign_a
    else:
        if mant_a >= mant_b:
            mant = mant_a - mant_b
            sign = sign_a
        else:
            mant = mant_b - mant_a
            sign = sign_b

    return sign, exp, mant

def normalize_and_round(sign, exp, mant):
    # Step 5: Normalize
    while mant > 0xFF:
        mant >>= 1
        exp += 1
    while mant < 0x80 and exp > 0:
        mant <<= 1
        exp -= 1

    # Step 6: Round to nearest even
    if (mant & 0x7) == 4:  # Tie-breaking case
        if mant & 0x8:
            mant += 4
    elif (mant & 0x7) > 4:
        mant += 4

    # Handle overflow from rounding
    if mant > 0xFF:
        mant >>= 1
        exp += 1

    # Step 7: Check for overflow/underflow
    if exp > 31:
        return sign, 31, 3  # Infinity
    if exp == 0 and mant < 0x80:
        return sign, 0, mant >> 6  # Subnormal

    # Pack result
    mant = (mant >> 6) & 0x3
    return sign, exp, mant

def fp8_add(a, b):
    sign, exp, mant = align_and_add(a, b)
    sign, exp, mant = normalize_and_round(sign, exp, mant)
    
    # Step 8: Pack result
    result = (sign << 7) | (exp << 2) | mant

    # Step 9: Handle special cases (simplified)
    if (a & 0x7F == 0x7F) or (b & 0x7F == 0x7F):  # NaN
        return 0x7F
    if a == 0:
        return b
    if b == 0:
        return a

    return result

def fp8_to_float(fp8):
    sign, exp, mant = extract_components(fp8)
    if exp == 31 and mant == 3:
        return float('nan')
    if exp == 31:
        return float('inf') if sign == 0 else float('-inf')
    if exp == 0:
        return ((-1)**sign) * (mant / 4) * (2**-14)
    return ((-1)**sign) * (1 + mant/4) * (2**(exp-15))

def float_to_fp8(f):
    if f == 0:
        return 0
    if f != f:  # NaN
        return 0x7F
    sign = 0 if f > 0 else 1
    f = abs(f)
    exp = int(log2(f)) + 15
    if exp > 30:
        return (sign << 7) | 0x7E  # Infinity
    if exp < 1:
        mant = int(f * 2**(14-exp) * 4)
        exp = 0
    else:
        mant = int((f / 2**(exp-15) - 1) * 4)
    mant = min(mant, 3)  # Ensure mantissa fits in 2 bits
    return (sign << 7) | (exp << 2) | mant

# Test the function
from math import log2

print("Testing FP8 Addition:")
test_cases = [
    (0b01100100, 0b01100010),  # Normal case
    (0b01111110, 0b00000001),  # Near infinity + small number
    (0b00000001, 0b00000001),  # Two small numbers
    (0b01111111, 0b01100100),  # NaN + normal number
    (0b00000000, 0b01100100),  # Zero + normal number
]

for a, b in test_cases:
    result = fp8_add(a, b)
    print(f"\nA: {bin(a)} ({fp8_to_float(a)})")
    print(f"B: {bin(b)} ({fp8_to_float(b)})")
    print(f"Result: {bin(result)} ({fp8_to_float(result)})")

print("\nTesting Float to FP8 and back:")
float_tests = [0.1, 1.0, 10.0, 100.0, 1000.0, 0.001, -0.1, -10.0]
for f in float_tests:
    fp8 = float_to_fp8(f)
    back_to_float = fp8_to_float(fp8)
    print(f"Original: {f}, FP8: {bin(fp8)}, Back to float: {back_to_float}")