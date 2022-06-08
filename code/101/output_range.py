def get_output_range(bits):
    alpha_q = -2 ** (bits - 1)
    beta_q = 2 ** (bits - 1) - 1
    return alpha_q, beta_q

print("For 16-bit quantization, the quantized range is ", get_output_range(16))
print("For 8-bit quantization, the quantized range is ", get_output_range(8))
print("For 3-bit quantization, the quantized range is ", get_output_range(3))
print("For 2-bit quantization, the quantized range is ", get_output_range(2))