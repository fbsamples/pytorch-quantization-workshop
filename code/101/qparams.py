def get_quantization_params(input_range, output_range):
    min_val, max_val = input_range
    alpha_q, beta_q = output_range
    S = (max_val - min_val) / (beta_q - alpha_q)
    Z = alpha_q - (min_val / S)
    return S, Z


def quantize(x, S, Z):
    x_q = 1/S * x + Z  
    x_q = torch.round(x_q).to(torch.int8)
    return x_q


def dequantize(x_q, S, Z):
    x = S * (x_q - Z)
    return x


def quantize_int8(x):
    S, Z = get_quantization_params(input_range=(x.min(), x.max(),), output_range=(-128, 127))
    x_q = quantize(x, S, Z)
    return x_q, S, Z


