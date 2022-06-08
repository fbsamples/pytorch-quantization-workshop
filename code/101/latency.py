def module_latency(mod, input, num_tests=10):
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(num_tests):
            mod(input)
    elapsed = time.time() - t0
    latency = elapsed/num_tests
    print("Average Latency: ", format(latency, 'g'))