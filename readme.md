# Khwarizmi

### What is this?
An INT8 implementation, hopefully eventually usable in python to build and infer neural networks. 

### Why?
To learn. And for the lulz. And because in INT8, modern CPUs are blazing fast. 

### So you think you will get as fast as a GPU?
No. Faster hardware is faster. The gains in speed should translate, eventually, to any hardware. The aim here is to run viable networks on the edge at viable speeds. 

#### So how do I run this thing?
As of this writing, you need an x86 CPU that supports AVX 512 intrinsics. 
```sh
gcc -fopenmp -march=native -O3 tester.c -o test &&./test
```
To compare against Numpy/Torch in FP32
```sh
python numpy_test.py
```

### Next Steps
(In no specific order)
1. Exploit TensorRT cores on NVIDIA gpus, if available - extend speed gains to GPUs
2. Make these methods available to Python. 
3. Implement Karpathy's llama.c using INT8s only. 
4. Define quantization strategies for NN models. 
5. Write a library to do Quantization-Aware training, and distillation. 

### Disclosures
This project is being built with the assistance of AI. 