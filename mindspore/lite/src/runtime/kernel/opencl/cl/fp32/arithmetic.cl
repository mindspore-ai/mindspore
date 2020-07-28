__kernel void ArithmeticAdd(__global float *input_a,
                            __global float *input_b,
                            __global float *output,
                            const unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        output[id] = input_a[id] + input_b[id];
    }
}
__kernel void ArithmeticSub(__global float *input_a,
                            __global float *input_b,
                            __global float *output,
                            const unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        output[id] = input_a[id] - input_b[id];
    }
}
__kernel void ArithmeticMul(__global float *input_a,
                            __global float *input_b,
                            __global float *output,
                            const unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        output[id] = input_a[id] * input_b[id];
    }
}
__kernel void ArithmeticDiv(__global float *input_a,
                            __global float *input_b,
                            __global float *output,
                            const unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        output[id] = input_a[id] * input_b[id];
    }
}

__kernel void ArithmeticBiasAdd(__global float4 *input,
                                __global float4 *output,
                                const float weight,
                                const float bias,
                                const unsigned int n) {
    int id = get_global_id(0);
    float4 bias_vec = (float4)(bias, 0.0f, .0f, .0f);
    float4 weight_vec = (float4)(weight, 0.0f, .0f, .0f);
    if (id < n) {
        output[id] = weight_vec * input[id] + bias_vec;
    }
}
