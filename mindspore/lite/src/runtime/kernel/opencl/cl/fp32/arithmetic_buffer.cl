__kernel void ElementAdd(__global float *input_a, __global float *input_b, __global float *output,
                         const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] + input_b[idx];
}

__kernel void ElementSub(__global float *input_a, __global float *input_b, __global float *output,
                         const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] - input_b[idx];
}

__kernel void ElementMul(__global float *input_a, __global float *input_b, __global float *output,
                         const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b[idx];
}

__kernel void ElementDiv(__global float *input_a, __global float *input_b, __global float *output,
                         const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b[idx];
}

__kernel void BoardcastArith(__global float *input_a, float weight, float bias, __global float *output,
                             const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = weight * input_a[idx] + bias;
}
