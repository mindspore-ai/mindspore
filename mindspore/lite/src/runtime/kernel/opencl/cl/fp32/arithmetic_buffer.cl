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

__kernel void BoardcastAdd(__global float *input_a, float input_b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] + input_b;
}

__kernel void BoardcastSub(__global float *input_a, float input_b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] - input_b;
}

__kernel void BoardcastMul(__global float *input_a, float input_b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b;
}

__kernel void BoardcastDiv(__global float *input_a, float input_b, __global float *output, const unsigned int n) {
  int idx = get_global_id(0);
  if (idx >= n) return;
  output[idx] = input_a[idx] * input_b;
}
