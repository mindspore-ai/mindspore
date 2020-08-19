__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

// what is mask and args.slices_x32
__kernel void SoftMax1x1_IMG(__read_only image2d_t input, __write_only image2d_t output, const float4 mask,
                             const int slices, const int slices_x32) {
  int tid = get_local_id(0);
  int slices_count = 0;
  int offset = 0;
  float sum = 0.0f;
  do {
    int z = offset + tid;
    if (z < slices) {
      float4 mask_temp = z == slices - 1 ? mask : (float4)(1.0f);
      float4 src = read_imagef(input, smp_none, (int2)(0, 0));
      sum += dot(mask_temp, exp(src));
      offset += 32;
    }
    slices_count++;
  } while (slices_count < slices_x32);

  __local float4 tmp[8];
  __local float *tmpx1 = (__local float *)tmp;
  tmpx1[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    sum = dot((float4)(1.0f), tmp[0]);
    sum += dot((float4)(1.0f), tmp[1]);
    sum += dot((float4)(1.0f), tmp[2]);
    sum += dot((float4)(1.0f), tmp[3]);
    sum += dot((float4)(1.0f), tmp[4]);
    sum += dot((float4)(1.0f), tmp[5]);
    sum += dot((float4)(1.0f), tmp[6]);
    sum += dot((float4)(1.0f), tmp[7]);
    tmpx1[0] = 1.0f / sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = tmpx1[0];

  offset = 0;
  slices_count = 0;
  do {
    int z = offset + tid;
    if (z < slices) {
      float4 res = convert_float4(exp(read_imagef(input, smp_none, (int2)(0, 0))) * sum);
      write_imagef(output, (int2)(0, 0), res);
      offset += 32;
    }
    slices_count++;
  } while (slices_count < slices_x32);
}

__kernel void SoftMax1x1_BUF(__read_only image2d_t input, __global float4 *output, const float4 mask, const int slices,
                             const int slices_x32) {
  int tid = get_local_id(0);
  float sum = 0.0f;
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 src = read_imagef(input, smp_none, (int2)(i, 0));
    sum += dot((float4)(1.0f), exp(src));
  }
  if ((slices - 1) % 32 == tid) {
    float4 src = read_imagef(input, smp_none, (int2)(slices - 1, 0));
    sum += dot(mask, exp(src));
  }

  __local float4 tmp[8];
  __local float *tmpx1 = (__local float *)tmp;
  tmpx1[tid] = sum;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (tid == 0) {
    sum = dot((float4)(1.0f), tmp[0]);
    sum += dot((float4)(1.0f), tmp[1]);
    sum += dot((float4)(1.0f), tmp[2]);
    sum += dot((float4)(1.0f), tmp[3]);
    sum += dot((float4)(1.0f), tmp[4]);
    sum += dot((float4)(1.0f), tmp[5]);
    sum += dot((float4)(1.0f), tmp[6]);
    sum += dot((float4)(1.0f), tmp[7]);
    tmpx1[0] = 1.0f / sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  sum = tmpx1[0];
  for (size_t i = tid; i < slices - 1; i += 32) {
    float4 result = read_imagef(input, smp_none, (int2)(i, 0));
    result = exp(result) * sum;
    output[i] = result;
  }
  if ((slices - 1) % 32 == tid) {
    float4 result = read_imagef(input, smp_none, (int2)(slices - 1, 0));
    result = exp(result) * sum;
    __global float4 *remain_ptr4 = output;
    remain_ptr4 += slices - 1;
    __global float *remain_ptr = (__global float *)remain_ptr4;
    remain_ptr[0] = result.x;
    if (mask.y > 0.f) {
      remain_ptr[1] = result.y;
    }
    if (mask.z > 0.f) {
      remain_ptr[2] = result.z;
    }
    if (mask.w > 0.f) {
      remain_ptr[3] = result.w;
    }
  }
}
