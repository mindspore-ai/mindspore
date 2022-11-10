#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define C4NUM 4

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void crop(__read_only image2d_t src_data, __write_only image2d_t dst_data,
                   int4 in_shape, int4 out_shape, int4 offset) {
  int out_w = get_global_id(0);
  int out_h = get_global_id(1);

  int out_batch_idx = out_h / out_shape.y;
  int out_height_idx = out_h % out_shape.y;
  int in_batch_idx = out_batch_idx + offset.x;
  int in_height_idx = out_height_idx + offset.y;
  int in_h = in_batch_idx * in_shape.y + in_height_idx;

  int out_width_idx = (out_w * C4NUM) / out_shape.w;
  int out_channel_idx = (out_w * C4NUM) % out_shape.w;
  int in_width_idx = out_width_idx + offset.z;
  int in_channel_idx = out_channel_idx + offset.w;
  int in_w = in_width_idx * in_shape.w + in_channel_idx;

  DTYPE4 res = READ_IMAGE(src_data, smp_zero, (int2)(in_w / C4NUM, in_h));
  WRITE_IMAGE(dst_data, (int2)(out_w, out_h), res);
}
