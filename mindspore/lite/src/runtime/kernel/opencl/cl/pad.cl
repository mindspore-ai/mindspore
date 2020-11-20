#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void Pad(__read_only image2d_t input, __write_only image2d_t output, int4 input_shape, int4 output_shape,
                  int2 io_slices, int4 pad_before, float constant_value) {
  int IN = input_shape.x, IH = input_shape.y, IW = input_shape.z, CI = input_shape.w;
  int ON = output_shape.x, OH = output_shape.y, OW = output_shape.z, CO = output_shape.w;
  int CI_SLICES = io_slices.x, CO_SLICES = io_slices.y;
  int on_oh = get_global_id(0);
  int ow = get_global_id(1);
  int co_slice = get_global_id(2);
  int on = on_oh / OH;
  int oh = on_oh % OH;
  if (on >= ON || oh >= OH || ow >= OW || co_slice >= CO_SLICES) {
    return;
  }

  int in = on - pad_before.x;
  int ih = oh - pad_before.y;
  int iw = ow - pad_before.z;
  int ci = co_slice * 4 - pad_before.w;
  if (in < 0 || in >= IN || ih < 0 || ih >= IH || iw < 0 || iw >= IW || ci + 3 < 0 || ci >= CI) {
    WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), (FLT4)(constant_value));
    return;
  }

  int offset = ci % 4;
  if (offset < 0) {
    offset += 4;
  }
  FLT4 src0 = READ_IMAGE(input, smp_zero, (int2)(iw * CI_SLICES + ci / 4, in * IH + ih));
  if (offset == 0 && ci >= 0 && ci + 3 < CI) {
    WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), src0);
    return;
  }
  FLT4 src1 = READ_IMAGE(input, smp_zero, (int2)(iw * CI_SLICES + (ci + 4) / 4, in * IH + ih));
  FLT4 src_f4;
  if (offset == 0) {
    src_f4 = (FLT4)(src0.x, src0.y, src0.z, src0.w);
  } else if (offset == 1) {
    src_f4 = (FLT4)(src0.y, src0.z, src0.w, src1.x);
  } else if (offset == 2) {
    src_f4 = (FLT4)(src0.z, src0.w, src1.x, src1.y);
  } else {  // if (offset==3)
    src_f4 = (FLT4)(src0.w, src1.x, src1.y, src1.z);
  }
  FLT src[4] = {src_f4.x, src_f4.y, src_f4.z, src_f4.w};
  FLT out[4] = {constant_value, constant_value, constant_value, constant_value};
  for (int i = 0; i < 4; ++i) {
    if (ci + i >= 0 && ci + i < CI) {
      out[i] = src[i];
    }
  }
  FLT4 out_f4 = (FLT4)(out[0], out[1], out[2], out[3]);
  WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), out_f4);
}
