#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__constant sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

__kernel void strided_slice(__read_only image2d_t input, __write_only image2d_t output, int4 input_shape,
                            int4 output_shape, int2 io_slices, int4 begin, int4 stride, int4 size) {
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

  FLT tmp[4];
  for (int i = 0; i < 4; ++i) {
    // output_shape idx -> size idx. because squeeze(output_shape)=squeeze(size)
    // for example:
    // python code: B = A[1, 1:16, 2:16, 3:16]
    // input_shape  = [16, 16, 16, 16]
    // begin        = [ 1,  1,  2,  3]
    // end          = [ 2, 16, 16, 16]
    // stride       = [ 1,  1,  1,  1]
    // size         = [ 1, 15, 14, 13] = ceil((end - begin) / stride)
    // output_shape = [    15, 14, 13]
    int idx = ((on * OH + oh) * OW + ow) * CO + co_slice * 4 + i;
    int co_ = idx % size.w;
    idx /= size.w;
    int ow_ = idx % size.z;
    idx /= size.z;
    int oh_ = idx % size.y;
    idx /= size.y;
    int on_ = idx;

    int in = begin.x + stride.x * on_;
    int ih = begin.y + stride.y * oh_;
    int iw = begin.z + stride.z * ow_;
    int ci = begin.w + stride.w * co_;

    FLT4 src = READ_IMAGE(input, smp_none, (int2)(iw * CI_SLICES + ci / 4, in * IH + ih));
    int offset = ci % 4;
    if (offset == 0) {
      tmp[i] = src.x;
    } else if (offset == 1) {
      tmp[i] = src.y;
    } else if (offset == 2) {
      tmp[i] = src.z;
    } else {
      tmp[i] = src.w;
    }
  }

  FLT4 out = (FLT4)(tmp[0], tmp[1], tmp[2], tmp[3]);
  WRITE_IMAGE(output, (int2)(ow * CO_SLICES + co_slice, on_oh), out);
}
