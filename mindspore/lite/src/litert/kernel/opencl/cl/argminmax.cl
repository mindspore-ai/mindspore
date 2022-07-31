#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define swap(a, b, c) \
  c = a;              \
  a = b;              \
  b = c;
#define swap_atomic(a, b, c) \
  c = atomic_xchg(a, *(b));  \
  c = atomic_xchg(b, c);
#define UP_ROUND(a, b) (((a + b - 1) / b) * b)
#define C4NUM 4
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void argminmax(__global FLT *src_data, __global FLT *dst_data, __global FLT *buf, __global int *ids,
                        int4 shape, int4 src_size, int4 cus_size, int4 strides, int4 flags) {
  int X = get_global_id(0);  // lower reduce stride
  int Y = get_global_id(1);  // upper axis accumulation
  if (X >= src_size.x || Y >= src_size.y) {
    return;
  }
  bool keep_dims = cus_size.y;
  int width = shape.z * shape.w;
  int offset = X + Y * src_size.z;
  int align_c4_in = (flags.z != 3) ? (X / shape.w) * (C4NUM - shape.w & 0x00000003) : 0;
  int align_c4_out =
    (flags.z == 3 && flags.w == 1 && !keep_dims) ? (Y / shape.z) * (C4NUM - shape.z & 0x00000003) : align_c4_in;
  int align_in = 0;
  int align_out = 0;
  if (flags.z == 3) {
    align_in = (Y / shape.z) * cus_size.z;
    align_out = (Y / ((flags.w > 1 || keep_dims) ? shape.z : shape.z * shape.y)) * cus_size.w;
  }
  if (flags.z == 0) {
    align_in = X / (width)*cus_size.z;
    align_out = align_in;
  }
  if (flags.z == 2 && !keep_dims) {
    align_out = (Y / shape.y) * cus_size.w;
  }
  for (int k = 0; k < src_size.w; ++k) {
    int idx0 = (X + k * strides.x) + Y * strides.y + (align_c4_in + align_in);
    int idx1 = offset + k * src_size.x;
    ids[idx1] = k;
    buf[idx1] = src_data[idx0];
  }
  for (unsigned int i = 2; i <= cus_size.x; i <<= 1) {
    for (unsigned int j = i >> 1; j > 0; j >>= 1) {
      for (int tid = 0; tid < src_size.w; ++tid) {
        unsigned int tid_comp = tid + j;
        if (tid_comp < src_size.w) {
          int lk = offset + tid * src_size.x;
          int rk = offset + tid_comp * src_size.x;
          if ((tid & i) == 0) {  // ascending
            if (buf[lk] > buf[rk]) {
              FLT tmpf;
              swap(buf[lk], buf[rk], tmpf);
              int tmpi;
              swap(ids[lk], ids[rk], tmpi);
            }
          } else {  // desending
            if (buf[lk] < buf[rk]) {
              FLT tmpf;
              swap(buf[lk], buf[rk], tmpf);
              int tmpi;
              swap(ids[lk], ids[rk], tmpi);
            }
          }
        }
      }
    }
  }
  for (int k = 0; k < flags.w; ++k) {
    int idx0 = (X + k * strides.z) + Y * strides.w + (align_c4_out + align_out);
    int idx1 = flags.y ? (offset + (src_size.w - k - 1) * src_size.x) : (offset + k * src_size.x);
    if (flags.x) {
      dst_data[idx0] = buf[idx1];
    } else {
      dst_data[idx0] = ids[idx1];
    }
  }
}
