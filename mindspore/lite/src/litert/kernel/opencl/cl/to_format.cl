#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

#define BUF_to_IMG(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, WRITE_IMAGE_OUT)                                          \
  __kernel void BUF_to_IMG_##src_dtype##_##dst_dtype(__global SRC_TYPE##4 * src_data, __write_only image2d_t dst_data, \
                                                     int4 size, int4 shape) {                                          \
    int X = get_global_id(0);                                                                                          \
    int Y = get_global_id(1);                                                                                          \
    int Z = get_global_id(2);                                                                                          \
    if (X >= size.x || Y >= size.y || Z >= size.z) {                                                                   \
      return;                                                                                                          \
    }                                                                                                                  \
    DST_TYPE##4 data = (DST_TYPE##4)(0.f);                                                                             \
    int offset = (X * shape.z + Y) * shape.w + Z * 4;                                                                  \
    __global SRC_TYPE *src_addr = (__global SRC_TYPE *)src_data;                                                       \
    src_addr += offset;                                                                                                \
    if ((Z + 1) * 4 <= shape.w) {                                                                                      \
      data = convert_##DST_TYPE##4(((__global SRC_TYPE##4 *)src_addr)[0]);                                             \
    } else {                                                                                                           \
      if ((shape.w - Z * 4) >= 1) {                                                                                    \
        data.x = (DST_TYPE)src_addr[0];                                                                                \
      }                                                                                                                \
      if ((shape.w - Z * 4) >= 2) {                                                                                    \
        data.y = (DST_TYPE)src_addr[1];                                                                                \
      }                                                                                                                \
      if ((shape.w - Z * 4) >= 3) {                                                                                    \
        data.z = (DST_TYPE)src_addr[2];                                                                                \
      }                                                                                                                \
    }                                                                                                                  \
    if (size.y * size.z <= MAX_IMAGE2D_WIDTH)                                                                          \
      WRITE_IMAGE_OUT(dst_data, (int2)(Y * size.z + Z, X), data);                                                      \
    else                                                                                                               \
      WRITE_IMAGE_OUT(dst_data, (int2)(Z, X * size.y + Y), data);                                                      \
  }

// BUF_to_IMG(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, WRITE_IMAGE_OUT)
BUF_to_IMG(float32, float32, float, float, write_imagef);
BUF_to_IMG(float32, float16, float, half, write_imageh);
BUF_to_IMG(float16, float16, half, half, write_imageh);
BUF_to_IMG(int32, int32, int, int, write_imagei);
BUF_to_IMG(uint32, uint32, int, int, write_imagei);
BUF_to_IMG(int8, int8, char, int, write_imagei);

#define IMG_to_BUF(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, READ_IMAGE_IN)                                           \
  __kernel void IMG_to_BUF_##src_dtype##_##dst_dtype(__read_only image2d_t src_data, __global DST_TYPE##4 * dst_data, \
                                                     int4 size, int4 shape) {                                         \
    int X = get_global_id(0);                                                                                         \
    int Y = get_global_id(1);                                                                                         \
    int Z = get_global_id(2);                                                                                         \
    if (X >= size.x || Y >= size.y || Z >= size.z) {                                                                  \
      return;                                                                                                         \
    }                                                                                                                 \
    DST_TYPE##4 data;                                                                                                 \
    if (size.y * size.z <= MAX_IMAGE2D_WIDTH)                                                                         \
      data = convert_##DST_TYPE##4(READ_IMAGE_IN(src_data, smp_zero, (int2)(Y * size.z + Z, X)));                     \
    else                                                                                                              \
      data = convert_##DST_TYPE##4(READ_IMAGE_IN(src_data, smp_zero, (int2)(Z, X * size.y + Y)));                     \
    int offset = (X * shape.z + Y) * shape.w + Z * 4;                                                                 \
    __global DST_TYPE *dst_addr = (__global DST_TYPE *)dst_data;                                                      \
    dst_addr += offset;                                                                                               \
    if ((Z + 1) * 4 <= shape.w) {                                                                                     \
      ((__global DST_TYPE##4 *)dst_addr)[0] = data;                                                                   \
    } else {                                                                                                          \
      if (shape.w - Z * 4 >= 1) {                                                                                     \
        dst_addr[0] = data.x;                                                                                         \
      }                                                                                                               \
      if (shape.w - Z * 4 >= 2) {                                                                                     \
        dst_addr[1] = data.y;                                                                                         \
      }                                                                                                               \
      if (shape.w - Z * 4 >= 3) {                                                                                     \
        dst_addr[2] = data.z;                                                                                         \
      }                                                                                                               \
    }                                                                                                                 \
  }

// IMG_to_BUF(src_dtype, dst_dtype, SRC_TYPE, DST_TYPE, READ_IMAGE_IN)
IMG_to_BUF(float32, float32, float, float, read_imagef);
IMG_to_BUF(float16, float32, half, float, read_imageh);
IMG_to_BUF(float16, float16, half, half, read_imageh);
IMG_to_BUF(int32, int32, int, int, read_imagei);
IMG_to_BUF(uint32, uint32, int, int, read_imagei);
IMG_to_BUF(int32, float32, int, float, read_imagei);
IMG_to_BUF(int8, int8, char, char, read_imagei);
