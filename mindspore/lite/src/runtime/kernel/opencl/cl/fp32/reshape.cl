#define FLT float
#define FLT4 float4
#define READ_IMAGE read_imagef
#define WRITE_IMAGE write_imagef
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void reshape(__read_only image2d_t src_data, __write_only image2d_t dst_data, int4 size) {
  int X = get_global_id(0);
  int Y = get_global_id(1);
  int Z = get_global_id(2);
  if (X >= size.x || Y >= size.y || Z >= size.z) {
    return;
  }
  WRITE_IMAGE(dst_data, (int2)(Y * size.z + Z, X), READ_IMAGE(src_data, smp_zero, (int2)(Y * size.z + Z, X)));
}
