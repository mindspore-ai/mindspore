#define FLT4 half4
#define FLT16 half16
#define READ_IMAGE read_imageh
#define WRITE_IMAGE write_imageh
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void MatMul(__read_only image2d_t input, __global FLT16 *weight, __read_only image2d_t bias,
                     __write_only image2d_t output, int2 offset_ci, int2 offset_co, int has_bias) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int2 lid = (int2)(get_local_id(0), get_local_id(1));
  FLT4 result = (FLT4)(0.0f);
  bool inside = gid.x < offset_co.y;
  for (uint i = lid.y; i < offset_ci.y && inside; i += 4) {
    FLT4 v = READ_IMAGE(input, smp_zero, (int2)(i, 0));
    FLT16 w = weight[gid.x + i * offset_co.y];
    result.x += dot(v, w.s0123);
    result.y += dot(v, w.s4567);
    result.z += dot(v, w.s89ab);
    result.w += dot(v, w.scdef);
  }
  __local FLT4 temp[64][4];
  temp[lid.x][lid.y] = result;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid.y == 0 && inside) {
    result += temp[lid.x][1];
    result += temp[lid.x][2];
    result += temp[lid.x][3];
    if (has_bias != 0) {
      result += READ_IMAGE(bias, smp_zero, (int2)(gid.x, 0));
    }
    WRITE_IMAGE(output, (int2)(gid.x, 0), result);
  }
}
