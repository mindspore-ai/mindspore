#define FLT4 float4
#define FLT16 float16
__kernel void MatMul(__global FLT4 *x, __global FLT16 *weight,
                        __global FLT4 *buffer, __global FLT4 *bias, int2 offset_ci,
                        int2 offset_co, int has_bias) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int2 lid = (int2)(get_local_id(0), get_local_id(1));
  FLT4 s = (FLT4)(0.0f);
  bool inside = gid.x < offset_co.y;
  for (uint i = lid.y; i < offset_ci.y && inside; i += 4) {
    FLT4 v = x[i];
    FLT16 w = weight[gid.x + i * offset_co.y];
    s.x += dot(v, w.s0123);
    s.y += dot(v, w.s4567);
    s.z += dot(v, w.s89ab);
    s.w += dot(v, w.scdef);
  }
  __local FLT4 temp[64][4];
  temp[lid.x][lid.y] = s;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid.y == 0 && inside) {
    s += temp[lid.x][1];
    s += temp[lid.x][2];
    s += temp[lid.x][3];
    if (has_bias != 0) {
      s += bias[gid.x];
    }
    buffer[gid.x] = s;
    // memory pollution? or protected by opencl
  }
}