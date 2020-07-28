#define CI_TILE 4
#define CO_TILE 4

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))

//#pragma OPENCL EXTENSION cl_arm_printf : enable
__kernel void convolution_NHWC_OHWI(__global float *input,
                                    __global float *weight,
                                    __global float *bias,
                                    __global float *output,
                                    const uint4 input_shape,  // NHWC
                                    const uint4 weight_shape, // OHWI
                                    const uint4 output_shape, // NHWC
                                    const uint2 stride,       // HW
                                    const uint4 pad)          // top bottom left right
{
    uint ow = get_global_id(0);
    uint oh = get_global_id(1);
    uint co_outer = get_global_id(2);

    uint CI = input_shape.w, IH = input_shape.y, IW = input_shape.z;
    uint CO = output_shape.w, OW = output_shape.z;
    uint KH = weight_shape.y, KW = weight_shape.z;
    uint stride_h = stride.x, stride_w = stride.y;
    uint pad_top = pad.x, pad_left = pad.z;
    uint CI_TILE_NUM = UP_DIV(CI, CI_TILE);
    uint CO_TILE_NUM = UP_DIV(CO, CO_TILE);

    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (uint kh = 0; kh < KH; ++kh)
    {
        uint ih = kh + oh * stride_h - pad_top;
        for (uint kw = 0; kw < KW; ++kw)
        {
            uint iw = kw + ow * stride_w - pad_left;
            for (uint ci_outer = 0; ci_outer < CI_TILE_NUM; ++ci_outer)
            {
                for (uint ci_inner = 0; ci_inner < CI_TILE; ++ci_inner)
                {
                    uint ci = ci_outer * CI_TILE + ci_inner;
                    if (ci >= CI)
                        break;

                    uint input_idx = ih * IW * CI + iw * CI + ci;
                    float value = 0;
                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW)
                        value = 0;
                    else
                        value = input[input_idx];

                    uint CO_TILE_OFFSET = KH * KW * CI;
                    uint weight_idx = (co_outer * CO_TILE) * CO_TILE_OFFSET +
                                      kh * KW * CI +
                                      kw * CI +
                                      ci;
                    acc.x += weight[weight_idx + 0 * CO_TILE_OFFSET] * value;
                    acc.y += weight[weight_idx + 1 * CO_TILE_OFFSET] * value;
                    acc.z += weight[weight_idx + 2 * CO_TILE_OFFSET] * value;
                    acc.w += weight[weight_idx + 3 * CO_TILE_OFFSET] * value;
                }
            }
        }
    }
    uint output_idx = oh * OW * CO + ow * CO + (co_outer * CO_TILE);
    if (co_outer < CO_TILE_NUM - 1 || CO % CO_TILE == 0)
    {
        output[output_idx + 0] = acc.x + bias[co_outer * CO_TILE + 0];
        output[output_idx + 1] = acc.y + bias[co_outer * CO_TILE + 1];
        output[output_idx + 2] = acc.z + bias[co_outer * CO_TILE + 2];
        output[output_idx + 3] = acc.w + bias[co_outer * CO_TILE + 3];
    }
    else if (CO % CO_TILE == 1)
    {
        output[output_idx + 0] = acc.x + bias[co_outer * CO_TILE + 0];
    }
    else if (CO % CO_TILE == 2)
    {
        output[output_idx + 0] = acc.x + bias[co_outer * CO_TILE + 0];
        output[output_idx + 1] = acc.y + bias[co_outer * CO_TILE + 1];
    }
    else if (CO % CO_TILE == 3)
    {
        output[output_idx + 0] = acc.x + bias[co_outer * CO_TILE + 0];
        output[output_idx + 1] = acc.y + bias[co_outer * CO_TILE + 1];
        output[output_idx + 2] = acc.z + bias[co_outer * CO_TILE + 2];
    }
}