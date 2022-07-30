__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void GLTexture2D_to_IMG(__read_only image2d_t imageSrc, __write_only image2d_t imageDst) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 color = read_imagef(imageSrc, smp_zero, (int2)(x, y));
    write_imagef(imageDst, (int2)(x, y), color);
}

__kernel void IMG_to_GLTexture2D(__read_only image2d_t imageSrc, __write_only image2d_t imageDst) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float4 color = read_imagef(imageSrc, smp_zero, (int2)(x, y));
    write_imagef(imageDst, (int2)(x, y), color);
}
