/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0dim_dst_width
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_max_pool_s8_opt.c
 * Description:  Pooling function implementations
 *
 * $Date:        February 27, 2020
 * $Revision:    V.1.0.1
 *
 * Target Processor:  Cortex-M
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

#if defined(ARM_MATH_DSP)

static void compare_and_replace_if_larger_q7(q7_t *base,
                                             const q7_t *target,
                                             const uint16_t length)
{
#if defined(ARM_MATH_MVEI)

    int loop_count = length / 16;
    while (loop_count)
    {
        const int8x16_t op_1 = vldrbq_s8(base);
        const int8x16_t op_2 = vldrbq_s8(target);
        const int8x16_t max = vmaxq_s8(op_1, op_2);
        vstrbq_s8(base, max);
        base += 16;
        target += 16;
        loop_count--;
    }

    if (((length & 0xF) / 8) > 0)
    {
        const int16x8_t op_1 = vldrbq_s16(base);
        const int16x8_t op_2 = vldrbq_s16(target);
        const int16x8_t max = vmaxq_s16(op_1, op_2);
        vstrbq_s16(base, max);

        base += 8;
        target += 8;
    }

    for (int i = 0; i < (length & 7); i++)
    {
        if (target[i] > base[i])
        {
            base[i] = target[i];
        }
    }

#else
    q7_t *dst = base;
    const q7_t *src = target;
    union arm_nnword ref_max;
    union arm_nnword comp_max;
    int32_t cnt = length >> 2;

    while (cnt > 0l)
    {
        ref_max.word = arm_nn_read_q7x4(dst);
        comp_max.word = arm_nn_read_q7x4_ia(&src);

        if (comp_max.bytes[0] > ref_max.bytes[0])
        {
            ref_max.bytes[0] = comp_max.bytes[0];
        }
        if (comp_max.bytes[1] > ref_max.bytes[1])
        {
            ref_max.bytes[1] = comp_max.bytes[1];
        }
        if (comp_max.bytes[2] > ref_max.bytes[2])
        {
            ref_max.bytes[2] = comp_max.bytes[2];
        }
        if (comp_max.bytes[3] > ref_max.bytes[3])
        {
            ref_max.bytes[3] = comp_max.bytes[3];
        }

        write_q7x4_ia(&dst, ref_max.word);

        cnt--;
    }

    cnt = length & 0x3;
    while (cnt > 0l)
    {
        if (*src > *dst)
        {
            *dst = *src;
        }
        dst++;
        src++;
        cnt--;
    }
#endif
}

static void clamp_output(q7_t *source, const uint16_t length, const int32_t act_min, const int32_t act_max)
{
#if defined(ARM_MATH_MVEI)

    int cnt = length / 16;
    while (cnt > 0)
    {
        const int8x16_t src = vldrbq_s8(source);
        int8x16_t res = vmaxq_s8(src, vdupq_n_s8((int8_t)act_min));
        res = vminq_s8(src, vdupq_n_s8((int8_t)act_max));
        vstrbq_s8(source, res);
        source += 16;
        cnt--;
    }

    if (((length & 0xF) / 8) > 0)
    {
        const int16x8_t src = vldrbq_s16(source);
        int16x8_t res = vmaxq_s16(src, vdupq_n_s16((int16_t)act_min));
        res = vminq_s16(src, vdupq_n_s16((int16_t)act_max));
        vstrbq_s16(source, res);
        source += 8;
    }

    cnt = length & 7;
    while (cnt > 0)
    {
        int32_t comp = *source;
        comp = MAX(comp, act_min);
        comp = MIN(comp, act_max);
        *source++ = (int8_t)comp;
        cnt--;
    }

#else
    union arm_nnword in;
    int32_t cnt = length >> 2;

    while (cnt > 0l)
    {
        in.word = arm_nn_read_q7x4(source);

        in.bytes[0] = MAX(in.bytes[0], act_min);
        in.bytes[0] = MIN(in.bytes[0], act_max);
        in.bytes[1] = MAX(in.bytes[1], act_min);
        in.bytes[1] = MIN(in.bytes[1], act_max);
        in.bytes[2] = MAX(in.bytes[2], act_min);
        in.bytes[2] = MIN(in.bytes[2], act_max);
        in.bytes[3] = MAX(in.bytes[3], act_min);
        in.bytes[3] = MIN(in.bytes[3], act_max);

        write_q7x4_ia(&source, in.word);
        cnt--;
    }

    cnt = length & 0x3;
    while (cnt > 0l)
    {
        int32_t comp = *source;
        comp = MAX(comp, act_min);
        comp = MIN(comp, act_max);
        *source++ = (int8_t)comp;
        cnt--;
    }
#endif
}
#endif

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Pooling
 * @{
 */

/*
   * Optimized s8 max pooling function
   *
   * Refer to header file for details.
   *
   */

arm_status arm_max_pool_s8_opt(const uint16_t input_y,
                               const uint16_t input_x,
                               const uint16_t output_y,
                               const uint16_t output_x,
                               const uint16_t stride_y,
                               const uint16_t stride_x,
                               const uint16_t kernel_y,
                               const uint16_t kernel_x,
                               const uint16_t pad_y,
                               const uint16_t pad_x,
                               const int8_t act_min,
                               const int8_t act_max,
                               const uint16_t depth,
                               int8_t *src,
                               int16_t *tmp_buffer,
                               int8_t *dst)
{

#if defined(ARM_MATH_DSP)

    /* Run the following code for Cortex-M4 and Cortex-M7 */
    (void)tmp_buffer;
    int32_t i_x, i_y;

    /* first does the pooling along x axis */
    for (i_y = 0; i_y < input_y; i_y++)
    {

        for (i_x = 0; i_x < output_x; i_x++)
        {
            /* for each output sample */
            q7_t *target = src + (i_y * input_x + i_x) * depth;
            q7_t *win_start;
            q7_t *win_stop;
            const int32_t x_origin = i_x * stride_x - pad_x;
            if (x_origin < 0)
            {
                win_start = target;
            }
            else
            {
                win_start = src + (i_y * input_x + x_origin) * depth;
            }

            if (x_origin + kernel_x >= input_x)
            {
                win_stop = src + (i_y * input_x + input_x) * depth;
            }
            else
            {
                win_stop = src + (i_y * input_x + x_origin + kernel_x) * depth;
            }

            /* first step is to copy over initial data(along channel) along the channel in  x direction */
            memmove(target, win_start, depth);

            /* Move over to next element along x axis and compare with the base(target)  */
            win_start += depth;
            for (; win_start < win_stop; win_start += depth)
            {
                compare_and_replace_if_larger_q7(target, win_start, depth);
            }
        }
    }

    /* then does the pooling along y axis */
    for (i_y = 0; i_y < output_y; i_y++)
    {
        /* for each output row */
        q7_t *target = dst + i_y * output_x * depth;
        q7_t *row_start;
        q7_t *row_end;
        const int32_t y_origin = i_y * stride_y - pad_y;
        /* setting the starting row */
        if (y_origin < 0)
        {
            row_start = src;
        }
        else
        {
            row_start = src + y_origin * input_x * depth;
        }
        /* setting the stopping row */
        if (y_origin + kernel_y >= input_y)
        {
            row_end = src + input_y * input_x * depth;
        }
        else
        {
            row_end = src + (y_origin + kernel_y) * input_x * depth;
        }

        /* copy over the complete first row. */
        memmove(target, row_start, output_x * depth);

        /* move over to next row and compare with the base row (target)*/
        row_start += depth * input_x;

        for (; row_start < row_end; row_start += input_x * depth)
        {
            compare_and_replace_if_larger_q7(target, row_start, output_x * depth);
        }
    }

    clamp_output(dst, output_x * output_y * depth, act_min, act_max);

#else
    /* Pure C implementation */
    arm_max_pool_s8(input_y,
                    input_x,
                    output_y,
                    output_x,
                    stride_y,
                    stride_x,
                    kernel_y,
                    kernel_x,
                    pad_y,
                    pad_x,
                    act_min,
                    act_max,
                    depth,
                    src,
                    tmp_buffer,
                    dst);
#endif
  return ARM_MATH_SUCCESS;
}

/**
 * @} end of Pooling group
 */
