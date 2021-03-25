/*
 * Copyright (C) 2010-2020 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
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
 * Title:        arm_convolve_1_x_n_s8.c
 * Description:  s8 version of 1xN convolution using symmetric quantization.
 *
 * $Date:        February 27, 2020
 * $Revision:    V.1.0.1
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/*
   * 1xN s8 convolution function.
   *
   * Refer header file for details.
   *
   */

arm_status arm_convolve_1_x_n_s8(const q7_t *input,
                                 const uint16_t input_x,
                                 const uint16_t input_ch,
                                 const uint16_t input_batches,
                                 const q7_t *kernel,
                                 const uint16_t output_ch,
                                 const uint16_t kernel_x,
                                 const uint16_t pad_x,
                                 const uint16_t stride_x,
                                 const int32_t *bias,
                                 q7_t *output,
                                 const int32_t *output_shift,
                                 const int32_t *output_mult,
                                 const int32_t out_offset,
                                 const int32_t input_offset,
                                 const int32_t out_activation_min,
                                 const int32_t out_activation_max,
                                 const uint16_t output_x,
                                 q15_t *buffer_a)
{
    (void)input_batches;

    arm_status status = ARM_MATH_SUCCESS;
    if (output_x % 4 != 0)
    {
        status = ARM_MATH_SIZE_MISMATCH;
        goto out;
    }

#if defined(ARM_MATH_MVEI)
    for (int i_out_x = 0; i_out_x <= (output_x - 4); i_out_x += 4)
    {
        int32_t input_begin_idx[4];
        int32_t ker_begin_idx[4];
        int32_t ker_end_idx[4];

        for (int i = 0; i < 4; i++)
        {
            const int32_t est_input_x_idx = stride_x * (i_out_x + i) - pad_x;
            input_begin_idx[i] = MAX(0, est_input_x_idx);
            ker_begin_idx[i] = MAX(0, -est_input_x_idx);
            ker_end_idx[i] = MIN(kernel_x, input_x - est_input_x_idx);
        }

        for (int i_out_ch = 0; i_out_ch < output_ch; i_out_ch++)
        {
            int32x4_t s_offset;
            int32_t acc[4];
            if ((ker_begin_idx[0] != 0) || (ker_end_idx[3] != kernel_x))
            {
                int32_t sum_row[4];

                (void)arm_nn_mat_mul_core_1x_s8((ker_end_idx[0] - ker_begin_idx[0]) * input_ch,
                                                input + input_begin_idx[0] * input_ch,
                                                kernel + (input_ch * kernel_x * i_out_ch) + (ker_begin_idx[0] * input_ch),
                                                &sum_row[0],
                                                &acc[0]);
                (void)arm_nn_mat_mul_core_1x_s8((ker_end_idx[1] - ker_begin_idx[1]) * input_ch,
                                                input + input_begin_idx[1] * input_ch,
                                                kernel + (input_ch * kernel_x * i_out_ch) + (ker_begin_idx[1] * input_ch),
                                                &sum_row[1],
                                                &acc[1]);

                (void)arm_nn_mat_mul_core_1x_s8((ker_end_idx[2] - ker_begin_idx[2]) * input_ch,
                                                input + input_begin_idx[2] * input_ch,
                                                kernel + (input_ch * kernel_x * i_out_ch) + (ker_begin_idx[2] * input_ch),
                                                &sum_row[2],
                                                &acc[2]);

                (void)arm_nn_mat_mul_core_1x_s8((ker_end_idx[3] - ker_begin_idx[3]) * input_ch,
                                                input + input_begin_idx[3] * input_ch,
                                                kernel + (input_ch * kernel_x * i_out_ch) + (ker_begin_idx[3] * input_ch),
                                                &sum_row[3],
                                                &acc[3]);

                s_offset = vldrwq_s32(sum_row);
            }
            else
            {
                int32_t sum_row;
                (void)arm_nn_mat_mul_core_4x_s8(kernel_x * input_ch,
                                                stride_x * input_ch,
                                                input + input_begin_idx[0] * input_ch,
                                                kernel + (input_ch * kernel_x * i_out_ch),
                                                &sum_row,
                                                acc);

                s_offset = vdupq_n_s32(sum_row);
            }
            int32x4_t res = vldrwq_s32(acc);
            s_offset = vmulq_n_s32(s_offset, input_offset);

            res = vaddq_n_s32(res, bias[i_out_ch]);
            res = vaddq_s32(res, s_offset);
            res = arm_requantize_mve(res, output_mult[i_out_ch], output_shift[i_out_ch]);
            res = vaddq_n_s32(res, out_offset);

            res = vmaxq_s32(res, vdupq_n_s32(out_activation_min));
            res = vminq_s32(res, vdupq_n_s32(out_activation_max));

            const uint32x4_t scatter_offset = {0, output_ch, output_ch * 2, output_ch * 3};
            vstrbq_scatter_offset_s32(output, scatter_offset, res);
            output++;
        }
        output += (3 * output_ch);
    }

#else
#define DIM_Y (1)
    status = arm_convolve_s8(input, input_x, DIM_Y,
                             input_ch, 1, kernel, output_ch,
                             kernel_x, DIM_Y,
                             pad_x, 0,
                             stride_x, DIM_Y,
                             bias, output,
                             output_shift, output_mult,
                             out_offset, input_offset,
                             out_activation_min, out_activation_max,
                             output_x, DIM_Y,
                             buffer_a);
#endif

out:
    /* Return to application */
    return status;
}

int32_t arm_convolve_1_x_n_s8_get_buffer_size(const uint16_t input_ch,
                                              const uint16_t kernel_x,
                                              const uint16_t kernel_y)
{
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    return (2 * input_ch * kernel_x * kernel_y) * sizeof(int16_t);
#else
    (void)input_ch;
    (void)kernel_x;
    (void)kernel_y;
    return 0;
#endif
}

/**
 * @} end of NNConv group
 */
