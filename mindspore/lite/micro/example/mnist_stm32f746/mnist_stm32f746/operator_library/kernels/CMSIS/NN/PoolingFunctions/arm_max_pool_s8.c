/*
 * Copyright (C) 2010-2019 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_max_pool_s8.c
 * Description:  Pure C max pool implementation
 *
 * $Date:        August 2019
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include <arm_math.h>

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Pooling
 * @{
 */

arm_status arm_max_pool_s8(const uint16_t input_y,
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
                           const uint16_t channel_in,
                           int8_t *input,
                           int16_t *tmp_buffer,
                           int8_t *output)
{
    int32_t i_ch_in, i_out_x, i_out_y;
    int32_t i_ker_x, i_ker_y;
    (void)tmp_buffer;

    for (i_out_y = 0; i_out_y < output_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < output_x; i_out_x++)
        {
            for (i_ch_in = 0; i_ch_in < channel_in; i_ch_in++)
            {
                /* Native data type for inner loop variables  */
                int32_t max_val = (int8_t)Q7_MIN;
                /* Condition for kernel start dimension: (base_idx_<x,y> + ker_<x,y>_start) >= 0 */
                const int32_t base_idx_y = (i_out_y * stride_y) - pad_y;
                const int32_t base_idx_x = (i_out_x * stride_x) - pad_x;
                const int32_t ker_y_start = MAX(0, -base_idx_y);
                const int32_t ker_x_start = MAX(0, -base_idx_x);

                /* Condition for kernel end dimension: (base_idx_<x,y> + ker_<x,y>_end) < input_<x,y> */
                const int32_t ker_y_end = MIN(kernel_y, input_y - base_idx_y);
                const int32_t ker_x_end = MIN(kernel_x, input_x - base_idx_x);

                for (i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
                {
                    for (i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
                    {
                        const int32_t col_idx = base_idx_x + i_ker_x;
                        const int32_t row_idx = base_idx_y + i_ker_y;

                        max_val = MAX(input[(row_idx * input_x + col_idx) * channel_in + i_ch_in], max_val);
                    }
                }

                /* Activation function */
                max_val = MAX(max_val, act_min);
                max_val = MIN(max_val, act_max);

                output[i_ch_in + channel_in * (i_out_x + i_out_y * output_x)] = (int8_t)max_val;
            }
        }
    }
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of Pooling group
 */
