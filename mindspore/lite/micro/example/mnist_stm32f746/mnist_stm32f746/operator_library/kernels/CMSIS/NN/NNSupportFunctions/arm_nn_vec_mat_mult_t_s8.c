/*
 * Copyright (C) 2020 Arm Limited or its affiliates. All rights reserved.
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
 * Title:        arm_nn_vec_mat_mult_t_s8
 * Description:  s8 vector by matrix (transposed) multiplication
 *
 * $Date:        March 17, 2020
 * $Revision:    V.1.0.1
 *
 * Target Processor:  Cortex-M
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 * @ingroup groupSupport
 */

/**
 * @addtogroup NNBasicMath
 * @{
 */

/*
   * s8 vector by matrix (transposed) multiplication
   *
   * Refer header file for details.
   *
   */
arm_status arm_nn_vec_mat_mult_t_s8(const q7_t *lhs,
                                    const q7_t *rhs,
                                    const q31_t *bias,
                                    q7_t *dst,
                                    const int32_t lhs_offset,
                                    const int32_t rhs_offset,
                                    const int32_t dst_offset,
                                    const int32_t dst_multiplier,
                                    const int32_t dst_shift,
                                    const int32_t rhs_cols,
                                    const int32_t rhs_rows,
                                    const int32_t activation_min,
                                    const int32_t activation_max)
{
#if defined(ARM_MATH_DSP)
    const int32_t off0 = rhs_cols - 4;
    const int16_t lhs_offset_s16 = lhs_offset;
    const int16_t rhs_offset_s16 = rhs_offset;

    const uint32_t lhs_offset_s16x2 = __PKHBT(lhs_offset_s16, lhs_offset_s16, 16);
    const uint32_t rhs_offset_s16x2 = __PKHBT(rhs_offset_s16, rhs_offset_s16, 16);

    for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = *bias++;
        q31_t res01 = *bias++;

        int32_t rhs_cols_idx = 0;

        q31_t val0, val1, val2, val3, val4, val5;
        for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
        {
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);

            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);

            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            // Read 4 x int8 values from the RHS matrix
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);

            val2 = __SXTAB16(rhs_offset_s16x2, val0);
            // Read 4 x int8 values from the LHS vector
            val1 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val1);
            // Read 4 x int8 values from the RHS matrix
            val4 = arm_nn_read_q7x4((const q7_t *)rhs_ptr + off0);
            val1 = __SXTAB16(lhs_offset_s16x2, __ROR(val1, 8));

            // Perform the accumulations
            res00 = __SMLAD(val3, val2, res00);
            val5  = __SXTAB16(rhs_offset_s16x2, val4);
            res00 = __SMLAD(val1, val0, res00);
            val4  = __SXTAB16(rhs_offset_s16x2, __ROR(val4, 8));
            res01 = __SMLAD(val3, val5, res01);
            res01 = __SMLAD(val1, val4, res01);
        }

        for (; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            q31_t rhs_value1 = rhs_ptr[rhs_cols] + rhs_offset;
            q31_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);
        res01 = arm_nn_requantize(res01, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);

        *dst++ = (q7_t)res00;
        *dst++ = (q7_t)res01;

        rhs += 2 * rhs_cols;
    }

    if (rhs_rows % 2)
    {
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = *bias++;

        int32_t rhs_cols_idx = 0;

        q31_t val0, val1, val2, val3;
        for (; rhs_cols_idx <= (rhs_cols - 16); rhs_cols_idx += 16)
        {
            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);

            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);

            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);

            val0 = arm_nn_read_q7x4_ia((const q7_t **)&rhs_ptr);
            val1 = __SXTAB16(rhs_offset_s16x2, val0);
            val2 = arm_nn_read_q7x4_ia((const q7_t **)&lhs_ptr);
            val0 = __SXTAB16(rhs_offset_s16x2, __ROR(val0, 8));
            val3 = __SXTAB16(lhs_offset_s16x2, val2);
            val2 = __SXTAB16(lhs_offset_s16x2, __ROR(val2, 8));

            // Partial accumulations
            res00 = __SMLAD(val3, val1, res00);
            res00 = __SMLAD(val2, val0, res00);
        }

        for (; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            q31_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);

        *dst = (q7_t)res00;
    }

#else

    for (int32_t rhs_rows_idx = 0; rhs_rows_idx <= (rhs_rows - 2); rhs_rows_idx += 2)
    {
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = *bias++;
        q31_t res01 = *bias++;

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            q31_t rhs_value1 = rhs_ptr[rhs_cols] + rhs_offset;
            q31_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;
            res01 += lhs_value * rhs_value1;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);
        res01 = arm_nn_requantize(res01, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;
        res01 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);
        res01 = MAX(res01, activation_min);
        res01 = MIN(res01, activation_max);

        *dst++ = (q7_t)res00;
        *dst++ = (q7_t)res01;

        rhs += 2 * rhs_cols;
    }

    if (rhs_rows % 2)
    {
        const q7_t *lhs_ptr = &lhs[0];
        const q7_t *rhs_ptr = &rhs[0];

        q31_t res00 = *bias++;

        for (int32_t rhs_cols_idx = 0; rhs_cols_idx < rhs_cols; ++rhs_cols_idx)
        {
            q31_t rhs_value0 = rhs_ptr[0] + rhs_offset;
            q31_t lhs_value  = lhs_ptr[0] + lhs_offset;

            res00 += lhs_value * rhs_value0;

            ++rhs_ptr;
            ++lhs_ptr;
        }

        // Quantize down
        res00 = arm_nn_requantize(res00, dst_multiplier, dst_shift);

        // Add offset
        res00 += dst_offset;

        // Clamp the result
        res00 = MAX(res00, activation_min);
        res00 = MIN(res00, activation_max);

        *dst = (q7_t)res00;
    }
#endif

    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNBasicMath group
 */