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
 * Title:        arm_nnfunctions.h
 * Description:  Public header file for CMSIS NN Library
 *
 * $Date:        April 6, 2020
 * $Revision:    V.2.0.0
 *
 * Target Processor:  Cortex-M cores
 * -------------------------------------------------------------------- */

/**
   \mainpage CMSIS NN Software Library
   *
   * Introduction
   * ------------
   *
   * This user manual describes the CMSIS NN software library,
   * a collection of efficient neural network kernels developed to maximize the
   * performance and minimize the memory footprint of neural networks on Cortex-M processor cores.
   *
   * The library is divided into a number of functions each covering a specific category:
   * - Convolution Functions
   * - Activation Functions
   * - Fully-connected Layer Functions
   * - Pooling Functions
   * - Softmax Functions
   * - Basic math Functions
   *
   * The library has separate functions for operating on different weight and activation data
   * types including 8-bit integers (q7_t) and 16-bit integers (q15_t). The descrition of the
   * kernels are included in the function description. The implementation details are also
   * described in this paper [1].
   *
   * Function Classification
   * --------
   * The functions can be classified into two segments
   * - Legacy functions supporting ARM's internal symmetric quantization(8 bits).
   * - Functions that support TensorFlow Lite framework with symmetric quantization(8 bits).
   *
   * The legacy functions can be identified with their suffix of _q7 or _q15 and are no new development is done there. The article in [2] describes in detail
   * how to run a network using the legacy functions.
   *
   * The functions supporting TensorFlow Lite framework is identified by the _s8 suffix and can be invoked from TFL micro. The functions are bit exact to
   * TensorFlow Lite. Refer to the TensorFlow's documentation in [3] on how to run a TensorFlow Lite model using optimized CMSIS-NN kernels.
   *
   * Block Diagram
   * --------
   * \image html CMSIS-NN-OVERVIEW.PNG
   *
   * Examples
   * --------
   *
   * The library ships with a number of examples which demonstrate how to use the library functions.
   *
   * Pre-processor Macros
   * ------------
   *
   * Each library project have different pre-processor macros.
   *
   * - ARM_MATH_DSP:
   *
   * Define macro ARM_MATH_DSP, If the silicon supports DSP instructions(DSP extension).
   *
   * - ARM_MATH_MVEI:
   *
   * Define macro ARM_MATH_MVEI, If the silicon supports M-Profile Vector Extension.

   * - ARM_MATH_AUTOVECTORIZE
   *  Used in conjucture with ARM_MATH_MVEI to let the compiler auto vectorize for the functions that uses inline assembly.
   *  It does not affect functions that use C or intrinsics.
   * - ARM_MATH_BIG_ENDIAN:
   *
   * Define macro ARM_MATH_BIG_ENDIAN to build the library for big endian targets. This is supported only for the legacy functions i.e, functions targetted at
   * TensorFlow Lite do not support big endianness. By default library builds for little endian targets.
   *
   * - ARM_NN_TRUNCATE:
   *
   * Define macro ARM_NN_TRUNCATE to use floor instead of round-to-the-nearest-int for the computation.
   *
   * Upcoming Interface Change
   * --------
   * Starting from the 1.4.0 next release, CMSIS-NN will gradually switch to a new API interface to:
   *
   * -# have a stable API
   * -# avoid passing many variables by value
   * -# improve security
   * -# improve validation
   * -# improve code readability
   *
   * The upcoming API interface change will be based on "struct" and only affect the TensorFlowLite micro compliant APIs [4] (functions with _s8 suffix)
   *
   * Below you can find a snapshot of how the new API interface will look like (names can change)
   *
   * i.e. arm_convolve_1x1_s8_fast
   *
   * Current API interface | New API interface proposal
   * ------------- | -------------
   * const q7_t *input                | const cmsis_nn_context &ctx
   * const uint16_t input_x           | const cmsis_nn_conv_params &params
   * const uint16_t input_y           | const cmsis_nn_dims &input_dims
   * const uint16_t input_ch          | const q7_t *input_data
   * const uint16_t input_batches     | const cmsis_nn_dims &filter_dims
   * const q7_t *kernel               | const q7_t *filter_data
   * const uint16_t output_ch         | const cmsis_nn_dims &bias_dims
   * const uint16_t pad_x             | const q31_t *bias_data
   * const uint16_t pad_y             | const cmsis_nn_dims &output_dims
   * const uint16_t stride_x          | q7_t *output_data
   * const uint16_t stride_y          | <br>
   * const int32_t *bias              | <br>
   * q7_t *output                     | <br>
   * const int32_t *output_shift      | <br>
   * const int32_t *output_mult       | <br>
   * const int32_t out_offset         | <br>
   * const int32_t input_offset       | <br>
   * const int32_t out_activation_min | <br>
   * const int32_t out_activation_max | <br>
   * const uint16_t output_x          | <br>
   * const uint16_t output_y          | <br>
   * q15_t *buffer_a                  | <br>
   *
   * Copyright Notice
   * ------------
   *
   * Copyright (C) 2010-2019 Arm Limited. All rights reserved.
   *
   * [1] CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs https://arxiv.org/abs/1801.06601
   *
   * [2] Converting a Neural Network for Arm Cortex-M with CMSIS-NN
   *     https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/converting-a-neural-network-for-arm-cortex-m-with-cmsis-nn/single-page
   * [3] https://www.tensorflow.org/lite/microcontrollers/library
   *
   * [4] https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN#legacy-vs-tfl-micro-compliant-apis
   */

/**
 * @defgroup groupNN Neural Network Functions
 * A collection of functions to perform basic operations for neural network layers. Functions with a _s8 suffix support
 * TensorFlow Lite framework.
 */

#ifndef _ARM_NNFUNCTIONS_H
#define _ARM_NNFUNCTIONS_H

#include "arm_nnsupportfunctions.h"
#include "arm_nn_tables.h"

#define USE_INTRINSIC

//#define ARM_NN_TRUNCATE /* This config the rounding model to floor or round to the nearest int */

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @defgroup NNConv Convolution Functions
 *
 * Collection of convolution, depthwise convolution functions and their variants.
 *
 * The convolution is implemented in 2 steps: im2col and GEMM
 *
 * im2col is a process of converting each patch of image data into
 * a column. After im2col, the convolution is computed as matrix-matrix
 * multiplication.
 *
 * To reduce the memory footprint, the im2col is performed partially.
 * Each iteration, only a few column (i.e., patches) are generated and
 * computed with GEMM kernels similar to CMSIS-DSP arm_mat_mult functions.
 *
 */

/**
   * @brief Basic s8 convolution function
   * @param[in]       input           pointer to input tensor. Range: int8, format: [N,H,W,in_ch]
   * @param[in]       input_x         input tensor width
   * @param[in]       input_y         input tensor height
   * @param[in]       input_ch        number of input tensor channels
   * @param[in]       input_batches   number of input batches
   * @param[in]       kernel          pointer to kernel weights. Range: int8, format: [out_ch, H, W, in_ch]
   * @param[in]       output_ch       number of filters, i.e., output tensor channels
   * @param[in]       kernel_x        filter/kernel width
   * @param[in]       kernel_y        filter/kernel height
   * @param[in]       pad_x           padding along width
   * @param[in]       pad_y           padding along height
   * @param[in]       stride_x        convolution stride x
   * @param[in]       stride_y        convolution stride y
   * @param[in]       bias            pointer to per output channel bias. Range: int32
   * @param[in,out]   output          pointer to output tensor. format: [H, W, out_ch]
   * @param[in]       output_shift    pointer to per output channel requantization shift parameter.
   * @param[in]       output_mult     pointer to per output channel requantization multiplier parameter.
   * @param[in]       out_offset      output tensor offset. Range: int8
   * @param[in]       input_offset    input tensor offset. Range: int8
   * @param[in]       output_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_x    output tensor width
   * @param[in]       output_y    output tensor height
   * @param[in]       buffer_a    pointer to buffer space used for input optimization(partial im2col) and is necessary
   *                              when ARM_MATH_DSP is defined.
   *                              Required space: (2 * input_ch * kernel_x * kernel_y) * sizeof(q15_t) bytes
   *                              Use arm_convolve_s8_get_buffer_size() to get the size.
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   * @details
   *    1. Supported framework: TensorFlow Lite micro
   *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
   *    3. Additional memory is required for optimization. Refer to argument 'buffer_a' for details.
   *
   */
    arm_status arm_convolve_s8(const q7_t *input,
                               const uint16_t input_x,
                               const uint16_t input_y,
                               const uint16_t input_ch,
                               const uint16_t input_batches,
                               const q7_t *kernel,
                               const uint16_t output_ch,
                               const uint16_t kernel_x,
                               const uint16_t kernel_y,
                               const uint16_t pad_x,
                               const uint16_t pad_y,
                               const uint16_t stride_x,
                               const uint16_t stride_y,
                               const int32_t *bias,
                               q7_t *output,
                               const int32_t *output_shift,
                               const int32_t *output_mult,
                               const int32_t out_offset,
                               const int32_t input_offset,
                               const int32_t output_activation_min,
                               const int32_t output_activation_max,
                               const uint16_t output_x,
                               const uint16_t output_y,
                               q15_t *buffer_a);

  /**
   * @brief Get the required buffer size for s8 convolution function
   * @param[in]       input_ch              number of input tensor channels
   * @param[in]       kernel_x              filter/kernel width
   * @param[in]       kernel_y              filter/kernel height
   * @return          The function returns  required buffer size
   *
   */
    int32_t arm_convolve_s8_get_buffer_size(const uint16_t input_ch,
                                            const uint16_t kernel_x,
                                            const uint16_t kernel_y);

  /**
   * @brief Basic Q7 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */
    arm_status arm_convolve_HWC_q7_basic(const q7_t * Im_in,
                                         const uint16_t dim_im_in,
                                         const uint16_t ch_im_in,
                                         const q7_t * wt,
                                         const uint16_t ch_im_out,
                                         const uint16_t dim_kernel,
                                         const uint16_t padding,
                                         const uint16_t stride,
                                         const q7_t * bias,
                                         const uint16_t bias_shift,
                                         const uint16_t out_shift,
                                         q7_t * Im_out,
                                         const uint16_t dim_im_out,
                                         q15_t * bufferA,
                                         q7_t * bufferB);

  /**
   * @brief Basic Q7 convolution function (non-square shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   */
    arm_status arm_convolve_HWC_q7_basic_nonsquare(const q7_t * Im_in,
                                                  const uint16_t dim_im_in_x,
                                                  const uint16_t dim_im_in_y,
                                                  const uint16_t ch_im_in,
                                                  const q7_t * wt,
                                                  const uint16_t ch_im_out,
                                                  const uint16_t dim_kernel_x,
                                                  const uint16_t dim_kernel_y,
                                                  const uint16_t padding_x,
                                                  const uint16_t padding_y,
                                                  const uint16_t stride_x,
                                                  const uint16_t stride_y,
                                                  const q7_t * bias,
                                                  const uint16_t bias_shift,
                                                  const uint16_t out_shift,
                                                  q7_t * Im_out,
                                                  const uint16_t dim_im_out_x,
                                                  const uint16_t dim_im_out_y,
                                                  q15_t * bufferA,
                                                  q7_t * bufferB);

  /**
   * @brief Basic Q15 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */
    arm_status arm_convolve_HWC_q15_basic(const q15_t * Im_in,
                                          const uint16_t dim_im_in,
                                          const uint16_t ch_im_in,
                                          const q15_t * wt,
                                          const uint16_t ch_im_out,
                                          const uint16_t dim_kernel,
                                          const uint16_t padding,
                                          const uint16_t stride,
                                          const q15_t * bias,
                                          const uint16_t bias_shift,
                                          const uint16_t out_shift,
                                          q15_t * Im_out,
                                          const uint16_t dim_im_out,
                                          q15_t * bufferA,
                                          q7_t * bufferB);

  /**
   * @brief Fast Q7 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 4
   *   ch_im_out is multiple of 2
   */
    arm_status arm_convolve_HWC_q7_fast(const q7_t * Im_in,
                                        const uint16_t dim_im_in,
                                        const uint16_t ch_im_in,
                                        const q7_t * wt,
                                        const uint16_t ch_im_out,
                                        const uint16_t dim_kernel,
                                        const uint16_t padding,
                                        const uint16_t stride,
                                        const q7_t * bias,
                                        const uint16_t bias_shift,
                                        const uint16_t out_shift,
                                        q7_t * Im_out,
                                        const uint16_t dim_im_out,
                                        q15_t * bufferA,
                                        q7_t * bufferB);

  /**
   * @brief Fast Q7 convolution function (non-sqaure shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 4
   *   ch_im_out is multiple of 2
   */

    arm_status arm_convolve_HWC_q7_fast_nonsquare(const q7_t * Im_in,
                                                  const uint16_t dim_im_in_x,
                                                  const uint16_t dim_im_in_y,
                                                  const uint16_t ch_im_in,
                                                  const q7_t * wt,
                                                  const uint16_t ch_im_out,
                                                  const uint16_t dim_kernel_x,
                                                  const uint16_t dim_kernel_y,
                                                  const uint16_t padding_x,
                                                  const uint16_t padding_y,
                                                  const uint16_t stride_x,
                                                  const uint16_t stride_y,
                                                  const q7_t * bias,
                                                  const uint16_t bias_shift,
                                                  const uint16_t out_shift,
                                                  q7_t * Im_out,
                                                  const uint16_t dim_im_out_x,
                                                  const uint16_t dim_im_out_y,
                                                  q15_t * bufferA,
                                                  q7_t * bufferB);

  /**
   * @brief Fast Q7 version of 1x1 convolution (non-sqaure shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns either
   *                          <code>ARM_MATH_SIZE_MISMATCH</code> if argument constraints fail. or,
   *                          <code>ARM_MATH_SUCCESS</code> on successful completion.
   *
   * This function implement convolution with 1x1 kernel size (i.e., dim_kernel_x=1
   * and dim_kernel_y=1). It can be used for
   * second half of MobileNets after depthwise separable convolution.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 4
   *   ch_im_out is multiple of 2
   */
    arm_status arm_convolve_1x1_HWC_q7_fast_nonsquare(const q7_t * Im_in,
                                                      const uint16_t dim_im_in_x,
                                                      const uint16_t dim_im_in_y,
                                                      const uint16_t ch_im_in,
                                                      const q7_t * wt,
                                                      const uint16_t ch_im_out,
                                                      const uint16_t dim_kernel_x,
                                                      const uint16_t dim_kernel_y,
                                                      const uint16_t padding_x,
                                                      const uint16_t padding_y,
                                                      const uint16_t stride_x,
                                                      const uint16_t stride_y,
                                                      const q7_t * bias,
                                                      const uint16_t bias_shift,
                                                      const uint16_t out_shift,
                                                      q7_t * Im_out,
                                                      const uint16_t dim_im_out_x,
                                                      const uint16_t dim_im_out_y,
                                                      q15_t * bufferA,
                                                      q7_t * bufferB);

  /**
   * @brief Fast s8 version for 1x1 convolution (non-square shape)
   * @param[in]      input                pointer to input tensor.  Format: [N, H, W, in_ch]
   * @param[in]      input_x              input tensor dimension x
   * @param[in]      input_y              input tensor dimension y
   * @param[in]      input_ch             number of input tensor channels
   * @param[in]      input_batches        number of input batches
   * @param[in]      kernel               pointer to kernel weights. Format: [out_ch, H, W, in_ch]
   * @param[in]      output_ch            number of filters, i.e., output tensor channels
   * @param[in]      pad_x                padding size x
   * @param[in]      pad_y                padding size y
   * @param[in]      stride_x             convolution stride x
   * @param[in]      stride_y             convolution stride y
   * @param[in]      bias                 pointer to per channel bias. Range : int32
   * @param[in,out]  output               pointer to output tensor.  Format: [H, W, out_ch]
   * @param[in]      output_shift         pointer to per output channel requantization shift parameter.
   * @param[in]      output_mult          pointer to per output channel requantization multiplier parameter.
   * @param[in]      out_offset           output tensor offset. Range: int8
   * @param[in]      input_offset         input tensor offset. Range: -127 to 128
   * @param[in]      out_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]      out_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]      output_x             output tensor width
   * @param[in]      output_y             output tensor height
   * @param[in]      buffer_a             pointer to buffer space used if required by the implementation
   *                                      Use arm_convolve_1x1_s8_fast_get_buffer_size() to get the size
   * @return     The function returns either
   *                  <code>ARM_MATH_SIZE_MISMATCH</code> if argument constraints fail. or,
   *                  <code>ARM_MATH_SUCCESS</code> on successful completion.
   *
   * @details
   *   - Supported framework : TensorFlow Lite Micro
   *   - The following constrains on the arguments apply
   *      -# input_ch is a multiple of 4
   *      -# padding equals 0
   *      -# Stride equals 1
   *      -# kernel dimension is 1x1 (Not provided in the argument list)
   *
   */
    arm_status arm_convolve_1x1_s8_fast(const q7_t *input,
                                        const uint16_t input_x,
                                        const uint16_t input_y,
                                        const uint16_t input_ch,
                                        const uint16_t input_batches,
                                        const q7_t *kernel,
                                        const uint16_t output_ch,
                                        const uint16_t pad_x,
                                        const uint16_t pad_y,
                                        const uint16_t stride_x,
                                        const uint16_t stride_y,
                                        const int32_t *bias,
                                        q7_t *output,
                                        const int32_t *output_shift,
                                        const int32_t *output_mult,
                                        const int32_t out_offset,
                                        const int32_t input_offset,
                                        const int32_t out_activation_min,
                                        const int32_t out_activation_max,
                                        const uint16_t output_x,
                                        const uint16_t output_y,
                                        q15_t *buffer_a);

  /**
   * @brief Get the required buffer size for the fast 1x1 convolution
   * (non-square shape) s8 convolution function
   * @param[in]       input_ch              number of input tensor channels
   * @return          The function returns  required buffer size
   *
   */
    int32_t arm_convolve_1x1_s8_fast_get_buffer_size(const uint16_t input_ch);

  /**
   * @brief 1xn convolution
   * @param[in]      input                pointer to input tensor.  Format: [N, H, W, in_ch]
   * @param[in]      input_x              input tensor dimension x
   * @param[in]      input_ch             number of input tensor channels
   * @param[in]      input_batches        argument is not used.
   * @param[in]      kernel               pointer to kernel weights. Format: [out_ch, H, W, in_ch]
   * @param[in]      output_ch            number of filters, i.e., output tensor channels
   * @param[in]      kernel_x             kernel width along x
   * @param[in]      pad_x                padding along x
   * @param[in]      stride_x             stride along x
   * @param[in]      bias                 pointer to per channel bias. Range : int32
   * @param[out]     output               pointer to output tensor.  Format: [H, W, out_ch]
   * @param[in]      output_shift         pointer to per output channel requantization shift parameter.
   * @param[in]      output_mult          pointer to per output channel requantization multiplier parameter.
   * @param[in]      out_offset           output tensor offset. Range: int8
   * @param[in]      input_offset         input tensor offset. Range: -127 to 128
   * @param[in]      out_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]      out_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]      output_x             output tensor width
   * @param[in]      buffer_a             pointer to buffer space used for input optimization and is necessary
   *                                      when ARM_MATH_DSP is defined but not ARM_MATH_MVEI.
   *                                      Required space: 2 * input_ch * sizeof(q15_t) bytes
   *                                      Use arm_convolve_1_x_n_s8_get_buffer_size() to get the size
   * @return     The function returns either
   *                  <code>ARM_MATH_SIZE_MISMATCH</code> if argument constraints fail. or,
   *                  <code>ARM_MATH_SUCCESS</code> on successful completion.
   *
   * @details
   *   - Supported framework : TensorFlow Lite Micro
   *   - The following constrains on the arguments apply
   *      -# input_batches equals 1
   *      -# ouput_x is a multiple of 4
   *      -# Explicit constraints(since it is for 1xN convolution)
   *      -## input_y equals 1
   *      -## output_y equals 1
   *      -## kernel_y equals 1
   *@todo  Remove constraint on output_x to make the function generic.
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
                                    q15_t *buffer_a);


  /**
   * @brief Get the required additional buffer size for 1xn convolution
   *
   * @param[in]       input_ch              number of input tensor channels
   * @param[in]       kernel_x              filter/kernel width
   * @param[in]       kernel_y              filter/kernel height
   * @return          The function returns  required buffer size(bytes)
   *
   */
    int32_t arm_convolve_1_x_n_s8_get_buffer_size(const uint16_t input_ch,
                                                  const uint16_t kernel_x,
                                                  const uint16_t kernel_y);


  /**
   * @brief Q7 version of convolution for RGB image
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This kernel is written exclusively for convolution with ch_im_in
   * equals 3. This applies on the first layer of CNNs which has input
   * image with RGB format.
   */

    arm_status arm_convolve_HWC_q7_RGB(const q7_t * Im_in,
                                       const uint16_t dim_im_in,
                                       const uint16_t ch_im_in,
                                       const q7_t * wt,
                                       const uint16_t ch_im_out,
                                       const uint16_t dim_kernel,
                                       const uint16_t padding,
                                       const uint16_t stride,
                                       const q7_t * bias,
                                       const uint16_t bias_shift,
                                       const uint16_t out_shift,
                                       q7_t * Im_out,
                                       const uint16_t dim_im_out,
                                       q15_t * bufferA,
                                       q7_t * bufferB);

  /**
   * @brief Fast Q15 convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 2
   *   ch_im_out is multiple of 2
   */

    arm_status arm_convolve_HWC_q15_fast(const q15_t * Im_in,
                                         const uint16_t dim_im_in,
                                         const uint16_t ch_im_in,
                                         const q15_t * wt,
                                         const uint16_t ch_im_out,
                                         const uint16_t dim_kernel,
                                         const uint16_t padding,
                                         const uint16_t stride,
                                         const q15_t * bias,
                                         const uint16_t bias_shift,
                                         const uint16_t out_shift,
                                         q15_t * Im_out,
                                         const uint16_t dim_im_out,
                                         q15_t * bufferA,
                                         q7_t * bufferB);

  /**
   * @brief Fast Q15 convolution function (non-sqaure shape)
   * @param[in]       Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       wt           pointer to kernel weights
   * @param[in]       ch_im_out    number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_y     convolution stride y
   * @param[in]       bias         pointer to bias
   * @param[in]       bias_shift   amount of left-shift for bias
   * @param[in]       out_shift    amount of right-shift for output
   * @param[in,out]   Im_out       pointer to output tensor
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   bufferB      pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size: 2*ch_im_in*dim_kernel*dim_kernel
   *
   * bufferB size: 0
   *
   * <b>Input dimension constraints:</b>
   *
   * ch_im_in is multiple of 2
   *
   * ch_im_out is multipe of 2
   *
   */

    arm_status
    arm_convolve_HWC_q15_fast_nonsquare(const q15_t * Im_in,
                              const uint16_t dim_im_in_x,
                              const uint16_t dim_im_in_y,
                              const uint16_t ch_im_in,
                              const q15_t * wt,
                              const uint16_t ch_im_out,
                              const uint16_t dim_kernel_x,
                              const uint16_t dim_kernel_y,
                              const uint16_t padding_x,
                              const uint16_t padding_y,
                              const uint16_t stride_x,
                              const uint16_t stride_y,
                              const q15_t * bias,
                              const uint16_t bias_shift,
                              const uint16_t out_shift,
                              q15_t * Im_out,
                              const uint16_t dim_im_out_x,
                              const uint16_t dim_im_out_y,
                              q15_t * bufferA,
                              q7_t * bufferB);

  /**
   * @brief Q7 depthwise separable convolution function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       wt          pointer to kernel weights
   * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       bias        pointer to bias
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in,out]   Im_out      pointer to output tensor
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   bufferB     pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 2
   *   ch_im_out is multiple of 2
   */

    arm_status arm_depthwise_separable_conv_HWC_q7(const q7_t * Im_in,
                                                   const uint16_t dim_im_in,
                                                   const uint16_t ch_im_in,
                                                   const q7_t * wt,
                                                   const uint16_t ch_im_out,
                                                   const uint16_t dim_kernel,
                                                   const uint16_t padding,
                                                   const uint16_t stride,
                                                   const q7_t * bias,
                                                   const uint16_t bias_shift,
                                                   const uint16_t out_shift,
                                                   q7_t * Im_out,
                                                   const uint16_t dim_im_out,
                                                   q15_t * bufferA,
                                                   q7_t * bufferB);

  /**
   * @brief Q7 depthwise separable convolution function (non-square shape)
   * @param[in]       Im_in         pointer to input tensor
   * @param[in]       dim_im_in_x   input tensor dimension x
   * @param[in]       dim_im_in_y   input tensor dimension y
   * @param[in]       ch_im_in      number of input tensor channels
   * @param[in]       wt            pointer to kernel weights
   * @param[in]       ch_im_out     number of filters, i.e., output tensor channels
   * @param[in]       dim_kernel_x  filter kernel size x
   * @param[in]       dim_kernel_y  filter kernel size y
   * @param[in]       padding_x     padding sizes x
   * @param[in]       padding_y     padding sizes y
   * @param[in]       stride_x      convolution stride x
   * @param[in]       stride_y      convolution stride y
   * @param[in]       bias          pointer to bias
   * @param[in]       bias_shift    amount of left-shift for bias
   * @param[in]       out_shift     amount of right-shift for output
   * @param[in,out]   Im_out        pointer to output tensor
   * @param[in]       dim_im_out_x  output tensor dimension x
   * @param[in]       dim_im_out_y  output tensor dimension y
   * @param[in,out]   bufferA       pointer to buffer space for input
   * @param[in,out]   bufferB       pointer to buffer space for output
   * @return     The function returns either
   * <code>ARM_MATH_SIZE_MISMATCH</code> or <code>ARM_MATH_SUCCESS</code> based on the outcome of size checking.
   *
   * This function is the version with full list of optimization tricks, but with
   * some contraints:
   *   ch_im_in is multiple of 2
   *   ch_im_out is multiple of 2
   */
    arm_status arm_depthwise_separable_conv_HWC_q7_nonsquare(const q7_t * Im_in,
                                                             const uint16_t dim_im_in_x,
                                                             const uint16_t dim_im_in_y,
                                                             const uint16_t ch_im_in,
                                                             const q7_t * wt,
                                                             const uint16_t ch_im_out,
                                                             const uint16_t dim_kernel_x,
                                                             const uint16_t dim_kernel_y,
                                                             const uint16_t padding_x,
                                                             const uint16_t padding_y,
                                                             const uint16_t stride_x,
                                                             const uint16_t stride_y,
                                                             const q7_t * bias,
                                                             const uint16_t bias_shift,
                                                             const uint16_t out_shift,
                                                             q7_t * Im_out,
                                                             const uint16_t dim_im_out_x,
                                                             const uint16_t dim_im_out_y,
                                                             q15_t * bufferA,
                                                             q7_t * bufferB);

/**
   * @brief Basic s8 depthwise convolution function
   * @param[in]       input      pointer to input tensor. Range: int8, format: [H,W,in_ch]
   * @param[in]       input_x    input tensor width
   * @param[in]       input_y    input tensor height
   * @param[in]       input_ch   number of input tensor channels
   * @param[in]       kernel     pointer to kernel weights. Range: int8, format: [in_ch, H, W, out_ch]
   * @param[in]       output_ch  Number of output channels. output_ch = ch_mult * input_ch
   * @param[in]       ch_mult    channel multiplier.
   * @param[in]       kernel_x   filter/kernel width
   * @param[in]       kernel_y   filter/kernel height
   * @param[in]       pad_x      padding along width
   * @param[in]       pad_y      padding along height
   * @param[in]       stride_x   convolution stride along width
   * @param[in]       stride_y   convolution stride along height
   * @param[in]       bias       pointer to per output channel bias. Range: int32
   * @param[in,out]   output     pointer to output tensor. Format: [H, W, out_ch]
   * @param[in]       output_shift pointer to per output channel requantization shift parameter.
   * @param[in]       output_mult  pointer to per output channel requantization multiplier parameter.
   * @param[in]       output_x     output tensor width
   * @param[in]       output_y     output tensor height
   * @param[in]       output_offset   offset to elements of output tensor. Range: int8
   * @param[in]       input_offset    offset to elements of input tensor. Range: -127 to 128
   * @param[in]       output_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]       dilation_x   dilation along x. Not used. Dilation factor of 1 is used.
   * @param[in]       dilation_y   dilation along y. Not used. Dilation factor of 1 is used.
   * @param[in]       buffer_a     Not used.
   *
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   * @details
   *    1. Supported framework: TensorFlow Lite
   *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
   *    3. Optimization using DSP extension is not available for the generic case where channel multiplier is > 1.
   *
   */

    arm_status arm_depthwise_conv_s8(const q7_t *input,
                                     const uint16_t input_x,
                                     const uint16_t input_y,
                                     const uint16_t input_ch,
                                     const q7_t *kernel,
                                     const uint16_t output_ch,
                                     const uint16_t ch_mult,
                                     const uint16_t kernel_x,
                                     const uint16_t kernel_y,
                                     const uint16_t pad_x,
                                     const uint16_t pad_y,
                                     const uint16_t stride_x,
                                     const uint16_t stride_y,
                                     const int32_t *bias,
                                     q7_t *output,
                                     const int32_t *output_shift,
                                     const int32_t *output_mult,
                                     const uint16_t output_x,
                                     const uint16_t output_y,
                                     const int32_t output_offset,
                                     const int32_t input_offset,
                                     const int32_t output_activation_min,
                                     const int32_t output_activation_max,
                                     const uint16_t dilation_x,
                                     const uint16_t dilation_y,
                                     q15_t *buffer_a);

/**
   * @brief Optimized s8 depthwise convolution function for 3x3 kernel size with constraint that in_channel equals out_channel
   * @param[in]       input      pointer to input tensor. Range: int8, format: [H,W,in_ch]
   * @param[in]       input_x    input tensor width
   * @param[in]       input_y    input tensor height
   * @param[in]       input_ch   number of input tensor channels
   * @param[in]       kernel     pointer to kernel weights. Range: int8, Format: [in_ch, H, W, out_ch]
   * @param[in]       output_ch  Number of output channels.
   * @param[in]       pad_x      padding along width
   * @param[in]       pad_y      padding along height
   * @param[in]       stride_x   convolution stride along width
   * @param[in]       stride_y   convolution stride along height
   * @param[in]       bias       pointer to per output channel bias. Range: int8
   * @param[in,out]   output     pointer to output tensor. Format: [H, W, out_ch]
   * @param[in]       output_shift pointer to per output channel requantization shift parameter.
   * @param[in]       output_mult  pointer to per output channel requantization multiplier parameter.
   * @param[in]       output_x     output tensor width
   * @param[in]       output_y     output tensor height
   * @param[in]       output_offset   offset to elements of output tensor
   * @param[in]       input_offset    offset to elements of input tensor
   * @param[in]       output_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]       dilation_x   dilation along x. Not used. Dilation factor of 1 is used.
   * @param[in]       dilation_y   dilation along y. Not used. Dilation factor of 1 is used.
   * @param[in]       buffer_a     Buffer for partial im2col optimization. Not used.
   *
   * @return     The function returns one of the following
   *                <code>ARM_MATH_SIZE_MISMATCH</code> - Unsupported dimension of tensors
   *                <code>ARM_MATH_ARGUMENT_ERROR</code> - Unsupported pad size along the x axis
   *                <code>ARM_MATH_SUCCESS</code> - Successful operation
   *
   * @details
   *    Supported framework: TensorFlow Lite
   *
   */
  arm_status arm_depthwise_conv_3x3_s8(const int8_t *input,
                                       const int32_t input_x,
                                       const int32_t input_y,
                                       const int32_t input_ch,
                                       const int8_t *kernel,
                                       const int32_t output_ch,
                                       const int32_t pad_x,
                                       const int32_t pad_y,
                                       const int32_t stride_x,
                                       const int32_t stride_y,
                                       const int32_t *bias,
                                       int8_t *output,
                                       const int32_t *output_shift,
                                       const int32_t *output_mult,
                                       const int32_t output_x,
                                       const int32_t output_y,
                                       const int32_t output_offset,
                                       const int32_t input_offset,
                                       const int32_t output_activation_min,
                                       const int32_t output_activation_max,
                                       const int32_t dilation_x,
                                       const int32_t dilation_y,
                                       int16_t *buffer_a);

/**
   * @brief Optimized s8 depthwise convolution function with constraint that in_channel equals out_channel
   * @param[in]       input      pointer to input tensor. Range: int8, format: [H,W,in_ch]
   * @param[in]       input_x    input tensor width
   * @param[in]       input_y    input tensor height
   * @param[in]       input_ch   number of input tensor channels
   * @param[in]       kernel     pointer to kernel weights. Range: int8, Format: [in_ch, H, W, out_ch]
   * @param[in]       output_ch  Number of output channels.
   * @param[in]       kernel_x   filter/kernel width
   * @param[in]       kernel_y   filter/kernel height
   * @param[in]       pad_x      padding along width
   * @param[in]       pad_y      padding along height
   * @param[in]       stride_x   convolution stride along width
   * @param[in]       stride_y   convolution stride along height
   * @param[in]       bias       pointer to per output channel bias. Range: int8
   * @param[in,out]   output     pointer to output tensor. Format: [H, W, out_ch]
   * @param[in]       output_shift pointer to per output channel requantization shift parameter.
   * @param[in]       output_mult  pointer to per output channel requantization multiplier parameter.
   * @param[in]       output_x     output tensor width
   * @param[in]       output_y     output tensor height
   * @param[in]       output_offset   offset to elements of output tensor
   * @param[in]       input_offset    offset to elements of input tensor
   * @param[in]       output_activation_min   Minimum value to clamp the output to. Range: int8
   * @param[in]       output_activation_max   Minimum value to clamp the output to. Range: int8
   * @param[in]       dilation_x   dilation along x. Not used. Dilation factor of 1 is used.
   * @param[in]       dilation_y   dilation along y. Not used. Dilation factor of 1 is used.
   * @param[in]       buffer_a     Buffer for partial im2col optimization. This is mandatory when
   *                               ARM_MATH_DSP is defined.
   *                               Required space: (2 * input_ch * kernel_x * kernel_y) * sizeof(q15_t) bytes
   *                               Use arm_depthwise_conv_s8_opt_get_buffer_size() to get the size.
   *
   * @return     The function returns one of the following
   *                <code>ARM_MATH_SIZE_MISMATCH</code> - Unsupported dimension of tensors
   *                <code>ARM_MATH_SUCCESS</code> - Successful operation
   *
   * @note       If number of channels is not a multiple of 4, upto 3 elements outside the boundary will be read out
   *             for the following if MVE optimizations(Arm Helium Technology) are used.
   *               - Output shift
   *               - Output multiplier
   *               - Output bias
   *               - kernel
   *
   * @details
   *    1. Supported framework: TensorFlow Lite
   *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
   *    3. Reccomended when number of channels is 4 or greater.
   *
   */
  arm_status arm_depthwise_conv_s8_opt(const q7_t *input,
                                       const uint16_t input_x,
                                       const uint16_t input_y,
                                       const uint16_t input_ch,
                                       const q7_t *kernel,
                                       const uint16_t output_ch,
                                       const uint16_t kernel_x,
                                       const uint16_t kernel_y,
                                       const uint16_t pad_x,
                                       const uint16_t pad_y,
                                       const uint16_t stride_x,
                                       const uint16_t stride_y,
                                       const int32_t *bias,
                                       q7_t *output,
                                       const int32_t *output_shift,
                                       const int32_t *output_mult,
                                       const uint16_t output_x,
                                       const uint16_t output_y,
                                       const int32_t output_offset,
                                       const int32_t input_offset,
                                       const int32_t output_activation_min,
                                       const int32_t output_activation_max,
                                       const uint16_t dilation_x,
                                       const uint16_t dilation_y,
                                       q15_t *buffer_a);

  /**
   * @brief Get the required buffer size for optimized s8 depthwise convolution
   * function with constraint that in_channel equals out_channel.
   * @param[in]       input_ch              number of input tensor channels
   * @param[in]       kernel_x              filter/kernel width
   * @param[in]       kernel_y              filter/kernel height
   * @return          The function returns  required buffer size
   *
   */
int32_t arm_depthwise_conv_s8_opt_get_buffer_size(const uint16_t input_ch,
                                                  const uint16_t kernel_x,
                                                  const uint16_t kernel_y);

/**
 * @defgroup FC Fully-connected Layer Functions
 *
 * Collection of fully-connected and matrix multiplication functions.
 *
 * Fully-connected layer is basically a matrix-vector multiplication
 * with bias. The matrix is the weights and the input/output vectors
 * are the activation values. Supported {weight, activation} precisions
 * include {8-bit, 8-bit}, {16-bit, 16-bit}, and {8-bit, 16-bit}.
 *
 * Here we have two types of kernel functions. The basic function
 * implements the function using regular GEMV approach. The opt functions
 * operates with weights in interleaved formats.
 *
 */

  /**
   * @brief Q7 basic fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q7(const q7_t * pV,
                                      const q7_t * pM,
                                      const uint16_t dim_vec,
                                      const uint16_t num_of_rows,
                                      const uint16_t bias_shift,
                                      const uint16_t out_shift,
                                      const q7_t * bias,
                                      q7_t * pOut,
                                      q15_t * vec_buffer);

  /**
   * @brief S8 basic fully-connected and matrix multiplication layer function for TF Lite
   * @param[in]       pInput                       pointer to pInput vector
   * @param[in]       pWeight                      pointer to matrix weights
   * @param[in]       col_dim                      dimension of the input vector
   * @param[in]       row_dim                      dimension of the output vector
   * @param[in]       nb_batches                   number of batches
   * @param[in]       input_offset                 tensor offset for input. Range: -127 to 128
   * @param[in]       filter_offset                tensor offset for filter. Range: -127 to 128
   * @param[in]       out_mult                     requantization parameter
   * @param[in]       out_shift                    requantization parameter
   * @param[in]       output_offset                tensor offset for output. Range: int8
   * @param[in]       pBias                        pointer to bias
   * @param[out]      pOut                         pointer to output vector
   * @param[in]       output_activation_min        for clamping
   * @param[in]       output_activation_max        for clamping
   * @param[in]       vec_buffer                   pointer to buffer space used for optimization and is necessary
   *                                               when ARM_MATH_DSP is defined but not
   *                                               ARM_MATH_MVEI.
   *                                               Required space: col_dim * sizeof(q15_t) bytes
   *                                               Use arm_fully_connected_s8_get_buffer_size() to get the size.
   * @return          The function returns         ARM_MATH_SUCCESS
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * vec_buffer size: col_dim of word16.
   *
   *  This basic function is designed to work with regular pWeight
   *  matrix without interleaving.
   *
   *    1. Supported framework: TensorFlow Lite
   *    2. q7 is used as data type eventhough it is s8 data. It is done so to be consistent with existing APIs.
   *
   */

    arm_status
    arm_fully_connected_s8(const int8_t *pInput,
                           const int8_t *pWeight,
                           const uint16_t col_dim,
                           const uint16_t row_dim,
                           const uint16_t nb_batches,
                           const int32_t input_offset,
                           const int32_t filter_offset,
                           const int32_t out_mult,
                           const int32_t out_shift,
                           const int32_t output_offset,
                           const int32_t *pBias,
                           int8_t *pOut,
                           const int32_t output_activation_min,
                           const int32_t output_activation_max,
                           q15_t *vec_buffer);

  /**
   * @brief Get the required buffer size for S8 basic fully-connected and
   * matrix multiplication layer function for TF Lite
   * @param[in]       col_dim                      dimension of the input vector
   * @return          The function returns         required buffer size
   *
   */
    int32_t arm_fully_connected_s8_get_buffer_size(const uint16_t col_dim);

  /**
   * @brief Q7 opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q7_opt(const q7_t * pV,
                                          const q7_t * pM,
                                          const uint16_t dim_vec,
                                          const uint16_t num_of_rows,
                                          const uint16_t bias_shift,
                                          const uint16_t out_shift,
                                          const q7_t * bias,
                                          q7_t * pOut,
                                          q15_t * vec_buffer);

  /**
   * @brief Q15 basic fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q15(const q15_t * pV,
                                       const q15_t * pM,
                                       const uint16_t dim_vec,
                                       const uint16_t num_of_rows,
                                       const uint16_t bias_shift,
                                       const uint16_t out_shift,
                                       const q15_t * bias,
                                       q15_t * pOut,
                                       q15_t * vec_buffer);

  /**
   * @brief Q15 opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_q15_opt(const q15_t * pV,
                                           const q15_t * pM,
                                           const uint16_t dim_vec,
                                           const uint16_t num_of_rows,
                                           const uint16_t bias_shift,
                                           const uint16_t out_shift,
                                           const q15_t * bias,
                                           q15_t * pOut,
                                           q15_t * vec_buffer);

  /**
   * @brief Mixed Q15-Q7 fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_mat_q7_vec_q15(const q15_t * pV,
                                                  const q7_t * pM,
                                                  const uint16_t dim_vec,
                                                  const uint16_t num_of_rows,
                                                  const uint16_t bias_shift,
                                                  const uint16_t out_shift,
                                                  const q7_t * bias,
                                                  q15_t * pOut,
                                                  q15_t * vec_buffer);

  /**
   * @brief Mixed Q15-Q7 opt fully-connected layer function
   * @param[in]       pV          pointer to input vector
   * @param[in]       pM          pointer to matrix weights
   * @param[in]       dim_vec     length of the vector
   * @param[in]       num_of_rows number of rows in weight matrix
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        pointer to bias
   * @param[in,out]   pOut        pointer to output vector
   * @param[in,out]   vec_buffer  pointer to buffer space for input
   * @return     The function returns <code>ARM_MATH_SUCCESS</code>
   *
   */

    arm_status arm_fully_connected_mat_q7_vec_q15_opt(const q15_t * pV,
                                                      const q7_t * pM,
                                                      const uint16_t dim_vec,
                                                      const uint16_t num_of_rows,
                                                      const uint16_t bias_shift,
                                                      const uint16_t out_shift,
                                                      const q7_t * bias,
                                                      q15_t * pOut,
                                                      q15_t * vec_buffer);

/**
 * @brief Matrix-Multiplication Kernels for Convolution
 *
 * These functions are used within convolution layer functions for
 * matrix multiplication.
 *
 * The implementation is similar to CMSIS-DSP arm_mat_mult functions
 * with one Q7 and one Q15 operands. The Q15 operand is the im2col
 * output which is always with 2 columns.
 *
 */

   /**
   * @brief Matrix-multiplication function for convolution
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always conssists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   */

    q7_t     *arm_nn_mat_mult_kernel_q7_q15(const q7_t * pA,
                                            const q15_t * pInBuffer,
                                            const uint16_t ch_im_out,
                                            const uint16_t numCol_A,
                                            const uint16_t bias_shift,
                                            const uint16_t out_shift,
                                            const q7_t * bias,
                                            q7_t * pOut);
   /**
   * @brief Matrix-multiplication function for convolution with per-channel requantization.
   * @param[in]       input_a     pointer to operand A
   * @param[in]       input_b     pointer to operand B, always consists of 2 vectors.
   * @param[in]       output_ch   number of rows of A
   * @param[in]       out_shift  pointer to per output channel requantization shift parameter.
   * @param[in]       out_mult   pointer to per output channel requantization multiplier parameter.
   * @param[in]       out_offset      output tensor offset.
   * @param[in]       activation_min   minimum value to clamp the output to. Range : int8
   * @param[in]       activation_max   maximum value to clamp the output to. Range : int8
   * @param[in]       num_col_a   number of columns of A
   * @param[in]       output_bias per output channel bias. Range : int32
   * @param[in,out]   out_0       pointer to output
   * @return     The function returns one of the two
   *              1. The incremented output pointer for a successful operation or
   *              2. NULL if implementation is not available.
   *
   * @details   This function does the matrix multiplication of weight matrix for all output channels
   *            with 2 columns from im2col and produces two elements/output_channel. The outputs are
   *            clamped in the range provided by activation min and max.
   *            Supported framework: TensorFlow Lite micro.
   */
    q7_t *arm_nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                        const q15_t *input_b,
                                        const uint16_t output_ch,
                                        const int32_t *out_shift,
                                        const int32_t *out_mult,
                                        const int32_t out_offset,
                                        const int16_t activation_min,
                                        const int16_t activation_max,
                                        const uint16_t num_col_a,
                                        const int32_t *const output_bias,
                                        q7_t *out_0);

   /**
   * @brief Matrix-multiplication of re-ordered input B with A.
   *
   * @details  For arguments, refer arm_nn_mat_mult_kernel_s8_s16. The re-ordering is a consequence
   *           of sign extension done by the SXTB16 command on input_b. The outputs are clamped in the range
   *           provided by activation min and max.
   *   * @details
   *   - Supported framework : TensorFlow Lite Micro
   *   - The following constrains on the arguments apply
   *      -# num_col_a is a multiple of 4
   *      -# output_ch is a multiple of 2
   *
   */
    q7_t *arm_nn_mat_mult_kernel_s8_s16_reordered(const q7_t *input_a,
                                                  const q15_t *input_b,
                                                  const uint16_t output_ch,
                                                  const int32_t *out_shift,
                                                  const int32_t *out_mult,
                                                  const int32_t out_offset,
                                                  const int16_t activation_min,
                                                  const int16_t activation_max,
                                                  const uint16_t num_col_a,
                                                  const int32_t *const output_bias,
                                                  q7_t *out_0);

    /**
   * @brief Matrix-multiplication function for convolution with reordered columns
   * @param[in]       pA          pointer to operand A
   * @param[in]       pInBuffer   pointer to operand B, always conssists of 2 vectors
   * @param[in]       ch_im_out   numRow of A
   * @param[in]       numCol_A    numCol of A
   * @param[in]       bias_shift  amount of left-shift for bias
   * @param[in]       out_shift   amount of right-shift for output
   * @param[in]       bias        the bias
   * @param[in,out]   pOut        pointer to output
   * @return     The function returns the incremented output pointer
   *
   * @details  This function assumes that data in pInBuffer are reordered
   */
    q7_t     *arm_nn_mat_mult_kernel_q7_q15_reordered(const q7_t * pA,
                                                      const q15_t * pInBuffer,
                                                      const uint16_t ch_im_out,
                                                      const uint16_t numCol_A,
                                                      const uint16_t bias_shift,
                                                      const uint16_t out_shift,
                                                      const q7_t * bias,
                                                      q7_t * pOut);

#ifdef __cplusplus
}
#endif

/*
 *  Other functions
 *  These layers are typically not timing critical
 *  Basic implementation is supported here
 */

#ifdef __cplusplus
extern    "C"
{
#endif

/**
 * @defgroup BasicMath Basic math functions
 *
 * Element wise add and multiplication functions.
 *
 */

/**
   * @brief s8 element wise add of two vectors
   * @param[in]       input_1_vect            pointer to input vector 1
   * @param[in]       input_2_vect            pointer to input vector 2
   * @param[in]       input_1_offset          offset for input 1. Range: Range: -127 to 128
   * @param[in]       input_1_mult            multiplier for input 1
   * @param[in]       input_1_shift           shift for input 1
   * @param[in]       input_2_offset          offset for input 2. Range: Range: -127 to 128
   * @param[in]       input_2_mult            multiplier for input 2
   * @param[in]       input_2_shift           shift for input 2
   * @param[in]       left_shift              input left shift
   * @param[in,out]   output                  pointer to output vector
   * @param[in]       out_offset              output offset
   * @param[in]       out_mult                output multiplier
   * @param[in]       out_shift               output shift
   * @param[in]       out_activation_min      minimum value to clamp output to
   * @param[in]       out_activation_max      maximum value to clamp output to
   * @param[in]       block_size              number of samples
   * @return          The function returns    ARM_MATH_SUCCESS
   */
    arm_status arm_elementwise_add_s8(const int8_t *input_1_vect,
                                      const int8_t *input_2_vect,
                                      const int32_t input_1_offset,
                                      const int32_t input_1_mult,
                                      const int32_t input_1_shift,
                                      const int32_t input_2_offset,
                                      const int32_t input_2_mult,
                                      const int32_t input_2_shift,
                                      const int32_t left_shift,
                                      int8_t *output,
                                      const int32_t out_offset,
                                      const int32_t out_mult,
                                      const int32_t out_shift,
                                      const int32_t out_activation_min,
                                      const int32_t out_activation_max,
                                      const uint32_t block_size);

/**
   * @brief s8 element wise multiplication
   * @param[in]       input_1_vect            pointer to input vector 1
   * @param[in]       input_2_vect            pointer to input vector 2
   * @param[in]       input_1_offset          offset for input 1. Range: Range: -127 to 128
   * @param[in]       input_2_offset          offset for input 2. Range: Range: -127 to 128
   * @param[in,out]   output                  pointer to output vector
   * @param[in]       out_offset              output offset
   * @param[in]       out_mult                output multiplier
   * @param[in]       out_shift               output shift
   * @param[in]       out_activation_min      minimum value to clamp output to
   * @param[in]       out_activation_max      maximum value to clamp output to
   * @param[in]       block_size              number of samples
   * @return          The function returns    ARM_MATH_SUCCESS
   *
   * @details   Supported framework: TensorFlow Lite micro
   */
  arm_status arm_elementwise_mul_s8(const int8_t *input_1_vect,
                                    const int8_t *input_2_vect,
                                    const int32_t input_1_offset,
                                    const int32_t input_2_offset,
                                    int8_t *output,
                                    const int32_t out_offset,
                                    const int32_t out_mult,
                                    const int32_t out_shift,
                                    const int32_t out_activation_min,
                                    const int32_t out_activation_max,
                                    const uint32_t block_size);
/**
 * @defgroup Acti Activation Functions
 *
 * Perform activation layers, including ReLU (Rectified Linear Unit),
 * sigmoid and tanh
 *
 */

  /**
   * @brief Q7 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @return none.
   */

    void      arm_relu_q7(q7_t *data, uint16_t size);

  /**
   * @brief s8 ReLU6 function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   */

    void      arm_relu6_s8(q7_t *data, uint16_t size);

  /**
   * @brief Q15 RELU function
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @return none.
   */

    void      arm_relu_q15(q15_t *data, uint16_t size);

  /**
   * @brief Q7 neural network activation function using direct table look-up
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
   * @param[in]       type        type of activation functions
   * @return none.
   */

    void      arm_nn_activations_direct_q7(q7_t * data, uint16_t size, uint16_t int_width,
                                           arm_nn_activation_type type);

  /**
   * @brief Q15 neural network activation function using direct table look-up
   * @param[in,out]   data        pointer to input
   * @param[in]       size        number of elements
   * @param[in]       int_width   bit-width of the integer part, assume to be smaller than 3
   * @param[in]       type        type of activation functions
   * @return none.
   *
   * @details
   *
   * This is the direct table look-up approach.
   *
   * Assume here the integer part of the fixed-point is <= 3.
   * More than 3 just not making much sense, makes no difference with
   * saturation followed by any of these activation functions.
   */

    void      arm_nn_activations_direct_q15(q15_t * data, uint16_t size, uint16_t int_width,
                                            arm_nn_activation_type type);

/**
 * @defgroup Pooling Pooling Functions
 *
 * Perform pooling functions, including max pooling and average pooling
 *
 */

  /**
   * @brief Q7 max pooling function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   */

    void      arm_maxpool_q7_HWC(q7_t * Im_in,
                                 const uint16_t dim_im_in,
                                 const uint16_t ch_im_in,
                                 const uint16_t dim_kernel,
                                 const uint16_t padding,
                                 const uint16_t stride,
                                 const uint16_t dim_im_out,
                                 q7_t * bufferA,
                                 q7_t * Im_out);

  /**
   * @brief Q7 average pooling function
   * @param[in]       Im_in       pointer to input tensor
   * @param[in]       dim_im_in   input tensor dimension
   * @param[in]       ch_im_in    number of input tensor channels
   * @param[in]       dim_kernel  filter kernel size
   * @param[in]       padding     padding sizes
   * @param[in]       stride      convolution stride
   * @param[in]       dim_im_out  output tensor dimension
   * @param[in,out]   bufferA     pointer to buffer space for input
   * @param[in,out]   Im_out      pointer to output tensor
   * @return none.
   *
   */

    void      arm_avepool_q7_HWC(q7_t * Im_in,
                                 const uint16_t dim_im_in,
                                 const uint16_t ch_im_in,
                                 const uint16_t dim_kernel,
                                 const uint16_t padding,
                                 const uint16_t stride,
                                 const uint16_t dim_im_out,
                                 q7_t * bufferA,
                                 q7_t * Im_out);

  /**
   * @brief s8 average pooling function
   * @param[in]       dim_src_height     input tensor dimension
   * @param[in]       dim_src_width      input tensor dimension
   * @param[in]       dim_dst_height     output tensor dimension
   * @param[in]       dim_dst_width      output tensor dimension
   * @param[in]       stride_height      stride along y
   * @param[in]       stride_width       stride along x
   * @param[in]       dim_kernel_height  filter kernel size along y
   * @param[in]       dim_kernel_width   filter kernel size along x
   * @param[in]       padding_height     padding size along y
   * @param[in]       padding_width      padding size along x
   * @param[in]       act_min            Min clamping
   * @param[in]       act_max            Max clamping
   * @param[in]       ch_src             number of input tensor channels
   * @param[in,out]   src                pointer to input tensor
   * @param[in]       bufferA            temporary buffer used for optimization and is necessary when
   *                                     ARM_MATH_DSP is defined.
   *                                     Required space: (ch_src * dim_dst_width) * sizeof(q15_t) bytes
   *                                     Use arm_avgpool_s8_get_buffer_size() to get the size
   * @param[in,out]   dst                pointer to output tensor
   * @return                             The function returns one of the following
   *                                     <code>ARM_MATH_SIZE_MISMATCH</code> - Unsupported dimension of tensors
   *                                     <code>ARM_MATH_SUCCESS</code> - Successful operation
   *                                     <code>ARM_MATH_ARGUMENT_ERROR</code> - Implementation not available
   *
   * @details
   *    - Supported Framework: TensorFlow Lite
   *
   */

    arm_status arm_avgpool_s8(const int dim_src_height,
                              const int dim_src_width,
                              const int dim_dst_height,
                              const int dim_dst_width,
                              const int stride_height,
                              const int stride_width,
                              const int dim_kernel_height,
                              const int dim_kernel_width,
                              const int padding_height,
                              const int padding_width,
                              const int act_min,
                              const int act_max,
                              const int ch_src,
                              int8_t *src,
                              int16_t *bufferA,
                              int8_t *dst);

  /**
   * @brief Get the required buffer size for S8 average pooling function
   * @param[in]       dim_dst_width         output tensor dimension
   * @param[in]       ch_src                number of input tensor channels
   * @return          The function returns  required buffer size
   *
   */
    int32_t arm_avgpool_s8_get_buffer_size(const int dim_dst_width,
                                           const int ch_src);

   /**
   * @brief s8 DSP optimized max pooling function
   * @param[in]       input_y     input tensor dimension along y
   * @param[in]       input_x     input tensor dimension along x
   * @param[in]       output_y    output tensor dimension along y
   * @param[in]       output_x    output tensor dimension along x
   * @param[in]       stride_y    stride along y
   * @param[in]       stride_x    stride along x
   * @param[in]       kernel_y    filter kernel size along y
   * @param[in]       kernel_x    filter kernel size along x
   * @param[in]       pad_y       padding size along y
   * @param[in]       pad_x       padding size along x
   * @param[in]       act_min     Activation min. Lower limit to clamp output to. Range: int8
   * @param[in]       act_max     Activation max. Upper limit to clamp output to. Range: int8
   * @param[in]       depth       number of input channels
   * @param[in]       input       pointer to input tensor
   * @param[in]       tmp_buffer  Not used.
   * @param[in,out]   output      pointer to output tensor
   * @return                      The function returns one of the following
   *                              <code>ARM_MATH_SIZE_MISMATCH</code> - Unsupported dimension of tensors
   *                              <code>ARM_MATH_SUCCESS</code> - Successful operation
   *                              <code>ARM_MATH_ARGUMENT_ERROR</code> - Implementation not available
   * @note The input data is corrupted by this function.
   * @details This optimized implementation is recommended when depth is >=  4 and dimensions are large.
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
                                   int8_t *input,
                                   int16_t *tmp_buffer,
                                   int8_t *output);

  /**
   * @brief s8 pure C max pooling function
   * @param[in]       input_y     input tensor dimension along y
   * @param[in]       input_x     input tensor dimension along x
   * @param[in]       output_y    output tensor dimension along y
   * @param[in]       output_x    output tensor dimension along x
   * @param[in]       stride_y    stride along y
   * @param[in]       stride_x    stride along x
   * @param[in]       kernel_y    filter kernel size along y
   * @param[in]       kernel_x    filter kernel size along x
   * @param[in]       pad_y       padding size along y
   * @param[in]       pad_x       padding size along x
   * @param[in]       act_min     Activation min. Lower limit to clamp output to. Range: int8
   * @param[in]       act_max     Activation max. Upper limit to clamp output to. Range: int8
   * @param[in]       channel_in  number of input channels
   * @param[in]       input       pointer to input tensor
   * @param[in]       tmp_buffer  Not used.
   * @param[in,out]   output      pointer to output tensor
   * @return                      The function returns one of the following
   *                              <code>ARM_MATH_SIZE_MISMATCH</code> - Unsupported dimension of tensors
   *                              <code>ARM_MATH_SUCCESS</code> - Successful operation
   *                              <code>ARM_MATH_ARGUMENT_ERROR</code> - Implementation not available
   *
   * @details
   *    - This basic implementation is recommended when number of channels is less than 4 and/or
   *      dimensions are small.
   *
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
                               int8_t *output);
/**
 * @defgroup Softmax Softmax Functions
 *
 * EXP(2) based softmax functions.
 *
 */

  /**
   * @brief Q7 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   *
   * @note This function is an optimized version which is not bit-accurate with
   *       TensorFlow Lite's kernel
   *
   */

void arm_softmax_q7(const q7_t * vec_in, const uint16_t dim_vec, q7_t * p_out);

  /**
   * @brief Q7 softmax function with batch parameter
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       nb_batches  number of batches
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   * @note This function is an optimized version which is not bit-accurate with
   *       TensorFlow Lite's kernel
   *
   */

void arm_softmax_with_batch_q7(const q7_t * vec_in, const uint16_t nb_batches,const uint16_t dim_vec, q7_t * p_out );
  /**
   * @brief Q15 softmax function
   * @param[in]       vec_in      pointer to input vector
   * @param[in]       dim_vec     input vector dimension
   * @param[out]      p_out       pointer to output vector
   * @return none.
   *
   * @note This function is an optimized version which is not bit-accurate with
   *       TensorFlow Lite's kernel
   *
   */

void arm_softmax_q15(const q15_t * vec_in, const uint16_t dim_vec, q15_t * p_out);

  /**
   * @brief S8 softmax function
   * @param[in]  input     Pointer to the input tensor
   * @param[in]  num_rows  Number of rows in the input tensor
   * @param[in]  row_size  Number of elements in each input row
   * @param[in]  mult      Input quantization multiplier
   * @param[in]  shift     Input quantization shift within the range [0, 31]
   * @param[in]  diff_min  Minimum difference with max in row. Used to check if
   *                       the quantized exponential operation can be performed
   * @param[out] output    Pointer to the output tensor
   *
   * @note Supported framework: TensorFlow Lite micro (bit-accurate)
   *
   */

void arm_softmax_s8(const int8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    int8_t *output);

  /**
   * @brief U8 softmax function
   * @param[in]  input     Pointer to the input tensor
   * @param[in]  num_rows  Number of rows in the input tensor
   * @param[in]  row_size  Number of elements in each input row
   * @param[in]  mult      Input quantization multiplier
   * @param[in]  shift     Input quantization shift within the range [0, 31]
   * @param[in]  diff_min  Minimum difference with max in row. Used to check if
   *                       the quantized exponential operation can be performed
   * @param[out] output    Pointer to the output tensor
   *
   * @note Supported framework: TensorFlow Lite micro (bit-accurate)
   *
   */

void arm_softmax_u8(const uint8_t *input,
                    const int32_t num_rows,
                    const int32_t row_size,
                    const int32_t mult,
                    const int32_t shift,
                    const int32_t diff_min,
                    uint8_t *output);

  /**
   * @brief uint8 depthwise convolution function with asymmetric quantization for even number of channel multiplier
   *        and input channels. Unless specified otherwise, arguments are mandatory.
   *
   * @param[in]     input     Pointer to input tensor
   * @param[in]     input_x   Width of input tensor
   * @param[in]     input_y   Height of input tensor
   * @param[in]     input_ch  Channels in input tensor
   * @param[in]     kernel    Pointer to kernel weights
   * @param[in]     kernel_x  Width of kernel
   * @param[in]     kernel_y  Height of kernel
   * @param[in]     ch_mult   Number of channel multiplier
   * @param[in]     pad_x     Padding sizes x
   * @param[in]     pad_y     Padding sizes y
   * @param[in]     stride_x  Convolution stride along the width
   * @param[in]     stride_y  Convolution stride along the height
   * @param[in]     dilation_x Dilation along width. Not used and intended for future enhancement.
   * @param[in]     dilation_y Dilation along height. Not used and intended for future enhancement.
   * @param[in]     bias       Pointer to optional bias values. If no bias is
   *                           availble, NULL is expected
   * @param[in]     input_offset  Input tensor zero offset
   * @param[in]     filter_offset Kernel tensor zero offset
   * @param[in]     output_offset Output tensor zero offset
   * @param[in,out] output        Pointer to output tensor
   * @param[in]     output_x  Width of output tensor
   * @param[in]     output_y  Height of output tensor
   * @param[in]     output_activation_min   Minimum value to clamp the output to. Range : {0, 255}
   * @param[in]     output_activation_max   Minimum value to clamp the output to. Range : {0, 255}
   * @param[in]     out_shift  Amount of right-shift for output
   * @param[in]     out_mult   Output multiplier for requantization
   * @return        The function returns one of the following
   *                <code>ARM_MATH_SIZE_MISMATCH</code> - Unsupported dimension of tensors
   *                <code>ARM_MATH_SUCCESS</code> - Successful operation
   *                <code>ARM_MATH_ARGUMENT_ERROR</code> - Implementation not available
   *
   * <b> Input constraints</b>
   * ch_mult  is multiple of 2
   * kernel_x is multiple of 2
   *
   */
    arm_status arm_depthwise_conv_u8_basic_ver1(const uint8_t *input,
                                                const uint16_t input_x,
                                                const uint16_t input_y,
                                                const uint16_t input_ch,
                                                const uint8_t *kernel,
                                                const uint16_t kernel_x,
                                                const uint16_t kernel_y,
                                                const int16_t ch_mult,
                                                const int16_t pad_x,
                                                const int16_t pad_y,
                                                const int16_t stride_x,
                                                const int16_t stride_y,
                                                const int16_t dilation_x,
                                                const int16_t dilation_y,
                                                const int32_t *bias,
                                                const int32_t input_offset,
                                                const int32_t filter_offset,
                                                const int32_t output_offset,
                                                uint8_t *output,
                                                const uint16_t output_x,
                                                const uint16_t output_y,
                                                const int32_t output_activation_min,
                                                const int32_t output_activation_max,
                                                const int32_t out_shift,
                                                const int32_t out_mult);

/**
 * @defgroup Reshape Reshape Functions
 *
 */

   /**
    * @brief Reshape a s8 vector into another with different shape
    * @param[in]  input      points to the s8 input vector
    * @param[out] output     points to the s8 output vector
    * @param[in]  total_size total size of the input and output vectors in bytes
    *
    * @note The output is expected to be in a memory area that does not overlap with the input's
    *
    */
    void arm_reshape_s8(const int8_t *input,
                        int8_t *output,
                        const uint32_t total_size);

/**
 * @defgroup Concatenation Concatenation Functions
 *
 */

  /**
   * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the X axis
   *        This function should be called for each input tensor to concatenate. The argument offset_x
   *        will be used to store the input tensor in the correct position in the output tensor
   *
   *        i.e.    offset_x = 0
   *                for(i = 0 i < num_input_tensors; ++i)
   *                {
   *                    arm_concatenation_s8_x(&input[i], ..., &output, ..., ..., offset_x)
   *                    offset_x += input_x[i]
   *                }
   *
   *        This function assumes that the output tensor has:
   *        -# The same height of the input tensor
   *        -# The same number of channels of the input tensor
   *        -# The same batch size of the input tensor
   *
   *        Unless specified otherwise, arguments are mandatory.
   *
   * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because does not involve any arithmetic operation
   *
   * @param[in]  input    Pointer to input tensor
   * @param[in]  input_x  Width of input tensor
   * @param[in]  input_y  Height of input tensor
   * @param[in]  input_z  Channels in input tensor
   * @param[in]  input_w  Batch size in input tensor
   * @param[out] output   Pointer to output tensor
   * @param[in]  output_x Width of output tensor
   * @param[in]  offset_x The offset (in number of elements) on the X axis to start concatenating the input tensor
   *                      It is user responsibility to provide the correct value
   *
   * <b> Input constraints</b>
   * offset_x is less than output_x
   *
   */
    void arm_concatenation_s8_x(const int8_t *input,
                                const uint16_t input_x,
                                const uint16_t input_y,
                                const uint16_t input_z,
                                const uint16_t input_w,
                                int8_t *output,
                                const uint16_t output_x,
                                const uint32_t offset_x);

  /**
   * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Y axis
   *        This function should be called for each input tensor to concatenate. The argument offset_y
   *        will be used to store the input tensor in the correct position in the output tensor
   *
   *        i.e.    offset_y = 0
   *                for(i = 0 i < num_input_tensors; ++i)
   *                {
   *                    arm_concatenation_s8_y(&input[i], ..., &output, ..., ..., offset_y)
   *                    offset_y += input_y[i]
   *                }
   *
   *        This function assumes that the output tensor has:
   *        -# The same width of the input tensor
   *        -# The same number of channels of the input tensor
   *        -# The same batch size of the input tensor
   *
   *        Unless specified otherwise, arguments are mandatory.
   *
   * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because does not involve any arithmetic operation
   *
   * @param[in]  input    Pointer to input tensor
   * @param[in]  input_x  Width of input tensor
   * @param[in]  input_y  Height of input tensor
   * @param[in]  input_z  Channels in input tensor
   * @param[in]  input_w  Batch size in input tensor
   * @param[out] output   Pointer to output tensor
   * @param[in]  output_y Height of output tensor
   * @param[in]  offset_y The offset on the Y axis to start concatenating the input tensor
   *                      It is user responsibility to provide the correct value
   *
   * <b> Input constraints</b>
   * offset_y is less than output_y
   *
   */
    void arm_concatenation_s8_y(const int8_t *input,
                                const uint16_t input_x,
                                const uint16_t input_y,
                                const uint16_t input_z,
                                const uint16_t input_w,
                                int8_t *output,
                                const uint16_t output_y,
                                const uint32_t offset_y);

  /**
   * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the Z axis
   *        This function should be called for each input tensor to concatenate. The argument offset_z
   *        will be used to store the input tensor in the correct position in the output tensor
   *
   *        i.e.    offset_z = 0
   *                for(i = 0 i < num_input_tensors; ++i)
   *                {
   *                    arm_concatenation_s8_z(&input[i], ..., &output, ..., ..., offset_z)
   *                    offset_z += input_z[i]
   *                }
   *
   *        This function assumes that the output tensor has:
   *        -# The same width of the input tensor
   *        -# The same height of the input tensor
   *        -# The same batch size of the input tensor
   *
   *        Unless specified otherwise, arguments are mandatory.
   *
   * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because does not involve any arithmetic operation
   *
   * @param[in]  input    Pointer to input tensor
   * @param[in]  input_x  Width of input tensor
   * @param[in]  input_y  Height of input tensor
   * @param[in]  input_z  Channels in input tensor
   * @param[in]  input_w  Batch size in input tensor
   * @param[out] output   Pointer to output tensor
   * @param[in]  output_z Channels in output tensor
   * @param[in]  offset_z The offset on the Z axis to start concatenating the input tensor
   *                      It is user responsibility to provide the correct value
   *
   * <b> Input constraints</b>
   * offset_z is less than output_z
   *
   */
    void arm_concatenation_s8_z(const int8_t *input,
                                const uint16_t input_x,
                                const uint16_t input_y,
                                const uint16_t input_z,
                                const uint16_t input_w,
                                int8_t *output,
                                const uint16_t output_z,
                                const uint32_t offset_z);

  /**
   * @brief int8/uint8 concatenation function to be used for concatenating N-tensors along the W axis (Batch size)
   *        This function should be called for each input tensor to concatenate. The argument offset_w
   *        will be used to store the input tensor in the correct position in the output tensor
   *
   *        i.e.    offset_w = 0
   *                for(i = 0 i < num_input_tensors; ++i)
   *                {
   *                    arm_concatenation_s8_w(&input[i], ..., &output, ..., ..., offset_w)
   *                    offset_w += input_w[i]
   *                }
   *
   *        This function assumes that the output tensor has:
   *        -# The same width of the input tensor
   *        -# The same height of the input tensor
   *        -# The same number o channels of the input tensor
   *
   *        Unless specified otherwise, arguments are mandatory.
   *
   * @note This function, data layout independent, can be used to concatenate either int8 or uint8 tensors because does not involve any arithmetic operation
   *
   * @param[in]  input    Pointer to input tensor
   * @param[in]  input_x  Width of input tensor
   * @param[in]  input_y  Height of input tensor
   * @param[in]  input_z  Channels in input tensor
   * @param[in]  input_w  Batch size in input tensor
   * @param[out] output   Pointer to output tensor
   * @param[in]  offset_w The offset on the W axis to start concatenating the input tensor
   *                      It is user responsibility to provide the correct value
   *
   */
    void arm_concatenation_s8_w(const int8_t *input,
                                const uint16_t input_x,
                                const uint16_t input_y,
                                const uint16_t input_z,
                                const uint16_t input_w,
                                int8_t *output,
                                const uint32_t offset_w);
#ifdef __cplusplus
}
#endif

#endif
