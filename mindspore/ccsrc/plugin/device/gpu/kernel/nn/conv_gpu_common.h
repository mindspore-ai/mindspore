/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV_GPU_COMMON_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV_GPU_COMMON_H_

#include <cuda.h>
#include <cudnn.h>
#include <unordered_map>
#include <string>
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
constexpr auto kConvNormalAlgoName = "normal";
constexpr auto kConvPerformanceAlgoName = "performance";

static std::unordered_map<std::string, cudnnConvolutionFwdAlgo_t> cudnn_fwd_algos = {
  {"implicit_gemm", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM},
  {"precomp_gemm", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM},
  {"gemm", CUDNN_CONVOLUTION_FWD_ALGO_GEMM},
  {"direct", CUDNN_CONVOLUTION_FWD_ALGO_DIRECT},
  {"fft", CUDNN_CONVOLUTION_FWD_ALGO_FFT},
  {"fft_tiling", CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING},
  {"winograd", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD},
  {"winograd_nonfused", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED}};

static std::unordered_map<std::string, cudnnConvolutionBwdDataAlgo_t> cudnn_bwd_data_algos = {
  {"algo_0", CUDNN_CONVOLUTION_BWD_DATA_ALGO_0},
  {"algo_1", CUDNN_CONVOLUTION_BWD_DATA_ALGO_1},
  {"fft", CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT},
  {"fft_tiling", CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING},
  {"winograd", CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD},
  {"winograd_nonfused", CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED}};

static std::unordered_map<std::string, cudnnConvolutionBwdFilterAlgo_t> cudnn_bwd_filter_algos = {
  {"algo_0", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0},
  {"algo_1", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1},
  {"fft", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT},
  {"algo_3", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3},
  {"winograd_nonfused", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED},
  {"fft_tiling", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING}};

static cudnnConvolutionFwdAlgo_t SelectForwardAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t &x_desc,
                                                        const cudnnFilterDescriptor_t &w_desc,
                                                        const cudnnConvolutionDescriptor_t &conv_desc,
                                                        const cudnnTensorDescriptor_t &y_desc, const int &group) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto algo = context_ptr->get_param<std::string>(MS_CTX_CONV_FPROP_ALGO);
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results;

  cudnnConvolutionFwdAlgo_t conv_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  if (cudnn_fwd_algos.find(algo) != cudnn_fwd_algos.end()) {
    conv_algorithm = cudnn_fwd_algos[algo];
  } else if (algo == kConvNormalAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionForwardAlgorithm_v7(handle, x_desc, w_desc, conv_desc, y_desc, requested_algo_count,
                                             &returned_algo_count, &perf_results),
      "cudnnGetConvolutionForwardAlgorithm_v7 failed");
    conv_algorithm = perf_results.algo;
  } else if (algo == kConvPerformanceAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, requested_algo_count,
                                           &returned_algo_count, &perf_results),
      "cudnnFindConvolutionForwardAlgorithm failed");
    conv_algorithm = perf_results.algo;
  } else {
    MS_LOG(EXCEPTION) << "Conv fprop algo type: " << algo << " is not supported.";
  }
#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc,
                                          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
      "cudnnGetConvolutionForwardAlgorithm failed");
  }
#endif
  return conv_algorithm;
}

static cudnnConvolutionBwdDataAlgo_t SelectBackwardDataAlgorithm(
  cudnnHandle_t handle, const cudnnFilterDescriptor_t &w_desc, const cudnnTensorDescriptor_t &dy_desc,
  const cudnnConvolutionDescriptor_t &conv_desc, const cudnnTensorDescriptor_t &dx_desc, const int &group) {
  auto context_ptr = MsContext::GetInstance();
  auto algo = context_ptr->get_param<std::string>(MS_CTX_CONV_DGRAD_ALGO);
  MS_EXCEPTION_IF_NULL(context_ptr);

  cudnnConvolutionBwdDataAlgo_t conv_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf_results;

  if (cudnn_bwd_data_algos.find(algo) != cudnn_bwd_data_algos.end()) {
    conv_algorithm = cudnn_bwd_data_algos[algo];
  } else if (algo == kConvNormalAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, w_desc, dy_desc, conv_desc, dx_desc, requested_algo_count,
                                                  &returned_algo_count, &perf_results),
      "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
    conv_algorithm = perf_results.algo;
  } else if (algo == kConvPerformanceAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc, conv_desc, dx_desc, requested_algo_count,
                                                &returned_algo_count, &perf_results),
      "cudnnFindConvolutionBackwardDataAlgorithm failed");
    conv_algorithm = perf_results.algo;
  } else {
    MS_LOG(EXCEPTION) << "Conv dgrad algo type: " << algo << " is not supported.";
  }
#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc, conv_desc, dx_desc,
                                               CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
      "cudnnGetConvolutionBackwardDataAlgorithm failed");
  }
#endif
  return conv_algorithm;
}

static cudnnConvolutionBwdFilterAlgo_t SelectBackwardFilterAlgorithm(
  cudnnHandle_t handle, const cudnnTensorDescriptor_t x_desc, const cudnnTensorDescriptor_t dy_desc,
  const cudnnConvolutionDescriptor_t conv_desc, const cudnnFilterDescriptor_t dw_desc, const int &group) {
  auto context_ptr = MsContext::GetInstance();
  auto algo = context_ptr->get_param<std::string>(MS_CTX_CONV_WGRAD_ALGO);
  MS_EXCEPTION_IF_NULL(context_ptr);
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perf_results;

  cudnnConvolutionBwdFilterAlgo_t conv_algorithm = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  if (cudnn_bwd_filter_algos.find(algo) != cudnn_bwd_filter_algos.end()) {
    conv_algorithm = cudnn_bwd_filter_algos[algo];
  } else if (algo == kConvNormalAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x_desc, dy_desc, conv_desc, dw_desc, requested_algo_count,
                                                    &returned_algo_count, &perf_results),
      "cudnnGetConvolutionBackwardFilterAlgorithm_v7 failed");
    conv_algorithm = perf_results.algo;
  } else if (algo == kConvPerformanceAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, dy_desc, conv_desc, dw_desc, requested_algo_count,
                                                  &returned_algo_count, &perf_results),
      "cudnnFindConvolutionBackwardFilterAlgorithm failed");
    conv_algorithm = perf_results.algo;
  } else {
    MS_LOG(EXCEPTION) << "Conv wgrad algo type: " << algo << " is not supported.";
  }
#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetConvolutionBackwardFilterAlgorithm(
                                          handle, x_desc, dy_desc, conv_desc, dw_desc,
                                          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
                                        "GetConvolutionBackwardFilterAlgorithm failed");
  }
#endif
  return conv_algorithm;
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV_GPU_COMMON_H_
