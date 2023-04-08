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

namespace mindspore {
namespace kernel {
constexpr auto kConv2dFwdAlgoName = "CUDNN_CONV2D_FWD_ALGO";
constexpr auto kConv2dBwdDataAlgoName = "CUDNN_CONV2D_BWD_DATA_ALGO";
constexpr auto kConv2dBwdFilterAlgoName = "CUDNN_CONV2D_BWD_FILTER_ALGO";

constexpr auto kConv3dFwdAlgoName = "CUDNN_CONV3D_FWD_ALGO";
constexpr auto kConv3dBwdDataAlgoName = "CUDNN_CONV3D_BWD_DATA_ALGO";
constexpr auto kConv3dBwdFilterAlgoName = "CUDNN_CONV3D_BWD_FILTER_ALGO";
constexpr auto kConv3dTransposeAlgoName = "CUDNN_CONV3D_TRANSPOSE_ALGO";
constexpr auto kEnableCudnnHeuristicSearch = "ENABLE_CUDNN_HEURISTIC_SEARCH";

static std::unordered_map<std::string, cudnnConvolutionFwdAlgo_t> cudnn_fwd_algos = {
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM},
  {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM", CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM},
  {"CUDNN_CONVOLUTION_FWD_ALGO_GEMM", CUDNN_CONVOLUTION_FWD_ALGO_GEMM},
  {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT", CUDNN_CONVOLUTION_FWD_ALGO_DIRECT},
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT", CUDNN_CONVOLUTION_FWD_ALGO_FFT},
  {"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING", CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING},
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD},
  {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED", CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED}};

static std::unordered_map<std::string, cudnnConvolutionBwdDataAlgo_t> cudnn_bwd_data_algos = {
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0", CUDNN_CONVOLUTION_BWD_DATA_ALGO_0},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1", CUDNN_CONVOLUTION_BWD_DATA_ALGO_1},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT", CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING", CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD", CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD},
  {"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED", CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED}};

static std::unordered_map<std::string, cudnnConvolutionBwdFilterAlgo_t> cudnn_bwd_filter_algos = {
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED},
  {"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING", CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING}};

static cudnnConvolutionFwdAlgo_t SelectForwardAlgorithm(cudnnHandle_t handle, const cudnnTensorDescriptor_t &x_desc,
                                                        const cudnnFilterDescriptor_t &w_desc,
                                                        const cudnnConvolutionDescriptor_t &conv_desc,
                                                        const cudnnTensorDescriptor_t &y_desc, const int &group,
                                                        const std::string &env_name) {
  cudnnConvolutionFwdAlgo_t conv_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc,
                                          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
      "cudnnGetConvolutionForwardAlgorithm failed");
  }
#else
  std::string cudnn_algo = common::GetEnv(env_name);
  if (!cudnn_algo.empty()) {
    if (cudnn_fwd_algos.find(cudnn_algo) == cudnn_fwd_algos.end()) {
      MS_LOG(EXCEPTION) << "cudnn algorithm type: " << cudnn_algo << " is not supported.";
    } else {
      conv_algorithm = cudnn_fwd_algos[cudnn_algo];
    }
  } else {
    std::string heuristic_search = common::GetEnv(kEnableCudnnHeuristicSearch);
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results;
    if (heuristic_search.empty() || heuristic_search == "True") {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionForwardAlgorithm_v7(handle, x_desc, w_desc, conv_desc, y_desc, requested_algo_count,
                                               &returned_algo_count, &perf_results),
        "cudnnGetConvolutionForwardAlgorithm_v7 failed");
    } else if (heuristic_search == "False") {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, requested_algo_count,
                                             &returned_algo_count, &perf_results),
        "cudnnFindConvolutionForwardAlgorithm failed");
    } else {
      MS_LOG(EXCEPTION) << "Enable cudnn heuristic search type: " << heuristic_search << " is not supported.";
    }
    conv_algorithm = perf_results.algo;
  }
#endif
  return conv_algorithm;
}

static cudnnConvolutionBwdDataAlgo_t SelectBackwardDataAlgorithm(cudnnHandle_t handle,
                                                                 const cudnnFilterDescriptor_t &w_desc,
                                                                 const cudnnTensorDescriptor_t &dy_desc,
                                                                 const cudnnConvolutionDescriptor_t &conv_desc,
                                                                 const cudnnTensorDescriptor_t &dx_desc,
                                                                 const int &group, const std::string &env_name) {
  cudnnConvolutionBwdDataAlgo_t conv_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;

#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc, conv_desc, dx_desc,
                                               CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
      "cudnnGetConvolutionBackwardDataAlgorithm failed");
  }
#else
  std::string cudnn_algo = common::GetEnv("CUDNN_CONV2D_BWD_DATA_ALGO");
  if (!cudnn_algo.empty()) {
    if (cudnn_bwd_data_algos.find(cudnn_algo) == cudnn_bwd_data_algos.end()) {
      MS_LOG(EXCEPTION) << "cudnn algorithm type: " << cudnn_algo << " is not supported.";
    } else {
      conv_algorithm = cudnn_bwd_data_algos[cudnn_algo];
    }
  } else {
    std::string heuristic_search = common::GetEnv(kEnableCudnnHeuristicSearch);
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results;
    if (heuristic_search.empty() || heuristic_search == "True") {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, w_desc, dy_desc, conv_desc, dx_desc, requested_algo_count,
                                                    &returned_algo_count, &perf_results),
        "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
    } else if (heuristic_search == "False") {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc, conv_desc, dx_desc, requested_algo_count,
                                                  &returned_algo_count, &perf_results),
        "cudnnFindConvolutionBackwardDataAlgorithm failed");
    } else {
      MS_LOG(EXCEPTION) << "Enable cudnn heuristic search type: " << heuristic_search << " is not supported.";
    }
    conv_algorithm = perf_results.algo;
  }

#endif
  return conv_algorithm;
}

static cudnnConvolutionBwdFilterAlgo_t SelectBackwardFilterAlgorithm(cudnnHandle_t handle,
                                                                     const cudnnTensorDescriptor_t x_desc,
                                                                     const cudnnTensorDescriptor_t dy_desc,
                                                                     const cudnnConvolutionDescriptor_t conv_desc,
                                                                     const cudnnFilterDescriptor_t dw_desc,
                                                                     const int &group, const std::string &env_name) {
  cudnnConvolutionBwdFilterAlgo_t conv_algorithm = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetConvolutionBackwardFilterAlgorithm(
                                          handle, x_desc, dy_desc, conv_desc, dw_desc,
                                          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
                                        "GetConvolutionBackwardFilterAlgorithm failed");
  }
#else
  std::string cudnn_algo = common::GetEnv("CUDNN_CONV2D_BWD_FILTER_ALGO");
  if (!cudnn_algo.empty()) {
    if (cudnn_bwd_filter_algos.find(cudnn_algo) == cudnn_bwd_filter_algos.end()) {
      MS_LOG(EXCEPTION) << "cudnn algorithm type: " << cudnn_algo << " is not supported.";
    } else {
      conv_algorithm = cudnn_bwd_filter_algos[cudnn_algo];
    }
  } else {
    std::string heuristic_search = common::GetEnv(kEnableCudnnHeuristicSearch);
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
    if (heuristic_search.empty() || heuristic_search == "True") {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x_desc, dy_desc, conv_desc, dw_desc, requested_algo_count,
                                                      &returned_algo_count, &perf_results),
        "cudnnGetConvolutionBackwardFilterAlgorithm_v7 failed");
    } else if (heuristic_search == "False") {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, dy_desc, conv_desc, dw_desc, requested_algo_count,
                                                    &returned_algo_count, &perf_results),
        "cudnnFindConvolutionBackwardFilterAlgorithm failed");
    } else {
      MS_LOG(EXCEPTION) << "Enable cudnn heuristic search type: " << heuristic_search << " is not supported.";
    }

    conv_algorithm = perf_results.algo;
  }
#endif
  return conv_algorithm;
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV_GPU_COMMON_H_
