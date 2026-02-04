/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <unordered_map>
#include "mindspore/ccsrc/plugin/gpu/res_manager/device_context_conf/op_precision_conf.h"
#include "mindspore/ccsrc/plugin/gpu/res_manager/device_context_conf/op_tuning_conf.h"

namespace mindspore {
namespace kernel {
enum class ConvType { kForward = 0, kInputGrad = 1, kFilterGrad = 2 };
enum class ConvKernelType { kCudnn = 0, kDepthWise = 1 };
constexpr size_t kShapeIndex = 2;
constexpr size_t kConv2dDimSize = 2;
constexpr size_t kConv2dInputDimSize = 4;
constexpr size_t kTop2DPadIndex = 0;
constexpr size_t kBottom2DPadIndex = 1;
constexpr size_t kLeft2DPadIndex = 2;
constexpr size_t kRight2DPadIndex = 3;
constexpr size_t kConv2dSymmetricCoef = 2;
constexpr size_t kHeight2DStrideIndex = 2;
constexpr size_t kWidth2DStrideIndex = 3;
constexpr size_t kHeight2DDilationIndex = 2;
constexpr size_t kWidth2DDilationIndex = 3;
constexpr size_t k2DHeightIndexNCHW = 2;
constexpr size_t k2DHeightIndexNHWC = 1;
constexpr size_t kAlgoCacheSize = 100;
constexpr int64_t kHashMagicNum = 0x9e3779b9;

struct VectorLongHash {
  std::size_t operator()(const std::vector<int64_t> &v) const {
    std::size_t seed = v.size();
    for (auto &i : v) {
      seed ^= i + kHashMagicNum + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

struct ConvolutionArgs {
  std::string kernel_name{""};
  size_t output_size{0};
  size_t type_id_size{0};
  int batch_size{0};
  int in_height{0};
  int in_width{0};
  int in_channel{0};
  int out_channel{0};
  int filter_height{0};
  int filter_width{0};
  int pad_height{0};
  int pad_width{0};
  int pad_top{0};
  int pad_left{0};
  int out_height{0};
  int out_width{0};
  int group{0};
  std::vector<int64_t> tensor0_shape;
  std::vector<int64_t> tensor1_shape;
  std::vector<int64_t> tensor2_shape;
  std::vector<int> stride;
  std::vector<int> dilation;
  std::vector<int> pad_list;
  std::string data_type{""};
  std::string data_format{""};
  std::string data_format_attr{""};
  std::string pad_mode{""};
  bool use_pad{false};
  float alpha{1};
  float beta{0};
};

static const std::unordered_map<std::string, size_t> kFormatIndexMap = {{"NCHW", 2}, {"HWCN", 0}, {"NHWC", 1}};
static const std::vector<int32_t> to_nhwc_axis = {0, 2, 3, 1};
static const std::vector<int32_t> to_nchw_axis = {0, 3, 1, 2};

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

template <typename T>
std::string GetArrayText(const T *values, const size_t len) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < len; ++i) {
    if (i > 0) {
      oss << ", ";
    }
    oss << values[i];
  }
  oss << "]";
  return oss.str();
}

constexpr size_t CONV2D_DIM_SIZE = 2;
constexpr size_t CONV2D_INPUT_DIM = 4;

static std::string GetCudnnDataTypeStr(cudnnDataType_t computeType) {
  if (computeType == CUDNN_DATA_FLOAT) {
    return "Float32";
  } else if (computeType == CUDNN_DATA_HALF) {
    return "Half";
  } else {
    return "";
  }
}

static std::string GetCudnnModeStr(cudnnConvolutionMode_t mode) {
  if (mode == CUDNN_CONVOLUTION) {
    return "CUDNN_CONVOLUTION";
  } else if (mode == CUDNN_CROSS_CORRELATION) {
    return "CUDNN_CROSS_CORRELATION";
  } else {
    return "";
  }
}

static std::string GetCudnnDataFormatStr(cudnnTensorFormat_t format) {
  if (format == CUDNN_TENSOR_NCHW) {
    return "NCHW";
  } else if (format == CUDNN_TENSOR_NHWC) {
    return "NHWC";
  } else {
    return "";
  }
}

// get conv_desc info
static std::ostream &operator<<(std::ostream &os, const cudnnConvolutionDescriptor_t conv_desc) {
  int arrayLength;
  int padA[CONV2D_DIM_SIZE];
  int strideA[CONV2D_DIM_SIZE];
  int dilationA[CONV2D_DIM_SIZE];
  cudnnConvolutionMode_t mode;
  cudnnDataType_t computeType;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetConvolutionNdDescriptor(conv_desc, CONV2D_DIM_SIZE, &arrayLength, padA,
                                                                      strideA, dilationA, &mode, &computeType),
                                      "cudnnGetConvolutionNdDescriptor failed");

  os << "padA = " << GetArrayText(padA, arrayLength) << ", strideA = " << GetArrayText(strideA, arrayLength)
     << ", dilationA = " << GetArrayText(dilationA, arrayLength)
     << ", computeType = " << GetCudnnDataTypeStr(computeType) << ", computeMode = " << GetCudnnModeStr(mode);
  return os;
}

// get tensor_desc info
static std::ostream &operator<<(std::ostream &os, const cudnnTensorDescriptor_t tensor_desc) {
  cudnnDataType_t dataType;
  int nbDims;
  int dimA[CONV2D_INPUT_DIM];
  int strideA[CONV2D_INPUT_DIM];
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetTensorNdDescriptor(tensor_desc, CONV2D_INPUT_DIM, &dataType, &nbDims, dimA, strideA),
    "cudnnGetTensorNdDescriptor failed");
  os << "dimA = " << GetArrayText(dimA, nbDims);

  return os;
}

// get filter_desc info
static std::ostream &operator<<(std::ostream &os, const cudnnFilterDescriptor_t dw_desc) {
  cudnnDataType_t dataType;
  cudnnTensorFormat_t format;
  int nbDims;
  int filterDimA[CONV2D_INPUT_DIM];
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetFilterNdDescriptor(dw_desc, CONV2D_INPUT_DIM, &dataType, &format, &nbDims, filterDimA),
    "cudnnGetFilterNdDescriptor failed");
  os << "filterDimA = " << GetArrayText(filterDimA, nbDims) << ", computeFormat = " << GetCudnnDataFormatStr(format);

  return os;
}

// get convolution forward info
static std::string GetConvForwardInfo(const std::string &msg, const cudnnTensorDescriptor_t x_desc,
                                      const cudnnFilterDescriptor_t w_desc,
                                      const cudnnConvolutionDescriptor_t conv_desc,
                                      const cudnnTensorDescriptor_t y_desc) {
  std::ostringstream oss;

  oss << msg << " conv_desc: " << conv_desc << " x_desc: " << x_desc << " w_desc: " << w_desc << " y_desc: " << y_desc;

  return oss.str();
}

// get convolution backward data info
static std::string GetConvBwdDataInfo(const std::string &msg, const cudnnFilterDescriptor_t &w_desc,
                                      const cudnnTensorDescriptor_t &dy_desc,
                                      const cudnnConvolutionDescriptor_t &conv_desc,
                                      const cudnnTensorDescriptor_t &dx_desc) {
  std::ostringstream oss;

  oss << msg << " conv_desc: " << conv_desc << " w_desc: " << w_desc << " dy_desc: " << dy_desc
      << " dx_desc: " << dx_desc;

  return oss.str();
}

// get convolution backward filter info
static std::string GetConvBwdFilterInfo(const std::string &msg, const cudnnTensorDescriptor_t x_desc,
                                        const cudnnTensorDescriptor_t dy_desc,
                                        const cudnnConvolutionDescriptor_t conv_desc,
                                        const cudnnFilterDescriptor_t dw_desc) {
  std::ostringstream oss;

  oss << msg << " conv_desc: " << conv_desc << " x_desc: " << x_desc << " dy_desc: " << dy_desc
      << " dw_desc: " << dw_desc;

  return oss.str();
}

static void SetConvolutionMathType(const cudnnConvolutionDescriptor_t &conv_desc,
                                   const cudnnDataType_t &cudnn_data_type) {
  auto op_precision_conf = device::gpu::GPUOpPrecisionConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_precision_conf);
  auto math_type = CUDNN_DEFAULT_MATH;
  // for gpu volta architecture
  if (cudnn_data_type == CUDNN_DATA_HALF) {
    math_type = CUDNN_TENSOR_OP_MATH;
  } else if (cudnn_data_type == CUDNN_DATA_FLOAT) {
// for gpu amper architecture
#if CUDNN_VERSION >= 8000
    auto conv_allow_tf32 = op_precision_conf->conv_allow_tf32();
    if (conv_allow_tf32) {
      math_type = CUDNN_TENSOR_OP_MATH;
    } else {
      math_type = CUDNN_FMA_MATH;
    }
#endif
  }
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionMathType(conv_desc, math_type),
                                      "cudnnSetConvolutionMathType failed.")
}

static cudnnConvolutionFwdAlgo_t SelectForwardAlgorithm(const cudnnHandle_t &handle,
                                                        const cudnnDataType_t &cudnn_data_type,
                                                        const cudnnTensorDescriptor_t &x_desc,
                                                        const cudnnFilterDescriptor_t &w_desc,
                                                        const cudnnConvolutionDescriptor_t &conv_desc,
                                                        const cudnnTensorDescriptor_t &y_desc, const int &group) {
  auto op_tuning_conf = device::gpu::GPUOpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  auto algo = op_tuning_conf->conv_fprop_algo();
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results;

  cudnnConvolutionFwdAlgo_t conv_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  if (cudnn_data_type == CUDNN_DATA_HALF) {
    return conv_algorithm;
  }
  if (cudnn_fwd_algos.find(algo) != cudnn_fwd_algos.end()) {
    conv_algorithm = cudnn_fwd_algos[algo];
  } else if (algo == kConvNormalAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionForwardAlgorithm_v7(handle, x_desc, w_desc, conv_desc, y_desc, requested_algo_count,
                                             &returned_algo_count, &perf_results),
      GetConvForwardInfo("cudnnGetConvolutionForwardAlgorithm_v7 failed", x_desc, w_desc, conv_desc, y_desc));
    conv_algorithm = perf_results.algo;
  } else if (algo == kConvPerformanceAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnFindConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc, requested_algo_count,
                                           &returned_algo_count, &perf_results),
      GetConvForwardInfo("cudnnFindConvolutionForwardAlgorithm failed", x_desc, w_desc, conv_desc, y_desc));
    conv_algorithm = perf_results.algo;
  } else {
    MS_LOG(EXCEPTION) << "Conv fprop algo type: " << algo << " is not supported.";
  }
#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionForwardAlgorithm(handle, x_desc, w_desc, conv_desc, y_desc,
                                          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
      GetConvForwardInfo("cudnnGetConvolutionForwardAlgorithm failed", x_desc, w_desc, conv_desc, y_desc));
  }
#endif
  return conv_algorithm;
}

static cudnnConvolutionBwdDataAlgo_t SelectBackwardDataAlgorithm(
  const cudnnHandle_t &handle, const cudnnDataType_t &cudnn_data_type, const cudnnFilterDescriptor_t &w_desc,
  const cudnnTensorDescriptor_t &dy_desc, const cudnnConvolutionDescriptor_t &conv_desc,
  const cudnnTensorDescriptor_t &dx_desc, const int &group) {
  auto op_tuning_conf = device::gpu::GPUOpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  auto algo = op_tuning_conf->conv_dgrad_algo();
  cudnnConvolutionBwdDataAlgo_t conv_algorithm = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  if (cudnn_data_type == CUDNN_DATA_HALF) {
    if (cudnn_bwd_data_algos.find(algo) != cudnn_bwd_data_algos.end()) {
      conv_algorithm = cudnn_bwd_data_algos[algo];
    }
    return conv_algorithm;
  }
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf_results;

  if (cudnn_bwd_data_algos.find(algo) != cudnn_bwd_data_algos.end()) {
    conv_algorithm = cudnn_bwd_data_algos[algo];
  } else if (algo == kConvNormalAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, w_desc, dy_desc, conv_desc, dx_desc, requested_algo_count,
                                                  &returned_algo_count, &perf_results),
      GetConvBwdDataInfo("cudnnGetConvolutionBackwardDataAlgorithm_v7 failed", w_desc, dy_desc, conv_desc, dx_desc));
    conv_algorithm = perf_results.algo;
  } else if (algo == kConvPerformanceAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnFindConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc, conv_desc, dx_desc, requested_algo_count,
                                                &returned_algo_count, &perf_results),
      GetConvBwdDataInfo("cudnnFindConvolutionBackwardDataAlgorithm failed", w_desc, dy_desc, conv_desc, dx_desc));
    conv_algorithm = perf_results.algo;
  } else {
    MS_LOG(EXCEPTION) << "Conv dgrad algo type: " << algo << " is not supported.";
  }
#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardDataAlgorithm(handle, w_desc, dy_desc, conv_desc, dx_desc,
                                               CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm),
      GetConvBwdDataInfo("cudnnGetConvolutionBackwardDataAlgorithm failed", w_desc, dy_desc, conv_desc, dx_desc));
  }
#endif
  return conv_algorithm;
}

static cudnnConvolutionBwdFilterAlgo_t SelectBackwardFilterAlgorithm(
  const cudnnHandle_t &handle, const cudnnDataType_t &cudnn_data_type, const cudnnTensorDescriptor_t x_desc,
  const cudnnTensorDescriptor_t dy_desc, const cudnnConvolutionDescriptor_t conv_desc,
  const cudnnFilterDescriptor_t dw_desc, const int &group) {
  auto op_tuning_conf = device::gpu::GPUOpTuningConf::GetInstance();
  MS_EXCEPTION_IF_NULL(op_tuning_conf);
  auto algo = op_tuning_conf->conv_wgrad_algo();
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perf_results;

  cudnnConvolutionBwdFilterAlgo_t conv_algorithm = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  if (cudnn_data_type == CUDNN_DATA_HALF) {
    return conv_algorithm;
  }
  if (cudnn_bwd_filter_algos.find(algo) != cudnn_bwd_filter_algos.end()) {
    conv_algorithm = cudnn_bwd_filter_algos[algo];
  } else if (algo == kConvNormalAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, x_desc, dy_desc, conv_desc, dw_desc, requested_algo_count,
                                                    &returned_algo_count, &perf_results),
      GetConvBwdFilterInfo("cudnnGetConvolutionBackwardFilterAlgorithm_v7 failed", x_desc, dy_desc, conv_desc,
                           dw_desc));
    conv_algorithm = perf_results.algo;
  } else if (algo == kConvPerformanceAlgoName) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnFindConvolutionBackwardFilterAlgorithm(handle, x_desc, dy_desc, conv_desc, dw_desc, requested_algo_count,
                                                  &returned_algo_count, &perf_results),
      GetConvBwdFilterInfo("cudnnFindConvolutionBackwardFilterAlgorithm failed", x_desc, dy_desc, conv_desc, dw_desc));
    conv_algorithm = perf_results.algo;
  } else {
    MS_LOG(EXCEPTION) << "Conv wgrad algo type: " << algo << " is not supported.";
  }
#if CUDNN_VERSION < 8000
  if (group > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardFilterAlgorithm(handle, x_desc, dy_desc, conv_desc, dw_desc,
                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0,
                                                 &conv_algorithm),
      GetConvBwdFilterInfo("GetConvolutionBackwardFilterAlgorithm failed", x_desc, dy_desc, conv_desc, dw_desc));
  }
#endif
  return conv_algorithm;
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV_GPU_COMMON_H_
