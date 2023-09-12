/**
 * Copyright 2021 Harbin Institute of Technology
 * Copyright 2021 Huawei Technologies Co., Ltd.
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

#include "maxpool.h"

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
namespace {
const char *MAXPOOL = "MaxPool";
constexpr uint32_t kMaxPoolInputNum = 1;
constexpr uint32_t kMaxPoolOutputNum = 1;
constexpr int64_t kParallelNum = 64 * 1024;
struct PoolParams {
  int depth;

  int tensor_cols;
  int tensor_rows;
  int tensor_batch;

  int ksize_rows;
  int ksize_cols;
  int ksize_depth;

  int strides_rows;
  int strides_cols;
  int strides_depth;

  int64_t out_height;
  int64_t out_width;
  int out_depth;

  int64_t pad_top;
  int64_t pad_bottom;
  int64_t pad_left;
  int64_t pad_right;
};
}  // namespace
namespace aicpu {
uint32_t GetOutputSize(int input_size, int kernel_size, int stride, const std::string &padding, int64_t *output_size,
                       int64_t *padding_before, int64_t *padding_after) {
  KERNEL_CHECK_FALSE(stride > 0, KERNEL_STATUS_PARAM_INVALID, "[MaxPool] Stride must be positive.");
  std::string same("SAME");
  std::string valid("VALID");
  if (valid == padding) {
    *output_size = (input_size - kernel_size + stride) / stride;
    *padding_before = 0;
    *padding_after = 0;
  } else if (same == padding) {
    *output_size = (input_size + stride - 1) / stride;
    const int64_t padding_need =
      std::max(static_cast<int64_t>(0), (*output_size - 1) * stride + kernel_size - input_size);
    *padding_before = padding_need / 2;
    *padding_after = padding_need - *padding_before;
  } else {
    KERNEL_LOG_ERROR("[MaxPool] Padding is invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (*output_size < 0) {
    KERNEL_LOG_ERROR("[MaxPool] Computed output size is negative.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t ConstructPoolParams(aicpu::CpuKernelContext &ctx, const aicpu::TensorShape &data_format, PoolParams &params) {
  Format format = data_format.GetFormat();
  KERNEL_CHECK_FALSE((format == FORMAT_NHWC || format == FORMAT_NCHW), KERNEL_STATUS_PARAM_INVALID,
                     "[MaxPool] Format is not NHWC or NCHW.");
  std::vector<int64_t> tensor_in_shapes = data_format.GetDimSizes();
  if (tensor_in_shapes.size() != 4) {
    KERNEL_LOG_ERROR("[MaxPool] Input tensor must have 2 spacial dimensions.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  std::vector<int64_t> ksize = ctx.GetAttr("ksize")->GetListInt();
  std::vector<int64_t> strides = ctx.GetAttr("strides")->GetListInt();
  std::string padding = ctx.GetAttr("padding")->GetString();
  std::string data_format_str = "";
  if (ctx.GetAttr("data_format") == nullptr) {
    KERNEL_LOG_INFO("[MaxPool] Attr data_format is empty, using default value NHWC.");
    format = FORMAT_NHWC;
  } else {
    std::map<std::string, aicpu::Format> format_str_to_enum_map = {{"NHWC", FORMAT_NHWC}, {"NCHW", FORMAT_NCHW}};
    data_format_str = ctx.GetAttr("data_format")->GetString();

    KERNEL_HANDLE_ERROR(format_str_to_enum_map.find(data_format_str) == format_str_to_enum_map.end(),
                        "[MaxPool] data_format string is invalid.");
    format = format_str_to_enum_map[data_format_str];
  }
  switch (format) {
    case FORMAT_NHWC:
      params.depth = tensor_in_shapes[kFormatNHWCIndexC];
      params.tensor_rows = tensor_in_shapes[kFormatNHWCIndexH];
      params.tensor_cols = tensor_in_shapes[kFormatNHWCIndexW];
      params.tensor_batch = tensor_in_shapes[kFormatNHWCIndexN];
      params.ksize_rows = ksize[kFormatNHWCIndexH];
      params.ksize_cols = ksize[kFormatNHWCIndexW];
      params.ksize_depth = ksize[kFormatNHWCIndexC];
      params.strides_rows = strides[kFormatNHWCIndexH];
      params.strides_cols = strides[kFormatNHWCIndexW];
      params.strides_depth = strides[kFormatNHWCIndexC];
      break;
    case FORMAT_NCHW:
      params.depth = tensor_in_shapes[kFormatNCHWIndexC];
      params.tensor_rows = tensor_in_shapes[kFormatNCHWIndexH];
      params.tensor_cols = tensor_in_shapes[kFormatNCHWIndexW];
      params.tensor_batch = tensor_in_shapes[kFormatNCHWIndexN];
      params.ksize_rows = ksize[kFormatNCHWIndexH];
      params.ksize_cols = ksize[kFormatNCHWIndexW];
      params.ksize_depth = ksize[kFormatNCHWIndexC];
      params.strides_rows = strides[kFormatNCHWIndexH];
      params.strides_cols = strides[kFormatNCHWIndexW];
      params.strides_depth = strides[kFormatNCHWIndexC];
      break;
    default:
      KERNEL_LOG_ERROR("[MaxPool] Format is not NHWC or NCHW, current is [%s].", FormatToSerialString(format).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  auto ret1 = GetOutputSize(params.tensor_rows, params.ksize_rows, params.strides_rows, padding, &params.out_height,
                            &params.pad_top, &params.pad_bottom);
  auto ret2 = GetOutputSize(params.tensor_cols, params.ksize_cols, params.strides_cols, padding, &params.out_width,
                            &params.pad_left, &params.pad_right);
  KERNEL_CHECK_FALSE(ret1 == KERNEL_STATUS_OK && ret2 == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                     "[MaxPool] An error occurred while calculating output size.");
  params.out_depth = params.depth;
  return KERNEL_STATUS_OK;
}
template <class T>
uint32_t SpacialMaxPool(CpuKernelContext &ctx, const PoolParams &params) {
  Tensor *input = ctx.Input(kFirstInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);

  const T *raw_input_data = static_cast<T *>(input->GetData());
  T *raw_output_data = static_cast<T *>(output->GetData());
  auto shard_NCHW = [&params, &raw_input_data, &raw_output_data](int64_t start, int64_t limit) {
    typedef Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenArrayMap;
    typedef Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> EigenArrayMap;
    const int64_t batch_size = limit;
    const int64_t X_W = static_cast<int64_t>(params.tensor_cols);
    const int64_t X_H = static_cast<int64_t>(params.tensor_rows);
    const int64_t Y_W = params.out_width;
    const int64_t Y_H = params.out_height;
    const int64_t X_HxW = X_H * X_W;
    const int64_t Y_HxW = Y_H * Y_W;
    const int64_t X_stride = X_HxW;
    const int64_t Y_stride = Y_HxW;
    const int64_t stride_h = static_cast<int64_t>(params.strides_rows);
    const int64_t stride_w = static_cast<int64_t>(params.strides_cols);
    const int64_t pad_t = params.pad_top;
    const int64_t pad_l = params.pad_left;
    const int64_t kernel_h = static_cast<int64_t>(params.ksize_rows);
    const int64_t kernel_w = static_cast<int64_t>(params.ksize_cols);
    const T *x_ptr = raw_input_data + start * X_stride;
    T *y_ptr = raw_output_data + start * Y_stride;
    for (int64_t i = start; i < batch_size; ++i) {
      ConstEigenArrayMap x_arr(x_ptr, X_W, X_H);
      EigenArrayMap y_arr(y_ptr, Y_W, Y_H);
      for (int64_t h = 0; h < Y_H; ++h) {
        const int64_t t = std::max(h * stride_h - pad_t, static_cast<int64_t>(0));
        const int64_t b = std::min(h * stride_h - pad_t + kernel_h, X_H);
        for (int64_t w = 0; w < Y_W; ++w) {
          const int64_t l = std::max(w * stride_w - pad_l, static_cast<int64_t>(0));
          const int64_t r = std::min(w * stride_w - pad_l + kernel_w, X_W);
          const int64_t y = h * Y_W + w;
          y_arr(y) = x_arr.block(l, t, r - l, b - t).maxCoeff();
        }
      }
      x_ptr += X_stride;
      y_ptr += Y_stride;
    }
  };
  auto shard_NHWC = [&params, &raw_input_data, &raw_output_data](int64_t start, int64_t limit) {
    typedef Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenArrayMap;
    typedef Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> EigenArrayMap;
    const int64_t batch_size = limit;
    const int64_t X_W = static_cast<int64_t>(params.tensor_cols);
    const int64_t X_H = static_cast<int64_t>(params.tensor_rows);
    const int64_t Y_W = params.out_width;
    const int64_t Y_H = params.out_height;
    const int64_t X_HxW = X_H * X_W;
    const int64_t Y_HxW = Y_H * Y_W;
    const int64_t C = static_cast<int64_t>(params.depth);
    const int64_t X_stride = X_HxW * C;
    const int64_t Y_stride = Y_HxW * C;
    const int64_t stride_h = static_cast<int64_t>(params.strides_rows);
    const int64_t stride_w = static_cast<int64_t>(params.strides_cols);
    const int64_t pad_t = params.pad_top;
    const int64_t pad_l = params.pad_left;
    const int64_t kernel_h = static_cast<int64_t>(params.ksize_rows);
    const int64_t kernel_w = static_cast<int64_t>(params.ksize_cols);
    const T *x_ptr = raw_input_data + start * X_stride;
    T *y_ptr = raw_output_data + start * Y_stride;
    for (int64_t i = start; i < batch_size; ++i) {
      ConstEigenArrayMap x_arr(x_ptr, C, X_HxW);
      EigenArrayMap y_arr(y_ptr, C, Y_HxW);
      for (int64_t h = 0; h < Y_H; ++h) {
        const int64_t t = std::max(h * stride_h - pad_t, static_cast<int64_t>(0));
        const int64_t b = std::min(h * stride_h - pad_t + kernel_h, X_H);
        for (int64_t w = 0; w < Y_W; ++w) {
          const int64_t l = std::max(w * stride_w - pad_l, static_cast<int64_t>(0));
          const int64_t r = std::min(w * stride_w - pad_l + kernel_w, X_W);
          const int64_t y = h * Y_W + w;
          y_arr.col(y).setConstant(Eigen::NumTraits<T>::lowest());
          for (int64_t xi = t; xi < b; ++xi) {
            for (int64_t yj = l; yj < r; ++yj) {
              y_arr.col(y) = y_arr.col(y).max(x_arr.col(xi * X_W + yj));
            }
          }
        }
      }
      x_ptr += X_stride;
      y_ptr += Y_stride;
    }
  };
  int64_t total_elements = params.tensor_batch * params.tensor_cols * params.tensor_rows * params.depth;
  if (ctx.GetAttr("data_format") != nullptr && ctx.GetAttr("data_format")->GetString() == "NCHW") {
    int64_t total_images = params.tensor_batch * params.depth;
    KERNEL_LOG_INFO("[MaxPool] Calling new shard_NCHW");
    if (total_elements <= kParallelNum) {
      shard_NCHW(0, total_images);
      return KERNEL_STATUS_OK;
    } else {
      uint32_t max_core_num = aicpu::CpuKernelUtils::GetCPUNum(ctx);
      max_core_num = std::min(total_images, static_cast<int64_t>(max_core_num));
      return CpuKernelUtils::ParallelFor(ctx, total_images, total_images / max_core_num, shard_NCHW);
    }
  } else {
    int64_t total_images_with_chann = params.tensor_batch;
    KERNEL_LOG_INFO("[MaxPool] Calling new shard_NHWC");
    if (total_elements <= kParallelNum) {
      shard_NHWC(0, total_images_with_chann);
      return KERNEL_STATUS_OK;
    } else {
      uint32_t max_core_num = aicpu::CpuKernelUtils::GetCPUNum(ctx);
      max_core_num = std::min(total_images_with_chann, static_cast<int64_t>(max_core_num));
      return CpuKernelUtils::ParallelFor(ctx, total_images_with_chann, total_images_with_chann / max_core_num,
                                         shard_NHWC);
    }
  }
}

template <class T>
uint32_t ComputeMaxPoolImpl(CpuKernelContext &ctx) {
  TensorShape ts = *(ctx.Input(kFirstInputIndex)->GetTensorShape());
  PoolParams params;
  KERNEL_CHECK_FALSE(ConstructPoolParams(ctx, ts, params) == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                     "[MaxPool] Pooling parameters construct failed.")
  return SpacialMaxPool<T>(ctx, params);
}
uint32_t MaxPoolCpuKernel::Compute(CpuKernelContext &ctx) {
  const std::vector<std::string> required_attrs = {"ksize", "strides", "padding"};
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kMaxPoolInputNum, kMaxPoolOutputNum, required_attrs),
                      "[MaxPool] Check input and output number failed.");
  DataType input_type = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeMaxPoolImpl<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeMaxPoolImpl<float>(ctx);
    case DT_DOUBLE:
      return ComputeMaxPoolImpl<double>(ctx);
    case DT_INT8:
      return ComputeMaxPoolImpl<int8_t>(ctx);
    case DT_INT16:
      return ComputeMaxPoolImpl<int16_t>(ctx);
    case DT_INT32:
      return ComputeMaxPoolImpl<int32_t>(ctx);
    case DT_INT64:
      return ComputeMaxPoolImpl<int64_t>(ctx);
    case DT_UINT8:
      return ComputeMaxPoolImpl<uint8_t>(ctx);
    case DT_UINT16:
      return ComputeMaxPoolImpl<uint16_t>(ctx);
    default:
      KERNEL_LOG_ERROR("[MaxPool] Data type [%s] is not supported.", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(MAXPOOL, MaxPoolCpuKernel);
}  // namespace aicpu
