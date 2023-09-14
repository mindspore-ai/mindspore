/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "cpu_kernel/ms_kernel/avgpoolgrad.h"

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <algorithm>
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "cpu_kernel/inc/cpu_context.h"

namespace {
const char *AVGPOOLGRAD = "AvgPoolGrad";
constexpr uint32_t kAvgPoolGradInputNum = 2;
constexpr uint32_t kAvgPoolGradOutputNum = 1;
constexpr int64_t kParallelNum_7K = 7 * 1024;
constexpr int64_t kParallelNum_16K = 16 * 1024;
}  // namespace

namespace aicpu {

uint32_t GetBroadcastSize(const int index, const int in_size, const int ksize, const int stride, const int pad_size,
                          int *bindex, int *bsize) {
  // Cannot have index beyond the input size.
  if (index * stride > in_size) {
    KERNEL_LOG_ERROR("index * stride must be less than or equal to input size");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  *bindex = index * stride;
  *bsize = ksize;
  if (*bindex < pad_size) {
    // If the current index is in the padding area, start broadcast  from index
    // 0 with broadcast size reduced by padding size.
    *bsize = ksize + *bindex - pad_size;
    *bindex = 0;
  } else {
    // Otherwise, start broadcast from current index reduced by padding size.
    *bindex -= pad_size;
  }
  if (*bindex + ksize > in_size) {
    *bsize = std::min((in_size - *bindex), ksize);
  }
  return KERNEL_STATUS_OK;
}

uint32_t GetOutputSize(int64_t input_size, int64_t kernel_size, int64_t stride, const std::string &padding,
                       int64_t *output_size, int64_t *padding_before, int64_t *padding_after) {
  KERNEL_CHECK_FALSE(stride > 0, KERNEL_STATUS_PARAM_INVALID, "[AvgPoolGrad] Stride must be positive.");
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
    KERNEL_LOG_ERROR("[AvgPoolGrad] Padding is invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (*output_size < 0) {
    KERNEL_LOG_ERROR("[AvgPoolGrad] Computed output size is negative.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

uint32_t CheckAvgPoolGrad(const CpuKernelContext &ctx) {
  // Check whether input or output is nullptr
  Tensor *tensor_in_shape = ctx.Input(0);
  Tensor *out_backprop = ctx.Input(1);
  const std::vector<std::string> attr = {"ksize", "strides", "padding"};

  KERNEL_CHECK_FALSE(NormalCheck(const_cast<CpuKernelContext &>(ctx), kAvgPoolGradInputNum, kAvgPoolGradOutputNum,
                                 attr) == KERNEL_STATUS_OK,
                     KERNEL_STATUS_PARAM_INVALID, "[AvgPoolGrad] NormalCheck input and output failed.");

  // For avgpooling, tensor_in_shape should have 1 dimension, and 4 elements.
  KERNEL_CHECK_FALSE(tensor_in_shape->GetTensorShape()->GetDims() == 1 && tensor_in_shape->NumElements() == 4,
                     KERNEL_STATUS_PARAM_INVALID,
                     "[AvgPoolGrad] origin_tensor_shape must be 1-dimensional and 4 "
                     "elements");

  // For avgpooling, input_grad should have 4 dimensions.
  KERNEL_CHECK_FALSE(out_backprop->GetTensorShape()->GetDims() == 4, KERNEL_STATUS_PARAM_INVALID,
                     "[AvgPoolGrad] input_grad must be 4-dimensional");

  // Check tensor_in_shape is int32 or not
  DataType tensor_in_shape_type = tensor_in_shape->GetDataType();
  if (tensor_in_shape_type != DT_INT32) {
    KERNEL_LOG_ERROR(
      "[AvgPoolGrad] Please make sure that type of orig_input_shape"
      "satisfied: int32_t");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Check size of ksize and other constraints
  AttrValue *attr_ksize = ctx.GetAttr("ksize");
  std::vector<int64_t> ksize = attr_ksize->GetListInt();
  if (ksize.size() != 4) {
    KERNEL_LOG_ERROR("[AvgPoolGrad] Size of ksize must be 4.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // Check size of strides and other constraints
  AttrValue *attr_strides = ctx.GetAttr("strides");
  std::vector<int64_t> strides = attr_strides->GetListInt();
  if (strides.size() != 4) {
    KERNEL_LOG_ERROR("[AvgPoolGrad] Size of strides must be 4.");
    return KERNEL_STATUS_PARAM_INVALID;
  }

  // check data format string, optional
  AttrValue *attr_data_format = ctx.GetAttr("data_format");
  std::string data_format_NCHW("NCHW");
  std::string data_format_NHWC("NHWC");
  std::string data_format;

  if (attr_data_format == nullptr) {
    data_format = data_format_NHWC;
  } else {
    data_format = attr_data_format->GetString();
    bool data_format_cond = (data_format_NCHW == data_format) || (data_format_NHWC == data_format);
    if (!data_format_cond) {
      KERNEL_LOG_ERROR(
        "[AvgPoolGrad] Parameter data_format must be one of the following: "
        "NCHW, "
        "NHWC.");
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }

  KERNEL_LOG_DEBUG("[AvgPoolGrad] Parameters check pass.");
  return KERNEL_STATUS_OK;
}

template <class T>
uint32_t ComputeAvgPoolGradImpl(const CpuKernelContext &ctx) {
  Tensor *tensor_in_shape = ctx.Input(0);
  EigenTensor tensor_in_shape_eigen_tensor(tensor_in_shape, tensor_in_shape->GetData());
  Tensor *out_backprop = ctx.Input(1);
  EigenTensor out_backprop_eigen_tensor(out_backprop, out_backprop->GetData());

  int64_t out_backprop_batch = 0;
  int64_t out_backprop_depth = 0;
  int64_t out_backprop_rows = 0;
  int64_t out_backprop_cols = 0;

  int32_t *dims;
  int64_t in_rows = 0;
  int64_t in_cols = 0;

  // ksize
  AttrValue *attr_ksize = ctx.GetAttr("ksize");
  std::vector<int64_t> ksize = attr_ksize->GetListInt();
  int64_t window_rows = ksize[1];
  int64_t window_cols = ksize[2];
  int64_t depth_window = ksize[3];

  // strides
  AttrValue *attr_strides = ctx.GetAttr("strides");
  std::vector<int64_t> strides = attr_strides->GetListInt();
  int64_t row_stride = strides[1];
  int64_t col_stride = strides[2];

  // data_format
  AttrValue *attr_data_format = ctx.GetAttr("data_format");
  std::string data_format_NCHW("NCHW");
  std::string data_format_NHWC("NHWC");
  std::string data_format;
  if (attr_data_format == nullptr) {
    data_format = data_format_NHWC;
  } else {
    data_format = attr_data_format->GetString();
  }

  if (data_format_NCHW == data_format) {
    out_backprop_batch = out_backprop->GetTensorShape()->GetDimSize(0);
    out_backprop_depth = out_backprop->GetTensorShape()->GetDimSize(1);
    out_backprop_rows = out_backprop->GetTensorShape()->GetDimSize(2);
    out_backprop_cols = out_backprop->GetTensorShape()->GetDimSize(3);

    dims = reinterpret_cast<int32_t *>(tensor_in_shape->GetData());
    in_rows = static_cast<int64_t>(*(dims + 2));
    in_cols = static_cast<int64_t>(*(dims + 3));

    depth_window = ksize[1];
    window_rows = ksize[2];
    window_cols = ksize[3];

    row_stride = strides[2];
    col_stride = strides[3];

  } else if (data_format_NHWC == data_format) {
    out_backprop_batch = out_backprop->GetTensorShape()->GetDimSize(0);
    out_backprop_rows = out_backprop->GetTensorShape()->GetDimSize(1);
    out_backprop_cols = out_backprop->GetTensorShape()->GetDimSize(2);
    out_backprop_depth = out_backprop->GetTensorShape()->GetDimSize(3);

    dims = reinterpret_cast<int *>(tensor_in_shape->GetData());
    in_rows = static_cast<int64_t>(*(dims + 1));
    in_cols = static_cast<int64_t>(*(dims + 2));

    window_rows = ksize[1];
    window_cols = ksize[2];
    depth_window = ksize[3];

    row_stride = strides[1];
    col_stride = strides[2];
  }

  Tensor *output = ctx.Output(kFirstOutputIndex);
  EigenTensor output_eigen_tensor(output, output->GetData());
  output_eigen_tensor.flat<T>().setZero();

  int64_t out_height;
  int64_t out_width;
  int64_t pad_rows;
  int64_t pad_cols;
  int64_t padding_rows_after;
  int64_t padding_cols_after;

  KERNEL_CHECK_FALSE(depth_window == 1, KERNEL_STATUS_PARAM_INVALID,
                     "Non-spatial pooling is not"
                     "yet supported. Volunteers? :)");
  KERNEL_CHECK_FALSE(GetOutputSize(in_rows, window_rows, row_stride, ctx.GetAttr("padding")->GetString(), &out_height,
                                   &pad_rows, &padding_rows_after) == KERNEL_STATUS_OK,
                     KERNEL_STATUS_PARAM_INVALID, "[AvgPoolingGrad] Getoutputsize error.")
  KERNEL_CHECK_FALSE(GetOutputSize(in_cols, window_cols, col_stride, ctx.GetAttr("padding")->GetString(), &out_width,
                                   &pad_cols, &padding_cols_after) == KERNEL_STATUS_OK,
                     KERNEL_STATUS_PARAM_INVALID, "[AvgPoolingGrad] Getoutputsize error.")

  auto out_backprop_ptr = out_backprop_eigen_tensor.flat<T>().data();
  auto input_backprop_ptr = output_eigen_tensor.flat<T>().data();

  // shard_NCHW's limit is batch_size * depth
  auto shard_NCHW = [&out_backprop_ptr, &input_backprop_ptr, &out_backprop_batch, &out_backprop_cols,
                     &out_backprop_rows, &in_cols, &in_rows, &row_stride, &col_stride, &window_cols, &window_rows,
                     &pad_cols, &pad_rows](int64_t start, int64_t limit) {
    typedef Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenArrayMap;
    typedef Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> EigenArrayMap;
    const int64_t X_W = in_cols;
    const int64_t X_H = in_rows;
    const int64_t Y_W = out_backprop_cols;
    const int64_t Y_H = out_backprop_rows;
    const int64_t batch_size = limit;
    const int64_t X_HxW = X_H * X_W;
    const int64_t Y_HxW = Y_H * Y_W;
    const int64_t X_stride = X_HxW;
    const int64_t Y_stride = Y_HxW;
    const T *dy_ptr = out_backprop_ptr + start * Y_stride;
    T *dx_ptr = input_backprop_ptr + start * X_stride;
    const int64_t stride_h = row_stride;
    const int64_t stride_w = col_stride;
    const int64_t kernel_h = window_rows;
    const int64_t kernel_w = window_cols;
    const int64_t pad_t = pad_rows;
    const int64_t pad_l = pad_cols;
    for (int64_t i = start; i < batch_size; ++i) {
      ConstEigenArrayMap dy_arr(dy_ptr, Y_W, Y_H);
      EigenArrayMap dx_arr(dx_ptr, X_W, X_H);
      for (int h = 0; h < Y_H; ++h) {
        const int64_t t = std::max(h * stride_h - pad_t, static_cast<int64_t>(0));
        const int64_t b = std::min(h * stride_h - pad_t + kernel_h, X_H);
        for (int w = 0; w < Y_W; ++w) {
          const int64_t l = std::max(w * stride_w - pad_l, static_cast<int64_t>(0));
          const int64_t r = std::min(w * stride_w - pad_l + kernel_w, X_W);
          const int64_t y = h * Y_W + w;
          const T scale = T(1) / static_cast<T>((b - t) * (r - l));
          dx_arr.block(l, t, r - l, b - t) += dy_arr(y) * scale;
        }
      }
      dy_ptr += Y_stride;
      dx_ptr += X_stride;
    }
  };
  auto shard_NHWC = [&ctx, &out_backprop_ptr, &input_backprop_ptr, &out_backprop_rows, &out_backprop_cols,
                     &out_backprop_depth, &in_rows, &in_cols, &window_rows, &window_cols, &row_stride, &col_stride,
                     &pad_rows, &pad_cols](int64_t start, int64_t limit) {
    for (int64_t b = start; b < limit; ++b) {
      for (int64_t r = 0; r < out_backprop_rows; ++r) {
        int rindex;
        int rsize;
        KERNEL_CHECK_FALSE(
          GetBroadcastSize(r, in_rows, window_rows, row_stride, pad_rows, &rindex, &rsize) == KERNEL_STATUS_OK,
          KERNEL_STATUS_INNER_ERROR, "[AvgPoolGrad] An error happened during calculation.")

        for (int64_t c = 0; c < out_backprop_cols; ++c) {
          int cindex;
          int csize;
          KERNEL_CHECK_FALSE(
            GetBroadcastSize(c, in_cols, window_cols, col_stride, pad_cols, &cindex, &csize) == KERNEL_STATUS_OK,
            KERNEL_STATUS_INNER_ERROR, "[AvgPoolGrad] An error happened during calculation.")

          T divide_coeff(1.0 / (rsize * csize));
          int64_t output_index = (b * out_backprop_rows + r) * out_backprop_cols + c;
          for (int64_t r_dst = rindex; r_dst < rindex + rsize; ++r_dst) {
            for (int64_t c_dst = cindex; c_dst < cindex + csize; ++c_dst) {
              int64_t input_index = (b * in_rows + r_dst) * in_cols + c_dst;
              const T *output_offset = out_backprop_ptr + output_index * out_backprop_depth;
              T *input_offset = input_backprop_ptr + input_index * out_backprop_depth;
              for (int64_t d = 0; d < out_backprop_depth; ++d) {
                *input_offset += *output_offset * divide_coeff;
                ++output_offset;
                ++input_offset;
              }
            }
          }
        }
      }
    }
    return KERNEL_STATUS_OK;
  };

  int64_t total_elements = out_backprop_batch * in_cols * in_rows * out_backprop_depth;

  if (data_format_NCHW == data_format) {
    KERNEL_LOG_INFO("[AvgPoolGrad] Calling new shard NCHW");
    int64_t total_images = out_backprop_batch * out_backprop_depth;
    if (total_elements <= kParallelNum_7K) {
      shard_NCHW(0, total_images);
      return KERNEL_STATUS_OK;
    } else {
      uint32_t min_core_num = 1;
      // Use CpuKernelUtils::GetCPUNum to get the core of AI CPU
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
      if (total_elements <= kParallelNum_16K) {
        max_core_num = std::min(max_core_num, 4U);
      }
      if (max_core_num > total_images) {
        max_core_num = total_images;
      }
      return CpuKernelUtils::ParallelFor(ctx, total_images, total_images / max_core_num, shard_NCHW);
    }
  } else {
    KERNEL_LOG_INFO("[AvgPoolGrad] Calling shard NHWC");
    if (total_elements <= kParallelNum_7K) {
      shard_NHWC(0, out_backprop_batch);
      return KERNEL_STATUS_OK;
    } else {
      uint32_t min_core_num = 1;
      // Use CpuKernelUtils::GetCPUNum to get the core of AI CPU
      uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
      if (total_elements <= kParallelNum_16K) {
        max_core_num = std::min(max_core_num, 4U);
      }
      if (max_core_num > out_backprop_batch) {
        max_core_num = out_backprop_batch;
      }
      return CpuKernelUtils::ParallelFor(ctx, out_backprop_batch, out_backprop_batch / max_core_num, shard_NHWC);
    }
  }
}

uint32_t AvgPoolGradCpuKernel::Compute(CpuKernelContext &ctx) {
  uint32_t check_status = CheckAvgPoolGrad(ctx);
  KERNEL_CHECK_FALSE(check_status == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID, "[AvgPoolGrad] check failure.");
  DataType input_type = ctx.Input(kSecondInputIndex)->GetDataType();
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeAvgPoolGradImpl<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeAvgPoolGradImpl<float>(ctx);
    case DT_DOUBLE:
      return ComputeAvgPoolGradImpl<double>(ctx);
    default:
      KERNEL_LOG_ERROR("[AvgPoolGrad] The data type of input_grad is not supported.", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(AVGPOOLGRAD, AvgPoolGradCpuKernel);
}  // namespace aicpu
