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
#include "maxpool_grad.h"

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "cpu_kernel_utils.h"
#include "utils/allocator_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
namespace {
const char *kMaxPoolGrad = "MaxPoolGrad";
constexpr uint32_t kInvalidMaxPoolingIndex = -1;
constexpr uint32_t kMaxPoolGradInputNum = 3;
constexpr uint32_t kMaxPoolGradOutputNum = 1;
constexpr int64_t kParallelNum_7K = 7 * 1024;
constexpr int64_t kParallelNum_16K = 16 * 1024;
constexpr int64_t kParallelNum_128K = 128 * 1024;
constexpr uint32_t kThirdInputIndex = 2;
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
template <typename T, typename Targmax>
uint32_t SpatialMaxPoolWithArgMaxHelper(CpuKernelContext &ctx, const PoolParams &params) {
  bool include_batch_in_index = true;

  Tensor *tensor_in = ctx.Input(kFirstInputIndex);
  EigenTensor input_eigen_tensor(tensor_in, tensor_in->GetData());
  Tensor *tensor_out = ctx.Input(kSecondInputIndex);
  EigenTensor output_eigen_tensor(tensor_out, tensor_out->GetData());
  Tensor *tensor_out_backprop = ctx.Input(2);
  EigenTensor out_backprop(tensor_out_backprop, tensor_out_backprop->GetData());
  Tensor *tensor_output_dup = ctx.Output(kFirstOutputIndex);
  EigenTensor input_backprop(tensor_output_dup, tensor_output_dup->GetData());

  // create a new aicpu::Tensor
  auto tensor_out_arg_max_tmp = CpuKernelUtils::CreateTensor();
  Targmax *arg_max = new Targmax[tensor_output_dup->NumElements()];

  TensorShape out_dup_ts = *(tensor_output_dup->GetTensorShape());
  tensor_out_arg_max_tmp->SetDataType(DT_INT64);
  tensor_out_arg_max_tmp->SetData(static_cast<void *>(arg_max));
  tensor_out_arg_max_tmp->SetDataSize(tensor_output_dup->GetDataSize());

  auto out_arg_max_ts = tensor_out_arg_max_tmp->GetTensorShape();
  out_arg_max_ts->SetFormat(out_dup_ts.GetFormat());
  out_arg_max_ts->SetUnknownRank(out_dup_ts.GetUnknownRank());
  out_arg_max_ts->SetDimSizes(out_dup_ts.GetDimSizes());

  auto tensor_out_arg_max = tensor_out_arg_max_tmp.get();
  EigenTensor output_arg_max(tensor_out_arg_max, tensor_out_arg_max->GetData());

  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<Targmax, Eigen::Dynamic, Eigen::Dynamic>> EigenIndexMatrixMap;

  ConstEigenMatrixMap in_mat(input_eigen_tensor.flat<T>().data(), params.depth,
                             params.tensor_cols * params.tensor_rows * params.tensor_batch);
  EigenMatrixMap out_mat(output_eigen_tensor.flat<T>().data(), params.depth,
                         params.out_width * params.out_height * params.tensor_batch);
  EigenIndexMatrixMap out_arg_max_mat(output_arg_max.flat<Targmax>().data(), params.depth,
                                      params.out_width * params.out_height * params.tensor_batch);

  input_backprop.flat<T>().setZero();
  auto orig_input_ptr = static_cast<T *>(tensor_in->GetData());
  auto orig_output_ptr = static_cast<T *>(tensor_out->GetData());
  auto grad_ptr = static_cast<T *>(tensor_out_backprop->GetData());
  auto output_ptr = static_cast<T *>(tensor_output_dup->GetData());
  // shard_NCHW's limit is params.tensor_batch * params.depth
  auto shard_NCHW = [&params, &orig_input_ptr, &orig_output_ptr, &grad_ptr, &output_ptr](int64_t start, int64_t limit) {
    typedef Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenArrayMap;
    typedef Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>> EigenArrayMap;
    const int64_t X_W = static_cast<int64_t>(params.tensor_cols), X_H = static_cast<int64_t>(params.tensor_rows);
    const int64_t Y_W = params.out_width, Y_H = params.out_height;
    const int64_t batch_size = limit;
    const int64_t X_HxW = X_H * X_W, Y_HxW = Y_H * Y_W;
    const int64_t X_stride = X_HxW, Y_stride = Y_HxW;
    const int64_t stride_h = static_cast<int64_t>(params.strides_rows),
                  stride_w = static_cast<int64_t>(params.strides_cols);
    const int64_t pad_t = params.pad_top, pad_l = params.pad_left;
    const int64_t kernel_h = static_cast<int64_t>(params.ksize_rows),
                  kernel_w = static_cast<int64_t>(params.ksize_cols);
    const T *dy_ptr = grad_ptr + start * Y_stride;
    const T *x_ptr = orig_input_ptr + start * X_stride;
    const T *y_ptr = orig_output_ptr + start * Y_stride;
    T *dx_ptr = output_ptr + start * X_stride;
    for (int64_t i = start; i < batch_size; i++) {
      ConstEigenArrayMap dy_arr(dy_ptr, Y_W, Y_H);
      ConstEigenArrayMap x_arr(x_ptr, X_W, X_H);
      ConstEigenArrayMap y_arr(y_ptr, Y_W, Y_H);
      EigenArrayMap dx_arr(dx_ptr, X_W, X_H);
      for (int64_t h = 0; h < Y_H; ++h) {
        const int64_t t = std::max(h * stride_h - pad_t, static_cast<int64_t>(0));
        const int64_t b = std::min(h * stride_h - pad_t + kernel_h, X_H);
        for (int64_t w = 0; w < Y_W; ++w) {
          const int64_t l = std::max(w * stride_w - pad_l, static_cast<int64_t>(0));
          const int64_t r = std::min(w * stride_w - pad_l + kernel_w, X_W);
          const int64_t y = h * Y_W + w;
          auto some_max_block = (x_arr.block(l, t, r - l, b - t) == y_arr(y)).template cast<T>();
          int64_t first_max_x_rel = 0, first_max_y_rel = 0;
          bool max_found = false;
          for (int64_t by = 0; by < b - t; ++by) {
            for (int64_t bx = 0; bx < r - l; ++bx) {
              if (some_max_block(bx, by) == static_cast<T>(1)) {
                first_max_x_rel = bx, first_max_y_rel = by, max_found = true;
                break;
              }
            }
            if (max_found) {
              break;
            }
          }
          const int64_t fact_index_h = t + first_max_y_rel, fact_index_w = l + first_max_x_rel;
          *(dx_ptr + fact_index_h * X_W + fact_index_w) += static_cast<T>(1) * dy_arr(y);
        }
      }
      dy_ptr += Y_stride;
      x_ptr += X_stride;
      y_ptr += Y_stride;
      dx_ptr += X_stride;
    }
  };
  auto shard = [&params, &in_mat, &out_mat, &out_arg_max_mat, &input_backprop, &output_arg_max, &out_backprop,
                &tensor_out_backprop, include_batch_in_index](int64_t start, int64_t limit) {
    const int32_t depth = params.depth;
    const int32_t in_rows = params.tensor_rows;
    const int32_t in_cols = params.tensor_cols;
    const int32_t pad_top = params.pad_top;
    const int32_t pad_left = params.pad_left;
    const int32_t window_rows = params.ksize_rows;
    const int32_t window_cols = params.ksize_cols;
    const int32_t row_stride = params.strides_rows;
    const int32_t col_stride = params.strides_cols;
    const int32_t out_height = params.out_height;
    const int32_t out_width = params.out_width;
    {
      const int32_t output_image_size = out_height * out_width * depth;
      EigenMatrixMap out_shard(out_mat.data() + start * output_image_size, 1, (limit - start) * output_image_size);
      out_shard.setConstant(Eigen::NumTraits<T>::lowest());
      EigenIndexMatrixMap out_arg_max_shard(out_arg_max_mat.data() + start * output_image_size, 1,
                                            (limit - start) * output_image_size);
      out_arg_max_shard.setConstant(kInvalidMaxPoolingIndex);
    }

    for (int64_t b = start; b < limit; ++b) {
      for (int h = 0; h < in_rows; ++h) {
        for (int w = 0; w < in_cols; ++w) {
          const int hpad = h + pad_top;
          const int wpad = w + pad_left;
          const int h_start = (hpad < window_rows) ? 0 : (hpad - window_rows) / row_stride + 1;
          const int h_end = std::min(hpad / row_stride + 1, out_height);
          const int w_start = (wpad < window_cols) ? 0 : (wpad - window_cols) / col_stride + 1;
          const int w_end = std::min(wpad / col_stride + 1, out_width);
          const int64_t in_index = (b * in_rows + h) * in_cols + w;
          for (int ph = h_start; ph < h_end; ++ph) {
            const int64_t out_index_base = (b * out_height + ph) * out_width;
            for (int pw = w_start; pw < w_end; ++pw) {
              const int64_t out_index = out_index_base + pw;
              for (int d = 0; d < depth; ++d) {
                const T &input_ref = in_mat.coeffRef(d, in_index);
                T &output_ref = out_mat.coeffRef(d, out_index);
                Targmax &out_arg_max_ref = out_arg_max_mat.coeffRef(d, out_index);
                if (output_ref < input_ref || out_arg_max_ref == kInvalidMaxPoolingIndex) {
                  output_ref = input_ref;
                  if (include_batch_in_index) {
                    out_arg_max_ref = in_index * depth + d;
                  } else {
                    out_arg_max_ref = (h * in_cols + w) * depth + d;
                  }
                }
              }
            }
          }
        }
      }
    }
    if (include_batch_in_index) {
      auto input_backprop_flat = input_backprop.flat<T>();
      auto out_arg_max_flat = output_arg_max.flat<int64_t>();
      auto out_backprop_flat = out_backprop.flat<T>();
      const int64_t in_size = in_rows * in_cols * depth;
      const int64_t in_start = start * in_size;
      const int64_t in_end = limit * in_size;
      EigenMatrixMap in_shard(input_backprop_flat.data() + in_start, 1, in_end - in_start);
      in_shard.setConstant(T(0));

      // Backpropagate.
      const int out_size = out_height * out_width * depth;
      const int out_start = start * out_size;
      const int out_end = limit * out_size;
      for (int index = out_start; index < out_end; ++index) {
        int input_backprop_index = out_arg_max_flat(index);
        // BoundsCheck
        if (input_backprop_index - in_start >= 0 && input_backprop_index - in_end < 0) {
          if (index < (tensor_out_backprop->NumElements())) {
            input_backprop_flat(input_backprop_index) += out_backprop_flat(index);
          }
        } else {
          KERNEL_LOG_ERROR("[MaxPoolGrad] Backpropagate boundsCheck failed");
          return KERNEL_STATUS_PARAM_INVALID;
        }
      }
    }
    return KERNEL_STATUS_PARAM_INVALID;
  };

  const int64_t total_elements = params.tensor_batch * params.tensor_rows * params.tensor_cols * params.depth;
  if (ctx.GetAttr("data_format") != nullptr && ctx.GetAttr("data_format")->GetString() == "NCHW") {
    const int64_t total_images = params.tensor_batch * params.depth;
    if (total_elements <= kParallelNum_16K) {
      shard_NCHW(0, total_images);
      return KERNEL_STATUS_OK;
    } else {
      return CpuKernelUtils::ParallelFor(ctx, total_images, 1, shard_NCHW);
    }
  }
  uint32_t tensor_batch = params.tensor_batch;
  if (total_elements <= kParallelNum_7K) {
    shard(0, params.tensor_batch);
    return KERNEL_STATUS_OK;
  } else {
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (total_elements <= kParallelNum_16K) {
      max_core_num = std::min(max_core_num, 4U);
    }
    if (total_elements >= kParallelNum_128K || max_core_num > tensor_batch) {
      max_core_num = params.tensor_batch;
    }
    return CpuKernelUtils::ParallelFor(ctx, params.tensor_batch, params.tensor_batch / max_core_num, shard);
  }
}
uint32_t CheckMaxPoolGrad(CpuKernelContext &ctx) {
  Tensor *tensor_in = ctx.Input(kFirstInputIndex);
  Tensor *tensor_out = ctx.Input(kSecondInputIndex);
  Tensor *out_backprop = ctx.Input(kThirdInputIndex);
  const std::vector<std::string> attr = {"ksize", "strides", "padding"};

  KERNEL_CHECK_FALSE(NormalCheck(ctx, kMaxPoolGradInputNum, kMaxPoolGradOutputNum, attr) == KERNEL_STATUS_OK,
                     KERNEL_STATUS_PARAM_INVALID, "[MaxPoolGrad] NormalCheck input and output failed.");
  // check tensor_in dims
  Tensor &input0 = *(tensor_in);
  auto input_shape_ptr = input0.GetTensorShape();
  KERNEL_CHECK_FALSE(input_shape_ptr->GetDims() == 4, KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] tensor expected for input(0).");
  // check tensor_out dims
  Tensor &input1 = *(tensor_out);
  auto output_shape_ptr = input1.GetTensorShape();
  KERNEL_CHECK_FALSE(output_shape_ptr->GetDims() == 4, KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] tensor expected for input(1).");
  // check out_backprop dims
  Tensor &input2 = *(out_backprop);
  auto grad_shape_ptr = input2.GetTensorShape();
  KERNEL_CHECK_FALSE(grad_shape_ptr->GetDims() == 4, KERNEL_STATUS_PARAM_INVALID,
                     "Non-empty [4D] tensor expected for input(2).");
  // check output data
  KERNEL_LOG_DEBUG("[MaxPoolGrad] Parameters check pass.");
  return KERNEL_STATUS_OK;
}
uint32_t GetOutputSizeGrad(int input_size, int kernel_size, int stride, const std::string &padding,
                           int64_t *output_size, int64_t *padding_before, int64_t *padding_after) {
  KERNEL_CHECK_FALSE(stride > 0, KERNEL_STATUS_PARAM_INVALID, "[MaxPoolGrad] Stride must be positive.");
  std::string same("SAME"), valid("VALID");
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
    KERNEL_LOG_ERROR("[MaxPoolGrad] Padding is invalid.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (*output_size < 0) {
    KERNEL_LOG_ERROR("[MaxPoolGrad] Computed output size is negative.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
uint32_t ConstructPoolParams(aicpu::CpuKernelContext &ctx, const aicpu::TensorShape &data_format, PoolParams &params) {
  Format format = data_format.GetFormat();
  KERNEL_CHECK_FALSE((format == FORMAT_NHWC || format == FORMAT_NCHW), KERNEL_STATUS_PARAM_INVALID,
                     "[MaxPoolGrad] Format is not NHWC or NCHW.");
  std::vector<int64_t> tensor_in_shapes = data_format.GetDimSizes();
  std::vector<int64_t> ksize = ctx.GetAttr("ksize")->GetListInt(), strides = ctx.GetAttr("strides")->GetListInt();
  std::string padding = ctx.GetAttr("padding")->GetString();
  std::string data_format_str = "";
  if (ctx.GetAttr("data_format") == nullptr) {
    KERNEL_LOG_INFO("[MaxPoolGrad] Attr data_format is empty, using default value NHWC.");
    format = FORMAT_NHWC;
  } else {
    std::map<std::string, aicpu::Format> format_str_to_enum_map = {{"NHWC", FORMAT_NHWC}, {"NCHW", FORMAT_NCHW}};
    data_format_str = ctx.GetAttr("data_format")->GetString();

    KERNEL_HANDLE_ERROR(format_str_to_enum_map.find(data_format_str) == format_str_to_enum_map.end(),
                        "[MaxPoolGrad] data_format string is invalid.");
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
      KERNEL_LOG_ERROR("[MaxPoolGrad] Format is not NHWC or NCHW, current is [%d].", format);
      return KERNEL_STATUS_PARAM_INVALID;
  }
  // 1 types of pooling is supported: 2d pooling on w/h
  // depth pooling on channel is not supported
  KERNEL_CHECK_FALSE(params.ksize_depth == 1, KERNEL_STATUS_PARAM_INVALID,
                     "[MaxPoolGrad] Only pooling on width/height is supported.");
  // Padding calc
  if (params.ksize_depth == 1) {
    uint32_t ret1 = GetOutputSizeGrad(params.tensor_rows, params.ksize_rows, params.strides_rows, padding,
                                      &params.out_height, &params.pad_top, &params.pad_bottom);
    uint32_t ret2 = GetOutputSizeGrad(params.tensor_cols, params.ksize_cols, params.strides_cols, padding,
                                      &params.out_width, &params.pad_left, &params.pad_right);
    KERNEL_CHECK_FALSE(ret1 == KERNEL_STATUS_OK && ret2 == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                       "[MaxPoolGrad] An error occurred while calculating output size.");
    params.out_depth = params.depth;
  }
  return KERNEL_STATUS_OK;
}
template <class T>
uint32_t ComputeMaxPoolGradImpl(CpuKernelContext &ctx) {
  TensorShape ts = *(ctx.Input(kFirstInputIndex)->GetTensorShape());
  PoolParams params;
  KERNEL_CHECK_FALSE(ConstructPoolParams(ctx, ts, params) == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                     "[MaxPoolGrad] Parameters construct failed.")
  return SpatialMaxPoolWithArgMaxHelper<T, int64_t>(ctx, params);
}
uint32_t MaxPoolGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_CHECK_FALSE(CheckMaxPoolGrad(ctx) == KERNEL_STATUS_OK, KERNEL_STATUS_PARAM_INVALID,
                     "[MaxPoolGrad] Parameters check failure.");
  DataType input_type = ctx.Input(kFirstInputIndex)->GetDataType();
  switch (input_type) {
    case DT_FLOAT16:
      return ComputeMaxPoolGradImpl<Eigen::half>(ctx);
    case DT_FLOAT:
      return ComputeMaxPoolGradImpl<float>(ctx);
    case DT_DOUBLE:
      return ComputeMaxPoolGradImpl<double>(ctx);
    case DT_INT8:
      return ComputeMaxPoolGradImpl<int8_t>(ctx);
    case DT_INT16:
      return ComputeMaxPoolGradImpl<int16_t>(ctx);
    case DT_INT32:
      return ComputeMaxPoolGradImpl<int32_t>(ctx);
    case DT_INT64:
      return ComputeMaxPoolGradImpl<int64_t>(ctx);
    case DT_UINT8:
      return ComputeMaxPoolGradImpl<uint8_t>(ctx);
    case DT_UINT16:
      return ComputeMaxPoolGradImpl<uint16_t>(ctx);
    case DT_UINT32:
      return ComputeMaxPoolGradImpl<uint32_t>(ctx);
    case DT_UINT64:
      return ComputeMaxPoolGradImpl<uint64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("[MaxPoolGrad] Input Data type [%s] is not supported.", DTypeStr(input_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kMaxPoolGrad, MaxPoolGradCpuKernel);
}  // namespace aicpu