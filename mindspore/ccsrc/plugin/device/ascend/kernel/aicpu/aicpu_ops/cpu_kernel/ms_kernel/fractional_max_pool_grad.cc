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
#include "fractional_max_pool_grad.h"

#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kFractionalMaxPoolGrad = "FractionalMaxPoolGrad";
const uint32_t k_InputNum = 5;
const uint32_t k_OutputNum = 1;
static const int kInvalidMaxPoolingIndex = -1;
const int64_t kParallelDataNum = 32 * 1024;
const uint32_t tensor_in_and_out_dims = 4;
}  // namespace

namespace aicpu {
uint32_t FractionalMaxPoolGradCpuKernel::FractionalMaxPoolGradParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, k_InputNum, k_OutputNum),
                      "FractionalMaxPoolGrad check input and output number failed.");
  Tensor *orig_input = ctx.Input(0);
  Tensor *orig_output = ctx.Input(1);
  Tensor *out_backprop = ctx.Input(2);
  auto orig_input_shape = orig_input->GetTensorShape();
  int32_t orig_input_dims = orig_input_shape->GetDims();
  auto orig_output_shape = orig_output->GetTensorShape();
  int32_t orig_output_dims = orig_output_shape->GetDims();
  auto out_backprop_shape = out_backprop->GetTensorShape();
  int32_t out_backprop_dims = out_backprop_shape->GetDims();
  if (orig_input->GetDataType() != orig_output->GetDataType()) {
    KERNEL_LOG_ERROR(
      "The data type of the orig_output [%s] need be the same as the "
      "orig_input [%s].",
      DTypeStr(orig_output->GetDataType()).c_str(), DTypeStr(orig_input->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (orig_input->GetDataType() != out_backprop->GetDataType()) {
    KERNEL_LOG_ERROR(
      "The data type of the out_backprop [%s] need be the same as the "
      "orig_input [%s].",
      DTypeStr(out_backprop->GetDataType()).c_str(), DTypeStr(orig_input->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((orig_input_dims == tensor_in_and_out_dims), KERNEL_STATUS_PARAM_INVALID,
                     "orig_input should be a tensor of rank 4.");
  KERNEL_CHECK_FALSE((orig_output_dims == tensor_in_and_out_dims), KERNEL_STATUS_PARAM_INVALID,
                     "orig_output should be a tensor of rank 4.");
  KERNEL_CHECK_FALSE((out_backprop_dims == tensor_in_and_out_dims), KERNEL_STATUS_PARAM_INVALID,
                     "out_backprop should be a tensor of rank 4.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FractionalMaxPoolGradCpuKernel::DoCompute(CpuKernelContext &ctx) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> EigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>> EigenIndexMatrixMap;
  const Tensor *tensor_in = ctx.Input(0);
  const Tensor *tensor_out = ctx.Input(1);
  const Tensor *out_backprop = ctx.Input(2);
  const Tensor *height_seq_tensor = ctx.Input(3);
  const Tensor *width_seq_tensor = ctx.Input(4);
  Tensor *output = ctx.Output(0);
  auto output_data = static_cast<T *>(output->GetData());
  AttrValue *overlapping_ = ctx.GetAttr("overlapping");
  bool overlapping = (overlapping_ == nullptr) ? false : (overlapping_->GetBool());
  auto tensor_in_shape = tensor_in->GetTensorShape();
  auto tensor_out_shape = tensor_out->GetTensorShape();
  std::vector<int64_t> input_size(tensor_in_and_out_dims);
  std::vector<int64_t> output_size(tensor_in_and_out_dims);
  for (uint32_t i = 0; i < tensor_in_and_out_dims; ++i) {
    input_size[i] = tensor_in_shape->GetDimSize(i);
  }
  for (uint32_t i = 0; i < tensor_in_and_out_dims; ++i) {
    output_size[i] = tensor_out_shape->GetDimSize(i);
  }
  int64_t tensor_in_num = tensor_in->NumElements();
  int64_t tensor_out_num = tensor_out->NumElements();
  std::vector<T> tensor_out_dup(tensor_out_num);
  std::vector<int64_t> tensor_out_arg_max(tensor_out_num);
  for (int i = 0; i < tensor_out_num; i++) {
    tensor_out_dup[i] = std::numeric_limits<T>::lowest();
    tensor_out_arg_max[i] = -1;
  }
  // Find arg_max for each tensor_out
  ConstEigenMatrixMap tensor_in_mat(reinterpret_cast<T *>(tensor_in->GetData()), input_size[3],
                                    input_size[2] * input_size[1] * input_size[0]);
  EigenMatrixMap tensor_out_dup_mat(tensor_out_dup.data(), output_size[3],
                                    output_size[2] * output_size[1] * output_size[0]);
  EigenIndexMatrixMap tensor_out_arg_max_mat(tensor_out_arg_max.data(), output_size[3],
                                             output_size[2] * output_size[1] * output_size[0]);
  auto height_seq_tensor_shape = height_seq_tensor->GetTensorShape();
  auto width_seq_tensor_shape = width_seq_tensor->GetTensorShape();
  auto height_seq_tensor_data = static_cast<int64_t *>(height_seq_tensor->GetData());
  auto width_seq_tensor_data = static_cast<int64_t *>(width_seq_tensor->GetData());
  /**
   * Now walk through the process of fractional max pooling again.
   * For both input and output,
   * 0: batch
   * 1: height / row
   * 2: width / col
   * 3: depth / channel
   */
  if (tensor_in_num < kParallelDataNum) {
    const int64_t height_max = input_size[1] - 1;
    const int64_t width_max = input_size[2] - 1;
    for (int64_t b = 0; b < input_size[0]; ++b) {
      // height sequence.
      for (int64_t hs = 0; hs < height_seq_tensor_shape->GetDimSize(0) - 1; ++hs) {
        // height start and end.
        const int64_t height_start = *(height_seq_tensor_data + hs);
        int64_t height_end = overlapping ? *(height_seq_tensor_data + hs + 1) : *(height_seq_tensor_data + hs + 1) - 1;
        height_end = std::min(height_end, height_max);
        // width sequence.
        for (int64_t ws = 0; ws < width_seq_tensor_shape->GetDimSize(0) - 1; ++ws) {
          const int64_t out_index = (b * output_size[1] + hs) * output_size[2] + ws;
          // width start and end.
          const int64_t width_start = *(width_seq_tensor_data + ws);
          int64_t width_end = overlapping ? *(width_seq_tensor_data + ws + 1) : *(width_seq_tensor_data + ws + 1) - 1;
          width_end = std::min(width_end, width_max);
          for (int64_t h = height_start; h <= height_end; ++h) {
            for (int64_t w = width_start; w <= width_end; ++w) {
              const int64_t in_index = (b * input_size[1] + h) * input_size[2] + w;
              // Walk through each channel (depth).
              for (int64_t d = 0; d < input_size[3]; ++d) {
                const T &input_ref = tensor_in_mat.coeffRef(d, in_index);
                T &output_ref = tensor_out_dup_mat.coeffRef(d, out_index);
                int64_t &out_arg_max_ref = tensor_out_arg_max_mat.coeffRef(d, out_index);
                if (output_ref < input_ref || out_arg_max_ref == kInvalidMaxPoolingIndex) {
                  output_ref = input_ref;
                  int input_offset = in_index * input_size[3] + d;
                  out_arg_max_ref = input_offset;
                }
              }
            }
          }
        }
      }
    }
  } else {
    uint64_t height_seq_len = height_seq_tensor_shape->GetDimSize(0) - 1;
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > height_seq_len) {
      max_core_num = height_seq_len;
    }
    const int64_t height_max = input_size[1] - 1;
    const int64_t width_max = input_size[2] - 1;
    for (int64_t b = 0; b < input_size[0]; ++b) {
      // height sequence.
      auto sharder_fractionalmaxpoolgrad_index = [&](size_t start, size_t end) {
        for (size_t hs = start; hs < end; ++hs) {
          // height start and end.
          const int64_t height_start = *(height_seq_tensor_data + hs);
          int64_t height_end =
            overlapping ? *(height_seq_tensor_data + hs + 1) : *(height_seq_tensor_data + hs + 1) - 1;
          height_end = std::min(height_end, height_max);
          // width sequence.
          for (int64_t ws = 0; ws < width_seq_tensor_shape->GetDimSize(0) - 1; ++ws) {
            const int64_t out_index = (b * output_size[1] + hs) * output_size[2] + ws;
            // width start and end.
            const int64_t width_start = *(width_seq_tensor_data + ws);
            int64_t width_end = overlapping ? *(width_seq_tensor_data + ws + 1) : *(width_seq_tensor_data + ws + 1) - 1;
            width_end = std::min(width_end, width_max);
            for (int64_t h = height_start; h <= height_end; ++h) {
              for (int64_t w = width_start; w <= width_end; ++w) {
                const int64_t in_index = (b * input_size[1] + h) * input_size[2] + w;
                // Walk through each channel (depth).
                for (int64_t d = 0; d < input_size[3]; ++d) {
                  const T &input_ref = tensor_in_mat.coeffRef(d, in_index);
                  T &output_ref = tensor_out_dup_mat.coeffRef(d, out_index);
                  int64_t &out_arg_max_ref = tensor_out_arg_max_mat.coeffRef(d, out_index);
                  if (output_ref < input_ref || out_arg_max_ref == kInvalidMaxPoolingIndex) {
                    output_ref = input_ref;
                    int input_offset = in_index * input_size[3] + d;
                    out_arg_max_ref = input_offset;
                  }
                }
              }
            }
          }
        }
      };
      KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, height_seq_len, height_seq_len / max_core_num,
                                                      sharder_fractionalmaxpoolgrad_index),
                          "FractionalMaxPoolGrad Index Compute failed.");
    }
  }
  for (int i = 0; i < tensor_in_num; i++) {
    *(output_data + i) = 0;
  }
  auto out_backprop_data = static_cast<T *>(out_backprop->GetData());
  int num_total_outputs = out_backprop->NumElements();
  int num_total_inputs = output->NumElements();
  for (int index = 0; index < num_total_outputs; ++index) {
    int input_backprop_index = tensor_out_arg_max[index];
    KERNEL_CHECK_FALSE((input_backprop_index >= 0 && input_backprop_index < num_total_inputs),
                       KERNEL_STATUS_PARAM_INVALID,
                       "Invalid input backprop index:[%d], The maximum number of output is: "
                       "[%d].",
                       input_backprop_index, num_total_inputs);
    *(output_data + input_backprop_index) += *(out_backprop_data + index);
  }
  return KERNEL_STATUS_OK;
}

uint32_t FractionalMaxPoolGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(FractionalMaxPoolGradParamCheck(ctx), "Check FractionalMaxPoolGrad params failed.");
  Tensor *input = ctx.Input(0);
  auto data_type = input->GetDataType();
  switch (data_type) {
    case DT_FLOAT:
      return DoCompute<float>(ctx);
    case DT_DOUBLE:
      return DoCompute<double>(ctx);
    case DT_INT32:
      return DoCompute<int32_t>(ctx);
    case DT_INT64:
      return DoCompute<int64_t>(ctx);
    default:
      KERNEL_LOG_ERROR("FractionalMaxPoolGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kFractionalMaxPoolGrad, FractionalMaxPoolGradCpuKernel);
}  // namespace aicpu
