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

#include "fractional_avg_pool_grad.h"

#include <iostream>
#include "Eigen/Dense"
#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const char *kFractionalAvgPoolGrad = "FractionalAvgPoolGrad";
const uint32_t k_InputNum = 4;
const uint32_t k_OutputNum = 1;
const int64_t kParallelDataNum = 32 * 1024;
}  // namespace

namespace aicpu {
uint32_t FractionalAvgPoolGradCpuKernel::FractionalAvgPoolGradParamCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, k_InputNum, k_OutputNum),
                      "FractionalAvgPoolGrad check input and output number failed.");
  Tensor *orig_input_tensor_shape = ctx.Input(0);
  Tensor *out_backprop = ctx.Input(1);
  Tensor *output = ctx.Output(0);
  auto orig_input_shape = orig_input_tensor_shape->GetTensorShape();
  int32_t orig_input_dims = orig_input_shape->GetDims();
  int32_t orig_input_shape_nums = orig_input_tensor_shape->NumElements();
  if (out_backprop->GetDataType() != output->GetDataType()) {
    KERNEL_LOG_ERROR(
      "The data type of the output [%s] need be the same as the out_backprop "
      "[%s]",
      DTypeStr(output->GetDataType()).c_str(), DTypeStr(out_backprop->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  KERNEL_CHECK_FALSE((orig_input_dims == 1 && orig_input_shape_nums == 4), KERNEL_STATUS_PARAM_INVALID,
                     "original input tensor shape must be 1-dimensional and 4 elements.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t FractionalAvgPoolGradCpuKernel::DoCompute(CpuKernelContext &ctx) {
  typedef Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> ConstEigenMatrixMap;
  typedef Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> EigenDoubleMatrixMap;
  const Tensor *orig_input_tensor_shape = ctx.Input(0);
  const Tensor *out_backprop = ctx.Input(1);
  const Tensor *row_pooling_sequence = ctx.Input(2);
  const Tensor *col_pooling_sequence = ctx.Input(3);
  Tensor *output = ctx.Output(0);
  auto output_data = static_cast<T *>(output->GetData());
  AttrValue *overlapping_ = ctx.GetAttr("overlapping");
  bool overlapping = (overlapping_ == nullptr) ? false : (overlapping_->GetBool());
  int32_t row_seq_nums = row_pooling_sequence->NumElements();
  int32_t col_seq_nums = col_pooling_sequence->NumElements();
  auto out_backprop_shape = out_backprop->GetTensorShape();
  const int64_t out_batch = out_backprop_shape->GetDimSize(0);
  const int64_t out_rows = out_backprop_shape->GetDimSize(1);
  const int64_t out_cols = out_backprop_shape->GetDimSize(2);
  const int64_t out_depth = out_backprop_shape->GetDimSize(3);
  KERNEL_CHECK_FALSE((row_seq_nums > out_rows), KERNEL_STATUS_PARAM_INVALID,
                     "Given out_backprop shape [%ld,%ld,%ld,%ld], row_seq_tensor must"
                     " have at least [%ld] elements, but got[%ld].",
                     out_batch, out_rows, out_cols, out_depth, out_rows + 1, row_seq_nums);
  KERNEL_CHECK_FALSE((col_seq_nums > out_cols), KERNEL_STATUS_PARAM_INVALID,
                     "Given out_backprop shape [%ld,%ld,%ld,%ld], col_seq_tensor must"
                     " have at least [%ld] elements, but got[%ld].",
                     out_batch, out_rows, out_cols, out_depth, out_cols + 1, col_seq_nums);
  auto row_seq_data = static_cast<int64_t *>(row_pooling_sequence->GetData());
  auto col_seq_data = static_cast<int64_t *>(col_pooling_sequence->GetData());
  auto orig_input_tensor_shape_data = static_cast<int64_t *>(orig_input_tensor_shape->GetData());
  const int64_t in_batch = *(orig_input_tensor_shape_data);
  const int64_t in_rows = *(orig_input_tensor_shape_data + 1);
  const int64_t in_cols = *(orig_input_tensor_shape_data + 2);
  const int64_t in_depth = *(orig_input_tensor_shape_data + 3);
  int32_t input_nums = orig_input_tensor_shape->NumElements();
  std::vector<int64_t> out_put_dims;
  for (int i = 0; i < input_nums; i++) {
    KERNEL_CHECK_FALSE((*(orig_input_tensor_shape_data + i) > 0), KERNEL_STATUS_PARAM_INVALID,
                       "Each dimension of input must be > 0.");
    out_put_dims.push_back(orig_input_tensor_shape_data[i]);
  }
  int64_t output_nums = in_batch * in_rows * in_cols * in_depth;
  // Create intermediate in_backprop.
  std::vector<double> in_backprop_tensor_temp(output_nums);
  for (int i = 0; i < output_nums; i++) {
    in_backprop_tensor_temp[i] = 0;
    *(output_data + i) = 0;
  }
  EigenDoubleMatrixMap in_backprop_tensor_temp_mat(in_backprop_tensor_temp.data(), in_depth,
                                                   in_cols * in_rows * in_batch);
  ConstEigenMatrixMap out_backprop_mat(reinterpret_cast<T *>(out_backprop->GetData()), out_depth,
                                       out_cols * out_rows * out_batch);
  // Loop through each element of out_backprop and evenly distribute the
  // element to the corresponding pooling cell.
  const int64_t in_max_row_index = in_rows - 1;
  const int64_t in_max_col_index = in_cols - 1;
  if (output_nums < kParallelDataNum) {
    for (int64_t b = 0; b < out_batch; ++b) {
      for (int64_t r = 0; r < out_rows; ++r) {
        const int64_t in_row_start = *(row_seq_data + r);
        int64_t in_row_end = overlapping ? *(row_seq_data + r + 1) : *(row_seq_data + r + 1) - 1;
        in_row_end = std::min(in_row_end, in_max_row_index);
        for (int64_t c = 0; c < out_cols; ++c) {
          const int64_t in_col_start = *(col_seq_data + c);
          int64_t in_col_end = overlapping ? *(col_seq_data + c + 1) : *(col_seq_data + c + 1) - 1;
          in_col_end = std::min(in_col_end, in_max_col_index);
          const int64_t num_elements_in_pooling_cell =
            (in_row_end - in_row_start + 1) * (in_col_end - in_col_start + 1);
          const int64_t out_index = (b * out_rows + r) * out_cols + c;
          // Now we can evenly distribute out_backprop(b, h, w, *) to
          // in_backprop(b, hs:he, ws:we, *).
          for (int64_t in_r = in_row_start; in_r <= in_row_end; ++in_r) {
            for (int64_t in_c = in_col_start; in_c <= in_col_end; ++in_c) {
              const int64_t in_index = (b * in_rows + in_r) * in_cols + in_c;
              // Walk through each channel (depth).
              for (int64_t d = 0; d < out_depth; ++d) {
                const double out_backprop_element = static_cast<double>(out_backprop_mat.coeffRef(d, out_index));
                double &in_backprop_ref = in_backprop_tensor_temp_mat.coeffRef(d, in_index);
                in_backprop_ref += out_backprop_element / num_elements_in_pooling_cell;
              }
            }
          }
        }
      }
    }
  } else {
    uint64_t row_len = out_rows;
    uint32_t min_core_num = 1;
    uint32_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx));
    if (max_core_num > row_len) {
      max_core_num = row_len;
    }
    for (int64_t b = 0; b < out_batch; ++b) {
      auto sharder_fractionalavgpoolgrad_index = [&](size_t start, size_t end) {
        for (size_t r = start; r < end; ++r) {
          const int64_t in_row_start = *(row_seq_data + r);
          int64_t in_row_end = overlapping ? *(row_seq_data + r + 1) : *(row_seq_data + r + 1) - 1;
          in_row_end = std::min(in_row_end, in_max_row_index);
          for (int64_t c = 0; c < out_cols; ++c) {
            const int64_t in_col_start = *(col_seq_data + c);
            int64_t in_col_end = overlapping ? *(col_seq_data + c + 1) : *(col_seq_data + c + 1) - 1;
            in_col_end = std::min(in_col_end, in_max_col_index);
            const int64_t num_elements_in_pooling_cell =
              (in_row_end - in_row_start + 1) * (in_col_end - in_col_start + 1);
            const int64_t out_index = (b * out_rows + r) * out_cols + c;
            // Now we can evenly distribute out_backprop(b, h, w, *) to
            // in_backprop(b, hs:he, ws:we, *).
            for (int64_t in_r = in_row_start; in_r <= in_row_end; ++in_r) {
              for (int64_t in_c = in_col_start; in_c <= in_col_end; ++in_c) {
                const int64_t in_index = (b * in_rows + in_r) * in_cols + in_c;
                // Walk through each channel (depth).
                for (int64_t d = 0; d < out_depth; ++d) {
                  const double out_backprop_element = static_cast<double>(out_backprop_mat.coeffRef(d, out_index));
                  double &in_backprop_ref = in_backprop_tensor_temp_mat.coeffRef(d, in_index);
                  in_backprop_ref += out_backprop_element / num_elements_in_pooling_cell;
                }
              }
            }
          }
        }
      };
      KERNEL_HANDLE_ERROR(
        CpuKernelUtils::ParallelFor(ctx, row_len, row_len / max_core_num, sharder_fractionalavgpoolgrad_index),
        "FractionalAvgPoolGrad Index Compute failed.");
    }
  }
  // Depending on the type, cast double to type T.
  for (int64_t i = 0; i < output_nums; ++i) {
    *(output_data + i) = static_cast<T>(in_backprop_tensor_temp[i]);
  }
  output->GetTensorShape()->SetDimSizes(out_put_dims);
  return KERNEL_STATUS_OK;
}

uint32_t FractionalAvgPoolGradCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(FractionalAvgPoolGradParamCheck(ctx), "Check FractionalAvgPoolGrad params failed.");
  Tensor *out_backprop = ctx.Input(1);
  auto data_type = out_backprop->GetDataType();
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
      KERNEL_LOG_ERROR("FractionalAvgPoolGrad kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kFractionalAvgPoolGrad, FractionalAvgPoolGradCpuKernel);
}  // namespace aicpu
