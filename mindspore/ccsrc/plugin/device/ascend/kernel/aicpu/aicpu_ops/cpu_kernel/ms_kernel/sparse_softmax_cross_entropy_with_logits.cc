/**
Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
#include "sparse_softmax_cross_entropy_with_logits.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include "cpu_kernel_utils.h"
#include "cpu_types.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
namespace {
const char *kSparseSoftmaxCrossEntropyWithLogits = "SparseSoftmaxCrossEntropyWithLogits";
const uint32_t kOutputNum{2};
const uint32_t kInputNum{2};
const uint32_t kDimSizeTwo{2};
const uint32_t kDimSizeOne{1};
const uint32_t paralledDataNum{2048};
}  // namespace

namespace aicpu {
template <typename data_type, typename label_type>
void SparseSoftmaxCrossEntropyWithLogitsSingleOp(data_type *input_features, label_type *input_labels,
                                                 data_type *output_loss, data_type *output_backprop, int64_t batch_size,
                                                 int64_t classes_num, size_t features_total) {
  double_t *dims_exp_sum = static_cast<double_t *>(malloc(batch_size * sizeof(double_t)));
  data_type *dims_maximum = static_cast<data_type *>(malloc(batch_size * sizeof(data_type)));
  memset(dims_exp_sum, 0, batch_size * sizeof(double_t));
  Eigen::TensorMap<Eigen::Tensor<data_type, kDimSizeTwo>, Eigen::Aligned> logits(input_features, batch_size,
                                                                                 classes_num);
  Eigen::TensorMap<Eigen::Tensor<double_t, 1>, Eigen::Aligned> dims_sum(dims_exp_sum, batch_size);
  Eigen::TensorMap<Eigen::Tensor<data_type, 1>, Eigen::Aligned> dims_max(dims_maximum, batch_size);
  Eigen::array<int, 1> axes{{1}};
  // compute softmax
  dims_max = logits.maximum(axes);
  const data_type constant_one(1.0);
  for (size_t index = 0, batch_idx = 0; index < features_total; index++) {
    output_backprop[index] = Eigen::numext::exp(input_features[index] - dims_maximum[batch_idx]);
    dims_exp_sum[batch_idx] += static_cast<double_t>(output_backprop[index]);
    if ((index + 1) % classes_num == 0) {
      batch_idx++;
    }
  }
  dims_sum = dims_sum.inverse();
  for (size_t index = 0, batch_idx = 0; index < features_total; index++) {
    *(output_backprop + index) =
      static_cast<data_type>(static_cast<double_t>(*(output_backprop + index)) * dims_exp_sum[batch_idx]);
    if ((index + 1) % classes_num == 0) {
      batch_idx++;
    }
  }
  label_type offset = 0;
  for (int64_t index = 0, batch_base = 0; index < batch_size; ++index, batch_base += classes_num) {
    offset = input_labels[index];
    *(output_loss + index) = -Eigen::numext::log(*(output_backprop + batch_base + offset));
    *(output_backprop + batch_base + offset) = *(output_backprop + batch_base + offset) - constant_one;
  }
  free(dims_exp_sum);
  free(dims_maximum);
}

template <typename data_type, typename label_type>
void SparseSoftmaxCrossEntropyWithLogitsMultiOp(data_type *input_features, label_type *input_labels,
                                                data_type *output_loss, data_type *output_backprop, size_t begin,
                                                size_t end, int64_t classes_num, size_t features_total) {
  for (size_t index = begin; index < end; index++) {
    size_t batch_begin = index * classes_num;
    size_t batch_end = batch_begin + classes_num;
    data_type max_value = input_features[batch_begin];
    double_t sum_value{0};
    data_type constant_one{1};
    for (size_t idx = batch_begin; idx < batch_end; idx++) {
      if (max_value < input_features[idx]) {
        max_value = input_features[idx];
      }
    }
    for (size_t idx = batch_begin; idx < batch_end; idx++) {
      output_backprop[idx] = Eigen::numext::exp(input_features[idx] - max_value);
      sum_value += static_cast<double_t>(output_backprop[idx]);
    }
    sum_value = double_t(1.0) / sum_value;
    for (size_t idx = batch_begin; idx < batch_end; idx++) {
      output_backprop[idx] = static_cast<data_type>(static_cast<double_t>(output_backprop[idx]) * sum_value);
      if (idx % classes_num == static_cast<size_t>(input_labels[index])) {
        output_loss[index] = -Eigen::numext::log(output_backprop[idx]);
        output_backprop[idx] = output_backprop[idx] - constant_one;
      }
    }
  }
}

std::uint32_t SparseSoftmaxCrossEntropyWithLogitsExtraCheck(CpuKernelContext &ctx) {
  Tensor *input_features = ctx.Input(0);
  Tensor *input_labels = ctx.Input(1);
  Tensor *output_loss = ctx.Output(0);
  Tensor *output_backprop = ctx.Output(1);
  std::vector<int64_t> features_dims = input_features->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> labels_dims = input_labels->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> loss_dims = output_loss->GetTensorShape()->GetDimSizes();
  std::vector<int64_t> backprop_dims = output_backprop->GetTensorShape()->GetDimSizes();
  if ((input_features->GetDataSize() == 0) || (input_labels->GetDataSize() == 0)) {
    KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }

  if (input_features->GetDataType() != output_loss->GetDataType() ||
      input_features->GetDataType() != output_backprop->GetDataType()) {
    KERNEL_LOG_ERROR(
      "The data type of the input features [%s], output loss [%s], output "
      "backprop [%s] must be the same type.",
      DTypeStr(ctx.Input(0)->GetDataType()).c_str(), DTypeStr(ctx.Output(0)->GetDataType()).c_str(),
      DTypeStr(ctx.Output(1)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (input_labels->GetDataType() != DT_INT32 && input_labels->GetDataType() != DT_INT64) {
    KERNEL_LOG_ERROR(
      "The data type of the input labels [%s], must be the int32 or int64 "
      "type.",
      DTypeStr(ctx.Input(1)->GetDataType()).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (features_dims.size() != kDimSizeTwo || labels_dims.size() != kDimSizeOne || loss_dims.size() != kDimSizeOne ||
      backprop_dims.size() != kDimSizeTwo) {
    KERNEL_LOG_ERROR(
      "The dims of the input features [%d], output backprop [%d] must be "
      "[batch_size x num_classes]. the dims of input labels [%d], output "
      "loss [%d] must be [batch_size].",
      features_dims.size(), backprop_dims.size(), labels_dims.size(), loss_dims.size());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  int64_t batch_size = features_dims[0];
  int64_t num_classes = features_dims[1];
  if (labels_dims[0] != batch_size) {
    KERNEL_LOG_ERROR("the size of label must be equal with batch_size[%d]", batch_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (loss_dims[0] != batch_size) {
    KERNEL_LOG_ERROR("the size of loss must be equal with batch_size[%d]", batch_size);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (backprop_dims[0] != batch_size || backprop_dims[1] != num_classes) {
    KERNEL_LOG_ERROR("the size of label must be equal with [%d x %d], but get [%d x %d]", batch_size, num_classes,
                     backprop_dims[0], backprop_dims[1]);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}
template <typename data_type, typename label_type>
inline uint32_t SparseSoftmaxCrossEntropyWithLogitsCompute(const CpuKernelContext &ctx) {
  size_t features_total = static_cast<size_t>(ctx.Input(0)->NumElements());
  uint64_t total_size = ctx.Input(0)->GetDataSize();
  uint32_t cores = aicpu::CpuKernelUtils::GetCPUNum(ctx);
  auto *input_features = static_cast<data_type *>(ctx.Input(0)->GetData());
  auto *input_labels = static_cast<label_type *>(ctx.Input(1)->GetData());
  auto *output_loss = static_cast<data_type *>(ctx.Output(0)->GetData());
  auto *output_backprop = static_cast<data_type *>(ctx.Output(1)->GetData());
  bool muilt_core_flag = false;
  if (total_size > paralledDataNum * sizeof(data_type)) {
    muilt_core_flag = true;
  }
  std::vector<std::int64_t> dims = ctx.Input(0)->GetTensorShape()->GetDimSizes();
  std::vector<std::int64_t> labels_dims = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  for (int64_t idx = 0; idx < labels_dims[0]; idx++) {
    if (input_labels[idx] >= dims[1]) {
      KERNEL_LOG_ERROR(
        "Received a label value of [%d] which is outside the valid range of "
        "[0, %d).",
        input_labels[idx], dims[1]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
  }
  // Determine whether to enable multi-core parallel computing
  size_t pivot, classes_num;
  int64_t batch_size{1};
  pivot = dims.size() - 1;
  classes_num = dims[pivot];
  for (size_t index = 0; index < dims.size(); index++) {
    if (index < pivot) {
      batch_size *= dims[index];
    }
  }
  // Eigen::Array
  if (muilt_core_flag) {
    std::int64_t per_unit_size{batch_size / std::min(std::max(1L, cores - 2L), batch_size)};
    auto shard = [&](size_t begin, size_t end) {
      SparseSoftmaxCrossEntropyWithLogitsMultiOp(input_features, input_labels, output_loss, output_backprop, begin, end,
                                                 classes_num, features_total);
    };
    CpuKernelUtils::ParallelFor(ctx, batch_size, per_unit_size, shard);
  } else if (cores != 0) {
    SparseSoftmaxCrossEntropyWithLogitsSingleOp<data_type, label_type>(
      input_features, input_labels, output_loss, output_backprop, batch_size, classes_num, features_total);
  } else {
    KERNEL_LOG_ERROR("SparseSoftmaxCrossEntropyWithLogits compute failed.");
    return KERNEL_STATUS_INNER_ERROR;
  }
  return KERNEL_STATUS_OK;
}

uint32_t SparseSoftmaxCrossEntropyWithLogitsCpuKernel::Compute(CpuKernelContext &ctx) {
  if (NormalCheck(ctx, kInputNum, kOutputNum) == KERNEL_STATUS_PARAM_INVALID) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  if (SparseSoftmaxCrossEntropyWithLogitsExtraCheck(ctx) == KERNEL_STATUS_PARAM_INVALID) {
    return KERNEL_STATUS_PARAM_INVALID;
  }
  // choose compute function depend on dataType
  auto data_type = static_cast<DataType>(ctx.Input(0)->GetDataType());
  auto labels_type = static_cast<DataType>(ctx.Input(1)->GetDataType());
  switch (data_type) {
    case DT_FLOAT16: {
      if (labels_type == DT_INT32) {
        return SparseSoftmaxCrossEntropyWithLogitsCompute<Eigen::half, std::int32_t>(ctx);
      } else if (labels_type == DT_INT64) {
        return SparseSoftmaxCrossEntropyWithLogitsCompute<Eigen::half, std::int64_t>(ctx);
      }
    }
    case DT_FLOAT: {
      if (labels_type == DT_INT32) {
        return SparseSoftmaxCrossEntropyWithLogitsCompute<std::float_t, std::int32_t>(ctx);
      } else if (labels_type == DT_INT64) {
        return SparseSoftmaxCrossEntropyWithLogitsCompute<std::float_t, std::int64_t>(ctx);
      }
    }
    case DT_DOUBLE: {
      if (labels_type == DT_INT32) {
        return SparseSoftmaxCrossEntropyWithLogitsCompute<std::double_t, std::int32_t>(ctx);
      } else if (labels_type == DT_INT64) {
        return SparseSoftmaxCrossEntropyWithLogitsCompute<std::double_t, std::int64_t>(ctx);
      }
    }
    default:
      KERNEL_LOG_ERROR("[%s] Data type of input is not support, input data type is [%s].", ctx.GetOpType().c_str(),
                       DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

REGISTER_CPU_KERNEL(kSparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogitsCpuKernel);
}  // namespace aicpu