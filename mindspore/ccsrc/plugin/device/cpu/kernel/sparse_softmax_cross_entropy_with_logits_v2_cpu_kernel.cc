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

#include "plugin/device/cpu/kernel/sparse_softmax_cross_entropy_with_logits_v2_cpu_kernel.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <functional>
namespace mindspore {
namespace kernel {
namespace {
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2InputNum{2};
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2OutputNum{2};
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2FeaturesShape{2};
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2LabelsShape{1};
}  // namespace

void SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  features_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
  labels_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
  loss_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
  backprop_shape = common::AnfAlgo::GetOutputInferShape(kernel_node, 1);
  int64_t features_batch = features_shape[kIndex0];
  int64_t labels_batch = labels_shape[kIndex0];
  if (features_shape.size() != kSparseSoftmaxCrossEntropyWithLogitsV2FeaturesShape ||
      labels_shape.size() != kSparseSoftmaxCrossEntropyWithLogitsV2LabelsShape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input logits(features) shape " << Vector2Str(features_shape)
                      << " must be same as [batch * classes] and the input labels shape " << Vector2Str(labels_shape)
                      << " must be same as [batch].";
  }
  if (features_batch != labels_batch) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input logits(features) batch " << features_batch
                      << " must be equal to the input label batch " << labels_batch;
  }
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SparseSoftmaxCrossEntropyWithLogitsV2 does not support this kernel data type: "
                      << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
}

template <typename data_type, typename label_type>
bool SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSoftmaxCrossEntropyWithLogitsV2InputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSoftmaxCrossEntropyWithLogitsV2OutputNum, kernel_name_);
  auto *features = static_cast<data_type *>(inputs[kIndex0]->addr);
  auto *labels = static_cast<label_type *>(inputs[kIndex1]->addr);
  auto *loss = static_cast<data_type *>(outputs[kIndex0]->addr);
  auto *backprop = static_cast<data_type *>(outputs[kIndex1]->addr);
  const size_t features_length = inputs[kIndex0]->size / sizeof(data_type);
  const size_t labels_length = inputs[kIndex1]->size / sizeof(label_type);
  const size_t batch_size = labels_length;
  const size_t classes_num = features_length / labels_length;
  for (size_t index = 0; index < labels_length; index++) {
    if (labels[index] >= SizeToInt(classes_num) || labels[index] < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the labels[" << index << "] = " << labels[index]
                        << " value is outside the valid range of [0, " << classes_num << ").";
      return false;
    }
  }

  float *dims_exp_sum = static_cast<float *>(malloc(batch_size * sizeof(float)));
  float *bp_fp32 = static_cast<float *>(malloc(features_length * sizeof(float)));
  data_type *dims_maximum = static_cast<data_type *>(malloc(batch_size * sizeof(data_type)));
  if (memset_s(dims_exp_sum, batch_size * sizeof(float), 0, batch_size * sizeof(float)) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset dims_exp_sum failed!";
  }
  Eigen::TensorMap<Eigen::Tensor<data_type, kSparseSoftmaxCrossEntropyWithLogitsV2FeaturesShape>, Eigen::Aligned>
    logits(features, batch_size, classes_num);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> dims_sum(dims_exp_sum, batch_size);
  Eigen::TensorMap<Eigen::Tensor<data_type, 1>, Eigen::Aligned> dims_max(dims_maximum, batch_size);
  Eigen::array<int, 1> axes{{1}};
  // compute softmax
  dims_max = logits.maximum(axes);
  const data_type constant_one(1.0);
  for (size_t index = 0, batch_idx = 0; index < features_length; index++) {
    bp_fp32[index] = Eigen::numext::exp(static_cast<float>(features[index] - dims_max(batch_idx)));
    dims_exp_sum[batch_idx] += bp_fp32[index];
    if ((index + 1) % classes_num == 0) {
      batch_idx++;
    }
  }
  dims_sum = dims_sum.inverse();
  for (size_t index = 0, batch_idx = 0; index < features_length; index++) {
    backprop[index] = static_cast<data_type>(bp_fp32[index] * dims_sum(batch_idx));
    if ((index + 1) % classes_num == 0) {
      batch_idx++;
    }
  }
  for (size_t index = 0, batch_base = 0; index < batch_size; ++index, batch_base += classes_num) {
    label_type offset = labels[index];
    loss[index] = -Eigen::numext::log(backprop[batch_base + offset]);
    backprop[batch_base + offset] = backprop[batch_base + offset] - constant_one;
  }
  free(bp_fp32);
  free(dims_exp_sum);
  free(dims_maximum);
  return true;
}

std::vector<
  std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::SparseSoftmaxCrossEntropyWithLogitsV2Func>>
  SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<Eigen::half, std::int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<Eigen::half, std::int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<float, std::int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<float, std::int64_t>}};
std::vector<KernelAttr> SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsV2Func> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSoftmaxCrossEntropyWithLogitsV2,
                      SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
