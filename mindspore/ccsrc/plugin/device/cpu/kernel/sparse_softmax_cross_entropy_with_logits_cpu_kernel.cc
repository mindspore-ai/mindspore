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

#include "plugin/device/cpu/kernel/sparse_softmax_cross_entropy_with_logits_cpu_kernel.h"
#include <numeric>
#include <limits>
#include <functional>
#include <cmath>
#include "kernel/ops_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "nnacl/fp32/softmax_fp32.h"
#include "nnacl/fp32/mul_fp32.h"

namespace mindspore {
namespace kernel {
bool SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  is_grad_ = GetValue<bool>(KernelMod::primitive_->GetAttr(ops::kIsGrad));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto logits_shape = inputs[0]->GetDeviceShapeVector();
  auto label_shape = inputs[1]->GetDeviceShapeVector();
  auto logits_dims = logits_shape.size();
  if (logits_dims <= 1) {
    MS_LOG(EXCEPTION) << "Labels shape length must be greater to 1";
  }
  auto labels_dims = label_shape.size();
  if (labels_dims + 1 != logits_dims) {
    MS_LOG(EXCEPTION) << "Labels shape length must be equal to Logits shape length minus 1";
  }
  auto is_same_shape_value = std::equal(label_shape.begin(), label_shape.end(), logits_shape.begin());
  if (!is_same_shape_value) {
    MS_LOG(EXCEPTION) << "Labels shape value must be equal to the Logits except the last dimension of Logits";
  }

  batch_size_ =
    static_cast<size_t>(std::accumulate(label_shape.begin(), label_shape.end(), 1, std::multiplies<size_t>()));
  class_num_ = static_cast<size_t>(logits_shape.back());
  if (batch_size_ == 0 || class_num_ == 0) {
    MS_LOG(EXCEPTION) << "Invalid batch size or class num input!";
  }

  size_t type_size = GetTypeByte(inputs[0]->dtype());
  size_t tensor_size = std::accumulate(logits_shape.begin(), logits_shape.end(), type_size, std::multiplies<size_t>());
  (void)workspace_size_list_.emplace_back(tensor_size);
  return KRET_OK;
}

void SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::SoftmaxFp32(const float *logits, float *losses) {
  auto task = [this, logits, losses](size_t start, size_t end) {
    int batch = SizeToInt(end - start);
    int len = SizeToInt(class_num_);
    (void)SoftmaxLastAxis(logits + start * class_num_, losses + start * class_num_, batch, len);
  };
  ParallelLaunchAutoSearch(task, batch_size_, this, &parallel_search_info_);
}

template <typename T, typename S>
void SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::ForwardPostExecute(const S *labels, const T *losses,
                                                                         T *output) const {
  T total_loss = 0;
  for (size_t i = 0; i < batch_size_; i++) {
    size_t offset = i * class_num_ + static_cast<size_t>(labels[i]);
    total_loss -= std::log(losses[offset]);
  }
  output[0] = total_loss / batch_size_;
}

template <typename T, typename S>
void SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::GradPostExecute(const S *labels, const T *losses, T *output) {
  auto task = [this, labels, losses, output](size_t start, size_t end) {
    T batch = 1.0 / batch_size_;
    if constexpr (std::is_same_v<T, float>) {
      int size = SizeToInt((end - start) * class_num_);
      (void)ElementOptMul(&batch, losses + start * class_num_, output + start * class_num_, size, true);
    } else {
      for (size_t i = start * class_num_; i < end * class_num_; i++) {
        output[i] = losses[i] / batch_size_;
      }
    }

    for (size_t i = start; i < end; i++) {
      size_t index = i * class_num_ + static_cast<size_t>(labels[i]);
      output[index] -= batch;
    }
  };
  ParallelLaunchAutoSearch(task, batch_size_, this, &grad_parallel_search_info_);
}

template <typename T, typename S>
bool SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                                   const std::vector<kernel::KernelTensor *> &workspace,
                                                                   const std::vector<kernel::KernelTensor *> &outputs) {
  const auto *logits = static_cast<T *>(inputs[kIndex0]->device_ptr());
  const auto *labels = static_cast<S *>(inputs[kIndex1]->device_ptr());
  auto *losses = static_cast<T *>(workspace[kIndex0]->device_ptr());
  auto *output = static_cast<T *>(outputs[kIndex0]->device_ptr());

  for (size_t i = 0; i < batch_size_; ++i) {
    if (labels[i] < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "' got error label! label[index] must >= 0, but got  index = " << i
                        << " label[index] = " << labels[i];
    }
    size_t label = IntToSize(labels[i]);
    if (label > class_num_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "' got error label! label[index] must <= class_num, but got index = " << i
                        << " label[index] = " << label << " and class_num = " << class_num_;
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    SoftmaxFp32(logits, losses);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the type of 'logits' must be {float}. but got "
                      << TypeIdToString(inputs[kIndex0]->dtype_id());
  }

  if (is_grad_) {
    GradPostExecute(labels, losses, output);
  } else {
    ForwardPostExecute(labels, losses, output);
  }
  return true;
}

std::vector<
  std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::SparseSoftmaxCrossEntropyWithLogitsFunc>>
  SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::LaunchKernel<float, int64_t>}};

std::vector<KernelAttr> SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSoftmaxCrossEntropyWithLogits,
                      SparseSoftmaxCrossEntropyWithLogitsCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
