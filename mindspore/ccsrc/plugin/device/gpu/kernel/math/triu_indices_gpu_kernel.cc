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

#include "plugin/device/gpu/kernel/math/triu_indices_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool TriuIndicesGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::TriuIndices>(base_operator);
  kernel_name_ = kernel_ptr_->name();
  if (outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in [int32, int64], "
                  << "but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  row_ = kernel_ptr_->get_row();
  col_ = kernel_ptr_->get_col();
  offset_ = kernel_ptr_->get_offset();
  return true;
}

int TriuIndicesGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  auto ret = KRET_OK;
  size_t tensor_size = 0;
  size_t type_size = GetTypeByte(TypeIdToType(outputs.at(kIndex0)->GetDtype()));
  auto shape = outputs.at(kIndex0)->GetShapeVector();
  if (!IsValidShape(shape)) {
    ret = KRET_UNKNOWN_OUT_SHAPE;
  } else {
    tensor_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
  }
  output_size_list_.emplace_back(tensor_size);
  const size_t matrix_dim = 2;
  triu_size_ = tensor_size / (type_size * matrix_dim);
  return ret;
}

void TriuIndicesGpuKernelMod::ResetResource() noexcept {
  triu_size_ = 0;
  input_size_list_.clear();
  workspace_size_list_.clear();
  output_size_list_.clear();
}

template <typename T>
bool TriuIndicesGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  if (triu_size_ > 0) {
    auto m_first_row = offset_ > 0 ? std::max<int64_t>(col_ - offset_, 0) : col_;
    int64_t rectangle_size = 0;
    if (offset_ < 0) {
      rectangle_size = std::min<int64_t>(row_, -offset_) * col_;
    }
    CalTriuIndices(std::max<int64_t>(0, offset_), m_first_row, col_, rectangle_size, triu_size_, output, device_id_,
                   reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  return true;
}

std::vector<std::pair<KernelAttr, TriuIndicesGpuKernelMod::TriuIndicesFunc>> TriuIndicesGpuKernelMod::func_list_ = {
  {KernelAttr().AddOutputAttr(kNumberTypeInt32), &TriuIndicesGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddOutputAttr(kNumberTypeInt64), &TriuIndicesGpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> TriuIndicesGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TriuIndicesFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TriuIndices, TriuIndicesGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
