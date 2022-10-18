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

#include "plugin/device/gpu/kernel/math/tril_indices_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool TrilIndicesGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr_ = std::dynamic_pointer_cast<ops::TrilIndices>(base_operator);
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

int TrilIndicesGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
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
  tril_size_ = tensor_size / (type_size * matrix_dim);
  return ret;
}

void TrilIndicesGpuKernelMod::ResetResource() noexcept {
  tril_size_ = 0;
  input_size_list_.clear();
  workspace_size_list_.clear();
  output_size_list_.clear();
}

template <typename T>
bool TrilIndicesGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs) {
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  if (tril_size_ > 0) {
    auto m_first_row = offset_ > 0 ? std::min<int64_t>(col_, 1 + offset_) : row_ + offset_ > 0;
    auto trapezoid_row_offset = std::max<int64_t>(0, -offset_);
    auto rectangle_row_offset = trapezoid_row_offset + col_ - m_first_row + 1;
    int64_t rectangle_size = 0;
    if (rectangle_row_offset < row_) {
      rectangle_size = (row_ - rectangle_row_offset) * col_;
    }
    CalTrilIndices(trapezoid_row_offset, m_first_row, col_, static_cast<int64_t>(tril_size_) - rectangle_size,
                   tril_size_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  }
  return true;
}

std::vector<std::pair<KernelAttr, TrilIndicesGpuKernelMod::TrilIndicesFunc>> TrilIndicesGpuKernelMod::func_list_ = {
  {KernelAttr().AddOutputAttr(kNumberTypeInt32), &TrilIndicesGpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddOutputAttr(kNumberTypeInt64), &TrilIndicesGpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> TrilIndicesGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TrilIndicesFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TrilIndices, TrilIndicesGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
