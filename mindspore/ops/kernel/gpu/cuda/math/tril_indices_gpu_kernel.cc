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

#include "kernel/gpu/cuda/math/tril_indices_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool TrilIndicesGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
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
  row_ = GetValue<int64_t>(primitive_->GetAttr("row"));
  col_ = GetValue<int64_t>(primitive_->GetAttr("col"));
  offset_ = GetValue<int64_t>(primitive_->GetAttr("offset"));
  return true;
}

int TrilIndicesGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(outputs[kIndex0]);
  ResetResource();
  auto ret = KRET_OK;
  size_t tensor_size = 0;
  size_t type_size = GetTypeByte(TypeIdToType(outputs.at(kIndex0)->dtype_id()));
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
  workspace_size_list_.clear();
  output_size_list_.clear();
}

template <typename T>
bool TrilIndicesGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs) {
  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(output);
  if (tril_size_ > 0) {
    auto m_first_row = offset_ > 0 ? std::min<int64_t>(col_, 1 + offset_) : row_ + offset_ > 0;
    auto trapezoid_row_offset = std::max<int64_t>(0, -offset_);
    auto rectangle_row_offset = trapezoid_row_offset + col_ - m_first_row + 1;
    int64_t rectangle_size = 0;
    if (rectangle_row_offset < row_) {
      rectangle_size = (row_ - rectangle_row_offset) * col_;
    }
    auto status =
      CalTrilIndices(trapezoid_row_offset, m_first_row, col_, static_cast<int64_t>(tril_size_) - rectangle_size,
                     tril_size_, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDA_STATUS(status, kernel_name_);
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
