/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/shape_calc_kernel.h"
#include <functional>

#include "infer/shape_calc.h"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
bool ShapeCalcCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (primitive_->HasAttr(kOutputRealTuple)) {
    is_dynamic_len_out_ = true;
  }
  inputs_size_ = inputs.size();
  return true;
}

int ShapeCalcCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  if (!primitive_->HasAttr(ops::kAttrCalcResult)) {
    MS_LOG(ERROR) << "For ShapeCalc, the calc result should be get here.";
    return KRET_RESIZE_FAILED;
  }
  outs_shape_ = GetValue<ShapeArray>(primitive_->GetAttr(ops::kAttrCalcResult));
  return ret;
}

bool ShapeCalcCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                   const std::vector<kernel::KernelTensor *> &,
                                   const std::vector<kernel::KernelTensor *> &outputs) {
  if (!is_dynamic_len_out_) {
    if (outputs.size() != outs_shape_.size()) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', outputs address list size must be equal to the number of outputs of shape func, but got "
                    << outputs.size() << " vs " << outs_shape_.size();
      return false;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      auto output_addr = GetDeviceAddress<int64_t>(outputs, i);
      for (size_t j = 0; j < outs_shape_[i].size(); ++j) {
        output_addr[j] = outs_shape_[i][j];
      }
    }
  } else {
    // Dynamic length, each out should have same shape for dynamic-length-out solution in runtime.
    if (outputs.size() != 1) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', dynamic length outputs address list size must be equal to 1, but got " << outputs.size();
      return false;
    }

    auto output_addr = GetDeviceAddress<int64_t>(outputs, 0);
    size_t offset_inner = outs_shape_[0].size();
    for (size_t i = 0; i < outs_shape_.size(); ++i) {
      for (size_t j = 0; j < outs_shape_[i].size(); ++j) {
        size_t cur_offset = i * offset_inner + j;
        *(output_addr + cur_offset) = outs_shape_[i][j];
      }
    }
  }

  return true;
}

std::vector<KernelAttr> ShapeCalcCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr().AddSkipCheckAttr(true)};
  return support_list;
}

std::vector<size_t> ShapeCalcCpuKernelMod::GetLaunchIgnoredInputAddressIdx() const {
  std::vector<size_t> ignored_idx(inputs_size_);
  std::iota(ignored_idx.begin(), ignored_idx.end(), kIndex0);
  return ignored_idx;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, ShapeCalc, ShapeCalcCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
