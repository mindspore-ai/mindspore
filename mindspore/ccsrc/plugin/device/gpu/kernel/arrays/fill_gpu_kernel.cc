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

#include <limits>
#include <memory>
#include <functional>
#include <algorithm>
#include "base/float16.h"
#include "abstract/utils.h"
#include "mindspore/core/ops/fill.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"
#include "plugin/device/gpu/kernel/arrays/fill_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/fill_impl.cuh"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace kernel {
#define FILL_GPU_REG(MS_T, MS_U, MS_V, T) \
  { KernelAttr().AddInputAttr(MS_T).AddInputAttr(MS_U).AddOutputAttr(MS_V), &FillGpuKernelMod::LaunchKernel<T> }

template <typename T>
using Complex = mindspore::utils::Complex<T>;
const std::vector<std::pair<KernelAttr, FillGpuKernelMod::KernelRunFunc>> &FillGpuKernelMod::GetFuncList() const {
  static std::vector<std::pair<KernelAttr, FillGpuKernelMod::KernelRunFunc>> func_list;
  std::vector<TypeId> shape_type_list = {kNumberTypeInt32, kNumberTypeInt64};
  std::vector<TypeId> value_type_list = {
    kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,     kNumberTypeInt64,     kNumberTypeFloat16,
    kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeUInt8,     kNumberTypeUInt16,    kNumberTypeUInt32,
    kNumberTypeUInt64,  kNumberTypeBool,    kNumberTypeComplex64, kNumberTypeComplex128};

  std::pair<KernelAttr, FillGpuKernelMod::KernelRunFunc> type_pair;
  for (auto i : shape_type_list) {
    for (auto j : value_type_list) {
      for (size_t k = 0; k < value_type_list.size(); k++) {
        if (value_type_list[k] == kNumberTypeInt8) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], int8_t);
        } else if (value_type_list[k] == kNumberTypeInt16) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], int16_t);
        } else if (value_type_list[k] == kNumberTypeInt32) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], int32_t);
        } else if (value_type_list[k] == kNumberTypeInt64) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], int64_t);
        } else if (value_type_list[k] == kNumberTypeFloat16) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], half);
        } else if (value_type_list[k] == kNumberTypeFloat32) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], float);
        } else if (value_type_list[k] == kNumberTypeFloat64) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], double);
        } else if (value_type_list[k] == kNumberTypeUInt8) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], uint8_t);
        } else if (value_type_list[k] == kNumberTypeUInt16) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], uint16_t);
        } else if (value_type_list[k] == kNumberTypeUInt32) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], uint32_t);
        } else if (value_type_list[k] == kNumberTypeUInt64) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], uint64_t);
        } else if (value_type_list[k] == kNumberTypeBool) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], bool);
        } else if (value_type_list[k] == kNumberTypeComplex64) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], Complex<float>);
        } else if (value_type_list[k] == kNumberTypeComplex128) {
          type_pair = FILL_GPU_REG(i, j, value_type_list[k], Complex<double>);
        }
        func_list.emplace_back(type_pair);
      }
    }
  }
  return func_list;
}

bool FillGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                            const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto tensor_attr = GetKernelAttrFromTensors(inputs, outputs);
  if (tensor_attr.GetInputAttr(kIndex1).first != outputs[kIndex0]->GetDtype()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' "
                      << "the datatype of the input [value] has to be exactly same as input[dtype].";
  }
  auto x_type_id = tensor_attr.GetInputAttr(kIndex1).first;
  x_type_str_ = TypeIdToString(x_type_id);
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int FillGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                             const std::vector<KernelTensorPtr> &outputs,
                             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != KRET_OK) {
    return ret;
  }
  auto shape = outputs.at(kIndex0)->GetShapeVector();
  input_elements_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  auto workspace_size = sizeof(double);
  workspace_size_list_.emplace_back(workspace_size);

  return KRET_OK;
}

template <typename T>
bool FillGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                    const std::vector<AddressPtr> &outputs) {
  auto value_ptr = GetDeviceAddress<T>(inputs, kIndex1);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  T value;
  auto cuda_stream = reinterpret_cast<cudaStream_t>(cuda_stream_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&value, value_ptr, sizeof(T), cudaMemcpyDeviceToHost, cuda_stream),
                                     "cudaMemcpy value variable failed.");
  Fill(input_elements_, 1, value_ptr, y_ptr, cuda_stream);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Fill, FillGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
