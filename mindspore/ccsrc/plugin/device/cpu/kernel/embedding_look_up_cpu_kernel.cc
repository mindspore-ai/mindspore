/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/embedding_look_up_cpu_kernel.h"
#include "mindspore/core/ops/embedding_lookup.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kEmbeddingLookupInputsNum = 2;
constexpr size_t kEmbeddingLookupDynamicShapeInputsNum = 3;
constexpr size_t kEmbeddingLookUpInputParamsMaxDim = 2;
using KernelRunFunc = EmbeddingLookUpCpuKernelMod::KernelRunFunc;

#define ADD_KERNEL(input_params_dtype, input_indices_dtype, output_dtype, input_params_type, input_indices_type) \
  {                                                                                                              \
    KernelAttr()                                                                                                 \
      .AddInputAttr(kNumberType##input_params_dtype)                                                             \
      .AddInputAttr(kNumberType##input_indices_dtype)                                                            \
      .AddOutputAttr(kNumberType##output_dtype),                                                                 \
      &EmbeddingLookUpCpuKernelMod::LaunchKernel<input_params_type, input_indices_type, int64_t>                 \
  }

#define ADD_KERNEL_DYNAMIC(input_params_dtype, input_indices_dtype, output_dtype, input_params_type, \
                           input_indices_type)                                                       \
  {                                                                                                  \
    KernelAttr()                                                                                     \
      .AddInputAttr(kNumberType##input_params_dtype)                                                 \
      .AddInputAttr(kNumberType##input_indices_dtype)                                                \
      .AddInputAttr(kNumberTypeInt64)                                                                \
      .AddOutputAttr(kNumberType##output_dtype),                                                     \
      &EmbeddingLookUpCpuKernelMod::LaunchKernel<input_params_type, input_indices_type, int64_t>     \
  }

#define ADD_KERNEL_DYNAMIC_INT32(input_params_dtype, input_indices_dtype, output_dtype, input_params_type, \
                                 input_indices_type)                                                       \
  {                                                                                                        \
    KernelAttr()                                                                                           \
      .AddInputAttr(kNumberType##input_params_dtype)                                                       \
      .AddInputAttr(kNumberType##input_indices_dtype)                                                      \
      .AddInputAttr(kNumberTypeInt32)                                                                      \
      .AddOutputAttr(kNumberType##output_dtype),                                                           \
      &EmbeddingLookUpCpuKernelMod::LaunchKernel<input_params_type, input_indices_type, int32_t>           \
  }

template <typename T, typename S>
void LookUpTableTask(const T *input_addr, const S *indices_addr, T *output_addr, size_t indices_lens,
                     size_t outer_dim_size, int64_t offset, size_t first_dim_size, std::string kernel_name_) {
  auto type_size = sizeof(T);
  size_t lens = outer_dim_size * type_size;
  for (size_t i = 0; i < indices_lens; ++i) {
    S index = indices_addr[i] - static_cast<S>(offset);
    if (index >= 0 && index < SizeToInt(first_dim_size)) {
      size_t pos = static_cast<size_t>(index) * outer_dim_size;
      auto ret = memcpy_s(output_addr, (indices_lens - i) * lens, input_addr + pos, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memcpy failed. Error no: " << ret;
      }
    } else {
      auto ret = memset_s(output_addr, (indices_lens - i) * lens, 0, lens);
      if (ret != EOK) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset failed. Error no: " << ret;
      }
    }
    output_addr += outer_dim_size;
  }
}
}  // namespace

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &EmbeddingLookUpCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    ADD_KERNEL(Bool, Int32, Bool, bool, int32_t),
    ADD_KERNEL(Int8, Int32, Int8, int8_t, int32_t),
    ADD_KERNEL(Int16, Int32, Int16, int16_t, int32_t),
    ADD_KERNEL(Int32, Int32, Int32, int32_t, int32_t),
    ADD_KERNEL(Int64, Int32, Int64, int64_t, int32_t),
    ADD_KERNEL(UInt8, Int32, UInt8, uint8_t, int32_t),
    ADD_KERNEL(UInt16, Int32, UInt16, uint16_t, int32_t),
    ADD_KERNEL(UInt32, Int32, UInt32, uint32_t, int32_t),
    ADD_KERNEL(UInt64, Int32, UInt64, uint64_t, int32_t),
    ADD_KERNEL(Float16, Int32, Float16, float16, int32_t),
    ADD_KERNEL(Float32, Int32, Float32, float, int32_t),
    ADD_KERNEL(Float64, Int32, Float64, double, int32_t),

    ADD_KERNEL(Bool, Int64, Bool, bool, int64_t),
    ADD_KERNEL(Int8, Int64, Int8, int8_t, int64_t),
    ADD_KERNEL(Int16, Int64, Int16, int16_t, int64_t),
    ADD_KERNEL(Int32, Int64, Int32, int32_t, int64_t),
    ADD_KERNEL(Int64, Int64, Int64, int64_t, int64_t),
    ADD_KERNEL(UInt8, Int64, UInt8, uint8_t, int64_t),
    ADD_KERNEL(UInt16, Int64, UInt16, uint16_t, int64_t),
    ADD_KERNEL(UInt32, Int64, UInt32, uint32_t, int64_t),
    ADD_KERNEL(UInt64, Int64, UInt64, uint64_t, int64_t),
    ADD_KERNEL(Float16, Int64, Float16, float16, int64_t),
    ADD_KERNEL(Float32, Int64, Float32, float, int64_t),
    ADD_KERNEL(Float64, Int64, Float64, double, int64_t),

    ADD_KERNEL_DYNAMIC(Bool, Int32, Bool, bool, int32_t),
    ADD_KERNEL_DYNAMIC(Int8, Int32, Int8, int8_t, int32_t),
    ADD_KERNEL_DYNAMIC(Int16, Int32, Int16, int16_t, int32_t),
    ADD_KERNEL_DYNAMIC(Int32, Int32, Int32, int32_t, int32_t),
    ADD_KERNEL_DYNAMIC(Int64, Int32, Int64, int64_t, int32_t),
    ADD_KERNEL_DYNAMIC(UInt8, Int32, UInt8, uint8_t, int32_t),
    ADD_KERNEL_DYNAMIC(UInt16, Int32, UInt16, uint16_t, int32_t),
    ADD_KERNEL_DYNAMIC(UInt32, Int32, UInt32, uint32_t, int32_t),
    ADD_KERNEL_DYNAMIC(UInt64, Int32, UInt64, uint64_t, int32_t),
    ADD_KERNEL_DYNAMIC(Float16, Int32, Float16, float16, int32_t),
    ADD_KERNEL_DYNAMIC(Float32, Int32, Float32, float, int32_t),
    ADD_KERNEL_DYNAMIC(Float64, Int32, Float64, double, int32_t),

    ADD_KERNEL_DYNAMIC(Bool, Int64, Bool, bool, int64_t),
    ADD_KERNEL_DYNAMIC(Int8, Int64, Int8, int8_t, int64_t),
    ADD_KERNEL_DYNAMIC(Int16, Int64, Int16, int16_t, int64_t),
    ADD_KERNEL_DYNAMIC(Int32, Int64, Int32, int32_t, int64_t),
    ADD_KERNEL_DYNAMIC(Int64, Int64, Int64, int64_t, int64_t),
    ADD_KERNEL_DYNAMIC(UInt8, Int64, UInt8, uint8_t, int64_t),
    ADD_KERNEL_DYNAMIC(UInt16, Int64, UInt16, uint16_t, int64_t),
    ADD_KERNEL_DYNAMIC(UInt32, Int64, UInt32, uint32_t, int64_t),
    ADD_KERNEL_DYNAMIC(UInt64, Int64, UInt64, uint64_t, int64_t),
    ADD_KERNEL_DYNAMIC(Float16, Int64, Float16, float16, int64_t),
    ADD_KERNEL_DYNAMIC(Float32, Int64, Float32, float, int64_t),
    ADD_KERNEL_DYNAMIC(Float64, Int64, Float64, double, int64_t),

    ADD_KERNEL_DYNAMIC_INT32(Int32, Int32, Int32, int32_t, int32_t),
    ADD_KERNEL_DYNAMIC_INT32(Float32, Int32, Float32, float, int32_t)};

  return func_list;
}

bool EmbeddingLookUpCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::EmbeddingLookup>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "For primitive[EmbeddingLookup], cast op from BaseOperator to EmbeddingLookup failed.";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  return MatchKernelFunc(base_operator, inputs, outputs);
}

int EmbeddingLookUpCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto kernel_ptr = std::dynamic_pointer_cast<ops::EmbeddingLookup>(base_operator);
  if ((inputs.size() != kEmbeddingLookupInputsNum && inputs.size() != kEmbeddingLookupDynamicShapeInputsNum) ||
      outputs.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output size must be " << kEmbeddingLookupInputsNum
                  << " and " << 1 << ", but got " << inputs.size() << " and " << outputs.size();
  }

  std::vector<int64_t> input_params_shape = inputs[kIndex0]->GetShapeVector();
  if (input_params_shape.empty() || input_params_shape.size() > kEmbeddingLookUpInputParamsMaxDim) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input must be 1-"
                      << kEmbeddingLookUpInputParamsMaxDim << "D, but got " << input_params_shape.size() << "D.";
  }
  first_dim_size_ = LongToSize(input_params_shape[0]);
  outer_dim_size_ = 1;
  for (size_t i = 1; i < input_params_shape.size(); ++i) {
    outer_dim_size_ *= LongToSize(input_params_shape[i]);
  }
  input_params_dtype_ = inputs[kIndex0]->GetDtype();

  std::vector<int64_t> input_indices_shape = inputs[kIndex1]->GetShapeVector();
  input_indices_lens_ = SizeOf(input_indices_shape);
  input_indices_dtype_ = inputs[kIndex1]->GetDtype();
  if (inputs.size() == kEmbeddingLookupInputsNum) {
    offset_ = kernel_ptr->get_offset();
  }
  return KRET_OK;
}

template <typename T, typename S, typename G>
bool EmbeddingLookUpCpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                               const std::vector<AddressPtr> &outputs) {
  T *input_params_addr = static_cast<T *>(inputs[0]->addr);
  S *input_indices_addr = static_cast<S *>(inputs[1]->addr);
  T *output_addr = static_cast<T *>(outputs[0]->addr);
  if (inputs.size() == kEmbeddingLookupDynamicShapeInputsNum) {
    G *input_offset_addr = static_cast<G *>(inputs[2]->addr);
    int ret = memcpy_s(&offset_, sizeof(G), input_offset_addr, sizeof(G));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "The memcpy_s error, errorno(" << ret << ")";
    }
  }
  auto task = [&](size_t start, size_t end) {
    size_t task_proc_lens = end - start;
    LookUpTableTask<T, S>(input_params_addr, input_indices_addr + start, output_addr + start * outer_dim_size_,
                          task_proc_lens, outer_dim_size_, offset_, first_dim_size_, kernel_name_);
  };

  ParallelLaunchAutoSearch(task, input_indices_lens_, this, &parallel_search_info_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, EmbeddingLookup, EmbeddingLookUpCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
