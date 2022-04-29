/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/reduce_cpu_kernel.h"
#include <complex>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <map>
#include "nnacl/fp32/reduce_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kReduceSmallVectorSize = 200000;
constexpr size_t kReduceInputsNum = 1;
constexpr size_t kReduceOutputsNum = 1;

using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

template <typename T>
class ReduceCpuKernelFunc : public DeprecatedCpuKernelFunc {
 public:
  ReduceCpuKernelFunc() = default;
  ~ReduceCpuKernelFunc() override = default;
  void InitFunc(const CNodePtr &kernel_node) override;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

 private:
  void AccelerateLongVector(T *input_addr, T *output_addr, size_t input_size);
  void ChooseFunc(const std::string &kernel_name_);

  enum class ReduceFuncType {
    kReduceAllType,
    kReduceAnyType,
    kReduceMaxType,
    kReduceMinType,
    kReduceSumType,
    kReduceMeanType,
    kReduceProdType
  };
  std::vector<size_t> input_shape_;
  std::vector<int64_t> axis_;
  ReduceFuncType reduce_type_{ReduceFuncType::kReduceAllType};
  std::function<void(const T *, size_t, T *)> reduce_func_;
  bool simple_execute_{false};
  std::string kernel_name_;
};

void UpdateAxis(const PrimitivePtr &prim, const CNodePtr &kernel_node, const std::string &kernel_name,
                std::vector<int64_t> *axis) {
  auto axis_addr = prim->GetAttr(AXIS);
  if (axis == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the 'axis' can not be null.";
  }
  if (axis_addr == nullptr) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the 'axis' can not be null, but got empty value.";
  }
  if (axis_addr->isa<ValueTuple>() || axis_addr->isa<ValueList>()) {
    *axis = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, AXIS);
  } else if (axis_addr->isa<Int64Imm>()) {
    (void)axis->emplace_back(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name
                      << "', the type of 'axis' must be tuple, list, or int, but got invalid type.";
  }
}

template <typename T>
void ReduceCpuKernelFunc<T>::ChooseFunc(const std::string &kernel_name_) {
  if constexpr (std::is_same<T, bool>::value) {
    if (kernel_name_ == prim::kPrimReduceAll->name()) {
      reduce_type_ = ReduceFuncType::kReduceAllType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out &= input[pos]; };
    } else if (kernel_name_ == prim::kPrimReduceAny->name()) {
      reduce_type_ = ReduceFuncType::kReduceAnyType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out |= input[pos]; };
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported reduce operation for bool.";
    }
  } else if constexpr (((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>))) {  // NOLINT
    if (kernel_name_ == prim::kPrimReduceProd->name()) {
      reduce_type_ = ReduceFuncType::kReduceProdType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out *= input[pos]; };
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported reduce operation for complex.";
    }
  } else {
    if (kernel_name_ == prim::kPrimReduceMax->name()) {
      reduce_type_ = ReduceFuncType::kReduceMaxType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::max(input[pos], *out); };
    } else if (kernel_name_ == prim::kPrimReduceMin->name()) {
      reduce_type_ = ReduceFuncType::kReduceMinType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out = std::min(input[pos], *out); };
    } else if (kernel_name_ == prim::kPrimReduceSum->name()) {
      reduce_type_ = ReduceFuncType::kReduceSumType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
    } else if (kernel_name_ == prim::kPrimReduceMean->name()) {
      reduce_type_ = ReduceFuncType::kReduceMeanType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out += input[pos]; };
    } else if (kernel_name_ == prim::kPrimReduceProd->name()) {
      reduce_type_ = ReduceFuncType::kReduceProdType;
      reduce_func_ = [](const T *input, size_t pos, T *out) { *out *= input[pos]; };
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', unsupported reduce operation.";
    }
  }
}

template <typename T>
void ReduceCpuKernelFunc<T>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  axis_.clear();
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  MS_EXCEPTION_IF_NULL(prim);
  UpdateAxis(prim, kernel_node, kernel_name_, &axis_);
  int64_t dimension = SizeToLong(input_shape_.size());
  (void)std::for_each(axis_.begin(), axis_.end(), [dimension](auto &a) {
    if (a < -dimension || a >= dimension) {
      MS_LOG(EXCEPTION) << "For reduce, the each axis element should be in [" << -dimension << ", " << dimension
                        << "), but got " << a;
    }
    a = a < 0 ? dimension + a : a;
  });
  // Delete the duplicate axis.
  sort(axis_.begin(), axis_.end());
  auto last = std::unique(axis_.begin(), axis_.end());
  axis_.erase(last, axis_.end());

  ChooseFunc(kernel_name_);

  // special accelerate for axis = 1 and input has 2 dims
  if constexpr (std::is_same<T, float>::value) {
    if ((reduce_type_ == ReduceFuncType::kReduceMeanType || reduce_type_ == ReduceFuncType::kReduceSumType) &&
        axis_.size() == 1 && axis_[0] == 1 && input_shape_.size() == 2) {
      simple_execute_ = true;
    }
  }
}

template <typename T>
bool ReduceCpuKernelFunc<T>::RunFunc(const std::vector<kernel::AddressPtr> &inputs,
                                     const std::vector<kernel::AddressPtr> &,
                                     const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kReduceInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kReduceOutputsNum, kernel_name_);
  size_t input_size = inputs[0]->size / sizeof(T);
  auto *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
  if (axis_.empty() || input_shape_.empty() || input_shape_.size() == 1) {
    if (input_size < kReduceSmallVectorSize) {
      // Get one ret
      *output_addr = input_addr[0];
      for (size_t i = 1; i < input_size; ++i) {
        reduce_func_(input_addr, i, output_addr);
      }
      if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
        *output_addr /= input_size;
      }
    } else {
      AccelerateLongVector(input_addr, output_addr, input_size);
    }
  } else {
    // Calculate transpose axes and stride
    int dimension = input_shape_.size();
    size_t stride = 1;
    std::vector<size_t> axes(input_shape_.size());
    size_t j = 0;
    size_t k = 0;
    for (int i = 0; i < dimension; ++i) {
      if (j == axis_.size() || i != axis_[j]) {
        axes[k] = i;
        ++k;
      } else {
        stride *= input_shape_[i];
        ++j;
      }
    }
    for (auto &it : axis_) {
      axes[k] = it;
      ++k;
    }

    size_t output_size = outputs[0]->size / sizeof(T);
    if constexpr (std::is_same<T, float>::value) {
      if (simple_execute_) {
        auto task = [&](size_t start, size_t end) {
          for (size_t i = start; i < end; ++i) {
            (void)ReduceSumDim2Axis1(stride, input_addr + i * stride, output_addr + i);
            if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
              output_addr[i] /= stride;
            }
          }
        };
        ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
        return true;
      }
    }
    // Calculate transpose shape
    std::vector<size_t> transpose_shape(input_shape_.size());
    for (int i = 0; i < dimension; ++i) {
      transpose_shape[i] = input_shape_[axes[i]];
    }
    TransposeIterator base_iter(std::move(transpose_shape), std::move(axes), input_shape_);
    auto task = [this, &base_iter, input_addr, output_addr, stride](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start * stride);
      for (size_t i = start; i < end; ++i) {
        output_addr[i] = input_addr[iter.GetPos()];
        iter.GenNextPos();
        for (size_t j = 1; j < stride; ++j) {
          reduce_func_(input_addr, iter.GetPos(), &output_addr[i]);
          iter.GenNextPos();
        }
        if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
          output_addr[i] /= stride;
        }
      }
    };
    ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  }
  return true;
}

template <typename T>
void ReduceCpuKernelFunc<T>::AccelerateLongVector(T *input_addr, T *output_addr, size_t input_size) {
  // init output_addr
  *output_addr = input_addr[0];
  std::mutex task_mutex;
  auto task = [this, input_addr, output_addr, &task_mutex](size_t start, size_t end) {
    if (start == 0) {
      ++start;
    }
    if (start == end) {
      return;
    }
    auto block_output = input_addr[start];
    size_t i = start + 1;
    while (i < end) {
      reduce_func_(input_addr, i, &block_output);
      ++i;
    }
    {
      std::lock_guard<std::mutex> task_lock(task_mutex);
      reduce_func_(&block_output, 0, output_addr);
    }
  };
  ParallelLaunchAutoSearch(task, input_size, this, &parallel_search_info_);
  if (reduce_type_ == ReduceFuncType::kReduceMeanType) {
    *output_addr /= input_size;
  }
}
template <typename T>
std::shared_ptr<DeprecatedCpuKernelFunc> SpecializeReduceFunc() {
  return std::make_shared<ReduceCpuKernelFunc<T>>();
}
using SpecializeReduceFuncCreator = std::function<std::shared_ptr<DeprecatedCpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, SpecializeReduceFuncCreator>>> kernel_attr_list = {
  {prim::kPrimReduceMean->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), SpecializeReduceFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), SpecializeReduceFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), SpecializeReduceFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), SpecializeReduceFunc<int64_t>}}},
  {prim::kPrimReduceMax->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), SpecializeReduceFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), SpecializeReduceFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), SpecializeReduceFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), SpecializeReduceFunc<int64_t>}}},
  {prim::kPrimReduceSum->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), SpecializeReduceFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), SpecializeReduceFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), SpecializeReduceFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), SpecializeReduceFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), SpecializeReduceFunc<bool>}}},
  {prim::kPrimReduceMin->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), SpecializeReduceFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), SpecializeReduceFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), SpecializeReduceFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), SpecializeReduceFunc<int64_t>}}},
  {prim::kPrimReduceProd->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), SpecializeReduceFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), SpecializeReduceFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8), SpecializeReduceFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16), SpecializeReduceFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32), SpecializeReduceFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64), SpecializeReduceFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8), SpecializeReduceFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16), SpecializeReduceFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32), SpecializeReduceFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64), SpecializeReduceFunc<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeComplex64),
     SpecializeReduceFunc<complex64>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddOutputAttr(kNumberTypeComplex128),
     SpecializeReduceFunc<complex128>}}},
  {prim::kPrimReduceAll->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), SpecializeReduceFunc<bool>}}},
  {prim::kPrimReduceAny->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool), SpecializeReduceFunc<bool>}}}};
}  // namespace

void ReduceCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Suppose to be " << kernel_type_ << " but got " << kernel_name_;
  }

  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(EXCEPTION) << "Reduce cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SpecializeReduceFuncCreator> &pair) { return pair.first; });

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, support_list);
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Reduce does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list[kernel_type_][index].second();
  func_obj_->InitFunc(kernel_node);
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceMean,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(prim::kPrimReduceMean->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceMax,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(prim::kPrimReduceMax->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceSum,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(prim::kPrimReduceSum->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceMin,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(prim::kPrimReduceMin->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceProd,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(prim::kPrimReduceProd->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceAll,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(prim::kPrimReduceAll->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ReduceAny,
                                 []() { return std::make_shared<ReduceCpuKernelMod>(prim::kPrimReduceAny->name()); });
}  // namespace kernel
}  // namespace mindspore
