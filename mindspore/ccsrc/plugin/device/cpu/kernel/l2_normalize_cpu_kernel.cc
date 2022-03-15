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

#include "plugin/device/cpu/kernel/l2_normalize_cpu_kernel.h"

#include <algorithm>
#include <utility>
#include <limits>
#include <string>

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kL2NormalizeInputsNum = 1;
constexpr size_t kL2NormalizeOutputsNum = 1;

template <typename T>
class L2NormalizeCpuFunc : public CpuKernelFunc {
 public:
  L2NormalizeCpuFunc() = default;
  ~L2NormalizeCpuFunc() override = default;

  void InitFunc(const CNodePtr &kernel_node) override;

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
               const std::vector<AddressPtr> &outputs) override;

  void CalcDenominator(const T *input_addr, const size_t reduce_size, const int dims,
                       std::unique_ptr<T[]> *denominator_addr);

  void CalcOutput(const T *input_addr, const std::vector<size_t> &reduce_shape, const size_t output_size,
                  T *output_addr, std::unique_ptr<T[]> const &denominator_addr);

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  T epsilon_{0};
  int axis_{0};
  void CheckParam(const CNodePtr &kernel_node);
  std::string kernel_name_;
};

template <typename T>
void L2NormalizeCpuFunc<T>::InitFunc(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  epsilon_ = static_cast<T>(common::AnfAlgo::GetNodeAttr<float>(kernel_node, EPSILON));
  axis_ = LongToInt(common::AnfAlgo::GetNodeAttr<int64_t>(kernel_node, AXIS));
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);

  int dims = SizeToInt(input_shape_.size());
  if (axis_ < -dims || axis_ >= dims) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'axis' should be in [" << -dims << ", " << dims
                      << "), but got: " << axis_;
  }
  if (epsilon_ == static_cast<T>(0.0)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the parameter of 'epsilon' can not be zero.";
  }
  if (axis_ < 0) {
    axis_ += SizeToInt(input_shape_.size());
  }
}

template <typename T>
void L2NormalizeCpuFunc<T>::CalcDenominator(const T *input_addr, const size_t reduce_size, const int dims,
                                            std::unique_ptr<T[]> *denominator_addr) {
  // Calculate transpose axes and stride
  size_t stride = 1;
  std::vector<size_t> axes(input_shape_.size());
  size_t k = 0;
  size_t axis_size = IntToSize(axis_);
  for (size_t i = 0; i < IntToSize(dims); ++i) {
    if (i != axis_size) {
      axes[k] = i;
      ++k;
    } else {
      stride *= input_shape_[i];
    }
  }
  axes[k] = axis_size;

  std::vector<size_t> transpose_shape(input_shape_.size());
  for (size_t i = 0; i < IntToSize(dims); ++i) {
    transpose_shape[i] = input_shape_[axes[i]];
  }

  TransposeIterator tran_base_iter(std::move(transpose_shape), std::move(axes), input_shape_);

  auto task = [this, &tran_base_iter, &input_addr, &denominator_addr, stride](size_t start, size_t end) {
    T temp = static_cast<T>(0.0);
    T denominator = static_cast<T>(0.0);
    auto iter = tran_base_iter;
    iter.SetPos(start * stride);
    for (size_t i = start; i < end; ++i) {
      denominator = input_addr[iter.GetPos()];
      denominator = denominator * denominator;
      iter.GenNextPos();
      for (size_t j = 1; j < stride; ++j) {
        temp = input_addr[iter.GetPos()];
        denominator += temp * temp;
        iter.GenNextPos();
      }
      denominator = (denominator > epsilon_) ? denominator : epsilon_;
      (*denominator_addr)[i] = sqrt(denominator);
    }
  };
  ParallelLaunchAutoSearch(task, reduce_size, this, &parallel_search_info_);
}

template <typename T>
void L2NormalizeCpuFunc<T>::CalcOutput(const T *input_addr, const std::vector<size_t> &reduce_shape,
                                       const size_t output_size, T *output_addr,
                                       std::unique_ptr<T[]> const &denominator_addr) {
  BroadcastIterator broad_base_iter(input_shape_, reduce_shape, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = broad_base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; ++i) {
      T dividend = input_addr[iter.GetInputPosA()];
      T divisor = denominator_addr[iter.GetInputPosB()];
      if (divisor == static_cast<T>(0)) {
        if (dividend == static_cast<T>(0)) {
          output_addr[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          output_addr[i] =
            dividend > static_cast<T>(0) ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          output_addr[i] = dividend > static_cast<T>(0) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      output_addr[i] = dividend / divisor;
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
}

template <typename T>
bool L2NormalizeCpuFunc<T>::RunFunc(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kL2NormalizeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kL2NormalizeOutputsNum, kernel_name_);
  auto input_addr = reinterpret_cast<T *>(inputs[0]->addr);
  auto output_addr = reinterpret_cast<T *>(outputs[0]->addr);

  int dims = SizeToInt(input_shape_.size());
  std::vector<size_t> reduce_shape = input_shape_;
  size_t reduce_size = 1;
  reduce_shape[axis_] = 1;
  for (size_t i = 0; i < input_shape_.size(); ++i) {
    reduce_size *= reduce_shape[i];
  }
  auto denominator_addr = std::make_unique<T[]>(reduce_size);

  L2NormalizeCpuFunc<T>::CalcDenominator(input_addr, reduce_size, dims, &denominator_addr);

  size_t output_size = outputs[0]->size / sizeof(T);
  L2NormalizeCpuFunc<T>::CalcOutput(input_addr, reduce_shape, output_size, output_addr, denominator_addr);

  return true;
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeL2NormFunc() {
  return std::make_shared<L2NormalizeCpuFunc<T>>();
}
using SpecializeL2NormFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::vector<std::pair<KernelAttr, SpecializeL2NormFuncCreator>> func_class_list = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16), SpecializeL2NormFunc<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32), SpecializeL2NormFunc<float>}};
}  // namespace

void L2NormalizeCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "L2Norm does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = func_class_list[index].second();
  func_obj_->InitFunc(kernel_node);
}

std::vector<KernelAttr> L2NormalizeCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  std::transform(func_class_list.begin(), func_class_list.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, SpecializeL2NormFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, L2Normalize, L2NormalizeCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
