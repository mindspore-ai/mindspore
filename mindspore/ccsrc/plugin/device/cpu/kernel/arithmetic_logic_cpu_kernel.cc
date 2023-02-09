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

#include "plugin/device/cpu/kernel/arithmetic_logic_cpu_kernel.h"

#include <string>
#include <cmath>
#include <unordered_map>
#include <functional>
#include <map>
#include <algorithm>
#include <utility>
#include <complex>

#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

constexpr size_t kMaxLessSerialSize = 15000;
constexpr auto kLess = "Less";
constexpr auto kLessEqual = "LessEqual";
constexpr auto kGreater = "Greater";
constexpr auto kGreaterEqual = "GreaterEqual";
constexpr auto kLogicalAnd = "LogicalAnd";
constexpr auto kLogicalOr = "LogicalOr";
constexpr auto kLogicalXor = "LogicalXor";
constexpr auto kEqual = "Equal";
constexpr auto kNotEqual = "NotEqual";

template <typename T>
class ArithLogicCpuTypeFunc : public CpuKernelFunc {
 public:
  ArithLogicCpuTypeFunc() = default;
  ~ArithLogicCpuTypeFunc() override = default;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
               const std::vector<AddressPtr> &outputs) override {
    const auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
    const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
    bool *output = reinterpret_cast<bool *>(outputs[0]->addr);
    compute_func_(this, input1, input2, output);
    return true;
  }

  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    dtype_ = inputs.at(kIndex0)->GetDtype();
    auto dtype_1 = inputs.at(kIndex1)->GetDtype();
    if (dtype_ != dtype_1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'input1' and 'input2' should have the same data type, but got type of 'input1': "
                        << dtype_ << ", and the type of 'input2': " << dtype_1;
    }
    static std::unordered_map<std::string, TypeComputeFunc> arithmetic_logic_func_map;
    if constexpr (!((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>))) {
      arithmetic_logic_func_map = {{kGreater, &ArithLogicCpuTypeFunc<T>::Greater},
                                   {kGreaterEqual, &ArithLogicCpuTypeFunc<T>::GreaterEqual},
                                   {kLogicalAnd, &ArithLogicCpuTypeFunc<T>::LogicalAnd},
                                   {kLessEqual, &ArithLogicCpuTypeFunc<T>::LessEqual},
                                   {kLogicalOr, &ArithLogicCpuTypeFunc<T>::LogicalOr},
                                   {kLogicalXor, &ArithLogicCpuTypeFunc<T>::LogicalXor},
                                   {kLess, &ArithLogicCpuTypeFunc<T>::Less},
                                   {kNotEqual, &ArithLogicCpuTypeFunc<T>::NotEqual}};
    } else {
      arithmetic_logic_func_map = {{kNotEqual, &ArithLogicCpuTypeFunc<T>::NotEqual}};
    }
    if (arithmetic_logic_func_map.find(kernel_name_) == arithmetic_logic_func_map.end()) {
      MS_LOG(EXCEPTION) << "For 'ArithmeticLogic', only supports operators in "
                        << Map2Str<std::unordered_map, TypeComputeFunc>(arithmetic_logic_func_map) << ", but got "
                        << kernel_name_;
    }
    compute_func_ = arithmetic_logic_func_map.at(kernel_name_);
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    input_shape1_ = inputs.at(kIndex0)->GetShapeVector();
    input_shape2_ = inputs.at(kIndex1)->GetShapeVector();
    output_shape_ = outputs.at(kIndex0)->GetShapeVector();
    if (output_shape_.empty()) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
    }
    output_size_ = SizeOf(output_shape_);
    size_t l = input_shape1_.size();
    for (size_t i = 0; i < output_shape_.size() - l; ++i) {
      (void)input_shape1_.insert(input_shape1_.begin(), 1);
    }
    l = input_shape2_.size();
    for (size_t i = 0; i < output_shape_.size() - l; ++i) {
      (void)input_shape2_.insert(input_shape2_.begin(), 1);
    }
    CPUKernelUtils::GetElementNumEveryDim(input_shape1_, &input_element_num1_);
    CPUKernelUtils::GetElementNumEveryDim(input_shape2_, &input_element_num2_);
    CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
    return KRET_OK;
  }

 private:
  template <typename Op>
  void BinaryOp(const T *input1, const T *input2, bool *out, Op op);

  void Less(const T *input1, const T *input2, bool *out);
  void NotEqual(const T *input1, const T *input2, bool *out);
  void Greater(const T *input1, const T *input2, bool *out);
  void GreaterEqual(const T *input1, const T *input2, bool *out);
  void LessEqual(const T *input1, const T *input2, bool *out);
  void LogicalAnd(const T *input1, const T *input2, bool *out);
  void LogicalOr(const T *input1, const T *input2, bool *out);
  void LogicalXor(const T *input1, const T *input2, bool *out);

  using TypeComputeFunc = std::function<void(ArithLogicCpuTypeFunc *, const T *, const T *, bool *)>;
  TypeComputeFunc compute_func_{nullptr};

  std::string kernel_name_;
  size_t output_size_{1};
  TypeId dtype_{kTypeUnknown};

  ShapeVector input_shape1_;
  ShapeVector input_shape2_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> input_element_num2_;
  ShapeVector output_shape_;
  std::vector<size_t> output_element_num_;
};

template <typename T>
class ArithComplexLogicCpuTypeFunc : public CpuKernelFunc {
 public:
  ArithComplexLogicCpuTypeFunc() = default;
  ~ArithComplexLogicCpuTypeFunc() override = default;
  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
               const std::vector<AddressPtr> &outputs) override {
    const auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
    const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
    bool *output = reinterpret_cast<bool *>(outputs[0]->addr);
    compute_func_(this, input1, input2, output);
    return true;
  }

  void InitFunc(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                const std::vector<KernelTensorPtr> &outputs) override {
    kernel_name_ = base_operator->name();
    dtype_ = inputs.at(kIndex0)->GetDtype();
    auto dtype_1 = inputs.at(kIndex1)->GetDtype();
    if (dtype_ != dtype_1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the 'input1' and 'input2' should have the same data type, but got type of 'input1': "
                        << dtype_ << ", and the type of 'input2': " << dtype_1;
    }
    static const std::unordered_map<std::string, ComplexTypeComputeFunc> arithmetic_logic_func_map{
      {kEqual, &ArithComplexLogicCpuTypeFunc<T>::Equal}};
    if (arithmetic_logic_func_map.find(kernel_name_) == arithmetic_logic_func_map.end()) {
      MS_LOG(EXCEPTION) << "For 'ArithmeticLogic', only supports operators in "
                        << Map2Str<std::unordered_map, ComplexTypeComputeFunc>(arithmetic_logic_func_map)
                        << ", but got " << kernel_name_;
    }
    compute_func_ = arithmetic_logic_func_map.at(kernel_name_);
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override {
    input_shape1_ = inputs.at(kIndex0)->GetShapeVector();
    input_shape2_ = inputs.at(kIndex1)->GetShapeVector();
    output_shape_ = outputs.at(kIndex0)->GetShapeVector();
    if (output_shape_.empty()) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
    }
    output_size_ = SizeOf(output_shape_);
    size_t l = input_shape1_.size();
    for (size_t i = 0; i < output_shape_.size() - l; ++i) {
      (void)input_shape1_.insert(input_shape1_.begin(), 1);
    }
    l = input_shape2_.size();
    for (size_t i = 0; i < output_shape_.size() - l; ++i) {
      (void)input_shape2_.insert(input_shape2_.begin(), 1);
    }
    CPUKernelUtils::GetElementNumEveryDim(input_shape1_, &input_element_num1_);
    CPUKernelUtils::GetElementNumEveryDim(input_shape2_, &input_element_num2_);
    CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
    return KRET_OK;
  }

 private:
  template <typename Op>
  void BinaryOp(const T *input1, const T *input2, bool *out, Op op);

  void Equal(const T *input1, const T *input2, bool *out);

  using ComplexTypeComputeFunc = std::function<void(ArithComplexLogicCpuTypeFunc *, const T *, const T *, bool *)>;
  ComplexTypeComputeFunc compute_func_{nullptr};

  std::string kernel_name_;
  size_t output_size_{1};
  TypeId dtype_{kTypeUnknown};

  ShapeVector input_shape1_;
  ShapeVector input_shape2_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> input_element_num2_;
  ShapeVector output_shape_;
  std::vector<size_t> output_element_num_;
};

template <typename T>
template <typename Op>
void ArithLogicCpuTypeFunc<T>::BinaryOp(const T *input1, const T *input2, bool *out, Op op) {
  int64_t input1_size = 1;
  int64_t input2_size = 1;

  for (size_t i = 0; i < output_shape_.size(); i++) {
    input1_size *= input_shape1_[i];
    input2_size *= input_shape2_[i];
  }

  if (input_shape1_ == input_shape2_) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[i], input2[i]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else if (input1_size == 1) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[0], input2[i]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else if (input2_size == 1) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[i], input2[0]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else {
    BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
    auto task = [this, input1, input2, out, op, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto x = input1[iter.GetInputPosA()];
        auto y = input2[iter.GetInputPosB()];
        out[i] = op(x, y);
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }
}

template <typename T>
template <typename Op>
void ArithComplexLogicCpuTypeFunc<T>::BinaryOp(const T *input1, const T *input2, bool *out, Op op) {
  int64_t input1_size = 1;
  int64_t input2_size = 1;

  for (size_t i = 0; i < output_shape_.size(); i++) {
    input1_size *= input_shape1_[i];
    input2_size *= input_shape2_[i];
  }

  if (input_shape1_ == input_shape2_) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[i], input2[i]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else if (input1_size == 1) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[0], input2[i]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else if (input2_size == 1) {
    auto task = [this, input1, input2, out, op](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        out[i] = op(input1[i], input2[0]);
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else {
    BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
    auto task = [this, input1, input2, out, op, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto x = input1[iter.GetInputPosA()];
        auto y = input2[iter.GetInputPosB()];
        out[i] = op(x, y);
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::Less(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::less<T>());
}

template <typename T>
void ArithComplexLogicCpuTypeFunc<T>::Equal(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::equal_to<T>());
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::NotEqual(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::not_equal_to<T>());
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::LogicalAnd(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::logical_and<T>());
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::LogicalOr(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::logical_or<T>());
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::LogicalXor(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::not_equal_to<T>());
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::Greater(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::greater<T>());
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::GreaterEqual(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::greater_equal<T>());
}

template <typename T>
void ArithLogicCpuTypeFunc<T>::LessEqual(const T *input1, const T *input2, bool *out) {
  BinaryOp(input1, input2, out, std::less_equal<T>());
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeArithLogFunc() {
  return std::make_shared<ArithLogicCpuTypeFunc<T>>();
}
template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeArithLogComplexFunc() {
  return std::make_shared<ArithComplexLogicCpuTypeFunc<T>>();
}
using ArithLogicCpuFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, ArithLogicCpuFuncCreator>>> kernel_attr_lists = {
  {kLess,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<double>}}},
  {kEqual,
   {{KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<complex128>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogComplexFunc<double>}}},
  {kNotEqual,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeComplex64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<complex64>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<complex128>}}},
  {kGreater,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<double>}}},
  {kGreaterEqual,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<bool>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<double>}}},
  {kLessEqual,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<uint64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<float16>}}},
  {kLogicalAnd,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<bool>}}},
  {kLogicalOr,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<bool>}}},
  {kLogicalXor,
   {{KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithLogFunc<bool>}}}};
}  // namespace

bool ArithmeticLogicCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(ERROR) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Arithmetic', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  func_obj_ = kernel_attr_lists[kernel_name_][index].second();
  func_obj_->InitFunc(base_operator, inputs, outputs);
  return true;
}

int ArithmeticLogicCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return func_obj_->Resize(base_operator, inputs, outputs);
}

std::vector<KernelAttr> ArithmeticLogicCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_lists.find(kernel_type_);
  if (iter == kernel_attr_lists.end()) {
    MS_LOG(EXCEPTION) << "Arithmetic logic cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ArithLogicCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}

bool ArithmeticComplexLogicCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (kernel_name_ != kernel_type_) {
    MS_LOG(ERROR) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
    return false;
  }

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Arithmetic', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  func_obj_ = kernel_attr_lists[kernel_name_][index].second();
  func_obj_->InitFunc(base_operator, inputs, outputs);
  return true;
}

int ArithmeticComplexLogicCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                               const std::vector<KernelTensorPtr> &inputs,
                                               const std::vector<KernelTensorPtr> &outputs,
                                               const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return func_obj_->Resize(base_operator, inputs, outputs);
}

std::vector<KernelAttr> ArithmeticComplexLogicCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_lists.find(kernel_type_);
  if (iter == kernel_attr_lists.end()) {
    MS_LOG(EXCEPTION) << "Arithmetic logic cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ArithLogicCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Less,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kLess); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Equal,
                                 []() { return std::make_shared<ArithmeticComplexLogicCpuKernelMod>(kEqual); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, NotEqual,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kNotEqual); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Greater,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kGreater); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, GreaterEqual,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kGreaterEqual); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LessEqual,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kLessEqual); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LogicalAnd,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kLogicalAnd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LogicalOr,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kLogicalOr); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LogicalXor,
                                 []() { return std::make_shared<ArithmeticLogicCpuKernelMod>(kLogicalXor); });
}  // namespace kernel
}  // namespace mindspore
