/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/arithmetic_ext_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <complex>
#include <unordered_map>
#include <utility>
#include "mindspore/core/ops/nn_optimizer_ops.h"
#include "mindspore/core/ops/math_ops.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/arithmetic_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/mul_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/power_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/sub_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/add_fp32.h"
#include "Eigen/Eigen"

namespace mindspore {
namespace kernel {
namespace {
constexpr float kMaxSubSerialSize = 10000.0;
constexpr float kMaxPowSerialSize = 700.0;
constexpr auto kAddExt = "AddExt";
constexpr auto kSubExt = "SubExt";

template <typename T1, typename T2>
class ArithmeticExtCpuTypeFunc : public CpuKernelFunc {
 public:
  ~ArithmeticExtCpuTypeFunc() override = default;
  ArithmeticExtCpuTypeFunc() = default;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    input_shape1_ = inputs[kIndex0]->GetShapeVector();
    input_shape2_ = inputs[kIndex1]->GetShapeVector();
    input_shape3_ = inputs[kIndex2]->GetShapeVector();
    output_shape_ = outputs[kIndex0]->GetShapeVector();
    if (output_shape_.empty()) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
    }
    output_size_ = SizeOf(output_shape_);
    op_para_.in_elements_num0_ = SizeToLong(SizeOf(input_shape1_));
    op_para_.in_elements_num1_ = SizeToLong(SizeOf(input_shape2_));
    size_t l = input_shape1_.size();
    if (l < output_shape_.size()) {
      for (size_t i = 0; i < output_shape_.size() - l; ++i) {
        (void)input_shape1_.insert(input_shape1_.begin(), 1);
      }
    }
    l = input_shape2_.size();
    if (l < output_shape_.size()) {
      for (size_t i = 0; i < output_shape_.size() - l; ++i) {
        (void)input_shape2_.insert(input_shape2_.begin(), 1);
      }
    }
    input_element_num1_.clear();
    CPUKernelUtils::GetElementNumEveryDim(input_shape1_, &input_element_num1_);
    input_element_num2_.clear();
    CPUKernelUtils::GetElementNumEveryDim(input_shape2_, &input_element_num2_);
    output_element_num_.clear();
    CPUKernelUtils::GetElementNumEveryDim(output_shape_, &output_element_num_);
    is_init_broadcast_ = false;
    return KRET_OK;
  }

  void InitFunc(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                const std::vector<KernelTensor *> &outputs) override {
    kernel_name_ = primitive->name();
    InitComputeFunc();
  }

  void AddExt(const T1 *input1, const T1 *input2, const T2 *alpha, T1 *out);
  void SubExt(const T1 *input1, const T1 *input2, const T2 *alpha, T1 *out);

  bool RunFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
               const std::vector<KernelTensor *> &outputs) override {
    auto *input1 = reinterpret_cast<T1 *>(inputs[0]->device_ptr());
    const auto *input2 = reinterpret_cast<T1 *>(inputs[1]->device_ptr());
    const auto *alpha = reinterpret_cast<T2 *>(inputs[2]->device_ptr());
    auto *output = reinterpret_cast<T1 *>(outputs[0]->device_ptr());
    if (output_size_ == 0) {
      MS_LOG(WARNING) << kernel_name_ << " output shape contain 0, output_shape: " << output_shape_;
      return true;
    }

    compute_func_(this, input1, input2, alpha, output);
    return true;
  }

 private:
  void InitBroadCast() {
    BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
    base_iter.SetPos(0);
    input_index1_.clear();
    input_index2_.clear();
    input_index1_.resize(output_size_);
    input_index2_.resize(output_size_);
    for (size_t i = 0; i < output_size_; i++) {
      input_index1_[i] = base_iter.GetInputPosA();
      input_index2_[i] = base_iter.GetInputPosB();
      base_iter.GenNextPos();
    }
    is_init_broadcast_ = true;
  }
  void InitComputeFunc() {
    string dtype_desc;
    static std::unordered_map<std::string, TypeComputeFunc> arithmeticMathFuncMap;
    if constexpr (!((std::is_same_v<T1, complex64>) || (std::is_same_v<T1, complex128>))) {
      dtype_desc = "real data";
      arithmeticMathFuncMap = {
        {kAddExt, &ArithmeticExtCpuTypeFunc<T1, T2>::AddExt},
        {kSubExt, &ArithmeticExtCpuTypeFunc<T1, T2>::SubExt},
      };
    } else {
      dtype_desc = "complex data";
      arithmeticMathFuncMap = {{kAddExt, &ArithmeticExtCpuTypeFunc<T1, T2>::AddExt},
                               {kSubExt, &ArithmeticExtCpuTypeFunc<T1, T2>::SubExt}};
    }
    if (arithmeticMathFuncMap.find(kernel_name_) == arithmeticMathFuncMap.end()) {
      MS_LOG(EXCEPTION) << "For 'Arithmetic', it only supports operators in "
                        << Map2Str<std::unordered_map, TypeComputeFunc>(arithmeticMathFuncMap) << ", but got "
                        << kernel_name_ << " for " << dtype_desc << ".";
    }
    compute_func_ = arithmeticMathFuncMap.at(kernel_name_);
  }

  std::string kernel_name_;

  size_t output_size_{1};
  ArithmeticParameter op_para_{};

  ShapeVector input_shape1_;
  ShapeVector input_shape2_;
  ShapeVector input_shape3_;
  std::vector<size_t> input_index1_;
  std::vector<size_t> input_index2_;
  std::vector<size_t> input_index3_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> input_element_num2_;
  std::vector<size_t> input_element_num3_;
  ShapeVector output_shape_;
  std::vector<size_t> output_element_num_;
  bool is_init_broadcast_{false};

  using TypeComputeFunc =
    std::function<void(ArithmeticExtCpuTypeFunc *, const T1 *in_x, const T1 *in_y, const T2 *alpha, T1 *out)>;
  TypeComputeFunc compute_func_{nullptr};
};

template <typename T1, typename T2>
void ArithmeticExtCpuTypeFunc<T1, T2>::AddExt(const T1 *input1, const T1 *input2, const T2 *alpha, T1 *out) {
  if constexpr (std::is_same_v<T1, float>) {
    if (input_shape1_ == input_shape2_) {
      auto task = [input1, input2, alpha, out](size_t start, size_t end) {
        (void)ElementAddExt(input1 + start, input2 + start, static_cast<T1>(*alpha), out + start, end - start);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
    if (op_para_.in_elements_num0_ == 1 || op_para_.in_elements_num1_ == 1) {
      auto task = [this, input1, input2, alpha, out](size_t start, size_t end) {
        if (op_para_.in_elements_num0_ == 1) {
          (void)ElementOptAddExt(input1, input2 + start, static_cast<T1>(*alpha), out + start, end - start, true);
        } else {
          (void)ElementOptAddExt(input1 + start, input2, static_cast<T1>(*alpha), out + start, end - start, false);
        }
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
  }
  if (!is_init_broadcast_) {
    InitBroadCast();
  }
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T1>(input1[input_index1_[i]] + input2[input_index2_[i]] * static_cast<T1>(*alpha));
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T1, typename T2>
void ArithmeticExtCpuTypeFunc<T1, T2>::SubExt(const T1 *input1, const T1 *input2, const T2 *alpha, T1 *out) {
  if constexpr (std::is_same_v<T1, float>) {
    if (input_shape1_ == input_shape2_) {
      auto task = [input1, input2, alpha, out](size_t start, size_t end) {
        (void)ElementSubExt(input1 + start, input2 + start, static_cast<T1>(*alpha), out + start, end - start);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }

    if (op_para_.in_elements_num0_ == 1 || op_para_.in_elements_num1_ == 1) {
      auto task = [this, input1, input2, alpha, out](size_t start, size_t end) {
        if (op_para_.in_elements_num0_ == 1) {
          (void)ElementOptSubExt(input1, input2 + start, static_cast<T1>(*alpha), out + start, end - start, true);
        } else {
          (void)ElementOptSubExt(input1 + start, input2, static_cast<T1>(*alpha), out + start, end - start, false);
        }
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
  }
  if (!is_init_broadcast_) {
    InitBroadCast();
  }
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T1>(input1[input_index1_[i]] - input2[input_index2_[i]] * static_cast<T1>(*alpha));
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T1, typename T2>
std::shared_ptr<CpuKernelFunc> SpecializeArithFunc() {
  return std::make_shared<ArithmeticExtCpuTypeFunc<T1, T2>>();
}

using ArithmeticExtCpuFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;
static std::map<std::string, std::vector<std::pair<KernelAttr, ArithmeticExtCpuFuncCreator>>> kernel_attr_list = {
  {kAddExt,
   {{KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt8),
     SpecializeArithFunc<int8_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt16),
     SpecializeArithFunc<int16_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt8),
     SpecializeArithFunc<uint8_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt16),
     SpecializeArithFunc<uint16_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt32),
     SpecializeArithFunc<uint32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt64),
     SpecializeArithFunc<uint64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeBool),
     SpecializeArithFunc<bool, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeBFloat16),
     SpecializeArithFunc<bfloat16, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeComplex64),
     SpecializeArithFunc<complex64, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeComplex128),
     SpecializeArithFunc<complex128, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     SpecializeArithFunc<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     SpecializeArithFunc<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     SpecializeArithFunc<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     SpecializeArithFunc<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     SpecializeArithFunc<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     SpecializeArithFunc<uint64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool),
     SpecializeArithFunc<bool, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBFloat16),
     SpecializeArithFunc<bfloat16, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64),
     SpecializeArithFunc<complex64, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     SpecializeArithFunc<complex128, int64_t>}}},
  {kSubExt,
   {{KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt8),
     SpecializeArithFunc<int8_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt16),
     SpecializeArithFunc<int16_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt8),
     SpecializeArithFunc<uint8_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt16),
     SpecializeArithFunc<uint16_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt32),
     SpecializeArithFunc<uint32_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeUInt64),
     SpecializeArithFunc<uint64_t, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeBool),
     SpecializeArithFunc<bool, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeBFloat16),
     SpecializeArithFunc<bfloat16, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeComplex64),
     SpecializeArithFunc<complex64, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeComplex128),
     SpecializeArithFunc<complex128, float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     SpecializeArithFunc<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     SpecializeArithFunc<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     SpecializeArithFunc<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     SpecializeArithFunc<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     SpecializeArithFunc<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     SpecializeArithFunc<uint64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool),
     SpecializeArithFunc<bool, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeBFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBFloat16),
     SpecializeArithFunc<bfloat16, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64),
     SpecializeArithFunc<complex64, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     SpecializeArithFunc<complex128, int64_t>}}}};
}  // namespace

bool ArithmeticExtCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto iter = kernel_attr_list.find(kernel_name_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(ERROR) << "For 'Arithmetic', the kernel name must be in "
                  << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, ArithmeticExtCpuFuncCreator>>>(
                       kernel_attr_list)
                  << ", but got " << kernel_name_;
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Arithmetic', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  func_obj_ = kernel_attr_list[kernel_name_][index].second();
  func_obj_->InitFunc(primitive_, inputs, outputs);
  return true;
}

int ArithmeticExtCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return func_obj_->Resize(inputs, outputs);
}

std::vector<KernelAttr> ArithmeticExtCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(EXCEPTION) << "Arithmetic cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ArithmeticExtCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AddExt,
                                 []() { return std::make_shared<ArithmeticExtCpuKernelMod>(kAddExt); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, SubExt,
                                 []() { return std::make_shared<ArithmeticExtCpuKernelMod>(kSubExt); });
}  // namespace kernel
}  // namespace mindspore
