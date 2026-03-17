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

#include "kernel/cpu/native/div_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <complex>
#include <unordered_map>
#include <utility>
#include "primitive/nn_optimizer_ops.h"
#include "primitive/math_ops.h"

#include "kernel/cpu/nnacl/fp32/arithmetic_fp32.h"

namespace mindspore {
namespace kernel {
namespace div_cpu {
namespace {
constexpr float kMaxSubSerialSize = 10000.0;
constexpr float kMaxPowSerialSize = 700.0;
constexpr auto kDiv = "Div";

template <typename T, typename S>
class DivCpuTypeFunc : public CpuKernelFunc {
 public:
  ~DivCpuTypeFunc() override = default;
  DivCpuTypeFunc() = default;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    input_shape1_ = inputs[kIndex0]->GetShapeVector();
    input_shape2_ = inputs[kIndex1]->GetShapeVector();
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
    InitComputeFunc();
  }

  void Div(const T *input1, const T *input2, S *out);
  void DivComplex(const T *input1, const T *input2, S *out);

  bool RunFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
               const std::vector<KernelTensor *> &outputs) override {
    auto *input1 = reinterpret_cast<T *>(inputs[0]->device_ptr());
    const auto *input2 = reinterpret_cast<T *>(inputs[1]->device_ptr());
    auto *output = reinterpret_cast<S *>(outputs[0]->device_ptr());
    if (output_size_ == 0) {
      MS_LOG(WARNING) << "Div output shape contain 0, output_shape: " << output_shape_;
      return true;
    }
    compute_func_(this, input1, input2, output);
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
    if constexpr (!((std::is_same_v<T, complex64>) || (std::is_same_v<T, complex128>))) {
      compute_func_ = &DivCpuTypeFunc<T, S>::Div;
    } else {
      compute_func_ = &DivCpuTypeFunc<T, S>::DivComplex;
    }
  }

  size_t output_size_{1};
  ArithmeticParameter op_para_{};

  ShapeVector input_shape1_;
  ShapeVector input_shape2_;
  std::vector<size_t> input_index1_;
  std::vector<size_t> input_index2_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> input_element_num2_;
  ShapeVector output_shape_;
  std::vector<size_t> output_element_num_;
  bool is_init_broadcast_{false};

  using TypeComputeFunc = std::function<void(DivCpuTypeFunc *, const T *in_x, const T *in_y, S *out)>;
  TypeComputeFunc compute_func_{nullptr};
};

template <typename T, typename S>
void DivCpuTypeFunc<T, S>::Div(const T *input1, const T *input2, S *out) {
  if (!is_init_broadcast_) {
    InitBroadCast();
  }
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[input_index1_[i]];
      auto divisor = input2[input_index2_[i]];

      auto zero = static_cast<T>(0);
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<S>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<S>::has_infinity) {
          out[i] = dividend > zero ? std::numeric_limits<S>::infinity() : -std::numeric_limits<S>::infinity();
        } else {
          out[i] = dividend > zero ? std::numeric_limits<S>::max() : std::numeric_limits<S>::min();
        }
        continue;
      }
      out[i] = static_cast<S>(static_cast<double>(dividend) / static_cast<double>(divisor));
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T, typename S>
void DivCpuTypeFunc<T, S>::DivComplex(const T *input1, const T *input2, S *out) {
  if (!is_init_broadcast_) {
    InitBroadCast();
  }
  auto task = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[input_index1_[i]];
      auto divisor = input2[input_index2_[i]];

      auto zero = static_cast<T>(0);
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<S>::quiet_NaN();
          continue;
        }
        continue;
      }
      out[i] = static_cast<S>(dividend / divisor);
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T, typename S>
std::shared_ptr<CpuKernelFunc> SpecializeArithFunc() {
  return std::make_shared<DivCpuTypeFunc<T, S>>();
}

template <typename IN1_T, typename IN2_T, typename OUT_T>
class DivMixedCpuTypeFunc : public CpuKernelFunc {
 public:
  ~DivMixedCpuTypeFunc() override = default;
  DivMixedCpuTypeFunc() = default;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    input_shape1_ = inputs[kIndex0]->GetShapeVector();
    input_shape2_ = inputs[kIndex1]->GetShapeVector();
    output_shape_ = outputs[kIndex0]->GetShapeVector();
    if (output_shape_.empty()) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
    }
    output_size_ = SizeOf(output_shape_);
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
    is_init_broadcast_ = false;
    return KRET_OK;
  }

  void InitFunc(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                const std::vector<KernelTensor *> &outputs) override {}

  bool RunFunc(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
               const std::vector<KernelTensor *> &outputs) override {
    auto *input1 = reinterpret_cast<IN1_T *>(inputs[0]->device_ptr());
    const auto *input2 = reinterpret_cast<IN2_T *>(inputs[1]->device_ptr());
    auto *output = reinterpret_cast<OUT_T *>(outputs[0]->device_ptr());
    if (output_size_ == 0) {
      MS_LOG(WARNING) << "Div output shape contain 0, output_shape: " << output_shape_;
      return true;
    }
    Div(input1, input2, output);
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

  void Div(const IN1_T *input1, const IN2_T *input2, OUT_T *out) {
    if (!is_init_broadcast_) {
      InitBroadCast();
    }
    auto task = [input1, input2, out, this](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto dividend = static_cast<OUT_T>(input1[input_index1_[i]]);
        auto divisor = static_cast<OUT_T>(input2[input_index2_[i]]);

        auto zero = static_cast<OUT_T>(0);
        if (divisor == zero) {
          if (dividend == zero) {
            out[i] = std::numeric_limits<OUT_T>::quiet_NaN();
            continue;
          }
          if (std::numeric_limits<OUT_T>::has_infinity) {
            out[i] = dividend > zero ? std::numeric_limits<OUT_T>::infinity() : -std::numeric_limits<OUT_T>::infinity();
          } else {
            out[i] = dividend > zero ? std::numeric_limits<OUT_T>::max() : std::numeric_limits<OUT_T>::min();
          }
          continue;
        }
        out[i] = static_cast<OUT_T>(static_cast<double>(dividend) / static_cast<double>(divisor));
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  }

  size_t output_size_{1};
  ShapeVector input_shape1_;
  ShapeVector input_shape2_;
  ShapeVector output_shape_;
  std::vector<size_t> input_index1_;
  std::vector<size_t> input_index2_;
  bool is_init_broadcast_{false};
};

template <typename IN1_T, typename IN2_T, typename OUT_T>
std::shared_ptr<CpuKernelFunc> SpecializeMixedArithFunc() {
  return std::make_shared<DivMixedCpuTypeFunc<IN1_T, IN2_T, OUT_T>>();
}

using DivCpuFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>()>;

#define REGISTER_MIXED_ARITH_FUNCS()                                                                                   \
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),   \
   SpecializeMixedArithFunc<float16, float, float>},                                                                   \
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), \
     SpecializeMixedArithFunc<float16, double, double>},                                                               \
    {KernelAttr()                                                                                                      \
       .AddInputAttr(kNumberTypeFloat16)                                                                               \
       .AddInputAttr(kNumberTypeBFloat16)                                                                              \
       .AddOutputAttr(kNumberTypeFloat32),                                                                             \
     SpecializeMixedArithFunc<float16, bfloat16, float>},                                                              \
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat16),    \
     SpecializeMixedArithFunc<float16, bool, float16>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat16),    \
     SpecializeMixedArithFunc<float16, int8_t, float16>},                                                              \
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<float16, int16_t, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<float16, int32_t, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<float16, int64_t, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<float16, uint8_t, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat32), \
     SpecializeMixedArithFunc<float, float16, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64), \
     SpecializeMixedArithFunc<float, double, double>},                                                                 \
    {KernelAttr()                                                                                                      \
       .AddInputAttr(kNumberTypeFloat32)                                                                               \
       .AddInputAttr(kNumberTypeBFloat16)                                                                              \
       .AddOutputAttr(kNumberTypeFloat32),                                                                             \
     SpecializeMixedArithFunc<float, bfloat16, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),    \
     SpecializeMixedArithFunc<float, bool, float>},                                                                    \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),    \
     SpecializeMixedArithFunc<float, int8_t, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<float, int16_t, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<float, int32_t, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<float, int64_t, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<float, uint8_t, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat64), \
     SpecializeMixedArithFunc<double, float16, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat64), \
     SpecializeMixedArithFunc<double, float, double>},                                                                 \
    {KernelAttr()                                                                                                      \
       .AddInputAttr(kNumberTypeFloat64)                                                                               \
       .AddInputAttr(kNumberTypeBFloat16)                                                                              \
       .AddOutputAttr(kNumberTypeFloat64),                                                                             \
     SpecializeMixedArithFunc<double, bfloat16, double>},                                                              \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat64),    \
     SpecializeMixedArithFunc<double, bool, double>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat64),    \
     SpecializeMixedArithFunc<double, int8_t, double>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<double, int16_t, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<double, int32_t, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<double, int64_t, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<double, uint8_t, double>},                                                               \
    {KernelAttr()                                                                                                      \
       .AddInputAttr(kNumberTypeBFloat16)                                                                              \
       .AddInputAttr(kNumberTypeFloat16)                                                                               \
       .AddOutputAttr(kNumberTypeFloat32),                                                                             \
     SpecializeMixedArithFunc<bfloat16, float16, float>},                                                              \
    {KernelAttr()                                                                                                      \
       .AddInputAttr(kNumberTypeBFloat16)                                                                              \
       .AddInputAttr(kNumberTypeFloat32)                                                                               \
       .AddOutputAttr(kNumberTypeFloat32),                                                                             \
     SpecializeMixedArithFunc<bfloat16, float, float>},                                                                \
    {KernelAttr()                                                                                                      \
       .AddInputAttr(kNumberTypeBFloat16)                                                                              \
       .AddInputAttr(kNumberTypeFloat64)                                                                               \
       .AddOutputAttr(kNumberTypeFloat64),                                                                             \
     SpecializeMixedArithFunc<bfloat16, double, double>},                                                              \
    {KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBFloat16),  \
     SpecializeMixedArithFunc<bfloat16, bool, bfloat16>},                                                              \
    {KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBFloat16),  \
     SpecializeMixedArithFunc<bfloat16, int8_t, bfloat16>},                                                            \
    {KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<bfloat16, int16_t, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<bfloat16, int32_t, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<bfloat16, int64_t, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<bfloat16, uint8_t, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),    \
     SpecializeMixedArithFunc<bool, float16, float16>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),    \
     SpecializeMixedArithFunc<bool, float, float>},                                                                    \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),    \
     SpecializeMixedArithFunc<bool, double, double>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16),  \
     SpecializeMixedArithFunc<bool, bfloat16, bfloat16>},                                                              \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),       \
     SpecializeMixedArithFunc<bool, int8_t, float>},                                                                   \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<bool, int16_t, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<bool, int32_t, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<bool, int64_t, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<bool, uint8_t, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),    \
     SpecializeMixedArithFunc<int8_t, float16, float16>},                                                              \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),    \
     SpecializeMixedArithFunc<int8_t, float, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),    \
     SpecializeMixedArithFunc<int8_t, double, double>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16),  \
     SpecializeMixedArithFunc<int8_t, bfloat16, bfloat16>},                                                            \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),       \
     SpecializeMixedArithFunc<int8_t, bool, float>},                                                                   \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int8_t, int16_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int8_t, int32_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int8_t, int64_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int8_t, uint8_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<int16_t, float16, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<int16_t, float, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<int16_t, double, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<int16_t, bfloat16, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int16_t, bool, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int16_t, int8_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int16_t, int32_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int16_t, int64_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int16_t, uint8_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<int32_t, float16, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<int32_t, float, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<int32_t, double, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<int32_t, bfloat16, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int32_t, bool, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int32_t, int8_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int32_t, int16_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int32_t, int64_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int32_t, uint8_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<int64_t, float16, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<int64_t, float, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<int64_t, double, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<int64_t, bfloat16, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int64_t, bool, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<int64_t, int8_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int64_t, int16_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int64_t, int32_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<int64_t, uint8_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),   \
     SpecializeMixedArithFunc<uint8_t, float16, float16>},                                                             \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),   \
     SpecializeMixedArithFunc<uint8_t, float, float>},                                                                 \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),   \
     SpecializeMixedArithFunc<uint8_t, double, double>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16), \
     SpecializeMixedArithFunc<uint8_t, bfloat16, bfloat16>},                                                           \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<uint8_t, bool, float>},                                                                  \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),      \
     SpecializeMixedArithFunc<uint8_t, int8_t, float>},                                                                \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<uint8_t, int16_t, float>},                                                               \
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),     \
     SpecializeMixedArithFunc<uint8_t, int32_t, float>},                                                               \
  {                                                                                                                    \
    KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),      \
      SpecializeMixedArithFunc<uint8_t, int64_t, float>                                                                \
  }

static std::vector<std::pair<KernelAttr, DivCpuFuncCreator>> kernel_attr_list = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
   SpecializeArithFunc<float16, float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<float, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   SpecializeArithFunc<double, double>},
  {KernelAttr().AddInputAttr(kNumberTypeBFloat16).AddInputAttr(kNumberTypeBFloat16).AddOutputAttr(kNumberTypeBFloat16),
   SpecializeArithFunc<bfloat16, bfloat16>},
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<bool, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<int8_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<int16_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<int32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<int64_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<uint8_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<uint16_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<uint32_t, float>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeFloat32),
   SpecializeArithFunc<uint64_t, float>},
  REGISTER_MIXED_ARITH_FUNCS(),
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex64)
     .AddInputAttr(kNumberTypeComplex64)
     .AddOutputAttr(kNumberTypeComplex64),
   SpecializeArithFunc<complex64, complex64>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeComplex128)
     .AddInputAttr(kNumberTypeComplex128)
     .AddOutputAttr(kNumberTypeComplex128),
   SpecializeArithFunc<complex128, complex128>}};
}  // namespace

bool DivCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For 'Div', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  func_obj_ = kernel_attr_list[index].second();
  func_obj_->InitFunc(primitive_, inputs, outputs);
  return true;
}

int DivCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return func_obj_->Resize(inputs, outputs);
}

std::vector<KernelAttr> DivCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(kernel_attr_list.begin(), kernel_attr_list.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, DivCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Div, DivCpuKernelMod);
}  // namespace div_cpu
}  // namespace kernel
}  // namespace mindspore
