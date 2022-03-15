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

#include "plugin/device/cpu/kernel/arithmetic_cpu_kernel.h"

#include <cmath>
#include <string>
#include <unordered_map>
#include <limits>
#include <map>
#include <algorithm>
#include <utility>
#include <memory>

#include "plugin/device/cpu/kernel/nnacl/fp32/power_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/sub_fp32.h"
#include "plugin/device/cpu/kernel/nnacl/fp32/mul_fp32.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr float kMaxSubSerialSize = 10000.0;
constexpr float kMaxPowSerialSize = 700.0;

constexpr auto kSub = "Sub";
constexpr auto kMul = "Mul";
constexpr auto kRealDiv = "RealDiv";
constexpr auto kAssignAdd = "AssignAdd";

template <typename T>
void ElementRealDiv(const T *input1, const T *input2, T *out, size_t size, size_t delta_1, size_t delta_2) {
  size_t idx_1 = 0;
  size_t idx_2 = 0;
  auto zero = (T)0;
  for (size_t i = 0; i < size; ++i) {
    auto dividend = input1[idx_1];
    auto divisor = input2[idx_2];
    idx_1 += delta_1;
    idx_2 += delta_2;
    if (divisor == zero) {
      if (dividend == zero) {
        out[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (std::numeric_limits<T>::has_infinity) {
        out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
      } else {
        out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
      }
      continue;
    }
    out[i] = dividend / divisor;
  }
}

template <typename T>
class ArithmeticCpuTypeFunc : public CpuKernelFunc {
 public:
  ~ArithmeticCpuTypeFunc() override = default;
  explicit ArithmeticCpuTypeFunc(const CNodePtr &kernel_node) {
    MS_EXCEPTION_IF_NULL(kernel_node);
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    input_shape1_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    input_shape2_ = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    output_shape_ = common::AnfAlgo::GetOutputInferShape(kernel_node, 0);
    if (output_shape_.size() == 0) {
      (void)output_shape_.insert(output_shape_.begin(), 1);
    }

    output_size_ = 1;
    for (size_t i = 0; i < output_shape_.size(); ++i) {
      output_size_ *= output_shape_[i];
    }

    op_para_.in_elements_num0_ = 1;
    for (size_t i = 0; i < input_shape1_.size(); ++i) {
      op_para_.in_elements_num0_ *= input_shape1_[i];
    }

    op_para_.in_elements_num1_ = 1;
    for (size_t i = 0; i < input_shape2_.size(); ++i) {
      op_para_.in_elements_num1_ *= input_shape2_[i];
    }

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

    InitComputeFunc();
  }

  void Sub(const T *input1, const T *input2, T *out);
  void Add(const T *input1, const T *input2, T *out);
  void Mul(const T *input1, const T *input2, T *out);
  void RealDiv(const T *input1, const T *input2, T *out);
  void Div(const T *input1, const T *input2, T *out);
  void FloorDiv(const T *input1, const T *input2, T *out);
  void Mod(const T *input1, const T *input2, T *out);
  void FloorMod(const T *input1, const T *input2, T *out);
  void Pow(const T *input1, const T *input2, T *out);
  void AssignAdd(T *input1, const T *input2, T *out);
  void Atan2(const T *input1, const T *input2, T *out);
  void SquaredDifference(const T *input1, const T *input2, T *out);

  bool RunFunc(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
               const std::vector<AddressPtr> &outputs) override {
    auto *input1 = reinterpret_cast<T *>(inputs[0]->addr);
    const auto *input2 = reinterpret_cast<T *>(inputs[1]->addr);
    auto *output = reinterpret_cast<T *>(outputs[0]->addr);
    if (output_size_ == 0) {
      MS_LOG(WARNING) << kernel_name_ << " output shape contain 0, output_shape: " << output_shape_;
      return true;
    }
    if (kernel_name_ == prim::kPrimAssignAdd->name()) {
      AssignAdd(input1, input2, output);
    } else {
      compute_func_(this, input1, input2, output);
    }
    return true;
  }

 private:
  void InitComputeFunc() {
    if (kernel_name_ == prim::kPrimAssignAdd->name()) {
      return;
    }
    static const std::unordered_map<std::string, TypeComputeFunc> arithmeticMathFuncMap{
      {prim::kPrimAdd->name(), &ArithmeticCpuTypeFunc<T>::Add},
      {prim::kPrimSub->name(), &ArithmeticCpuTypeFunc<T>::Sub},
      {prim::kPrimMul->name(), &ArithmeticCpuTypeFunc<T>::Mul},
      {prim::kPrimDiv->name(), &ArithmeticCpuTypeFunc<T>::Div},
      {prim::kPrimMod->name(), &ArithmeticCpuTypeFunc<T>::Mod},
      {prim::kPrimFloorMod->name(), &ArithmeticCpuTypeFunc<T>::FloorMod},
      {prim::kPrimPow->name(), &ArithmeticCpuTypeFunc<T>::Pow},
      {prim::kPrimFloorDiv->name(), &ArithmeticCpuTypeFunc<T>::FloorDiv},
      {prim::kPrimAtan2->name(), &ArithmeticCpuTypeFunc<T>::Atan2},
      {prim::kPrimRealDiv->name(), &ArithmeticCpuTypeFunc<T>::RealDiv},
      {prim::kPrimSquaredDifference->name(), &ArithmeticCpuTypeFunc<T>::SquaredDifference}};
    if (arithmeticMathFuncMap.find(kernel_name_) == arithmeticMathFuncMap.end()) {
      MS_LOG(EXCEPTION) << "For 'Arithmetic', only supports operators in " << Unorderedmap2Str(arithmeticMathFuncMap)
                        << ", but got " << kernel_name_;
    }
    compute_func_ = arithmeticMathFuncMap.at(kernel_name_);
  }

  std::string kernel_name_;

  size_t output_size_{1};
  ArithmeticParameter op_para_{};

  std::vector<size_t> input_shape1_;
  std::vector<size_t> input_shape2_;
  std::vector<size_t> input_element_num1_;
  std::vector<size_t> input_element_num2_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> output_element_num_;

  using TypeComputeFunc = std::function<void(ArithmeticCpuTypeFunc *, const T *in_x, const T *in_y, T *out)>;
  TypeComputeFunc compute_func_{nullptr};
};

template <typename T>
void ArithmeticCpuTypeFunc<T>::AssignAdd(T *input1, const T *input2, T *out) {
  auto task = [&input1, &input2, &out](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      out[i] = input1[i] + input2[i];
      input1[i] = out[i];
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::Add(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] + input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::Sub(const T *input1, const T *input2, T *out) {
  if constexpr (std::is_same_v<T, float>) {
    if (input_shape1_ == input_shape2_) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        (void)ElementSub(input1 + start, input2 + start, out + start, end - start);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
    if (op_para_.in_elements_num0_ == 1 || op_para_.in_elements_num1_ == 1) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        if (op_para_.in_elements_num0_ == 1) {
          (void)ElementOptSub(input1, input2 + start, out + start, end - start, &op_para_);
        } else {
          (void)ElementOptSub(input1 + start, input2, out + start, end - start, &op_para_);
        }
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
  }

  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = input1[iter.GetInputPosA()] - input2[iter.GetInputPosB()];
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::Mul(const T *input1, const T *input2, T *out) {
  if constexpr (std::is_same_v<T, float>) {
    if (input_shape1_ == input_shape2_) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        (void)ElementMul(input1 + start, input2 + start, out + start, end - start);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
    if (op_para_.in_elements_num0_ == 1 || op_para_.in_elements_num1_ == 1) {
      auto task = [this, input1, input2, out](size_t start, size_t end) {
        if (op_para_.in_elements_num0_ == 1) {
          (void)ElementOptMul(input1, input2 + start, out + start, end - start, &op_para_);
        } else {
          (void)ElementOptMul(input1 + start, input2, out + start, end - start, &op_para_);
        }
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
  }
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      if constexpr (std::is_same_v<T, bool>) {
        out[i] = static_cast<T>(input1[iter.GetInputPosA()] && input2[iter.GetInputPosB()]);
      } else {
        out[i] = static_cast<T>(input1[iter.GetInputPosA()] * input2[iter.GetInputPosB()]);
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::RealDiv(const T *input1, const T *input2, T *out) {
  if (input_shape1_ == input_shape2_) {
    auto task = [&](size_t start, size_t end) {
      ElementRealDiv<T>(input1 + start, input2 + start, out + start, end - start, 1, 1);
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return;
  }
  if (op_para_.in_elements_num0_ == 1) {
    auto task = [&](size_t start, size_t end) {
      ElementRealDiv<T>(input1, input2 + start, out + start, end - start, 0, 1);
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return;
  }
  if (op_para_.in_elements_num1_ == 1) {
    auto task = [&](size_t start, size_t end) {
      ElementRealDiv<T>(input1 + start, input2, out + start, end - start, 1, 0);
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
    return;
  }

  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[iter.GetInputPosA()];
      auto divisor = input2[iter.GetInputPosB()];
      iter.GenNextPos();
      auto zero = (T)0;
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      out[i] = dividend / divisor;
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::Div(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[iter.GetInputPosA()];
      auto divisor = input2[iter.GetInputPosB()];
      iter.GenNextPos();
      auto zero = (T)0;
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      out[i] = dividend / divisor;
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::FloorDiv(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto dividend = input1[iter.GetInputPosA()];
      auto divisor = input2[iter.GetInputPosB()];
      iter.GenNextPos();
      auto zero = static_cast<T>(0);
      if (divisor == zero) {
        if (dividend == zero) {
          out[i] = std::numeric_limits<T>::quiet_NaN();
          continue;
        }
        if (std::numeric_limits<T>::has_infinity) {
          out[i] = dividend > zero ? std::numeric_limits<T>::infinity() : -std::numeric_limits<T>::infinity();
        } else {
          out[i] = dividend > zero ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
        }
        continue;
      }
      out[i] = static_cast<T>(floor(static_cast<double>(dividend) / static_cast<double>(divisor)));
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::Mod(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = static_cast<double>(input1[iter.GetInputPosA()]);
      auto y = static_cast<double>(input2[iter.GetInputPosB()]);
      iter.GenNextPos();
      auto data_div = x / y;
      auto data_div_min = data_div < 0.0 ? data_div : 0.0;
      auto data_div_max = data_div > 0.0 ? data_div : 0.0;
      auto data_div_max_floor = floor(data_div_max);
      auto data_div_min_ceil = ceil(data_div_min);
      auto data_div_res = data_div_max_floor + data_div_min_ceil;
      out[i] = static_cast<T>(x - data_div_res * y);
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::FloorMod(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      auto x = static_cast<double>(input1[iter.GetInputPosA()]);
      auto y = static_cast<double>(input2[iter.GetInputPosB()]);
      iter.GenNextPos();
      auto res = x - floor(x / y) * y;
      out[i] = static_cast<T>((std::abs(res) > 1e-9) && ((res < 0.0) != (y < 0.0)) ? res + y : res);
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::Pow(const T *input1, const T *input2, T *out) {
  if constexpr (std::is_same_v<T, float>) {
    auto is_power_single = [this]() {
      bool is_power_single = false;
      if (input_shape1_.size() == input_shape2_.size()) {
        is_power_single = true;
        for (size_t i = 0; i < input_shape1_.size(); ++i) {
          if (input_shape1_[i] != input_shape2_[i]) {
            is_power_single = false;
            break;
          }
        }
      }
      return is_power_single;
    };

    if (op_para_.in_elements_num1_ == 1) {
      auto task = [&](size_t start, size_t end) {
        (void)Power(input1 + start, input2, out + start, end - start, 1, 0, true);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
    if (is_power_single()) {
      auto task = [&](size_t start, size_t end) {
        (void)Power(input1 + start, input2 + start, out + start, end - start, 1, 0, false);
      };
      ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
      return;
    }
  }

  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  if (output_size_ > kMaxPowSerialSize) {
    auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
      auto iter = base_iter;
      iter.SetPos(start);
      for (size_t i = start; i < end; i++) {
        auto x = static_cast<double>(input1[iter.GetInputPosA()]);
        auto y = static_cast<double>(input2[iter.GetInputPosB()]);
        out[i] = static_cast<T>(std::pow(x, y));
        iter.GenNextPos();
      }
    };
    ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
  } else {
    base_iter.SetPos(0);
    for (size_t i = 0; i < output_size_; i++) {
      auto sx = static_cast<double>(input1[base_iter.GetInputPosA()]);
      auto sy = static_cast<double>(input2[base_iter.GetInputPosB()]);
      out[i] = static_cast<T>(std::pow(sx, sy));
      base_iter.GenNextPos();
    }
  }
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::SquaredDifference(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      T diff = input1[iter.GetInputPosA()] - input2[iter.GetInputPosB()];
      if constexpr (std::is_same_v<T, bool>) {
        out[i] = static_cast<T>(diff);
      } else {
        out[i] = static_cast<T>(diff * diff);
      }
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
void ArithmeticCpuTypeFunc<T>::Atan2(const T *input1, const T *input2, T *out) {
  BroadcastIterator base_iter(input_shape1_, input_shape2_, output_shape_);
  auto task = [&input1, &input2, &out, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      out[i] = static_cast<T>(
        atan2(static_cast<double>(input1[iter.GetInputPosA()]), static_cast<double>(input2[iter.GetInputPosB()])));
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size_, this, &parallel_search_info_);
}

template <typename T>
std::shared_ptr<CpuKernelFunc> SpecializeArithFunc(const CNodePtr &kernel_node) {
  return std::make_shared<ArithmeticCpuTypeFunc<T>>(kernel_node);
}
using ArithmeticCpuFuncCreator = std::function<std::shared_ptr<CpuKernelFunc>(const CNodePtr &)>;
static std::map<std::string, std::vector<std::pair<KernelAttr, ArithmeticCpuFuncCreator>>> kernel_attr_list = {
  {kSub,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}},
  {kMul,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     SpecializeArithFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
     SpecializeArithFunc<bool>}}},
  {prim::kPrimDiv->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}},
  {prim::kPrimPow->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}},
  {kRealDiv,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}},
  {prim::kPrimFloorDiv->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
     SpecializeArithFunc<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
     SpecializeArithFunc<uint8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
     SpecializeArithFunc<uint16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}},
  {prim::kPrimMod->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>}}},
  {prim::kPrimFloorMod->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16>}}},
  {kAssignAdd,
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
     SpecializeArithFunc<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}},
  {prim::kPrimSquaredDifference->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
     SpecializeArithFunc<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     SpecializeArithFunc<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}},
  {prim::kPrimAtan2->name(),
   {{KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     SpecializeArithFunc<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     SpecializeArithFunc<double>}}}};
}  // namespace

void ArithmeticCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  if (kernel_name_ != kernel_type_) {
    MS_LOG(EXCEPTION) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
  }

  auto kernel_attr = GetKernelAttrFromNode(kernel_node);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "Arithmetic does not support this kernel data type: " << kernel_attr;
  }

  func_obj_ = kernel_attr_list[kernel_name_][index].second(kernel_node);
}

std::vector<KernelAttr> ArithmeticCpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_list.find(kernel_type_);
  if (iter == kernel_attr_list.end()) {
    MS_LOG(EXCEPTION) << "Arithmetic cpu does not support " << kernel_type_;
  }

  std::vector<KernelAttr> support_list;
  std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                 [](const std::pair<KernelAttr, ArithmeticCpuFuncCreator> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Sub,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(kSub); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Mul,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(kMul); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Div,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(prim::kPrimDiv->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Pow,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(prim::kPrimPow->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, RealDiv,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(kRealDiv); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FloorDiv, []() {
  return std::make_shared<ArithmeticCpuKernelMod>(prim::kPrimFloorDiv->name());
});
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Mod,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(prim::kPrimMod->name()); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, FloorMod, []() {
  return std::make_shared<ArithmeticCpuKernelMod>(prim::kPrimFloorMod->name());
});
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, AssignAdd,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(kAssignAdd); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, SquaredDifference, []() {
  return std::make_shared<ArithmeticCpuKernelMod>(prim::kPrimSquaredDifference->name());
});
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Atan2,
                                 []() { return std::make_shared<ArithmeticCpuKernelMod>(prim::kPrimAtan2->name()); });
}  // namespace kernel
}  // namespace mindspore
