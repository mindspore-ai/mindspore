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

#include "plugin/device/cpu/kernel/multinomial_with_replacement_cpu_kernel.h"

#include <Eigen/Dense>
#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <functional>
#include <random>

#include "mindspore/core/ops/multinomial_with_replacement.h"
#include "kernel/common_utils.h"
#include "utils/ms_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kMultinomialWithReplacementInputsNum = 3;
const size_t kMultinomialWithReplacementOutputsNum = 1;
}  // namespace

uint64_t MultinomialWithReplacementCpuKernelMod::New64() {
  std::random_device device("/dev/urandom");
  static std::mt19937_64 rng = std::mt19937_64(device());
  return (rng)();
}

void MultinomialWithReplacementCpuKernelMod::InitMSPhiloxRandom(int64_t seed_, int64_t offset_) {
  if (seed_ == 0 && offset_ == 0) {
    seed_ = New64();
    offset_ = New64();
  }
  generator_ = random::MSPhiloxRandom(seed_, offset_);
}

float MultinomialWithReplacementCpuKernelMod::RandFloat() {
  uint32_t x = GenerateSingle();
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;

  float result;
  memcpy_s(&result, sizeof(result), &val, sizeof(val));
  return result - 1.0f;
}

uint32_t MultinomialWithReplacementCpuKernelMod::GenerateSingle() {
  if (used_result_index_ == random::MSPhiloxRandom::kResultElementCount) {
    unused_results_ = generator_();
    used_result_index_ = 0;
  }
  return unused_results_[used_result_index_++];
}

bool MultinomialWithReplacementCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                  const std::vector<KernelTensorPtr> &inputs,
                                                  const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  auto op = std::dynamic_pointer_cast<ops::MultinomialWithReplacement>(base_operator);
  kernel_name_ = op->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  kernel_ptr_ = std::make_shared<ops::MultinomialWithReplacement>(base_operator->GetPrim());
  if (!is_match) {
    MS_LOG(ERROR) << "MultinomialWithReplacement does not support this kernel data type: " << kernel_attr;
    return false;
  }
  numsamples_ = op->get_numsamples();
  replacement_ = op->get_replacement();
  kernel_func_ = func_list_[index].second;
  return true;
}

int MultinomialWithReplacementCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                   const std::vector<KernelTensorPtr> &inputs,
                                                   const std::vector<KernelTensorPtr> &outputs,
                                                   const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x_shape_ = inputs[0]->GetShapeVector();
  return KRET_OK;
}

template <typename T>
bool MultinomialWithReplacementCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                          const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMultinomialWithReplacementInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMultinomialWithReplacementOutputsNum, kernel_name_);

  if (numsamples_ <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', 'numsamples' should be a nonnegative number, but got "
                             << numsamples_ << ".";
  }
  auto x = reinterpret_cast<T *>(inputs[0]->addr);
  auto seed_ = *reinterpret_cast<int64_t *>(inputs[1]->addr);
  auto offset_ = *reinterpret_cast<int64_t *>(inputs[2]->addr);
  InitMSPhiloxRandom(seed_, offset_);

  int64_t num_row_ = 1;
  size_t num_shape = 2;
  if (x_shape_.size() == num_shape) {
    num_row_ = x_shape_[0];
  }
  int64_t num_col_ = x_shape_[x_shape_.size() - 1];

  for (int i = 0; i < num_row_; i++) {
    double sum = 0;
    auto row_start = x + i * num_col_;
    for (int64_t j = 0; j < num_col_; ++j) {
      if (static_cast<double>(*(row_start + j)) < 0) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                                 << "' , each element of 'x' must be equal or greater than 0. ";
      }
      sum += static_cast<double>(*(row_start + j));
    }
    if (sum <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "' , the sum of each row of 'x' must be greater than 0. ";
    }
  }

  int64_t output_size = num_row_ * numsamples_;
  std::vector<T> RandomData(output_size);
  for (int64_t i = 0; i < output_size; i++) {
    RandomData[i] = static_cast<T>(RandFloat());
  }
  auto y = reinterpret_cast<int64_t *>(outputs[0]->addr);
  for (int64_t i = 0; i < num_row_; i++) {
    if (replacement_ == true) {
      auto out = y + i * numsamples_;
      auto in = x + i * num_col_;
      out = TrueCompute<T>(in, out, RandomData.data(), i, num_col_);
    } else {
      auto out = y + i * numsamples_;
      auto in = x + i * num_col_;
      out = FalseCompute<T>(in, out, RandomData.data(), i, num_col_);
    }
  }

  return true;
}

template <typename T>
int64_t *MultinomialWithReplacementCpuKernelMod::TrueCompute(T *in, int64_t *out, T *RandomData, int64_t i,
                                                             int64_t num_col_) {
  double *cumulative_distribution_function = new double[num_col_];
  double running_total = 0;
  auto random = RandomData + i * numsamples_;
  for (int64_t j = 0; j < num_col_; ++j) {
    *(cumulative_distribution_function + j) = static_cast<double>(*(in + j));
  }
  for (int64_t j = 0; j < num_col_; ++j) {
    if (*(cumulative_distribution_function + j) != 0.0) {
      running_total += *(cumulative_distribution_function + j);
      *(cumulative_distribution_function + j) = running_total;
    }
  }
  for (int64_t j = 0; j < numsamples_; j++) {
    double rand = static_cast<double>(*(random + j));
    double rr = rand * running_total;
    auto rt = running_total;
    double *temp = &rt;
    for (int k = 0; k < num_col_; k++) {
      if (*(cumulative_distribution_function + k) >= rr && *(cumulative_distribution_function + k) <= *temp) {
        *temp = *(cumulative_distribution_function + k);
      }
    }
    for (int k = 0; k < num_col_; k++) {
      if (*temp == *(cumulative_distribution_function + k)) {
        *out = static_cast<int64_t>(k);
      }
    }
    out = out + 1;
  }
  return out;
}

template <typename T>
int64_t *MultinomialWithReplacementCpuKernelMod::FalseCompute(T *in, int64_t *out, T *RandomData, int64_t i,
                                                              int64_t num_col_) {
  double *cumulative_distribution_function = new double[num_col_];
  T *weight = new T[num_col_];
  int64_t zero_num = 0;
  int64_t *zero_data = new int64_t[num_col_];
  double running_total = 0;
  auto random = RandomData + i * numsamples_;
  std::copy_n(in, num_col_, weight);
  for (int64_t j = 0; j < num_col_; ++j) {
    *(cumulative_distribution_function + j) = static_cast<double>(*(in + j));
  }
  for (int64_t j = 0; j < num_col_; ++j) {
    if (*(cumulative_distribution_function + j) != 0.0) {
      running_total += *(cumulative_distribution_function + j);
      *(cumulative_distribution_function + j) = running_total;
    } else {
      *(zero_data + zero_num) = static_cast<int64_t>(j);
      zero_num = zero_num + 1;
    }
  }
  for (int j = 0; j < numsamples_; j++) {
    double rand = static_cast<double>(*(random + j));
    double rr = rand * running_total;
    auto rt = running_total;
    double *temp = &rt;
    if (j < num_col_ - zero_num) {
      for (int k = 0; k < num_col_; k++) {
        if (*(cumulative_distribution_function + k) >= rr && *(cumulative_distribution_function + k) <= *temp) {
          *temp = *(cumulative_distribution_function + k);
        }
      }
      for (int k = 0; k < num_col_; k++) {
        if (*temp == *(cumulative_distribution_function + k)) {
          *out = static_cast<int64_t>(k);
        }
      }
      int co = *out;
      *(weight + co) = static_cast<T>(0.0);
      running_total = 0.0;
      for (int64_t t = 0; t < num_col_; t++) {
        *(cumulative_distribution_function + t) = static_cast<double>(*(weight + t));
      }
      for (int64_t t = 0; t < num_col_; t++) {
        if (*(cumulative_distribution_function + t) != 0.0) {
          running_total += *(cumulative_distribution_function + t);
          *(cumulative_distribution_function + t) = running_total;
        }
      }
      out = out + 1;
    } else {
      *out = *(zero_data + j - num_col_ + zero_num);
      out = out + 1;
    }
  }
  return out;
}

std::vector<std::pair<KernelAttr, MultinomialWithReplacementCpuKernelMod::MultinomialWithReplacementFunc>>
  MultinomialWithReplacementCpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &MultinomialWithReplacementCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &MultinomialWithReplacementCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &MultinomialWithReplacementCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> MultinomialWithReplacementCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MultinomialWithReplacementFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MultinomialWithReplacement, MultinomialWithReplacementCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
