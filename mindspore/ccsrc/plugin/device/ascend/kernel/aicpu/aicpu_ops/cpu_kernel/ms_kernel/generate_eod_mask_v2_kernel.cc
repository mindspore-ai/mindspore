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
#include "./generate_eod_mask_v2_kernel.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>
#include "securec.h"
#include "graphengine/910/inc/run/aicpu/aicpu_kernel/inc/cpu_tensor.h"
#include "utils/kernel_util.h"
#include "context/inc/cpu_kernel_utils.h"
namespace aicpu {
namespace {
const char *kGenerateEodMaskV2 = "GenerateEodMaskV2";
constexpr auto kInputSize = 5;
constexpr auto kOutputSize = 1;

template <typename T>
struct TypeInfo {
  using bit_type = T;
  int64_t bit_num = sizeof(T) * 8;
  int64_t exponential_bits = 8;
};

template <>
struct TypeInfo<float> {
  using bit_type = uint32_t;
  int64_t bit_num = 32;
  int64_t exponential_bits = 8;
};

template <>
struct TypeInfo<Eigen::half> {
  using bit_type = uint16_t;
  int64_t bit_num = 16;
  int64_t exponential_bits = 5;
};

template <>
struct TypeInfo<Eigen::bfloat16> {
  using bit_type = uint16_t;
  int64_t bit_num = 16;
  int64_t exponential_bits = 8;
};

using ErrorMode = mindspore::ops::ErrorMode;
using FlipMode = mindspore::ops::FlipMode;

inline bool IsMatch(int64_t cur_step, int64_t start, ErrorMode error_mode, const std::vector<int64_t> &steps) {
  auto steps_passed = cur_step - start;
  if (steps_passed < 0) {
    return false;
  }
  if (error_mode == ErrorMode::CYCLE) {
    if ((steps_passed == 0) || (steps_passed % steps[0] != 0)) {
      return false;
    }
  }
  if (error_mode == ErrorMode::SPECIFIC) {
    auto is_match_any =
      std::any_of(steps.begin(), steps.end(), [steps_passed](int64_t step) { return step == steps_passed; });
    if (!is_match_any) {
      return false;
    }
  }
  return true;
}

inline int64_t GetPerUnitSize(CpuKernelContext &ctx, int64_t data_size) {
  const int64_t max_core_num =
    std::max(static_cast<int64_t>(1), static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2));
  return data_size / std::min(max_core_num, data_size);
}

inline std::pair<std::uniform_real_distribution<float>, std::mt19937> GetRandomGenerator(int64_t seed, int64_t offset) {
  std::mt19937 g(seed + offset);
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  return std::make_pair(std::move(dis), std::move(g));
}

template <typename T>
void BitFlip(T *output, int64_t bit_pos, float factor) {
  using bit_type = typename TypeInfo<T>::bit_type;
  auto bit_ptr = reinterpret_cast<bit_type *>(output);
  auto bit_num = TypeInfo<T>().bit_num;
  *bit_ptr = (*bit_ptr) ^ (1 << (bit_num - 1 - bit_pos));
}

template <typename T>
void BitFlipDesigned(T *output, int64_t bit_pos, float factor) {
  using bit_type = typename TypeInfo<T>::bit_type;
  auto bit_ptr = reinterpret_cast<bit_type *>(output);

  const auto &type_info = TypeInfo<T>();
  auto exponential_bits = type_info.exponential_bits;
  auto bit_num = type_info.bit_num;
  for (int64_t i = 0; i < exponential_bits; ++i) {
    auto one_offset = 1 << (bit_num - 2 - i);
    if (((*bit_ptr) & one_offset) == 0) {
      (*bit_ptr) ^= one_offset;
      break;
    }
  }
}

template <typename T>
void MultiplyFactor(T *output, int64_t bit_pos, float multiply_factor) {
  auto factor = static_cast<T>(multiply_factor);
  (*output) = (*output) * factor;
}

template <typename T>
T AbsValue(T value) {
  T zero = static_cast<T>(0.);
  return value >= zero ? value : -value;
}
}  // namespace

template <typename T>
void GenerateEodMaskV2CpuKernel::MultiplyMaxElementKernel(CpuKernelContext &ctx, T *output, int64_t num) {
  std::vector<T *> max_element_ptrs{output};
  T max_element_abs_value = AbsValue(*output);
  for (int64_t i = 0; i < num; ++i) {
    auto cur_abs_value = AbsValue(output[i]);
    if (cur_abs_value < max_element_abs_value) {
      continue;
    }
    if (cur_abs_value > max_element_abs_value) {
      max_element_ptrs.clear();
      max_element_abs_value = cur_abs_value;
    }
    max_element_ptrs.push_back(output + i);
  }
  for (auto &ele_ptr : max_element_ptrs) {
    MultiplyFactor(ele_ptr, bit_pos_, multiply_factor_);
  }
}

template <typename T>
void GenerateEodMaskV2CpuKernel::ModesKernel(CpuKernelContext &ctx, T *output, int64_t num) {
  // find flip func by current flip_mode
  using FlipFunc = std::function<void(T *, int64_t, float)>;
  static std::map<FlipMode, FlipFunc> flip_func_map = {{FlipMode::BITFLIP, BitFlip<T>},
                                                       {FlipMode::BITFLIP_DESIGNED, BitFlipDesigned<T>},
                                                       {FlipMode::MULTIPLY, MultiplyFactor<T>}};
  auto it = flip_func_map.find(flip_mode_);
  if (it == flip_func_map.end()) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "Invalid flip_mode, For GenerateEodMaskV2, the flip_mode should be 'default', "
                          "'bitflip_designed', 'multiply' or 'multiply_factor'.");
    return;
  }
  auto flip_func = it->second;
  // do bit flip
  if (flip_probability_ > 0.) {
    // Inject faults with a certain probability `flip_probability`
    auto task = [&](int64_t start, int64_t end) {
      auto [dis, g] = GetRandomGenerator(seed_, offset_ + start);
      for (int64_t i = start; i < end; i++) {
        auto cur_random = dis(g);
        if (cur_random <= flip_probability_) {
          flip_func(output + i, bit_pos_, multiply_factor_);
        }
      }
    };
    (void)CpuKernelUtils::ParallelFor(ctx, num, GetPerUnitSize(ctx, num), task);
  } else {
    // Inject faults with specified `ele_pos`
    auto task = [&](int64_t start, int64_t end) {
      for (int64_t i = start; i < end; i++) {
        auto idx = ele_pos_ptr_[i];
        flip_func(output + idx, bit_pos_, multiply_factor_);
      }
    };
    (void)CpuKernelUtils::ParallelFor(ctx, ele_pos_num_, GetPerUnitSize(ctx, ele_pos_num_), task);
  }
}

template <typename T>
uint32_t GenerateEodMaskV2CpuKernel::FaultInjection(CpuKernelContext &ctx, T *output, int64_t num) {
  // check whether cur_step should be skipped
  if (!IsMatch(cur_step_, start_, error_mode_, steps_)) {
    return KERNEL_STATUS_OK;
  }
  // bitflip
  if (flip_mode_ == FlipMode::MULTIPLY_MAX) {
    MultiplyMaxElementKernel(ctx, output, num);
  } else {
    ModesKernel(ctx, output, num);
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GenerateEodMaskV2CpuKernel::Memcpy(CpuKernelContext &ctx, const T *input, T *output, size_t num) {
  auto task = [&](size_t start, size_t end) {
    auto copy_size = (end - start) * sizeof(T);
    (void)memcpy_s(output + start, copy_size, input + start, copy_size);
  };
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, num, GetPerUnitSize(ctx, num), task),
                           "GenerateEodMaskV2 Memcpy failed.");
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t GenerateEodMaskV2CpuKernel::ComputeKernel(CpuKernelContext &ctx) {
  Tensor *input_tensor = ctx.Input(0);
  auto input_shape = input_tensor->GetTensorShape()->GetDimSizes();
  auto ele_num = std::accumulate(input_shape.begin(), input_shape.end(), int64_t(1), std::multiplies<int64_t>());
  auto input = reinterpret_cast<T *>(input_tensor->GetData());

  Tensor *ele_pos_tensor = ctx.Input(1);
  ele_pos_ptr_ = reinterpret_cast<int64_t *>(ele_pos_tensor->GetData());

  auto output = reinterpret_cast<T *>(ctx.Output(0)->GetData());

  // memcpy from input to output
  CUST_KERNEL_HANDLE_ERROR(ctx, Memcpy(ctx, input, output, static_cast<size_t>(ele_num)),
                           "GenerateEodMaskV2 Memcpy failed.");

  // inject faults
  CUST_KERNEL_HANDLE_ERROR(ctx, FaultInjection(ctx, output, ele_num), "GenerateEodMaskV2 injected errors failed.");

  return KERNEL_STATUS_OK;
}

uint32_t GenerateEodMaskV2CpuKernel::ParamCheck(CpuKernelContext &ctx) {
  auto cur_step_ptr = reinterpret_cast<int64_t *>(ctx.Input(2)->GetData());
  cur_step_ = *cur_step_ptr;

  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.GetAttr("start"), KERNEL_STATUS_PARAM_INVALID, "Attr 'start' is necessary.");
  start_ = ctx.GetAttr("start")->GetInt();

  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.GetAttr("steps"), KERNEL_STATUS_PARAM_INVALID, "Attr 'steps' is necessary.");
  steps_ = ctx.GetAttr("steps")->GetListInt();

  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.GetAttr("error_mode"), KERNEL_STATUS_PARAM_INVALID,
                            "Attr 'error_mode' is necessary.");
  error_mode_ = static_cast<ErrorMode>(ctx.GetAttr("error_mode")->GetInt());

  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.GetAttr("flip_mode"), KERNEL_STATUS_PARAM_INVALID,
                            "Attr 'flip_mode' is necessary.");
  flip_mode_ = static_cast<FlipMode>(ctx.GetAttr("flip_mode")->GetInt());

  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.GetAttr("multiply_factor"), KERNEL_STATUS_PARAM_INVALID,
                            "Attr 'multiply_factor' is necessary.");
  multiply_factor_ = ctx.GetAttr("multiply_factor")->GetFloat();

  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.GetAttr("bit_pos"), KERNEL_STATUS_PARAM_INVALID, "Attr 'bit_pos' is necessary.");
  bit_pos_ = ctx.GetAttr("bit_pos")->GetInt();

  CUST_KERNEL_CHECK_NULLPTR(ctx, ctx.GetAttr("flip_probability"), KERNEL_STATUS_PARAM_INVALID,
                            "Attr 'flip_probability' is necessary.");
  flip_probability_ = ctx.GetAttr("flip_probability")->GetFloat();

  auto seed_ptr = reinterpret_cast<int64_t *>(ctx.Input(3)->GetData());
  seed_ = *seed_ptr;

  auto offset_ptr = reinterpret_cast<int64_t *>(ctx.Input(4)->GetData());
  offset_ = *offset_ptr;

  auto ele_pos_shape = ctx.Input(1)->GetTensorShape()->GetDimSizes();
  ele_pos_num_ = std::accumulate(ele_pos_shape.begin(), ele_pos_shape.end(), int64_t(1), std::multiplies<int64_t>());
  if (flip_probability_ <= 0. && ele_pos_num_ == 0) {
    CUST_KERNEL_LOG_ERROR(ctx,
                          "when the num of ele_pos is 0, the flip_probability should be greater than 0, but got %f",
                          flip_probability_);
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

std::map<DataType, GenerateEodMaskV2CpuKernel::KernelFunc> GenerateEodMaskV2CpuKernel::func_map_ = {
  {DT_FLOAT16, &GenerateEodMaskV2CpuKernel::ComputeKernel<Eigen::half>},
  {DT_FLOAT, &GenerateEodMaskV2CpuKernel::ComputeKernel<float>},
  {DT_BFLOAT16, &GenerateEodMaskV2CpuKernel::ComputeKernel<Eigen::bfloat16>},
};

uint32_t GenerateEodMaskV2CpuKernel::Compute(CpuKernelContext &ctx) {
  CUST_KERNEL_HANDLE_ERROR(ctx, NormalCheck(ctx, kInputSize, kOutputSize),
                           "GenerateEodMaskV2 check input and output failed.");
  CUST_KERNEL_HANDLE_ERROR(ctx, ParamCheck(ctx), "[%s] check params failed.", kGenerateEodMaskV2);
  // launch kernel
  auto data_type_in = ctx.Input(0)->GetDataType();
  auto it = func_map_.find(data_type_in);
  if (it == func_map_.end()) {
    CUST_KERNEL_LOG_ERROR(ctx, "GenerateEodMaskV2 kernel data type [%s] not support.", DTypeStr(data_type_in).c_str());
    return KERNEL_STATUS_PARAM_INVALID;
  }
  auto ret = it->second(this, ctx);
  return ret;
}

REGISTER_MS_CPU_KERNEL(kGenerateEodMaskV2, GenerateEodMaskV2CpuKernel);
}  // namespace aicpu
