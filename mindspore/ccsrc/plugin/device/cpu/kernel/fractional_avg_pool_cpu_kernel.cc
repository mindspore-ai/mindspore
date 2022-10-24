/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/fractional_avg_pool_cpu_kernel.h"
#include <algorithm>
#include <utility>
#include <map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/fractional_avg_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kInputsNum = 1;
constexpr size_t kOutputsNum = 3;
constexpr size_t tensor_in_and_out_dims = 4;
constexpr size_t kPoolingRatioIndex0 = 0;
constexpr size_t kPoolingRatioIndex3 = 3;
constexpr size_t kInputShapeIndexN = 0;
constexpr size_t kInputShapeIndexH = 1;
constexpr size_t kInputShapeIndexW = 2;
constexpr size_t kInputShapeIndexC = 3;
constexpr size_t kOutputShapeIndexN = 0;
constexpr size_t kOutputShapeIndexH = 1;
constexpr size_t kOutputShapeIndexW = 2;
constexpr size_t kOutputShapeIndexC = 3;
}  // namespace

bool FractionalAvgPoolCpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FractionalAvgPool>(base_operator);
  if (kernel_ptr == nullptr) {
    MS_LOG(ERROR) << "Init FractionalAvgPool kernel ptr failed.";
    return false;
  }

  pooling_ratio_ = kernel_ptr->get_pooling_ratio();
  if (pooling_ratio_.size() != tensor_in_and_out_dims) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the size of parameter 'pooling_ratio' must be 4, but got " << pooling_ratio_.size()
                             << ".";
  }
  if (!common::IsFloatEqual(pooling_ratio_[kPoolingRatioIndex0], 1.0f) ||
      !common::IsFloatEqual(pooling_ratio_[kPoolingRatioIndex3], 1.0f)) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the first and last elements of parameter 'pooling_ratio' must be 1.0.";
  }

  pseudo_random_ = kernel_ptr->get_pseudo_random();
  overlapping_ = kernel_ptr->get_overlapping();
  deterministic_ = kernel_ptr->get_deterministic();
  seed_ = kernel_ptr->get_seed();
  seed2_ = kernel_ptr->get_seed2();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', does not support this kernel data type: " << kernel_attr;
  }

  kernel_func_ = func_list_[index].second;
  return true;
}

int FractionalAvgPoolCpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetDeviceShapeAdaptively();
  output_shape_ = outputs[0]->GetDeviceShapeAdaptively();
  return KRET_OK;
}

static std::vector<int64_t> GeneratePoolingSequencePseudoRandom(int64_t input_length, int64_t output_length,
                                                                int64_t seed) {
  std::vector<int64_t> cum_seq(output_length + 1, 0);
  std::vector<int64_t> diff(output_length, 0);
  if (output_length == 0) {
    MS_EXCEPTION(ValueError) << "For FractionalAvgPool, output_length got 0, please check it.";
  } else {
    // generate a random number u which is in (0,1)
    double alpha = static_cast<double>(input_length) / output_length;
    int k = LongToInt(input_length / output_length);
    double u_max1 = (k + 2) / alpha - 1;
    if (common::IsDoubleEqual(alpha, LongToDouble(output_length - 1))) {
      MS_EXCEPTION(ValueError) << "For FractionalAvgPool, the input_length and output_length is wrong, please check "
                                  "the parameter 'pooling ratio'.";
    } else {
      double u_max2 = (input_length + 1 - k) / alpha - (output_length - 1);
      double max_u = std::min(u_max1, u_max2);
      std::default_random_engine random(seed);
      std::uniform_real_distribution<double> dis2(0.0, 1.0);
      const double u = dis2(random) * max_u;
      cum_seq[0] = 1;
      cum_seq[LongToSize(output_length)] = input_length + 1;
      for (size_t i = 1; i < LongToSize(output_length); ++i) {
        cum_seq[i] = static_cast<int>(ceil(alpha * (i + u)));
      }
      for (size_t i = 0; i < LongToSize(output_length); ++i) {
        diff[i] = cum_seq[i + 1] - cum_seq[i];
      }
      return diff;
    }
  }
}

static std::vector<int64_t> GeneratePoolingSequenceRandom(int64_t input_length, int64_t output_length, int64_t seed) {
  if (output_length == 0) {
    MS_EXCEPTION(ValueError) << "For FractionalAvgPool, output_length got 0, please check it.";
  } else {
    int k = LongToInt(input_length / output_length);
    size_t num_random_spot = LongToSize(input_length % output_length);
    std::vector<int64_t> diff(output_length, k);
    for (size_t i = 0; i < num_random_spot; ++i) {
      diff[i] += 1;
    }
    std::shuffle(diff.begin(), diff.end(), std::default_random_engine(seed));
    return diff;
  }
}

std::vector<int64_t> GeneratePoolingSequence(int64_t input_length, int64_t output_length, bool pseudo_random,
                                             int64_t seed) {
  std::vector<int64_t> diff;
  if (output_length == 0) {
    MS_EXCEPTION(ValueError) << "For FractionalAvgPool, output_length got 0, please check it.";
  } else {
    if (input_length % output_length == 0) {
      diff = std::vector<int64_t>(output_length, input_length / output_length);
    } else {
      if (pseudo_random) {
        diff = GeneratePoolingSequencePseudoRandom(input_length, output_length, seed);
      } else {
        diff = GeneratePoolingSequenceRandom(input_length, output_length, seed);
      }
    }
    int k = LongToInt(input_length / output_length);
    for (size_t i = 0; i < LongToSize(output_length); i++) {
      if (diff[i] < k || diff[i] > IntToLong(k) + 1) {
        MS_EXCEPTION(ValueError) << "For FractionalAvgPool, GeneratePoolingSequence diff[" << i << "] is error";
      }
    }
    std::vector<int64_t> cum_seq(output_length + 1, 0);
    for (size_t i = 1; i < cum_seq.size(); ++i) {
      cum_seq[i] = cum_seq[i - 1] + diff[i - 1];
    }
    return cum_seq;
  }
}

template <typename T>
bool FractionalAvgPoolCpuKernelMod::FractionalAvgPoolLaunch(const std::vector<AddressPtr> &inputs,
                                                            const std::vector<AddressPtr> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_ptr);
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_ptr);
  int64_t *row_pooling_sequence_ptr = reinterpret_cast<int64_t *>(outputs[1]->addr);
  MS_EXCEPTION_IF_NULL(row_pooling_sequence_ptr);
  int64_t *col_pooling_sequence_ptr = reinterpret_cast<int64_t *>(outputs[2]->addr);
  MS_EXCEPTION_IF_NULL(col_pooling_sequence_ptr);
  for (size_t i = 0; i < tensor_in_and_out_dims; i++) {
    output_shape_[i] = static_cast<int>(std::floor(input_shape_[i] / pooling_ratio_[i]));
    if (output_shape_[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', outputsize[" << i << "] cannot be 0.";
    }
  }
  std::random_device rd;
  std::mt19937 generator(rd());
  int seed = seed_;
  int seed2 = seed2_;
  if (deterministic_) {
    // If both seeds are not set when deterministic is true, force set seeds.
    if ((seed == 0) && (seed2 == 0)) {
      seed = static_cast<int>(generator());
      seed2 = static_cast<int>(generator());
    }
  } else {
    if ((seed != 0) || (seed2 != 0)) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', both 'seed' and 'seed2' must be 0 if 'deterministic' is false.";
    }
  }
  if (seed == 0 && seed2 != 0) {
    seed = seed2;
  }
  // Generate pooling sequence.
  std::vector<int64_t> height_cum_seq;
  std::vector<int64_t> width_cum_seq;
  height_cum_seq =
    GeneratePoolingSequence(input_shape_[kInputShapeIndexH], output_shape_[kOutputShapeIndexH], pseudo_random_, seed);
  width_cum_seq =
    GeneratePoolingSequence(input_shape_[kInputShapeIndexW], output_shape_[kOutputShapeIndexW], pseudo_random_, seed);
  for (size_t i = 0; i < height_cum_seq.size(); ++i) {
    *(row_pooling_sequence_ptr + i) = height_cum_seq[i];
  }
  for (size_t i = 0; i < width_cum_seq.size(); ++i) {
    *(col_pooling_sequence_ptr + i) = width_cum_seq[i];
  }
  size_t batch_len = static_cast<size_t>(input_shape_[kInputShapeIndexN]);
  const int64_t height_max = input_shape_[kInputShapeIndexH] - 1;
  auto shard_fractional_avg_pool = [this, input_ptr, output_ptr, height_cum_seq, width_cum_seq, height_max](
                                     size_t start, size_t end) {
    for (size_t b = start; b < end; ++b) {
      for (size_t hs = 0; hs < height_cum_seq.size() - 1; ++hs) {
        const int64_t height_start = height_cum_seq[hs];
        int64_t height_end = overlapping_ ? height_cum_seq[hs + 1] : height_cum_seq[hs + 1] - 1;
        height_end = std::min(height_end, height_max);
        (void)FractionalAvgPoolDoCompute(input_ptr, output_ptr, b, hs, height_start, height_end, width_cum_seq);
      }
    }
  };
  CPUKernelUtils::ParallelFor(shard_fractional_avg_pool, batch_len);
  return true;
}

template <typename T>
bool FractionalAvgPoolCpuKernelMod::FractionalAvgPoolDoCompute(const T *input_ptr, T *output_ptr, size_t b, size_t hs,
                                                               const int64_t height_start, int64_t height_end,
                                                               std::vector<int64_t> width_cum_seq) {
  const int64_t width_max = input_shape_[kInputShapeIndexW] - 1;
  const int64_t depth_max = input_shape_[kInputShapeIndexC] - 1;
  for (size_t ws = 0; ws < width_cum_seq.size() - 1; ++ws) {
    for (int64_t c = 0; c <= depth_max; ++c) {
      const int64_t out_offset =
        ((SizeToLong(b) * output_shape_[kOutputShapeIndexH] + SizeToLong(hs)) * output_shape_[kOutputShapeIndexW] +
         SizeToLong(ws)) *
          output_shape_[kOutputShapeIndexC] +
        c;
      // Initializes the output tensor with 0.
      T sum = static_cast<T>(0);
      int count = 0;
      const int64_t width_start = width_cum_seq[ws];
      int64_t width_end = overlapping_ ? width_cum_seq[ws + 1] : width_cum_seq[ws + 1] - 1;
      width_end = std::min(width_end, width_max);
      for (int64_t h = height_start; h <= height_end; ++h) {
        for (int64_t w = width_start; w <= width_end; ++w) {
          const int64_t in_offset =
            ((SizeToLong(b) * input_shape_[kInputShapeIndexH] + h) * input_shape_[kInputShapeIndexW] + w) *
              output_shape_[kOutputShapeIndexC] +
            c;
          sum += *(input_ptr + in_offset);
          count++;
        }
      }
      if (count == 0) {
        MS_EXCEPTION(ValueError) << "The 'count' cannot be 0.";
      }
      T avg = sum / static_cast<T>(count);
      *(output_ptr + out_offset) = avg;
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, FractionalAvgPoolCpuKernelMod::FractionalAvgPoolFunc>>
  FractionalAvgPoolCpuKernelMod::func_list_ = {{KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat32)
                                                  .AddOutputAttr(kNumberTypeFloat32)
                                                  .AddOutputAttr(kNumberTypeInt64)
                                                  .AddOutputAttr(kNumberTypeInt64),
                                                &FractionalAvgPoolCpuKernelMod::FractionalAvgPoolLaunch<float>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeFloat64)
                                                  .AddOutputAttr(kNumberTypeFloat64)
                                                  .AddOutputAttr(kNumberTypeInt64)
                                                  .AddOutputAttr(kNumberTypeInt64),
                                                &FractionalAvgPoolCpuKernelMod::FractionalAvgPoolLaunch<double>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeInt32)
                                                  .AddOutputAttr(kNumberTypeInt32)
                                                  .AddOutputAttr(kNumberTypeInt64)
                                                  .AddOutputAttr(kNumberTypeInt64),
                                                &FractionalAvgPoolCpuKernelMod::FractionalAvgPoolLaunch<int32_t>},
                                               {KernelAttr()
                                                  .AddInputAttr(kNumberTypeInt64)
                                                  .AddOutputAttr(kNumberTypeInt64)
                                                  .AddOutputAttr(kNumberTypeInt64)
                                                  .AddOutputAttr(kNumberTypeInt64),
                                                &FractionalAvgPoolCpuKernelMod::FractionalAvgPoolLaunch<int64_t>}};

std::vector<KernelAttr> FractionalAvgPoolCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FractionalAvgPoolFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FractionalAvgPool, FractionalAvgPoolCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
