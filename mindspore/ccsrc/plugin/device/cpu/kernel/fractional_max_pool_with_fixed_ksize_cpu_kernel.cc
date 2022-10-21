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
#include "plugin/device/cpu/kernel/fractional_max_pool_with_fixed_ksize_cpu_kernel.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kInputsNum = 2;
const size_t kOutputsNum = 2;
const size_t kInputIndex0 = 0;
const size_t kInputIndex1 = 1;
const size_t kOutputIndex1 = 1;
const size_t kInputDimIndexN = 0;
const size_t kInputDimIndexC = 1;
const size_t kInputDimIndexH = 2;
const size_t kInputDimIndexW = 3;
const size_t kDimSize1 = 1;
const size_t kDimSize2 = 2;
const size_t kDimSize3 = 3;
const size_t kDimSize4 = 4;
const size_t kKszieIndexH = 0;
const size_t kKszieIndexW = 1;
const size_t kOutputShapeIndexH = 0;
const size_t kOutputShapeIndexW = 1;
const size_t kRandomSimplesLastDimIndex = 2;
const int64_t kRandomSimplesThirdDimSize = 2;
const size_t kKsizeLength1 = 1;
const size_t kKsizeLength2 = 2;
const size_t kOutputShapeLength1 = 1;
const size_t kOutputShapeLength2 = 2;

#define ADD_KERNEL(t1, t2, t3, t4)  \
  KernelAttr()                      \
    .AddInputAttr(kNumberType##t1)  \
    .AddInputAttr(kNumberType##t2)  \
    .AddOutputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)
}  // namespace

bool FractionalMaxPoolWithFixedKsizeCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                       const std::vector<KernelTensorPtr> &inputs,
                                                       const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = kInputsNum;
  constexpr size_t output_num = kOutputsNum;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  input_type_ = inputs[kInputIndex0]->GetDtype();
  random_samples_type_ = inputs[kInputIndex1]->GetDtype();
  argmax_type_ = outputs[kOutputIndex1]->GetDtype();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', does not support this kernel data type: " << kernel_attr;
    return false;
  }
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FractionalMaxPoolWithFixedKsize>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  output_shape_ = kernel_ptr->get_output_shape();
  ksize_ = kernel_ptr->get_ksize();
  data_format_ = kernel_ptr->get_data_format();
  if (data_format_ != "NCHW") {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the attr data_format must be NCHW.";
  }
  if (output_shape_.size() == kOutputShapeLength1) {
    output_h_ = output_shape_[kOutputShapeIndexH];
    output_w_ = output_shape_[kOutputShapeIndexH];
  } else if (output_shape_.size() == kOutputShapeLength2) {
    output_h_ = output_shape_[kOutputShapeIndexH];
    output_w_ = output_shape_[kOutputShapeIndexW];
  } else {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', the size of attr output_shape must be equal to 1 or 2, but got "
                             << output_shape_.size() << ".";
  }
  if (ksize_.size() == kKsizeLength1) {
    ksize_h_ = ksize_[kKszieIndexH];
    ksize_w_ = ksize_[kKszieIndexH];
  } else if (ksize_.size() == kKsizeLength2) {
    ksize_h_ = ksize_[kKszieIndexH];
    ksize_w_ = ksize_[kKszieIndexW];
  } else {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the size of attr kszie must be equal to 1 or 2, but got "
                             << ksize_.size() << ".";
  }
  return true;
}

int FractionalMaxPoolWithFixedKsizeCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                        const std::vector<KernelTensorPtr> &inputs,
                                                        const std::vector<KernelTensorPtr> &outputs,
                                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kInputIndex0]->GetDeviceShapeAdaptively();
  random_samples_shape_ = inputs[kInputIndex1]->GetDeviceShapeAdaptively();

  input_n_ = input_shape_[kInputDimIndexN];
  input_c_ = input_shape_[kInputDimIndexC];
  input_h_ = input_shape_[kInputDimIndexH];
  input_w_ = input_shape_[kInputDimIndexW];

  if (output_h_ + ksize_h_ - 1 > input_h_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', ksize height [" << ksize_h_ << "] + output_shape_h ["
                             << output_h_ << "] too large relative to input height [" << input_h_
                             << "], conflict with the rule: ksize_h + output_shape_h - 1 <= input_h";
  }
  if (output_w_ + ksize_w_ - 1 > input_w_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', ksize width [" << ksize_w_ << "] + output_shape_w ["
                             << output_w_ << "] too large relative to input width [" << input_w_
                             << "], conflict with the rule: ksize_w + output_shape_w - 1 <= input_w";
  }
  if (random_samples_shape_[kInputDimIndexN] != input_n_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', The first dim of input[x] and input[random_samples] must be equal, but "
                             << "got x=[" << input_n_ << "] and random_samples=["
                             << random_samples_shape_[kInputDimIndexN] << "].";
  }
  if (random_samples_shape_[kInputDimIndexC] != input_c_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', The second dim of input[x] and input[random_samples] must be equal, but "
                             << "got x=[" << input_c_ << "] and random_samples=["
                             << random_samples_shape_[kInputDimIndexC] << "].";
  }
  if (random_samples_shape_[kRandomSimplesLastDimIndex] != kRandomSimplesThirdDimSize) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', The third dim of input[random_samples] must be 2, but got "
                             << random_samples_shape_[kRandomSimplesLastDimIndex] << ".";
  }
  return ret;
}

bool FractionalMaxPoolWithFixedKsizeCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                                         const std::vector<AddressPtr> &workspace,
                                                         const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputsNum, kernel_name_);
  switch (input_type_) {
    case kNumberTypeFloat16:
      return DoComputeWithRandomSamplesType<float16>(inputs, outputs, random_samples_type_);
    case kNumberTypeFloat32:
      return DoComputeWithRandomSamplesType<float>(inputs, outputs, random_samples_type_);
    case kNumberTypeFloat64:
      return DoComputeWithRandomSamplesType<double>(inputs, outputs, random_samples_type_);
    case kNumberTypeInt32:
      return DoComputeWithRandomSamplesType<int32_t>(inputs, outputs, random_samples_type_);
    case kNumberTypeInt64:
      return DoComputeWithRandomSamplesType<int64_t>(inputs, outputs, random_samples_type_);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the data type of input not support.";
  }
  return true;
}

template <typename scalar_t>
bool FractionalMaxPoolWithFixedKsizeCPUKernelMod::DoComputeWithRandomSamplesType(const std::vector<AddressPtr> &inputs,
                                                                                 const std::vector<AddressPtr> &outputs,
                                                                                 TypeId random_samples_type_) const {
  switch (random_samples_type_) {
    case kNumberTypeFloat16:
      return ComputeTemplate<scalar_t, float16>(inputs, outputs);
    case kNumberTypeFloat32:
      return ComputeTemplate<scalar_t, float>(inputs, outputs);
    case kNumberTypeFloat64:
      return ComputeTemplate<scalar_t, double>(inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', random_samples_type" << random_samples_type_
                              << "not support, must be in [{DT_FLOAT16, DT_FLOAT, DT_DOUBLE}].";
  }
}

template <typename scalar_t, typename random_sample_t>
bool FractionalMaxPoolWithFixedKsizeCPUKernelMod::ComputeTemplate(const std::vector<AddressPtr> &inputs,
                                                                  const std::vector<AddressPtr> &outputs) const {
  scalar_t *input_ptr = static_cast<scalar_t *>(inputs[0]->addr);
  random_sample_t *random_samples_ptr = static_cast<random_sample_t *>(inputs[1]->addr);
  scalar_t *output_ptr = static_cast<scalar_t *>(outputs[0]->addr);
  int64_t *argmax_ptr = static_cast<int64_t *>(outputs[1]->addr);
  MS_EXCEPTION_IF_NULL(input_ptr);
  MS_EXCEPTION_IF_NULL(random_samples_ptr);
  MS_EXCEPTION_IF_NULL(output_ptr);
  MS_EXCEPTION_IF_NULL(argmax_ptr);

  auto shard_fractional_max_pool_with_fixed_ksize = [&](size_t start, size_t end) {
    for (size_t n = start; n < end; n++) {
      scalar_t *inputForPlane = input_ptr + n * input_c_ * input_h_ * input_w_;
      random_sample_t *random_samplesForPlane = random_samples_ptr + n * input_c_ * kRandomSimplesThirdDimSize;
      scalar_t *outputForPlane = output_ptr + n * input_c_ * output_h_ * output_w_;
      int64_t *argmaxForPlane = argmax_ptr + n * input_c_ * output_h_ * output_w_;

      FractionalMaxPoolWithFixedKsizeCompute<scalar_t, random_sample_t>(inputForPlane, random_samplesForPlane,
                                                                        outputForPlane, argmaxForPlane);
    }
  };
  CPUKernelUtils::ParallelFor(shard_fractional_max_pool_with_fixed_ksize, LongToSize(input_n_));

  return true;
}

template <typename scalar_t, typename random_sample_t>
void FractionalMaxPoolWithFixedKsizeCPUKernelMod::FractionalMaxPoolWithFixedKsizeCompute(
  scalar_t *inputForPlane, random_sample_t *random_samplesForPlane, scalar_t *outputForPlane,
  int64_t *argmaxForPlane) const {
  for (int64_t plane = 0; plane < input_c_; plane++) {
    random_sample_t *random_samplesPlane = random_samplesForPlane + plane * 2;
    std::vector<int> sequenceW = GenerateIntervals<random_sample_t>(
      random_samplesPlane[0], static_cast<int>(input_w_), static_cast<int>(output_w_), static_cast<int>(ksize_w_));
    std::vector<int> sequenceH = GenerateIntervals<random_sample_t>(
      random_samplesPlane[1], static_cast<int>(input_h_), static_cast<int>(output_h_), static_cast<int>(ksize_h_));

    scalar_t *inputPlane = inputForPlane + plane * input_h_ * input_w_;
    scalar_t *outputPlane = outputForPlane + plane * output_h_ * output_w_;
    int64_t *argmaxPlane = argmaxForPlane + plane * output_h_ * output_w_;

    int h, w;
    for (h = 0; h < output_h_; h++) {
      int inputHStart = sequenceH[h];
      for (w = 0; w < output_w_; w++) {
        int inputWStart = sequenceW[w];
        int h2 = inputHStart;
        int w2 = inputWStart;
        scalar_t maxValue = -std::numeric_limits<scalar_t>::infinity();
        int64_t maxIndex = h2 * input_w_ + w2;

        for (h2 = inputHStart; h2 < inputHStart + ksize_h_; h2++) {
          for (w2 = inputWStart; w2 < inputWStart + ksize_w_; w2++) {
            if (h2 < 0 || h2 >= input_h_) {
              MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', index H value is illegal.";
            }
            if (w2 < 0 || w2 >= input_w_) {
              MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', index W value is illegal.";
            }

            int index = h2 * input_w_ + w2;
            scalar_t value = inputPlane[index];
            if (value > maxValue) {
              maxValue = value;
              maxIndex = index;
            }
          }
        }

        outputPlane[h * output_w_ + w] = maxValue;
        argmaxPlane[h * output_w_ + w] = maxIndex;
      }
    }
  }
}

template <typename random_sample_t>
std::vector<int> FractionalMaxPoolWithFixedKsizeCPUKernelMod::GenerateIntervals(random_sample_t sample, int input_size,
                                                                                int output_size,
                                                                                int kernel_size) const {
  std::vector<int> sequence(output_size);
  if (output_size > 1) {
    random_sample_t alpha =
      static_cast<random_sample_t>(input_size - kernel_size) / static_cast<random_sample_t>(output_size - 1);

    for (int i = 0; i < output_size - 1; i++) {
      sequence[i] =
        static_cast<int>((static_cast<random_sample_t>(i) + sample) * alpha) - static_cast<int>(sample * alpha);
    }
  }
  sequence[output_size - 1] = input_size - kernel_size;

  return sequence;
}

std::vector<KernelAttr> FractionalMaxPoolWithFixedKsizeCPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int32, Float32, Int32, Int64),     ADD_KERNEL(Int64, Float32, Int64, Int64),
    ADD_KERNEL(Float16, Float32, Float16, Int64), ADD_KERNEL(Float32, Float32, Float32, Int64),
    ADD_KERNEL(Float64, Float32, Float64, Int64), ADD_KERNEL(Int32, Float16, Int32, Int64),
    ADD_KERNEL(Int64, Float16, Int64, Int64),     ADD_KERNEL(Float16, Float16, Float16, Int64),
    ADD_KERNEL(Float32, Float16, Float32, Int64), ADD_KERNEL(Float64, Float16, Float64, Int64),
    ADD_KERNEL(Int32, Float64, Int32, Int64),     ADD_KERNEL(Int64, Float64, Int64, Int64),
    ADD_KERNEL(Float16, Float64, Float16, Int64), ADD_KERNEL(Float32, Float64, Float32, Int64),
    ADD_KERNEL(Float64, Float64, Float64, Int64)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FractionalMaxPoolWithFixedKsize, FractionalMaxPoolWithFixedKsizeCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
