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
#include "plugin/device/cpu/kernel/fractional_max_pool3d_with_fixed_ksize_cpu_kernel.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <string>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDimSize2 = 2;
constexpr size_t kDimSize3 = 3;
constexpr size_t kDimSize4 = 4;
constexpr size_t kDimSize5 = 5;
constexpr size_t kInputIndex0 = 0;
constexpr size_t kInputIndex1 = 1;
constexpr size_t kOutputIndex1 = 1;
constexpr size_t kkernelsizeIndexD = 0;
constexpr size_t kkernelsizeIndexH = 1;
constexpr size_t kkernelsizeIndexW = 2;
constexpr size_t kOutputshapeIndexD = 0;
constexpr size_t kOutputshapeIndexH = 1;
constexpr size_t kOutputshapeIndexW = 2;
constexpr size_t kDimSize5FormatNCDHWIndexN = 0;
constexpr size_t kDimSize5FormatNCDHWIndexC = 1;
constexpr size_t kDimSize5FormatNCDHWIndexD = 2;
constexpr size_t kDimSize5FormatNCDHWIndexH = 3;
constexpr size_t kDimSize5FormatNCDHWIndexW = 4;
constexpr size_t kDimSize5FormatNDHWCIndexN = 0;
constexpr size_t kDimSize5FormatNDHWCIndexD = 1;
constexpr size_t kDimSize5FormatNDHWCIndexH = 2;
constexpr size_t kDimSize5FormatNDHWCIndexW = 3;
constexpr size_t kDimSize5FormatNDHWCIndexC = 4;
constexpr size_t kDimSize4FormatNCDHWIndexC = 0;
constexpr size_t kDimSize4FormatNCDHWIndexD = 1;
constexpr size_t kDimSize4FormatNCDHWIndexH = 2;
constexpr size_t kDimSize4FormatNCDHWIndexW = 3;
constexpr size_t kDimSize4FormatNDHWCIndexD = 0;
constexpr size_t kDimSize4FormatNDHWCIndexH = 1;
constexpr size_t kDimSize4FormatNDHWCIndexW = 2;
constexpr size_t kDimSize4FormatNDHWCIndexC = 3;
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 2;

#define ADD_KERNEL(t1, t2, t3, t4)  \
  KernelAttr()                      \
    .AddInputAttr(kNumberType##t1)  \
    .AddInputAttr(kNumberType##t2)  \
    .AddOutputAttr(kNumberType##t3) \
    .AddOutputAttr(kNumberType##t4)
}  // namespace

bool FractionalMaxPool3DWithFixedKsizeCPUKernelMod::Init(const BaseOperatorPtr &base_operator,
                                                         const std::vector<KernelTensorPtr> &inputs,
                                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  constexpr size_t input_num = kInputsNum;
  constexpr size_t output_num = kOutputsNum;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto match = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!match.first) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  return true;
}

int FractionalMaxPool3DWithFixedKsizeCPUKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                                          const std::vector<KernelTensorPtr> &inputs,
                                                          const std::vector<KernelTensorPtr> &outputs,
                                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_type_ = inputs[kInputIndex0]->GetDtype();
  input_shape_ = inputs[kInputIndex0]->GetDeviceShapeAdaptively();
  random_samples_type_ = inputs[kInputIndex1]->GetDtype();
  random_samples_shape_ = inputs[kInputIndex1]->GetDeviceShapeAdaptively();
  argmax_type_ = outputs[kOutputIndex1]->GetDtype();
  auto kernel_ptr = std::dynamic_pointer_cast<ops::FractionalMaxPool3DWithFixedKsize>(base_operator);
  MS_EXCEPTION_IF_NULL(kernel_ptr);
  output_shape_ = kernel_ptr->get_output_shape();
  ksize_ = kernel_ptr->get_ksize();
  data_format_ = kernel_ptr->get_data_format();
  size_t input_num_dims = input_shape_.size();
  size_t random_samples_dims = random_samples_shape_.size();
  size_t output_shape_dims = output_shape_.size();
  size_t ksize_dims = ksize_.size();
  outputD_ = output_shape_[kOutputshapeIndexD];
  outputH_ = output_shape_[kOutputshapeIndexH];
  outputW_ = output_shape_[kOutputshapeIndexW];
  kernelsizeD_ = ksize_[kkernelsizeIndexD];
  kernelsizeH_ = ksize_[kkernelsizeIndexH];
  kernelsizeW_ = ksize_[kkernelsizeIndexW];
  if (data_format_ == "NCDHW") {
    if (input_shape_.size() == kDimSize5) {
      inputN_ = input_shape_[kDimSize5FormatNCDHWIndexN];
      inputC_ = input_shape_[kDimSize5FormatNCDHWIndexC];
      inputD_ = input_shape_[kDimSize5FormatNCDHWIndexD];
      inputH_ = input_shape_[kDimSize5FormatNCDHWIndexH];
      inputW_ = input_shape_[kDimSize5FormatNCDHWIndexW];
    } else {
      inputC_ = input_shape_[kDimSize4FormatNCDHWIndexC];
      inputD_ = input_shape_[kDimSize4FormatNCDHWIndexD];
      inputH_ = input_shape_[kDimSize4FormatNCDHWIndexH];
      inputW_ = input_shape_[kDimSize4FormatNCDHWIndexW];
    }
  } else {
    if (input_shape_.size() == kDimSize5) {
      inputN_ = input_shape_[kDimSize5FormatNDHWCIndexN];
      inputC_ = input_shape_[kDimSize5FormatNDHWCIndexC];
      inputD_ = input_shape_[kDimSize5FormatNDHWCIndexD];
      inputH_ = input_shape_[kDimSize5FormatNDHWCIndexH];
      inputW_ = input_shape_[kDimSize5FormatNDHWCIndexW];
    } else {
      inputC_ = input_shape_[kDimSize4FormatNDHWCIndexC];
      inputD_ = input_shape_[kDimSize4FormatNDHWCIndexD];
      inputH_ = input_shape_[kDimSize4FormatNDHWCIndexH];
      inputW_ = input_shape_[kDimSize4FormatNDHWCIndexW];
    }
  }
  if (outputD_ + kernelsizeD_ - 1 >= inputD_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', out(): pool depth ," << kernelsizeD_
                             << "too large relative to input depth" << inputD_ << ".";
  }
  if (outputH_ + kernelsizeH_ - 1 >= inputH_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', out(): pool height ," << kernelsizeH_
                             << "too large relative to input height" << inputH_ << ".";
  }
  if (outputW_ + kernelsizeW_ - 1 >= inputW_) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', out(): pool width ," << kernelsizeW_
                             << "too large relative to input width" << inputW_ << ".";
  }
  if (!(input_num_dims == kDimSize4 || input_num_dims == kDimSize5)) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the dimension of 'x' must be equal to 4 or 5, but got "
                            << input_num_dims << ".";
  }
  if (random_samples_dims != kDimSize3) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the dimension of 'random_samples' must be equal to 3, but got "
                            << random_samples_dims << ".";
  }
  if (output_shape_dims != kDimSize3) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_
                            << "', the dimension of 'output_shape' must be equal to 3, but got " << output_shape_dims
                            << ".";
  }
  if (ksize_dims != kDimSize3) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the dimension of 'ksize' must be equal to 3, but got "
                            << ksize_dims << ".";
  }
  for (size_t i = 0; i < input_num_dims; i++) {
    if (input_shape_[i] <= 0) {
      MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                               << "', expected 'x' have non-empty spatial dimensions, but 'x' has sizes "
                               << input_shape_[i] << " with dimension " << i << " being empty.";
    }
  }
  if (random_samples_shape_[kDimSize2] != kDimSize3) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                             << "', expected the third dimension of 'random_samples' must be 3, but got "
                             << random_samples_shape_[kDimSize2] << ".";
  }
  return ret;
}

template <typename random_sample_t>
std::vector<int> generate_intervals(random_sample_t random_sample, int input_size, int output_size, int kernel_size) {
  std::vector<int> sequence(output_size);
  if (output_size > 1) {
    random_sample_t alpha =
      static_cast<random_sample_t>(input_size - kernel_size) / static_cast<random_sample_t>(output_size - 1);

    for (int i = 0; i < output_size - 1; ++i) {
      sequence[IntToSize(i)] = static_cast<int>((static_cast<random_sample_t>(i) + random_sample) * alpha) -
                               static_cast<int>(random_sample * alpha);
    }
  }
  sequence[IntToSize(output_size) - 1] = input_size - kernel_size;

  return sequence;
}

template <>
std::vector<int> generate_intervals(float16 random_sample, int input_size, int output_size, int kernel_size) {
  std::vector<int> sequence(output_size);
  if (output_size > 1) {
    float alpha = static_cast<float>(input_size - kernel_size) / static_cast<float>(output_size - 1);

    for (int i = 0; i < output_size - 1; ++i) {
      sequence[IntToSize(i)] = static_cast<int>((static_cast<float>(i) + static_cast<float>(random_sample)) * alpha) -
                               static_cast<int>(static_cast<float>(random_sample) * alpha);
    }
  }
  sequence[IntToSize(output_size) - 1] = input_size - kernel_size;

  return sequence;
}

template <typename scalar_t, typename random_sample_t, typename argmax_t>
bool FractionalMaxPool3DWithFixedKsizeCPUKernelMod::ComputeTemplate(const std::vector<AddressPtr> &inputs,
                                                                    const std::vector<AddressPtr> &outputs) {
  auto input_data = reinterpret_cast<scalar_t *>(inputs[0]->addr);
  auto random_samples_data = reinterpret_cast<random_sample_t *>(inputs[1]->addr);
  auto output_data = reinterpret_cast<scalar_t *>(outputs[0]->addr);
  auto argmax_data = reinterpret_cast<argmax_t *>(outputs[1]->addr);

  if (input_shape_.size() == kDimSize4) {
    auto shard_fractional_max_pool3d_with_fixed_ksize = [&](size_t start, size_t end) {
      for (auto plane = start; plane < end; ++plane) {
        /* each plane contains 3 random random_samples,
           one for T, one for W, and one for H */
        scalar_t *inputForPlane = input_data + plane * inputD_ * inputH_ * inputW_;
        scalar_t *outputForPlane = output_data + plane * outputD_ * outputH_ * outputW_;
        argmax_t *argmaxForPlane = argmax_data + plane * outputD_ * outputH_ * outputW_;
        random_sample_t *random_samplesForPlane = random_samples_data + plane * 3;
        FractionalMaxPool3DWithFixedKsizeCompute<scalar_t, random_sample_t, argmax_t>(
          inputForPlane, random_samplesForPlane, argmaxForPlane, outputForPlane, outputD_, outputH_, outputW_,
          kernelsizeD_, kernelsizeH_, kernelsizeW_, inputC_, inputD_, inputH_, inputW_);
      }
    };
    CPUKernelUtils::ParallelFor(shard_fractional_max_pool3d_with_fixed_ksize, inputC_);
  } else {
    auto shard_fractional_max_pool3d_with_fixed_ksize = [&](size_t start, size_t end) {
      for (auto batch = start; batch < end; ++batch) {
        for (int64_t plane = 0; plane < inputC_; ++plane) {
          auto intput_data_n = input_data + batch * inputC_ * inputW_ * inputH_ * inputD_;
          auto output_data_n = output_data + batch * inputC_ * outputW_ * outputH_ * outputD_;
          auto argmax_data_n = argmax_data + batch * inputC_ * outputW_ * outputH_ * outputD_;
          scalar_t *inputForPlane = intput_data_n + plane * inputD_ * inputH_ * inputW_;
          scalar_t *outputForPlane = output_data_n + plane * outputD_ * outputH_ * outputW_;
          argmax_t *argmaxForPlane = argmax_data_n + plane * outputD_ * outputH_ * outputW_;
          auto random_samples_data_n = random_samples_data + batch * inputC_ * 3;
          random_sample_t *random_samplesForPlane = random_samples_data_n + plane * 3;
          FractionalMaxPool3DWithFixedKsizeCompute<scalar_t, random_sample_t, argmax_t>(
            inputForPlane, random_samplesForPlane, argmaxForPlane, outputForPlane, outputD_, outputH_, outputW_,
            kernelsizeD_, kernelsizeH_, kernelsizeW_, inputC_, inputD_, inputH_, inputW_);
        }
      }
    };
    CPUKernelUtils::ParallelFor(shard_fractional_max_pool3d_with_fixed_ksize, LongToSize(inputN_));
  }
  return true;
}
template <typename scalar_t, typename random_sample_t, typename argmax_t>
bool FractionalMaxPool3DWithFixedKsizeCPUKernelMod::FractionalMaxPool3DWithFixedKsizeCompute(
  scalar_t *inputForPlane, random_sample_t *random_samplesForPlane, argmax_t *argmaxForPlane, scalar_t *outputForPlane,
  int64_t outputD_, int64_t outputH_, int64_t outputW_, int64_t kernelsizeD_, int64_t kernelsizeH_,
  int64_t kernelsizeW_, int64_t inputC_, int64_t inputD_, int64_t inputH_, int64_t inputW_) {
  // Generate interval sequence
  auto sequenceT = generate_intervals<random_sample_t>(random_samplesForPlane[0], inputD_, outputD_, kernelsizeD_);
  auto sequenceH = generate_intervals<random_sample_t>(random_samplesForPlane[1], inputH_, outputH_, kernelsizeH_);
  auto sequenceW = generate_intervals<random_sample_t>(random_samplesForPlane[2], inputW_, outputW_, kernelsizeW_);
  // loop over output
  int64_t t, h, w;
  for (t = 0; t < outputD_; ++t) {
    int64_t inputD_Start = sequenceT[t];
    for (h = 0; h < outputH_; ++h) {
      int64_t inputH_Start = sequenceH[h];
      for (w = 0; w < outputW_; ++w) {
        int64_t inputW_Start = sequenceW[w];
        int64_t t2 = inputD_Start, h2 = inputH_Start, w2 = inputW_Start;
        scalar_t maxVal = std::numeric_limits<scalar_t>::lowest();
        argmax_t maxIndex = t2 * inputH_ * inputW_ + h2 * inputW_ + w2;
        for (t2 = inputD_Start; t2 < inputD_Start + kernelsizeD_; ++t2) {
          for (h2 = inputH_Start; h2 < inputH_Start + kernelsizeH_; ++h2) {
            for (w2 = inputW_Start; w2 < inputW_Start + kernelsizeW_; ++w2) {
              if (t2 < 0 || t2 >= inputD_) {
                MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', index T value is illegal.";
              }
              if (h2 < 0 || h2 >= inputH_) {
                MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', index H value is illegal.";
              }
              if (w2 < 0 || w2 >= inputW_) {
                MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', index W value is illegal.";
              }
              argmax_t planeIndex = t2 * inputH_ * inputW_ + h2 * inputW_ + w2;
              scalar_t val = inputForPlane[planeIndex];
              if (val > maxVal || std::isnan(static_cast<double>(val))) {
                maxVal = val;
                maxIndex = planeIndex;
              }
            }
          }
        }
        outputForPlane[t * outputH_ * outputW_ + h * outputW_ + w] = maxVal;
        argmaxForPlane[t * outputH_ * outputW_ + h * outputW_ + w] = maxIndex;
      }
    }
  }
  return true;
}

template <typename scalar_t, typename random_sample_t>
bool FractionalMaxPool3DWithFixedKsizeCPUKernelMod::DoComputeWithArgmaxType(const std::vector<AddressPtr> &inputs,
                                                                            const std::vector<AddressPtr> &outputs,
                                                                            TypeId argmax_type) {
  switch (argmax_type) {
    case kNumberTypeInt32:
      return ComputeTemplate<scalar_t, random_sample_t, int32_t>(inputs, outputs);
    case kNumberTypeInt64:
      return ComputeTemplate<scalar_t, random_sample_t, int64_t>(inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the type of 'argmax'" << argmax_type
                              << "not support, must be in [{DT_INT32, DT_INT64}].";
      return false;
  }
}

template <typename scalar_t>
bool FractionalMaxPool3DWithFixedKsizeCPUKernelMod::DoComputeWithRandomSamplesType(
  const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs, TypeId random_samples_type) {
  switch (random_samples_type) {
    case kNumberTypeFloat16:
      return DoComputeWithArgmaxType<scalar_t, float16>(inputs, outputs, argmax_type_);
    case kNumberTypeFloat32:
      return DoComputeWithArgmaxType<scalar_t, float>(inputs, outputs, argmax_type_);
    case kNumberTypeFloat64:
      return DoComputeWithArgmaxType<scalar_t, double>(inputs, outputs, argmax_type_);
    default:
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', the type of 'random_samples'" << random_samples_type
                              << "not support, must be in [{DT_FLOAT16, DT_FLOAT, DT_DOUBLE}].";
      return false;
  }
}

bool FractionalMaxPool3DWithFixedKsizeCPUKernelMod::Launch(const std::vector<AddressPtr> &inputs,
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
      MS_EXCEPTION(TypeError) << "For '" << kernel_name_ << "', does not support this kernel data type.";
  }
  return true;
}

std::vector<KernelAttr> FractionalMaxPool3DWithFixedKsizeCPUKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    ADD_KERNEL(Int32, Float32, Int32, Int32),     ADD_KERNEL(Int64, Float32, Int64, Int32),
    ADD_KERNEL(Float16, Float32, Float16, Int32), ADD_KERNEL(Float32, Float32, Float32, Int32),
    ADD_KERNEL(Float64, Float32, Float64, Int32), ADD_KERNEL(Int32, Float16, Int32, Int32),
    ADD_KERNEL(Int64, Float16, Int64, Int32),     ADD_KERNEL(Float16, Float16, Float16, Int32),
    ADD_KERNEL(Float32, Float16, Float32, Int32), ADD_KERNEL(Float64, Float16, Float64, Int32),
    ADD_KERNEL(Int32, Float64, Int32, Int32),     ADD_KERNEL(Int64, Float64, Int64, Int32),
    ADD_KERNEL(Float16, Float64, Float16, Int32), ADD_KERNEL(Float32, Float64, Float32, Int32),
    ADD_KERNEL(Float64, Float64, Float64, Int32), ADD_KERNEL(Int32, Float32, Int32, Int64),
    ADD_KERNEL(Int64, Float32, Int64, Int64),     ADD_KERNEL(Float16, Float32, Float16, Int64),
    ADD_KERNEL(Float32, Float32, Float32, Int64), ADD_KERNEL(Float64, Float32, Float64, Int64),
    ADD_KERNEL(Int32, Float16, Int32, Int64),     ADD_KERNEL(Int64, Float16, Int64, Int64),
    ADD_KERNEL(Float16, Float16, Float16, Int64), ADD_KERNEL(Float32, Float16, Float32, Int64),
    ADD_KERNEL(Float64, Float16, Float64, Int64), ADD_KERNEL(Int32, Float64, Int32, Int64),
    ADD_KERNEL(Int64, Float64, Int64, Int64),     ADD_KERNEL(Float16, Float64, Float16, Int64),
    ADD_KERNEL(Float32, Float64, Float32, Int64), ADD_KERNEL(Float64, Float64, Float64, Int64)};
  return kernel_attr_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FractionalMaxPool3DWithFixedKsize,
                      FractionalMaxPool3DWithFixedKsizeCPUKernelMod);
}  // namespace kernel
}  // namespace mindspore
