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
#include "plugin/device/cpu/kernel/hsv_to_rgb_cpu_kernel.h"
#include <vector>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
void HSVToRGBCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  const size_t kNumDims = 3;
  const size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  const size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (input_num != kInputNum) {
    MS_LOG(EXCEPTION) << "Needs " << kInputNum << " input, but got " << input_num << ".";
  }
  if (output_num != kOutputNum) {
    MS_LOG(EXCEPTION) << "Needs " << kOutputNum << " output, but got " << output_num << ".";
  }
  shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input_dtype = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  if (shape.cend()[-1] != kNumDims) {
    MS_LOG(EXCEPTION) << "The last dimension of the input tensor must be size 3.";
  }
}

template <typename T1>
void HSVToRGBCpuKernelMod::ConvertOnePixel(T1 h, T1 s, T1 v, T1 *r, T1 *g, T1 *b) {
  const T1 kNumber6 = 6;
  T1 c = s * v;
  T1 m = v - c;
  T1 dh = h * kNumber6;
  T1 rr, gg, bb;
  const int32_t h_category = static_cast<int32_t>(std::floor(dh));
  T1 fmodu = std::abs(dh);
  const T1 kLimit = 2;
  if (fmodu >= kLimit) {
    fmodu = std::fmod(fmodu, kLimit);
  }
  const int32_t h_category_value_0 = 0;
  const int32_t h_category_value_1 = 1;
  const int32_t h_category_value_2 = 2;
  const int32_t h_category_value_3 = 3;
  const int32_t h_category_value_4 = 4;
  const int32_t h_category_value_5 = 5;

  T1 x = c * (1 - std::abs(fmodu - 1));
  switch (h_category) {
    case h_category_value_0:
      rr = c;
      gg = x;
      bb = 0;
      break;
    case h_category_value_1:
      rr = x;
      gg = c;
      bb = 0;
      break;
    case h_category_value_2:
      rr = 0;
      gg = c;
      bb = x;
      break;
    case h_category_value_3:
      rr = 0;
      gg = x;
      bb = c;
      break;
    case h_category_value_4:
      rr = x;
      gg = 0;
      bb = c;
      break;
    case h_category_value_5:
      rr = c;
      gg = 0;
      bb = x;
      break;
    default:
      rr = c;
      gg = 0;
      bb = 0;
  }
  *r = rr + m;
  *g = gg + m;
  *b = bb + m;
}

template <typename T1>
void HSVToRGBCpuKernelMod::ComputeFloat(void *input, void *output, int64_t pixel_num) {
  T1 *input_ptr = reinterpret_cast<T1 *>(input);
  T1 *output_ptr = reinterpret_cast<T1 *>(output);
  auto shard_hsv_to_rgb = [&input_ptr, &output_ptr, this](size_t start, size_t end) {
    constexpr size_t pixel_stride = 3;
    constexpr size_t first_value = 0;
    constexpr size_t second_value = 1;
    constexpr size_t third_value = 2;
    for (size_t i = start; i < end; ++i) {
      T1 *h = input_ptr + pixel_stride * i + first_value;
      T1 *s = input_ptr + pixel_stride * i + second_value;
      T1 *v = input_ptr + pixel_stride * i + third_value;
      T1 *r = output_ptr + pixel_stride * i + first_value;
      T1 *g = output_ptr + pixel_stride * i + second_value;
      T1 *b = output_ptr + pixel_stride * i + third_value;
      ConvertOnePixel<T1>(*h, *s, *v, r, g, b);
    }
  };
  CPUKernelUtils::ParallelFor(shard_hsv_to_rgb, static_cast<size_t>(pixel_num));
}

void HSVToRGBCpuKernelMod::ComputeHalf(void *input, void *output, int64_t pixel_num) {
  float16 *input_ptr = reinterpret_cast<float16 *>(input);
  float16 *output_ptr = reinterpret_cast<float16 *>(output);
  auto shard_hsv_to_rgb = [&input_ptr, &output_ptr, this](size_t start, size_t end) {
    constexpr size_t pixel_stride = 3;
    constexpr size_t first_value = 0;
    constexpr size_t second_value = 1;
    constexpr size_t third_value = 2;
    float tmp[3];
    for (size_t i = start; i < end; ++i) {
      float h = static_cast<float>(input_ptr[pixel_stride * i + first_value]);
      float s = static_cast<float>(input_ptr[pixel_stride * i + second_value]);
      float v = static_cast<float>(input_ptr[pixel_stride * i + third_value]);
      ConvertOnePixel<float>(h, s, v, tmp + first_value, tmp + second_value, tmp + third_value);
      output_ptr[pixel_stride * i + first_value] = float16(tmp[first_value]);
      output_ptr[pixel_stride * i + second_value] = float16(tmp[second_value]);
      output_ptr[pixel_stride * i + third_value] = float16(tmp[third_value]);
    }
  };
  CPUKernelUtils::ParallelFor(shard_hsv_to_rgb, static_cast<size_t>(pixel_num));
}

bool HSVToRGBCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                  const std::vector<AddressPtr> &outputs) {
  const int64_t pixel_num =
    accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), [=](int64_t a, int64_t b) { return a * b; }) / 3;
  void *input = inputs[0]->addr;
  void *output = outputs[0]->addr;
  switch (input_dtype) {
    case kNumberTypeFloat16:
      ComputeHalf(input, output, pixel_num);
      break;
    case kNumberTypeFloat32:
      ComputeFloat<float>(input, output, pixel_num);
      break;
    case kNumberTypeFloat64:
      ComputeFloat<double>(input, output, pixel_num);
      break;
    default:
      MS_LOG(EXCEPTION) << "Input Tensor data type is not surpported.";
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HSVToRGB, HSVToRGBCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
