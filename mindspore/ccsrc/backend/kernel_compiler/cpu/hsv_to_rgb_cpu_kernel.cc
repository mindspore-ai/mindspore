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
#include "backend/kernel_compiler/cpu/hsv_to_rgb_cpu_kernel.h"
#include <iostream>
#include <vector>
#include "runtime/device/cpu/cpu_device_address.h"

namespace mindspore {
namespace kernel {
template <typename T>
void HSVToRGBCpuKernelMod<T>::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  const size_t kNumDims = 3;
  const size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
  const size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
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

template <typename T>
template <typename T1>
void HSVToRGBCpuKernelMod<T>::ConvertOnePixel(T1 h, T1 s, T1 v, T1 *r, T1 *g, T1 *b) {
  T1 c = s * v;
  T1 m = v - c;
  T1 dh = h * 6;
  T1 rr, gg, bb;
  const int32_t h_category = static_cast<int32_t>(std::floor(dh));
  T1 fmodu = dh;
  const int32_t kLimitMin = 0;
  const int32_t kLimitMax = 2;
  if (fmodu <= kLimitMin || fmodu >= kLimitMax) {
    const int32_t tmp = static_cast<int32_t>(fmodu);
    fmodu -= static_cast<T1>((tmp / kLimitMax) * kLimitMax);
    if (fmodu <= kLimitMin) {
      fmodu += kLimitMax;
    } else if (fmodu >= kLimitMax) {
      fmodu -= kLimitMax;
    }
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

template <typename T>
template <typename T1>
void HSVToRGBCpuKernelMod<T>::ComputeFloat(void *input, void *output, int64_t pixel_num) {
  T1 *input_ptr = reinterpret_cast<T1 *>(input);
  T1 *output_ptr = reinterpret_cast<T1 *>(output);
  auto shard_hsv_to_rgb = [&](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      T1 *h = input_ptr + 3 * i + 0;
      T1 *s = input_ptr + 3 * i + 1;
      T1 *v = input_ptr + 3 * i + 2;
      T1 *r = output_ptr + 3 * i + 0;
      T1 *g = output_ptr + 3 * i + 1;
      T1 *b = output_ptr + 3 * i + 2;
      ConvertOnePixel<T1>(*h, *s, *v, r, g, b);
    }
  };
  CPUKernelUtils::ParallelFor(shard_hsv_to_rgb, pixel_num);
}

template <typename T>
void HSVToRGBCpuKernelMod<T>::ComputeHalf(void *input, void *output, int64_t pixel_num) {
  float16 *input_ptr = reinterpret_cast<float16 *>(input);
  float16 *output_ptr = reinterpret_cast<float16 *>(output);
  auto shard_hsv_to_rgb = [&](size_t start, size_t end) {
    float tmp[3];
    for (size_t i = start; i < end; ++i) {
      float h = static_cast<float>(input_ptr[3 * i + 0]);
      float s = static_cast<float>(input_ptr[3 * i + 1]);
      float v = static_cast<float>(input_ptr[3 * i + 2]);
      ConvertOnePixel<float>(h, s, v, tmp, tmp + 1, tmp + 2);
      output_ptr[3 * i + 0] = float16(tmp[0]);
      output_ptr[3 * i + 1] = float16(tmp[1]);
      output_ptr[3 * i + 2] = float16(tmp[2]);
    }
  };
  CPUKernelUtils::ParallelFor(shard_hsv_to_rgb, pixel_num);
}

template <typename T>
bool HSVToRGBCpuKernelMod<T>::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
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
      break;
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore
