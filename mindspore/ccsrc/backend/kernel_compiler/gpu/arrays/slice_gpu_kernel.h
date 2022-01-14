/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/slice_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kRank1 = 1;
constexpr auto kRank2 = 2;
constexpr auto kRank3 = 3;
constexpr auto kRank4 = 4;
constexpr auto kRank5 = 5;
constexpr auto kRank6 = 6;
constexpr auto kRank7 = 7;

constexpr auto kIdx2 = 2;
constexpr auto kIdx3 = 3;
constexpr auto kIdx4 = 4;
constexpr auto kIdx5 = 5;
constexpr auto kIdx6 = 6;

template <typename T>
class SliceFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  SliceFwdGpuKernelMod()
      : is_null_input_(false), input_size_(0), output_size_(0), workspace_size_(0), kernel_name_("Slice") {}
  ~SliceFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);

    size_t input_rank = input_shape_.size();
    switch (input_rank) {
      case kRank1:
        Slice1DKernel(begin_[0], size_[0], input_shape_[0], input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case kRank2:
        Slice2DKernel(begin_[0], begin_[1], size_[0], size_[1], input_shape_[0], input_shape_[1], input, output,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case kRank3:
        Slice3DKernel(begin_[0], begin_[1], begin_[kIdx2], size_[0], size_[1], size_[kIdx2], input_shape_[0],
                      input_shape_[1], input_shape_[kIdx2], input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case kRank4:
        Slice4DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], size_[0], size_[1], size_[kIdx2],
                      size_[kIdx3], input_shape_[0], input_shape_[1], input_shape_[kIdx2], input_shape_[kIdx3], input,
                      output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case kRank5:
        Slice5DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], begin_[kIdx4], size_[0], size_[1],
                      size_[kIdx2], size_[kIdx3], size_[kIdx4], input_shape_[0], input_shape_[1], input_shape_[kIdx2],
                      input_shape_[kIdx3], input_shape_[kIdx4], input, output,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case kRank6:
        Slice6DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], begin_[kIdx4], begin_[kIdx5], size_[0],
                      size_[1], size_[kIdx2], size_[kIdx3], size_[kIdx4], size_[kIdx5], input_shape_[0],
                      input_shape_[1], input_shape_[kIdx2], input_shape_[kIdx3], input_shape_[kIdx4],
                      input_shape_[kIdx5], input, output, reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      case kRank7:
        Slice7DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], begin_[kIdx4], begin_[kIdx5], begin_[kIdx6],
                      size_[0], size_[1], size_[kIdx2], size_[kIdx3], size_[kIdx4], size_[kIdx5], size_[kIdx6],
                      input_shape_[0], input_shape_[1], input_shape_[kIdx2], input_shape_[kIdx3], input_shape_[kIdx4],
                      input_shape_[kIdx5], input_shape_[kIdx6], input, output,
                      reinterpret_cast<cudaStream_t>(stream_ptr));
        break;
      default:
        MS_LOG(EXCEPTION) << "gpu Slice operator does not support inputs with rank >= " << input_rank << ".";
    }

    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    (void)CheckParam(kernel_node);

    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto out_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(out_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_),
                         [](const int64_t &e) { return static_cast<int32_t>(e); });

    input_size_ = sizeof(T);
    for (size_t x : input_shape) {
      input_size_ *= x;
    }

    output_size_ = sizeof(T);
    for (size_t x : out_shape) {
      output_size_ *= x;
    }

    // transpose begin and size for NHWC data
    auto data_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    if (data_format == "NHWC") {
      std::swap(begin_[1], begin_[kIdx3]);
      std::swap(begin_[1], begin_[kIdx2]);
      std::swap(size_[1], size_[kIdx3]);
      std::swap(size_[1], size_[kIdx2]);
    } else if (data_format == "NDHWC") {
      std::swap(begin_[1], begin_[kIdx4]);
      std::swap(begin_[1], begin_[kIdx3]);
      std::swap(begin_[1], begin_[kIdx2]);
      std::swap(size_[1], size_[kIdx4]);
      std::swap(size_[1], size_[kIdx3]);
      std::swap(size_[1], size_[kIdx2]);
    }

    InitSizeLists();

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    const size_t kInputNumUpperLimit = 7;
    if (input_shape.size() > kInputNumUpperLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than 7, but got "
                        << input_shape.size();
    }
    if (input_shape.size() == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to 0, but got "
                        << input_shape.size();
    }
    auto size = GetAttr<std::vector<int64_t>>(kernel_node, "size");
    auto begin = GetAttr<std::vector<int64_t>>(kernel_node, "begin");

    if (size.size() != input_shape.size() || begin.size() != input_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of size, begin and input_x should be the same, but got the dimension "
                        << "of size: " << size.size() << ", the dimension of begin: " << begin.size()
                        << ", the dimension of input_x: " << input_shape.size();
    }
    const int64_t kDynamicShape = -1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (size[i] == kDynamicShape) {
        size[i] = input_shape[i] - begin[i];
      }
      if (input_shape[i] <= 0 || size[i] <= 0) {
        MS_LOG(WARNING) << "For '" << kernel_name_
                        << "', the element of 'size' and the shape of input_x should be greater than 0, but got "
                        << "size[" << i << "]: " << size[i] << ", input_x.shape[" << i << "] " << input_shape[i];
        is_null_input_ = true;
      }
    }

    (void)std::transform(size.begin(), size.end(), std::back_inserter(size_),
                         [](const int64_t &e) { return static_cast<int32_t>(e); });
    (void)std::transform(begin.begin(), begin.end(), std::back_inserter(begin_),
                         [](const int64_t &e) { return static_cast<int32_t>(e); });
  }

  // use int32_t, a smaller type than the typical size_t, so that we can add higher
  // dimension later on. cuda kernel arguments' total size cannot exceed 256 bytes
  std::vector<int32_t> begin_;
  std::vector<int32_t> size_;
  std::vector<int32_t> input_shape_;

  bool is_null_input_;

  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::string kernel_name_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_ARRAYS_SLICE_GPU_KERNEL_H_
