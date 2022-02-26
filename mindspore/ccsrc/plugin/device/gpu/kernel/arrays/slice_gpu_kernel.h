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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"

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
  SliceFwdGpuKernelMod() { kernel_name_ = "Slice"; }
  ~SliceFwdGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }

    T *input = GetDeviceAddress<T>(inputs, 0);
    T *output = GetDeviceAddress<T>(outputs, 0);
    if (is_dynamic_attr_ && !get_dynamic_attr_value_) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', fail to get value of the dynamic attr!";
    }

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
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    (void)CheckParam(kernel_node);

    auto input_shape = AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, 0);
    auto out_shape = AnfAlgo::GetOutputRealDeviceShapeIfExist(kernel_node, 0);
    is_null_input_ =
      CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(out_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_),
                         [](const int64_t &e) { return static_cast<int32_t>(e); });

    size_t input_size = sizeof(T);
    for (size_t x : input_shape) {
      input_size *= x;
    }
    input_size_list_.push_back(input_size);
    if (is_dynamic_attr_) {
      std::vector<size_t> dynamic_attr_indexs = {kBeginIndex_, kSizeIndex_};
      for (size_t index : dynamic_attr_indexs) {
        input_size = sizeof(T);
        for (size_t x : AnfAlgo::GetInputRealDeviceShapeIfExist(kernel_node, index)) {
          input_size *= x;
        }
        input_size_list_.push_back(input_size);
      }
    }

    size_t output_size = sizeof(T);
    for (size_t x : out_shape) {
      output_size *= x;
    }
    output_size_list_.push_back(output_size);

    // transpose begin and size for NHWC data
    auto data_format = AnfAlgo::GetInputFormat(kernel_node, 0);
    if (data_format == kOpFormat_NHWC) {
      std::swap(begin_[1], begin_[kIdx3]);
      std::swap(begin_[1], begin_[kIdx2]);
      std::swap(size_[1], size_[kIdx3]);
      std::swap(size_[1], size_[kIdx2]);
    } else if (data_format == kOpFormat_NDHWC) {
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

  void ResetResource() noexcept override {
    ResetSizeLists();
    begin_.clear();
    size_.clear();
    input_shape_.clear();
    is_null_input_ = false;
    kernel_name_ = "Slice";
    is_dynamic_attr_ = false;
    get_dynamic_attr_value_ = false;
  }

 protected:
  void InitSizeLists() override {}

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    constexpr size_t kDynamicSliceInputNum = 3;
    if (input_num != 1 && input_num != kDynamicSliceInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 1 or " << kDynamicSliceInputNum
                        << ", but got " << input_num;
    }
    if (input_num == kDynamicSliceInputNum) {
      is_dynamic_attr_ = true;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
    auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    const size_t kInputNumUpperLimit = 7;
    if (input_shape.size() > kInputNumUpperLimit) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be greater than 7, but got "
                        << input_shape.size();
    }
    if (input_shape.size() == 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of input cannot be equal to 0, but got "
                        << input_shape.size();
    }
    std::vector<int64_t> size, begin;
    if (!is_dynamic_attr_) {
      size = GetAttr<std::vector<int64_t>>(kernel_node, "size");
      begin = GetAttr<std::vector<int64_t>>(kernel_node, "begin");
    } else {
      // The value of dynamic attr can only be obtained after the InferShape() of dynamic kernel is executed
      if (DynamicKernel() == nullptr) {
        return;
      }
      begin = GetDynamicAttrIntValue(kernel_node, kBeginIndex_);
      size = GetDynamicAttrIntValue(kernel_node, kSizeIndex_);
      get_dynamic_attr_value_ = true;
    }

    if (size.size() != input_shape.size() || begin.size() != input_shape.size()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of size, begin and input_x should be the same, but got the dimension "
                        << "of size: " << size.size() << ", the dimension of begin: " << begin.size()
                        << ", the dimension of input_x: " << input_shape.size();
    }
    const int64_t NEG_ONE = -1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (size[i] == NEG_ONE) {
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

  bool is_null_input_{false};
  bool is_dynamic_attr_{false};
  bool get_dynamic_attr_value_{false};
  static constexpr size_t kBeginIndex_{1};
  static constexpr size_t kSizeIndex_{2};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SLICE_GPU_KERNEL_H_
