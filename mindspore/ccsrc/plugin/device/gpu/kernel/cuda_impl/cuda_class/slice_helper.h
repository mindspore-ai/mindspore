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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SLICE_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SLICE_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/slice_impl.cuh"

namespace mindspore {
namespace cukernel {
class SliceAttr : public GpuKernelAttrBase {
 public:
  SliceAttr() = default;
  ~SliceAttr() override = default;
  std::vector<int64_t> begin;
  std::vector<int64_t> size;
};

template <typename T, typename S>
class SliceHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit SliceHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {}

  virtual ~SliceHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    std::vector<std::vector<int64_t>> input_tensor_shapes{input_shapes[0]};
    int inp_flag = CalShapesSizeInBytes<T>(input_tensor_shapes, 1, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    constexpr size_t kDynamicSliceInputNum = 3;
    if (input_shapes.size() == kDynamicSliceInputNum) {
      std::vector<std::vector<int64_t>> input_attr_shapes{input_shapes[1], input_shapes[2]};
      int inp_flag = CalShapesSizeInBytes<S>(input_attr_shapes, input_attr_shapes.size(), kernel_name_, "input_shapes",
                                             &input_size_list_);
      if (inp_flag == -1) {
        return inp_flag;
      }
    }
    input_shape_.assign(input_shapes[0].begin(), input_shapes[0].end());
    int out_flag = CalShapesSizeInBytes<T>(output_shapes, 1, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);
    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    T *input = nullptr;
    T *output = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
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
    size_t input_rank = input_shape_.size();
    switch (input_rank) {
      case kRank1:
        Slice1DKernel(begin_[0], size_[0], input_shape_[0], input, output, device_id_,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
        break;
      case kRank2:
        Slice2DKernel(begin_[0], begin_[1], size_[0], size_[1], input_shape_[0], input_shape_[1], input, output,
                      device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
        break;
      case kRank3:
        Slice3DKernel(begin_[0], begin_[1], begin_[kIdx2], size_[0], size_[1], size_[kIdx2], input_shape_[0],
                      input_shape_[1], input_shape_[kIdx2], input, output, device_id_,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
        break;
      case kRank4:
        Slice4DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], size_[0], size_[1], size_[kIdx2],
                      size_[kIdx3], input_shape_[0], input_shape_[1], input_shape_[kIdx2], input_shape_[kIdx3], input,
                      output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
        break;
      case kRank5:
        Slice5DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], begin_[kIdx4], size_[0], size_[1],
                      size_[kIdx2], size_[kIdx3], size_[kIdx4], input_shape_[0], input_shape_[1], input_shape_[kIdx2],
                      input_shape_[kIdx3], input_shape_[kIdx4], input, output, device_id_,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
        break;
      case kRank6:
        Slice6DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], begin_[kIdx4], begin_[kIdx5], size_[0],
                      size_[1], size_[kIdx2], size_[kIdx3], size_[kIdx4], size_[kIdx5], input_shape_[0],
                      input_shape_[1], input_shape_[kIdx2], input_shape_[kIdx3], input_shape_[kIdx4],
                      input_shape_[kIdx5], input, output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
        break;
      case kRank7:
        Slice7DKernel(begin_[0], begin_[1], begin_[kIdx2], begin_[kIdx3], begin_[kIdx4], begin_[kIdx5], begin_[kIdx6],
                      size_[0], size_[1], size_[kIdx2], size_[kIdx3], size_[kIdx4], size_[kIdx5], size_[kIdx6],
                      input_shape_[0], input_shape_[1], input_shape_[kIdx2], input_shape_[kIdx3], input_shape_[kIdx4],
                      input_shape_[kIdx5], input_shape_[kIdx6], input, output, device_id_,
                      reinterpret_cast<cudaStream_t>(cuda_stream));
        break;
      default:
        MS_LOG(EXCEPTION) << "gpu Slice operator does not support inputs with rank >= " << input_rank << ".";
    }
    return 0;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<SliceAttr>(kernel_attr);
  }

 protected:
  int CheckKernelParam() override {
    begin_.assign(attr_ptr_->begin.begin(), attr_ptr_->begin.end());
    size_.assign(attr_ptr_->size.begin(), attr_ptr_->size.end());
    return 0;
  }

 private:
  // use int32_t, a smaller type than the typical size_t, so that we can add higher
  // dimension later on. cuda kernel arguments' total size cannot exceed 256 bytes
  std::vector<int32_t> begin_;
  std::vector<int32_t> size_;
  std::vector<int32_t> input_shape_;

  std::shared_ptr<SliceAttr> attr_ptr_{nullptr};
  bool is_null_input_{false};
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_SLICE_HELPER_H_
