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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_AVG_POOL3D_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_AVG_POOL3D_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool3d_impl.cuh"

namespace mindspore {
namespace cukernel {
class AdaptiveAvgPool3DAttr : public GpuKernelAttrBase {
 public:
  AdaptiveAvgPool3DAttr() = default;
  ~AdaptiveAvgPool3DAttr() override = default;
  std::vector<int64_t> output_size;
};

template <typename T>
class AdaptiveAvgPool3DHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit AdaptiveAvgPool3DHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~AdaptiveAvgPool3DHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 1;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();

    int inp_flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || out_flag == 1);

    constexpr int INPUT_SHAPE_SIZE = 3;
    constexpr int OUTPUT_SHAPE_SIZE = 3;
    auto input_rank = input_shapes[0].size();
    auto output_rank = output_shapes[0].size();
    if (input_rank < INPUT_SHAPE_SIZE) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 3, but got "
                    << input_rank;
      return -1;
    }
    if (output_rank < OUTPUT_SHAPE_SIZE) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 3, but got "
                    << output_rank;
      return -1;
    }

    constexpr int DEPTH = 1;
    constexpr int WIDTH = 2;
    constexpr int HEIGHT = 3;
    constexpr int CHANNEL = 4;
    constexpr int DIMENSION = 4;
    input_channel_ = input_shapes[0][input_rank - CHANNEL];
    input_height_ = input_shapes[0][input_rank - HEIGHT];
    input_width_ = input_shapes[0][input_rank - WIDTH];
    input_depth_ = input_shapes[0][input_rank - DEPTH];
    output_channel_ = output_shapes[0][output_rank - CHANNEL];
    output_height_ = output_shapes[0][output_rank - HEIGHT];
    output_width_ = output_shapes[0][output_rank - WIDTH];
    output_depth_ = output_shapes[0][output_rank - DEPTH];
    out_size_ = output_rank == DIMENSION
                  ? output_shapes[0][0] * output_height_ * output_width_ * output_depth_
                  : output_shapes[0][0] * output_shapes[0][1] * output_height_ * output_width_ * output_depth_;

    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    T *input_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    auto status =
      ApplyAdaptiveAvgPool3D((uint)out_size_, (uint)input_channel_, (uint)input_height_, (uint)input_width_,
                             (uint)input_depth_, (uint)output_channel_, (uint)output_height_, (uint)output_width_,
                             (uint)output_depth_, input_ptr, output_ptr, reinterpret_cast<cudaStream_t>(cuda_stream));
    return status;
  }

  void SetKernelParam(const GpuKernelAttrBasePtr &kernel_attr) override {
    attr_ptr_ = std::dynamic_pointer_cast<AdaptiveAvgPool3DAttr>(kernel_attr);
  }

 private:
  std::shared_ptr<AdaptiveAvgPool3DAttr> attr_ptr_;
  std::vector<int64_t> input_shape_;
  int64_t len_{0};
  int64_t input_channel_{0};
  int64_t input_height_{0};
  int64_t input_width_{0};
  int64_t input_depth_{0};
  int64_t output_channel_{0};
  int64_t output_height_{0};
  int64_t output_width_{0};
  int64_t output_depth_{0};
  int64_t in_size_{0};
  int64_t out_size_{0};
  bool is_null_input_{false};
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_AVG_POOL3D_HELPER_H_
