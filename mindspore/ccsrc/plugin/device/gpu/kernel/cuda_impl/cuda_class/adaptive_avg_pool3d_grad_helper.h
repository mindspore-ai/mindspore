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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_AVG_POOL3D_GRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_AVG_POOL3D_GRAD_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/adaptive_avg_pool3d_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
template <typename T>
class AdaptiveAvgPool3DGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit AdaptiveAvgPool3DGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
  }

  virtual ~AdaptiveAvgPool3DGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();

    std::vector<int64_t> in_input_shape_ = input_shapes[0];
    std::vector<int64_t> in_orig_input_size_shape_ = input_shapes[1];
    int inp_flap = 0;
    size_t cur_size_T = sizeof(T);
    for (const auto &val : in_input_shape_) {
      cur_size_T *= val;
    }
    if (cur_size_T == 0 && inp_flap == 0) {
      inp_flap = 1;
    }
    input_size_list_.emplace_back(cur_size_T);

    size_t cur_size = sizeof(int32_t);
    if (in_orig_input_size_shape_[0] == 0 && inp_flap == 0) {
      inp_flap = 1;
    }
    input_size_list_.emplace_back(cur_size * in_orig_input_size_shape_[0]);

    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flap == 1 || out_flag == 1);

    if (is_null_input_) {
      return -1;
    }

    constexpr int INPUT_SHAPE_SIZE = 3;
    constexpr int OUTPUT_SHAPE_SIZE = 3;
    auto input_rank = input_shapes[0].size();
    auto output_rank = output_shapes[0].size();

    if (input_rank < INPUT_SHAPE_SIZE) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of input cannot be less than 3, but got "
                    << input_rank;
      return -1;
    }

    constexpr int DEPTH = 1;
    constexpr int WIDTH = 2;
    constexpr int HEIGHT = 3;
    constexpr int CHANNEL = 4;
    constexpr int DIMENSION = 4;
    input_channel = input_shapes[0][input_rank - CHANNEL];
    input_height = input_shapes[0][input_rank - HEIGHT];
    input_width = input_shapes[0][input_rank - WIDTH];
    input_depth = input_shapes[0][input_rank - DEPTH];
    output_channel = output_shapes[0][output_rank - CHANNEL];
    output_height = output_shapes[0][output_rank - HEIGHT];
    output_width = output_shapes[0][output_rank - WIDTH];
    output_depth = output_shapes[0][output_rank - DEPTH];
    in_size = input_rank == DIMENSION
                ? input_shapes[0][0] * input_height * input_width * input_depth
                : input_shapes[0][0] * input_shapes[0][1] * input_height * input_width * input_depth;
    out_size = output_rank == DIMENSION
                 ? output_shapes[0][0] * output_height * output_width * output_depth
                 : output_shapes[0][0] * output_shapes[0][1] * output_height * output_width * output_depth;

    if (output_rank < OUTPUT_SHAPE_SIZE) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of output cannot be less than 3, but got "
                    << output_rank;
      return -1;
    }

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

    auto status = ApplyAdaptiveAvgPool3DGrad((uint)in_size, (uint)out_size, (uint)input_channel, (uint)input_height,
                                             (uint)input_width, (uint)input_depth, (uint)output_channel,
                                             (uint)output_height, (uint)output_width, (uint)output_depth, input_ptr,
                                             output_ptr, reinterpret_cast<cudaStream_t>(cuda_stream));
    return status;
  }

 private:
  std::vector<int64_t> input_shape_;
  int64_t input_channel{0};
  int64_t input_height{0};
  int64_t input_width{0};
  int64_t input_depth{0};
  int64_t output_channel{0};
  int64_t output_height{0};
  int64_t output_width{0};
  int64_t output_depth{0};
  int64_t in_size{0};
  int64_t out_size{0};
  bool is_null_input_{false};
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_ADAPTIVE_AVG_POOL3D_GRAD_HELPER_H_
