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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RANDOM_GAMMA_GRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RANDOM_GAMMA_GRAD_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_gamma_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr int MAX_DIMS = 7;
template <typename T>
class RandomGammaGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit RandomGammaGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    need_broadcast_ = false;
  }

  virtual ~RandomGammaGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 2;
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

    auto alpha_shape = input_shapes[0];
    auto sample_shape = input_shapes[1];
    auto output_shape = output_shapes[0];
    auto a_size = std::accumulate(alpha_shape.begin(), alpha_shape.end(), 1, std::multiplies{});
    auto x_size = std::accumulate(sample_shape.begin(), sample_shape.end(), 1, std::multiplies{});
    auto output_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies{});
    if (a_size < output_num || x_size < output_num) {
      need_broadcast_ = true;
    }
    for (size_t i = 0; i < alpha_shape.size(); i++) {
      if (alpha_shape[i] != sample_shape[i]) {
        need_broadcast_ = true;
      }
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    output_num_ = 1;
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (need_broadcast_) {
        output_shape_[i] = output_shape[i];
      }
      output_num_ *= output_shape[i];
    }

    int lhs_offset = output_shape.size() - alpha_shape.size();
    for (size_t j = 0; j < alpha_shape.size(); j++) {
      if (need_broadcast_) {
        if ((j + lhs_offset) >= 0 && (j + lhs_offset) < MAX_DIMS) {
          lhs_shape_[j + lhs_offset] = alpha_shape[j];
        }
      }
    }
    int rhs_offset = output_shape.size() - sample_shape.size();
    for (size_t k = 0; k < sample_shape.size(); k++) {
      if (need_broadcast_) {
        if ((k + rhs_offset) >= 0 && (k + rhs_offset) < MAX_DIMS) {
          rhs_shape_[k + rhs_offset] = sample_shape[k];
        }
      }
    }

    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    T *alpha_ptr = nullptr;
    T *sample_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &alpha_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &sample_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    // call cuda kernel
    if (need_broadcast_) {
      BroadcastRandomGammaGrad(lhs_shape_, rhs_shape_, output_shape_, alpha_ptr, sample_ptr, output_ptr, device_id_,
                               reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      CalRandomGammaGrad(alpha_ptr, sample_ptr, output_ptr, output_num_, device_id_,
                         reinterpret_cast<cudaStream_t>(cuda_stream));
    }

    return 0;
  }

 private:
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
  bool need_broadcast_;
  bool is_null_input_;
  size_t output_num_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RIGHTSHIFT_HELPER_H_
