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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LERP_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LERP_HELPER_H_
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lerp_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr int MAX_DIMS = 7;
template <typename T, typename S>
class LerpHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit LerpHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    weight_is_float_ = false;
    need_broadcast_ = false;
  }

  virtual ~LerpHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    ResetResource();
    constexpr int weight_pos = 2;
    std::vector<int64_t> inputx_shape = input_shapes[0];
    std::vector<int64_t> inputy_shape = input_shapes[1];
    std::vector<int64_t> inputz_shape = input_shapes[weight_pos];
    std::vector<int64_t> output_shape = output_shapes[0];
    weight_is_float_ = inputz_shape.size() == 0 ? true : false;
    std::vector<std::vector<int64_t>> input_tensor_shapes{input_shapes[0], input_shapes[1]};
    int inp_flag = CalShapesSizeInBytes<T>(input_tensor_shapes, 2, kernel_name_, "input_shapes", &input_size_list_);
    if (inp_flag == -1) {
      return inp_flag;
    }
    int inp_flag1 = 0;
    if (weight_is_float_) {
      size_t weight_size = sizeof(S);
      input_size_list_.emplace_back(weight_size);
    } else {
      std::vector<std::vector<int64_t>> input_weight_shapes{input_shapes[2]};
      int inp_flag1 =
        CalShapesSizeInBytes<S>(input_weight_shapes, 1, kernel_name_, "input_weight_shapes", &input_size_list_);
      if (inp_flag1 == -1) {
        return inp_flag1;
      }
    }
    int out_flag = CalShapesSizeInBytes<T>(output_shapes, 1, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (inp_flag == 1 || inp_flag1 == 1 || out_flag == 1);
    Broadcast(inputx_shape, inputy_shape, inputz_shape);
    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    whs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    output_num_ = 1;
    for (size_t i = 0; i < output_shape.size(); i++) {
      if (need_broadcast_) {
        output_shape_[i] = output_shape[i];
      }
      output_num_ *= output_shape[i];
    }
    CalLhs(inputx_shape, output_shape);
    CalRhs(inputy_shape, output_shape);
    CalWhs(inputz_shape, output_shape);
    return CheckKernelParam();
  }
  void Broadcast(const std::vector<int64_t> &start_shape, const std::vector<int64_t> &end_shape,
                 const std::vector<int64_t> &weight_shape) {
    size_t min_xy_size_ = start_shape.size() > end_shape.size() ? end_shape.size() : start_shape.size();
    for (size_t i = 0; i < min_xy_size_; i++) {
      if (start_shape[i] != end_shape[i]) {
        need_broadcast_ = true;
        break;
      }
    }
    if (start_shape.size() != end_shape.size()) {
      need_broadcast_ = true;
    }
    if (!weight_is_float_) {
      size_t min_size_ = weight_shape.size() > min_xy_size_ ? min_xy_size_ : weight_shape.size();
      for (size_t i = 0; i < min_size_; i++) {
        if (start_shape[i] != weight_shape[i]) {
          need_broadcast_ = true;
          break;
        }
      }
    }
    if (!weight_is_float_ && start_shape.size() != weight_shape.size()) {
      need_broadcast_ = true;
    }
  }
  void CalLhs(const std::vector<int64_t> &inputx_shape, const std::vector<int64_t> &output_shape) {
    int lhs_offset = output_shape.size() - inputx_shape.size();
    for (size_t n = 0; n < inputx_shape.size(); n++) {
      if (need_broadcast_) {
        if ((n + lhs_offset) >= 0 && (n + lhs_offset) < MAX_DIMS) {
          lhs_shape_[n + lhs_offset] = inputx_shape[n];
        }
      }
    }
  }
  void CalRhs(const std::vector<int64_t> &inputy_shape, const std::vector<int64_t> &output_shape) {
    int rhs_offset = output_shape.size() - inputy_shape.size();
    for (size_t n = 0; n < inputy_shape.size(); n++) {
      if (need_broadcast_) {
        if ((n + rhs_offset) >= 0 && (n + rhs_offset) < MAX_DIMS) {
          rhs_shape_[n + rhs_offset] = inputy_shape[n];
        }
      }
    }
  }
  void CalWhs(const std::vector<int64_t> &inputz_shape, const std::vector<int64_t> &output_shape) {
    int whs_offset = output_shape.size() - inputz_shape.size();
    for (size_t n = 0; n < inputz_shape.size(); n++) {
      if (need_broadcast_) {
        if ((n + whs_offset) >= 0 && (n + whs_offset) < MAX_DIMS) {
          whs_shape_[n + whs_offset] = inputz_shape[n];
        }
      }
    }
  }
  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    T *inputstart_ptr = nullptr;
    T *inputend_ptr = nullptr;
    S *inputweight_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &inputstart_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &inputend_ptr);
    if (flag != 0) {
      return flag;
    }

    constexpr int weight_pos = 2;
    flag = GetDeviceAddress<S>(input_ptrs, weight_pos, kernel_name_, &inputweight_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_ptr);
    if (flag != 0) {
      return flag;
    }

    cudaError_t status = cudaErrorNotReady;
    // call cuda kernel
    if (need_broadcast_) {
      if (weight_is_float_) {
        status =
          BroadcastLerpWeightFloat(lhs_shape_, rhs_shape_, output_shape_, inputstart_ptr, inputend_ptr, inputweight_ptr,
                                   output_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
      } else {
        status = BroadcastLerpWeightTensor(lhs_shape_, rhs_shape_, whs_shape_, output_shape_, inputstart_ptr,
                                           inputend_ptr, inputweight_ptr, output_ptr, device_id_,
                                           reinterpret_cast<cudaStream_t>(cuda_stream));
      }
    } else {
      if (weight_is_float_) {
        status = LerpWeightFloat(output_num_, inputstart_ptr, inputend_ptr, inputweight_ptr, output_ptr, device_id_,
                                 reinterpret_cast<cudaStream_t>(cuda_stream));
      } else {
        status = LerpWeightTensor(output_num_, inputstart_ptr, inputend_ptr, inputweight_ptr, output_ptr, device_id_,
                                  reinterpret_cast<cudaStream_t>(cuda_stream));
      }
    }
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

 private:
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> whs_shape_;
  std::vector<size_t> output_shape_;
  size_t output_num_;
  bool weight_is_float_;
  bool need_broadcast_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_LERP_HELPER_H_
