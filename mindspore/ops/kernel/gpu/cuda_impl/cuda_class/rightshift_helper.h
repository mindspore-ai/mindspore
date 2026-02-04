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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RIGHTSHIFT_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_RIGHTSHIFT_HELPER_H_
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "kernel/gpu/cuda_impl/cuda_class/helper_base.h"
#include "kernel/gpu/cuda_impl/cuda_ops/rightshift_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr int MAX_DIMS = 7;
template <typename T>
class RightShiftHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit RightShiftHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    need_broadcast_ = false;
  }

  virtual ~RightShiftHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    int out_flag =
      CalShapesSizeInBytes<T>(output_shapes, OUTPUT_NUM, kernel_name_, "output_shapes", &output_size_list_);
    if (out_flag == -1) {
      return out_flag;
    }
    is_null_input_ = (HasZeroInShapes(input_shapes) || out_flag == 1);

    auto inputx_shape = input_shapes[0];
    auto inputy_shape = input_shapes[1];
    auto output_shape = output_shapes[0];
    ProcessScalar(&inputx_shape, &inputy_shape, &output_shape);

    for (size_t i = 0; i < std::min(inputx_shape.size(), inputy_shape.size()); i++) {
      if (inputx_shape[i] != inputy_shape[i]) {
        need_broadcast_ = true;
      }
    }

    if (inputx_shape.size() != inputy_shape.size() && need_broadcast_ == false) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the shape of 'inputx_shape' and 'inputy_shape' should be the same or can be broadcast.";
      return -1;
    }

    if (inputx_shape.size() > MAX_DIMS || inputy_shape.size() > MAX_DIMS || output_shape.size() > MAX_DIMS) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the rank of 'inputx_shape', 'inputy_shape' and 'output_shape' should not be greater than "
                    << MAX_DIMS << ", but got " << inputx_shape.size() << ", " << inputy_shape.size() << ", "
                    << output_shape.size();
      return -1;
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    output_num_ = 1;

    // Calculate output_num_ for all cases
    for (size_t i = 0; i < output_shape.size(); i++) {
      output_num_ *= output_shape[i];
    }

    // Only perform broadcast-specific operations if needed
    if (need_broadcast_) {
      // Set output shape for broadcast
      for (size_t i = 0; i < output_shape.size(); i++) {
        output_shape_[i] = output_shape[i];
      }

      // Set left-hand side shape
      int lhs_offset = output_shape.size() - inputx_shape.size();
      for (size_t j = 0; j < inputx_shape.size(); j++) {
        if ((j + lhs_offset) >= 0 && (j + lhs_offset) < MAX_DIMS) {
          lhs_shape_[j + lhs_offset] = inputx_shape[j];
        }
      }

      // Set right-hand side shape
      int rhs_offset = output_shape.size() - inputy_shape.size();
      for (size_t k = 0; k < inputy_shape.size(); k++) {
        if ((k + rhs_offset) >= 0 && (k + rhs_offset) < MAX_DIMS) {
          rhs_shape_[k + rhs_offset] = inputy_shape[k];
        }
      }
    }

    return CheckKernelParam();
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }

    T *inputx_ptr = nullptr;
    T *inputy_ptr = nullptr;
    T *output_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &inputx_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &inputy_ptr);
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
      status = BroadcastRightShift(lhs_shape_, rhs_shape_, output_shape_, inputx_ptr, inputy_ptr, output_ptr,
                                   device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      status = CalRightShift(output_num_, inputx_ptr, inputy_ptr, output_ptr, device_id_,
                             reinterpret_cast<cudaStream_t>(cuda_stream));
    }
    CHECK_CUDA_STATUS(status, kernel_name_);
    return 0;
  }

  void ProcessScalar(std::vector<int64_t> *x1_shape, std::vector<int64_t> *x2_shape, std::vector<int64_t> *y_shape) {
    // If there is a scalar in the inputs, its shape will be [], so it will be treated as [1].
    if (x1_shape->size() == 0) {
      x1_shape->insert(x1_shape->begin(), 1);
    }
    if (x2_shape->size() == 0) {
      x2_shape->insert(x2_shape->begin(), 1);
    }
    if (y_shape->size() == 0) {
      y_shape->insert(y_shape->begin(), 1);
    }
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
