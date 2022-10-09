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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_GCD_LCM_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_GCD_LCM_HELPER_H_
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gcd_lcm_impl.cuh"

namespace mindspore {
namespace cukernel {
enum GcdLcmOptype { GCD_OP = 0, LCM_OP = 1, OP_INVALID_TYPE = 2 };

static const std::map<std::string, GcdLcmOptype> kGcdLcmOpTypeMap = {{"Gcd", GCD_OP}, {"Lcm", LCM_OP}};

constexpr int MAX_DIMS = 7;
template <typename T>
class GcdLcmHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit GcdLcmHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id) {
    is_null_input_ = false;
    need_broadcast_ = false;
    gcd_lcm_op_type_ = OP_INVALID_TYPE;
  }

  virtual ~GcdLcmHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    constexpr size_t INPUT_NUM = 2;
    constexpr size_t OUTPUT_NUM = 1;
    ResetResource();
    auto iter = kGcdLcmOpTypeMap.find(kernel_name_);
    if (iter == kGcdLcmOpTypeMap.end()) {
      MS_LOG(ERROR) << "For 'GcdLcmOp', only support these types: " << kernel::Map2Str(kGcdLcmOpTypeMap)
                    << " currently, but got " << kernel_name_;
      return -1;
    }
    gcd_lcm_op_type_ = iter->second;
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

    auto x1_shape = input_shapes[0];
    auto x2_shape = input_shapes[1];
    auto y_shape = output_shapes[0];

    ProcessScalar(&x1_shape, &x2_shape, &y_shape);

    for (size_t i = 0; i < x1_shape.size(); i++) {
      if (x1_shape[i] != x2_shape[i]) {
        need_broadcast_ = true;
      }
    }

    lhs_shape_.resize(MAX_DIMS, 1);
    rhs_shape_.resize(MAX_DIMS, 1);
    output_shape_.resize(MAX_DIMS, 1);
    output_num_ = 1;
    for (size_t i = 0; i < y_shape.size(); i++) {
      if (need_broadcast_) {
        output_shape_[i] = y_shape[i];
      }
      output_num_ *= y_shape[i];
    }
    int lhs_offset = y_shape.size() - x1_shape.size();
    for (size_t j = 0; j < x1_shape.size(); j++) {
      if (need_broadcast_) {
        if ((j + lhs_offset) >= 0 && (j + lhs_offset) < MAX_DIMS) {
          lhs_shape_[j + lhs_offset] = x1_shape[j];
        }
      }
    }
    int rhs_offset = y_shape.size() - x2_shape.size();
    for (size_t k = 0; k < x2_shape.size(); k++) {
      if (need_broadcast_) {
        if ((k + rhs_offset) >= 0 && (k + rhs_offset) < MAX_DIMS) {
          rhs_shape_[k + rhs_offset] = x2_shape[k];
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
    T *x1_ptr = nullptr;
    T *x2_ptr = nullptr;
    T *y_ptr = nullptr;
    int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &x1_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &x2_ptr);
    if (flag != 0) {
      return flag;
    }

    flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &y_ptr);
    if (flag != 0) {
      return flag;
    }
    // call cuda kernel
    if (kernel_name_.find("Gcd") != std::string::npos) {
      if (need_broadcast_) {
        BroadcastGcd(lhs_shape_, rhs_shape_, output_shape_, x1_ptr, x2_ptr, y_ptr, device_id_,
                     reinterpret_cast<cudaStream_t>(cuda_stream));
      } else {
        CalGcd(output_num_, x1_ptr, x2_ptr, y_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
      }
    } else if (kernel_name_.find("Lcm") != std::string::npos) {
      if (need_broadcast_) {
        BroadcastLcm(lhs_shape_, rhs_shape_, output_shape_, x1_ptr, x2_ptr, y_ptr, device_id_,
                     reinterpret_cast<cudaStream_t>(cuda_stream));
      } else {
        CalLcm(output_num_, x1_ptr, x2_ptr, y_ptr, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
      }
    } else {
      MS_LOG(ERROR) << "For 'GcdLcmOp', only support these types: " << kernel::Map2Str(kGcdLcmOpTypeMap)
                    << " currently, but got " << kernel_name_;
      return -1;
    }

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
  GcdLcmOptype gcd_lcm_op_type_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_GCD_LCM_HELPER_H_
