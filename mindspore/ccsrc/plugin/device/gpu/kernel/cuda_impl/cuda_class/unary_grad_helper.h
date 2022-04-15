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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNARY_GRAD_HELPER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNARY_GRAD_HELPER_H_
#include <string>
#include <vector>
#include <map>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_class/helper_base.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_grad_impl.cuh"

namespace mindspore {
namespace cukernel {
constexpr size_t INPUT_NUM = 2;

enum UnaryGradOpType {
  UNARY_OP_SQRT_GRAD = 0,
  UNARY_OP_RSQRT_GRAD,
  UNARY_OP_ASIN_GRAD,
  UNARY_OP_ACOS_GRAD,
  UNARY_OP_ATAN_GRAD,
  UNARY_OP_ASINH_GRAD,
  UNARY_OP_ACOSH_GRAD,
  UNARY_OP_RECIPROCAL_GRAD,
  UNARY_OP_INV_GRAD,
  UNARY_OP_GRAD_INVALID_TYPE = 255
};

static const std::map<std::string, UnaryGradOpType> kUnaryGradOpTypeMap = {
  {"SqrtGrad", UNARY_OP_SQRT_GRAD},   {"RsqrtGrad", UNARY_OP_RSQRT_GRAD},
  {"AsinGrad", UNARY_OP_ASIN_GRAD},   {"ACosGrad", UNARY_OP_ACOS_GRAD},
  {"AtanGrad", UNARY_OP_ATAN_GRAD},   {"AsinhGrad", UNARY_OP_ASINH_GRAD},
  {"AcoshGrad", UNARY_OP_ACOSH_GRAD}, {"ReciprocalGrad", UNARY_OP_RECIPROCAL_GRAD},
  {"InvGrad", UNARY_OP_INV_GRAD}};

template <typename T>
class UnaryGradHelperGpuKernel : public GpuKernelHelperBase {
 public:
  explicit UnaryGradHelperGpuKernel(const std::string &kernel_name, const uint32_t &device_id)
      : GpuKernelHelperBase(kernel_name, device_id_) {
    unary_grad_op_type_ = UNARY_OP_GRAD_INVALID_TYPE;
  }
  virtual ~UnaryGradHelperGpuKernel() = default;
  int CalMemSize(const std::vector<std::vector<int64_t>> &input_shapes,
                 const std::vector<std::vector<int64_t>> &output_shapes) override {
    auto iter = kUnaryGradOpTypeMap.find(kernel_name_);
    if (iter == kUnaryGradOpTypeMap.end()) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: SqrtGrad, RsqrtGrad, AsinGrad, "
                        << "ACosGrad, AtanGrad, AsinhGrad, AcoshGrad, ReciprocalGrad, InvGrad currently, but got "
                        << kernel_name_;
      return -1;
    }
    unary_grad_op_type_ = iter->second;
    int flag = CalShapesSizeInBytes<T>(input_shapes, INPUT_NUM, kernel_name_, "input_shapes", &input_size_list_);
    output_size_list_.emplace_back(input_size_list_[0]);
    is_null_input_ = (flag == 1);
    if (flag != 0) {
      return flag;
    }
    return 0;
  }

  int Process(const std::vector<void *> &input_ptrs, const std::vector<void *> &output_ptrs,
              const std::vector<void *> &work_ptrs, void *cuda_stream) override {
    if (is_null_input_) {
      return 0;
    }
    static std::map<UnaryGradOpType, std::function<void(const T *, const T *, T *, const size_t, cudaStream_t)>>
      func_map = {{UNARY_OP_SQRT_GRAD, SqrtGrad<T>},   {UNARY_OP_RSQRT_GRAD, RsqrtGrad<T>},
                  {UNARY_OP_ASIN_GRAD, AsinGrad<T>},   {UNARY_OP_ACOS_GRAD, ACosGrad<T>},
                  {UNARY_OP_ATAN_GRAD, AtanGrad<T>},   {UNARY_OP_ASINH_GRAD, AsinhGrad<T>},
                  {UNARY_OP_ACOSH_GRAD, AcoshGrad<T>}, {UNARY_OP_RECIPROCAL_GRAD, ReciprocalGrad<T>},
                  {UNARY_OP_INV_GRAD, InvGrad<T>}};

    auto iter = func_map.find(unary_grad_op_type_);
    if (iter != func_map.end()) {
      T *input_x_addr;
      T *input_dx_addr;
      T *output_y_addr;
      int flag = GetDeviceAddress<T>(input_ptrs, 0, kernel_name_, &input_x_addr);
      if (flag != 0) {
        return flag;
      }
      flag = GetDeviceAddress<T>(input_ptrs, 1, kernel_name_, &input_dx_addr);
      if (flag != 0) {
        return flag;
      }
      flag = GetDeviceAddress<T>(output_ptrs, 0, kernel_name_, &output_y_addr);
      if (flag != 0) {
        return flag;
      }
      iter->second(input_x_addr, input_dx_addr, output_y_addr, input_size_list_[0] / sizeof(T),
                   reinterpret_cast<cudaStream_t>(cuda_stream));
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", only support these types: SqrtGrad, RsqrtGrad, AsinGrad, "
                        << "ACosGrad, AtanGrad, AsinhGrad, AcoshGrad, ReciprocalGrad, InvGrad currently, but got "
                        << unary_grad_op_type_;
      return -1;
    }

    return 0;
  }

 private:
  UnaryGradOpType unary_grad_op_type_;
  bool is_null_input_;
};
}  // namespace cukernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_CUDA_IMPL_CUDA_CLASS_UNARY_GRAD_HELPER_H_
