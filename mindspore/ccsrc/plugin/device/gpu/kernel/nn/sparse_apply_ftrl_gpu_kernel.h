/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_FTRL_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_FTRL_GPU_KERNEL_H_

#include <vector>
#include <map>
#include "ops/sparse_apply_ftrl.h"
#include "utils/check_convert_utils.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_ftrl_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t INPUT_NUM = 5;
template <typename T, typename S>
class SparseFtrlGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseFtrlGpuKernelMod() { ResetResource(); }
  ~SparseFtrlGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    T *variable = GetDeviceAddress<T>(inputs, 0);
    T *accumulation = GetDeviceAddress<T>(inputs, 1);
    T *linear = GetDeviceAddress<T>(inputs, 2);
    T *gradient = GetDeviceAddress<T>(inputs, 3);
    S *indices = GetDeviceAddress<S>(inputs, 4);
    T *variable_out = GetDeviceAddress<T>(outputs, 0);
    T *accumulation_out = GetDeviceAddress<T>(outputs, 1);
    T *linear_out = GetDeviceAddress<T>(outputs, 2);
    auto status = CalSparseApplyFtrl(gradient, indices, num_index_, n_stride_, lr_, l1_, l2_, lr_power_, use_locking_,
                                     variable, accumulation, linear, reinterpret_cast<cudaStream_t>(stream_ptr));
    CHECK_CUDA_STATUS(status, kernel_name_);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(variable_out, variable, variable_size_, cudaMemcpyDeviceToDevice,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cudaMemcpyAsync output failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(accumulation_out, accumulation, accumulation_size_, cudaMemcpyDeviceToDevice,
                      reinterpret_cast<cudaStream_t>(stream_ptr)),
      "cudaMemcpyAsync output failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(linear_out, linear, linear_size_, cudaMemcpyDeviceToDevice,
                                                       reinterpret_cast<cudaStream_t>(stream_ptr)),
                                       "cudaMemcpyAsync output failed");
    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) {
    MS_EXCEPTION_IF_NULL(base_operator);
    kernel_name_ = base_operator->name();
    auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseApplyFtrl>(base_operator);
    MS_EXCEPTION_IF_NULL(kernel_ptr);
    lr_ = kernel_ptr->get_lr();
    l1_ = kernel_ptr->get_l1();
    l2_ = kernel_ptr->get_l2();
    lr_power_ = kernel_ptr->get_lr_power();
    use_locking_ = kernel_ptr->get_use_locking();
    return true;
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) {
    if (auto ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
      return ret;
    }

    (void)CheckAndConvertUtils::CheckInteger("input num", inputs.size(), kEqual, INPUT_NUM, kernel_name_);

    variable_size_ = sizeof(T);
    accumulation_size_ = sizeof(T);
    linear_size_ = sizeof(T);
    n_stride_ = 1;

    auto variable_shape = inputs.at(kIndex0)->GetShapeVector();
    auto accumulation_shape = inputs.at(kIndex1)->GetShapeVector();
    auto linear_shape = inputs.at(kIndex2)->GetShapeVector();
    auto indices_shape = inputs.at(kIndex4)->GetShapeVector();

    for (size_t i = 0; i < variable_shape.size(); i++) {
      variable_size_ *= variable_shape[i];
      if (i > 0) {
        n_stride_ *= variable_shape[i];
      }
    }
    accumulation_size_ *= SizeOf(accumulation_shape);
    linear_size_ *= SizeOf(linear_shape);
    num_index_ = indices_shape[0];

    return KRET_OK;
  }

 protected:
  void ResetResource() noexcept {
    lr_ = 0.0f;
    l1_ = 0.0f;
    l2_ = 0.0f;
    lr_power_ = 0.0f;
    use_locking_ = false;
    num_index_ = 0;
    n_stride_ = 1;
  }

 private:
  size_t variable_size_;
  size_t accumulation_size_;
  size_t linear_size_;
  float lr_;
  float l1_;
  float l2_;
  float lr_power_;
  bool use_locking_;
  int num_index_;
  size_t n_stride_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SPARSE_FTRL_GPU_KERNEL_H_
