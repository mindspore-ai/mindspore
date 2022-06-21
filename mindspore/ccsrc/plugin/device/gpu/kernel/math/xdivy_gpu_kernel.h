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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_XDIVY_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_XDIVY_GPU_KERNEL_H
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <map>
#include <utility>
#include <complex>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"

namespace mindspore {
namespace kernel {
class XdivyGpuKernelMod : public NativeGpuKernelMod {
 public:
  XdivyGpuKernelMod() { ResetResource(); }
  ~XdivyGpuKernelMod() = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    T *lhs = GetDeviceAddress<T>(inputs, 0);
    T *rhs = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);
    if (need_broadcast_) {
      BroadcastArith(lhs_shape_, rhs_shape_, output_shape_, BROADCAST_TYPE_XDIVY, lhs, rhs, output,
                     reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      ElewiseArith(out_ele_num_, BROADCAST_TYPE_XDIVY, lhs, rhs, output, reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  template <typename T>
  bool LaunchKernelComplex(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    T *lhs = GetDeviceAddress<T>(inputs, 0);
    T *rhs = GetDeviceAddress<T>(inputs, 1);
    T *output = GetDeviceAddress<T>(outputs, 0);

    if (need_broadcast_) {
      BroadcastComplexArith(lhs_shape_, rhs_shape_, output_shape_, BROADCAST_TYPE_DIV, lhs, rhs, output,
                            reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      ElewiseComplexArith(out_ele_num_, BROADCAST_TYPE_DIV, lhs, rhs, output,
                          reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }
  std::vector<KernelAttr> GetOpSupport() override;

  void ResetResource() noexcept {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    x_ele_num_ = 1;
    y_ele_num_ = 1;
    out_ele_num_ = 1;
    lhs_shape_.clear();
    rhs_shape_.clear();
    output_shape_.clear();
  }

 protected:
  static std::vector<KernelAttr> support_ops_;
  using XdivyFunc = std::function<bool(XdivyGpuKernelMod *, const std::vector<AddressPtr> &inputs,
                                       const std::vector<AddressPtr> &workspace, const std::vector<AddressPtr> &outputs,
                                       void *stream_ptr)>;
  static std::map<mindspore::TypeId, XdivyFunc> func_map_;
  XdivyFunc kernel_func_;
  static const size_t INPUT_NUM = 2;
  static const size_t OUTPUT_NUM = 1;
  static const int MAX_DIMS = 7;
  size_t x_ele_num_ = 1;
  size_t y_ele_num_ = 1;
  size_t out_ele_num_ = 1;
  std::vector<size_t> lhs_shape_;
  std::vector<size_t> rhs_shape_;
  std::vector<size_t> output_shape_;
  bool need_broadcast_ = false;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_XDIVY_GPU_KERNEL_H
