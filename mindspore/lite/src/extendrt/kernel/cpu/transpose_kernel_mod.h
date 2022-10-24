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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_CPU_TRANSPOSE_KERNEL_MOD_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_CPU_TRANSPOSE_KERNEL_MOD_H_

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "plugin/device/cpu/kernel/nnacl/transpose.h"
#include "kernel/common_utils.h"

namespace mindspore::kernel {
class TransposeKernelMod : public CpuKernelMod {
 public:
  TransposeKernelMod() = default;
  ~TransposeKernelMod() override = default;

  explicit TransposeKernelMod(const std::string name) { kernel_name_ = name; }

  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr);

  virtual bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                    const std::vector<KernelTensorPtr> &outputs);

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  std::vector<KernelAttr> GetOpSupport() override { return {}; }

 private:
  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename T>
  int DoTranspose(const T *in_data, T *out_data, const int *output_shape, const TransposeParameter *transpose_param);
  template <typename T>
  void TransposeDim2(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim3(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim4(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim5(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim6(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDim7(const T *in_data, T *out_data, const int *strides, const int *out_strides, const int *perm,
                     const int *output_shape);
  template <typename T>
  void TransposeDims(const T *in_data, T *out_data, const int *output_shape, const TransposeParameter *transpose_param,
                     int task_id, int thread_num);

  TransposeParameter transpose_param_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<size_t> axes_;
  TypeId dtype_{kTypeUnknown};
  using TypeKernel =
    std::function<void(TransposeKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  std::unordered_map<TypeId, TypeKernel> launch_map_;
  TypeKernel launch_func_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_CPU_TRANSPOSE_KERNEL_MOD_H_
