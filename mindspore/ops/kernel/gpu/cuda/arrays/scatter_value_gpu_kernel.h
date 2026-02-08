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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SCATTER_VALUE_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SCATTER_VALUE_GPU_KERNEL_H

#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <utility>
#include <memory>
#include "kernel/gpu/cuda_impl/cuda_ops/scatter_value.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnKnown = "UnKnown";
constexpr auto kScatterValue = "ScatterValue";

class ScatterValueGpuKernelMod : public NativeGpuKernelMod {
 public:
  ScatterValueGpuKernelMod() {}
  ~ScatterValueGpuKernelMod() { FreeResource(); }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 protected:
  using ScatterValueFunc = std::function<bool(ScatterValueGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &, void *)>;
  void MallocResource();
  void FreeResource();
  std::vector<KernelAttr> GetOpSupport() override;
  void GetSize();
  int ShapeCheck();
  int AxisCheck();
  void GetFuncList() const;

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

 private:
  ScatterValueFunc kernel_func_;
  Reduce type_{Reduce::REDUCE_NONE};
  static std::vector<std::pair<KernelAttr, ScatterValueFunc>> func_list_;

  bool sync_resource_ = false;

  std::vector<size_t> src_shape_{};
  std::vector<size_t> indices_shape_{};
  std::vector<size_t> input_shape_{};
  std::vector<size_t> output_shape_{};
  std::vector<size_t> indices_stride_{};
  std::vector<size_t> output_stride_{};

  int64_t axis_{0};
  int input_axis_size_{0};
  size_t input_dims_{0};

  size_t input_byte_size_{1};
  size_t indices_byte_size_{1};

  size_t data_unit_size_{0};    /* sizeof(T) */
  size_t indices_unit_size_{0}; /* sizeof(S) */

  size_t *d_indices_stride_{nullptr};
  size_t *d_output_stride_{nullptr};

  std::string input_name_{"input"};
  std::string axis_name_{"dim"};
  std::string index_name_{"index"};
  std::string src_name_{"src"};
  size_t input_idx_ = kIndex0;
  size_t axis_idx_ = kIndex1;
  size_t index_idx_ = kIndex2;
  size_t src_idx_ = kIndex3;
  size_t reduce_idx_ = kIndex4;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_SCATTER_VALUE_GPU_KERNEL_H
