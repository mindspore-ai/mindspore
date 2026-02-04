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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_LUUNPACK_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_LUUNPACK_GPU_KERNEL_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include "mindspore/ops/infer/lu_unpack.h"
#include "kernel/gpu/gpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/lu_unpack_impl.cuh"
#include "utils/check_convert_utils.h"

namespace mindspore {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 3;
namespace kernel {
class LuUnpackGpuKernelMod : public NativeGpuKernelMod {
 public:
  LuUnpackGpuKernelMod() { ResetResource(); }
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using LuUnpackFunc =
    std::function<bool(LuUnpackGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  LuUnpackFunc kernel_func_;
  void *cuda_stream_{nullptr};
  static std::vector<std::pair<KernelAttr, LuUnpackFunc>> func_list_;

  size_t lu_data_size_{0};
  size_t lu_pivots_size_{0};
  int64_t lu_data_dim1_{0};
  int64_t lu_data_dim2_{0};
  int64_t l_stride_{0};
  int64_t u_stride_{0};
  int64_t lu_pivots_dim_{0};
  int64_t batch_num_{0};
  size_t unit_size1_{0};
  size_t unit_size2_{0};

  void ResetResource() noexcept;
  template <typename T_data, typename T_pivots>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_LUUNPACK_GPU_KERNEL_H_
