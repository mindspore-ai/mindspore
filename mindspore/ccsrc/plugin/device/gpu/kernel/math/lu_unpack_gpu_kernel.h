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
#include "mindspore/core/ops/lu_unpack.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/lu_unpack_impl.cuh"
#include "plugin/device/gpu/hal/device/gpu_device_address.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
constexpr size_t kInputNum = 2;
constexpr size_t kOutputNum = 3;
namespace kernel {
class LuUnpackGpuKernelMod : public NativeGpuKernelMod {
 public:
  LuUnpackGpuKernelMod() { ResetResource(); }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using LuUnpackFunc =
    std::function<bool(LuUnpackGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  LuUnpackFunc kernel_func_;
  void *cuda_stream_{nullptr};
  static std::vector<std::pair<KernelAttr, LuUnpackFunc>> func_list_;

  size_t lu_data_size_;
  size_t lu_pivots_size_;
  int64_t lu_data_dim1_;
  int64_t lu_data_dim2_;
  int64_t l_stride_;
  int64_t u_stride_;
  int64_t lu_pivots_dim_;
  int64_t batch_num_;
  size_t unit_size1_;
  size_t unit_size2_;

  void ResetResource() noexcept;
  template <typename T_data, typename T_pivots>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_LUUNPACK_GPU_KERNEL_H_
