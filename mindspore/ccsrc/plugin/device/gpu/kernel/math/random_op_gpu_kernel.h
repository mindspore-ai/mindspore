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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_RANDOM_OP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_RANDOM_OP_GPU_KERNEL_H_

#include <curand_kernel.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/random_op_impl.cuh"
#include "include/curand.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
enum RandomOptype {
  RANDOM_OP_NORMAL = 0,
  RANDOM_OP_UNIFORM_INT,
  RANDOM_OP_UNIFORM_REAL,
  RANDOM_OP_CUDNN_UNIFORM_REAL,
  RANDOM_OP_INVALID_TYPE = 255
};

const std::map<std::string, RandomOptype> kRandomOpTypeMap = {{"StandardNormal", RANDOM_OP_NORMAL},
                                                              {"UniformInt", RANDOM_OP_UNIFORM_INT},
                                                              {"UniformReal", RANDOM_OP_UNIFORM_REAL},
                                                              {"CudnnUniformReal", RANDOM_OP_CUDNN_UNIFORM_REAL}};

class RandomOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  RandomOpGpuKernelMod() = default;
  explicit RandomOpGpuKernelMod(const std::string &kernel_name) : kernel_type_(kernel_name) {}
  ~RandomOpGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    cuda_stream_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using OpFunc =
    std::function<bool(RandomOpGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &workspace, const std::vector<kernel::AddressPtr> &)>;
  static std::map<std::string, std::vector<std::pair<KernelAttr, RandomOpGpuKernelMod::OpFunc>>> kernel_attr_map_;
  RandomOptype random_op_type_{RANDOM_OP_INVALID_TYPE};
  size_t input_num_{0};
  int seed_{0};
  int seed2_{0};
  OpFunc kernel_func_;
  curandGenerator_t mask_generator_{nullptr};
  bool states_init_{false};
  bool use_curand_{false};
  void *cuda_stream_{nullptr};
  std::string kernel_type_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_RANDOM_OP_GPU_KERNEL_H_
