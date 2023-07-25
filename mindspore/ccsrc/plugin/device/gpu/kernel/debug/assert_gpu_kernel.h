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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_ASSERT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_ASSERT_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <utility>
#include <map>
#include "mindspore/core/ops/assert.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/assert_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "kernel/common_utils.h"

namespace mindspore {
namespace kernel {
class AssertGpuKernelMod : public NativeGpuKernelMod {
 public:
  AssertGpuKernelMod() {}
  ~AssertGpuKernelMod() override = default;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    auto input_data_num = inputs.size() - 1;
    void **inputs_device = GetDeviceAddress<void *>(workspaces, 0);
    int *summarizes_device = GetDeviceAddress<int>(workspaces, 1);
    int *types_device = GetDeviceAddress<int>(workspaces, 2);
    bool *cond_device = GetDeviceAddress<bool>(inputs, 0);
    for (size_t i = 0; i < input_data_num; i++) {
      input_addrs_[i] = inputs[i + 1]->addr;
    }

    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(inputs_device, input_addrs_.data(), sizeof(void *) * input_data_num, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "assert cudaMemcpyAsync inputs failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(summarizes_device, summarizes_.data(), sizeof(int) * input_data_num, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "assert  cudaMemcpyAsync summarizes failed");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(types_device, types_.data(), sizeof(int) * input_data_num, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream)),
      "assert  cudaMemcpyAsync types failed");
    auto status = AssertKernel(cond_device, inputs_device, summarizes_device, types_device, input_data_num, device_id_,
                               reinterpret_cast<cudaStream_t>(cuda_stream));
    CHECK_CUDA_STATUS(status, kernel_name_);
    bool cond = true;
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(&cond, cond_device, sizeof(bool), cudaMemcpyDeviceToHost,
                                                       reinterpret_cast<cudaStream_t>(cuda_stream)),
                                       "copy condition failed");
    if (!cond) {
      CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream)),
                                         "assert cudaStreamSynchronized failed");
      MS_LOG(EXCEPTION) << "assert failed";
    }

    return true;
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 private:
  std::vector<void *> input_addrs_;
  std::vector<int> summarizes_;
  std::vector<int> types_;
  int summarize_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_DEBUG_ASSERT_GPU_KERNEL_H_
