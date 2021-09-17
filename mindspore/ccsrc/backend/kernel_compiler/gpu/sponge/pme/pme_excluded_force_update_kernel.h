/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
/**
 * Note:
 *  PMEExcludedForceUpdate. This is an experimental interface that is subject to change and/or deletion.
 */
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_PME_EXCLUDED_FORCE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_PME_EXCLUDED_FORCE_KERNEL_H_

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <map>
#include <string>
#include <vector>
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_excluded_force_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class PMEExcludedForceUpdateGpuKernel : public GpuKernel {
 public:
  PMEExcludedForceUpdateGpuKernel() : ele_uint_crd(1) {}
  ~PMEExcludedForceUpdateGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    excluded_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "excluded_numbers"));
    beta = static_cast<float>(GetAttr<float_t>(kernel_node, "beta"));
    need_update = static_cast<int>(GetAttr<int64_t>(kernel_node, "need_update"));
    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto uint_crd = GetDeviceAddress<int>(inputs, 0);
    auto sacler = GetDeviceAddress<T>(inputs, 1);
    auto charge = GetDeviceAddress<T>(inputs, 2);
    auto excluded_list_start = GetDeviceAddress<int>(inputs, 3);
    auto excluded_list = GetDeviceAddress<int>(inputs, 4);
    auto excluded_atom_numbers = GetDeviceAddress<int>(inputs, 5);
    auto d_beta = GetDeviceAddress<T>(inputs, 6);
    if (need_update) {
      cudaMemcpyAsync(&beta, d_beta, sizeof(float), cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream_ptr));
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_ptr));
    }

    auto force = GetDeviceAddress<T>(outputs, 0);
    PMEExcludedForce(atom_numbers, beta, uint_crd, sacler, charge, excluded_list_start, excluded_list,
                     excluded_atom_numbers, force, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(atom_numbers * sizeof(UNSIGNED_INT_VECTOR));
    input_size_list_.push_back(sizeof(VECTOR));
    input_size_list_.push_back(atom_numbers * sizeof(T));
    input_size_list_.push_back(atom_numbers * sizeof(T1));
    input_size_list_.push_back(excluded_numbers * sizeof(T1));
    input_size_list_.push_back(atom_numbers * sizeof(T1));
    input_size_list_.push_back(sizeof(T));

    output_size_list_.push_back(atom_numbers * 3 * sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int atom_numbers;
  int excluded_numbers;
  float beta;
  float factor;
  int need_update;
  struct VECTOR {
    float x;
    float y;
    float z;
  };

  struct UNSIGNED_INT_VECTOR {
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
  };
};
}  // namespace kernel
}  // namespace mindspore
#endif
