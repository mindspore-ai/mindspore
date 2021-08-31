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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_CRDMCMAP_REFRESH_BOXMAPTIMES_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_CRDMCMAP_REFRESH_BOXMAPTIMES_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/crdmcmap/refresh_boxmaptimes_impl.cuh"

#include <cuda_runtime_api.h>
#include <map>
#include <string>
#include <vector>

#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class RefreshBoxmaptimesGpuKernel : public GpuKernel {
 public:
  RefreshBoxmaptimesGpuKernel() : ele_crd(1) {}
  ~RefreshBoxmaptimesGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    // get bond_numbers
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    auto shape_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_old_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_box_length_inverse = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_box_map_times = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);

    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];
    for (size_t i = 0; i < shape_old_crd.size(); i++) ele_old_crd *= shape_old_crd[i];
    for (size_t i = 0; i < shape_box_length_inverse.size(); i++) ele_box_length_inverse *= shape_box_length_inverse[i];
    for (size_t i = 0; i < shape_box_map_times.size(); i++) ele_box_map_times *= shape_box_map_times[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto crd = GetDeviceAddress<T>(inputs, 0);
    auto old_crd = GetDeviceAddress<T>(inputs, 1);
    auto box_length_inverse = GetDeviceAddress<T>(inputs, 2);
    auto box_map_times = GetDeviceAddress<T1>(inputs, 3);

    refresh_boxmaptimes(atom_numbers, box_length_inverse, crd, old_crd, box_map_times,
                        reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_crd * sizeof(T));
    input_size_list_.push_back(ele_old_crd * sizeof(T));
    input_size_list_.push_back(ele_box_length_inverse * sizeof(T));
    input_size_list_.push_back(ele_box_map_times * sizeof(T1));

    output_size_list_.push_back(sizeof(T));
  }

 private:
  size_t ele_crd = 1;
  size_t ele_box_length_inverse = 1;
  size_t ele_old_crd = 1;
  size_t ele_box_map_times = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int atom_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_CRDMCMAP_CAL_NO_WRAP_CRD_KERNEL_H_
