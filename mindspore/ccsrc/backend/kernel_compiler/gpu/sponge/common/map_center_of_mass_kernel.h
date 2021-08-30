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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_MAP_CENTER_OF_MASS_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_MAP_CENTER_OF_MASS_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/map_center_of_mass_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class MapCenterOfMassGpuKernel : public GpuKernel {
 public:
  MapCenterOfMassGpuKernel() : ele_start(1) {}
  ~MapCenterOfMassGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    residue_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "residue_numbers"));

    auto shape_start = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_end = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_center_of_mass = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    auto shape_box_length = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 3);
    auto shape_no_wrap_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 4);
    auto shape_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 5);

    for (size_t i = 0; i < shape_start.size(); i++) ele_start *= shape_start[i];
    for (size_t i = 0; i < shape_end.size(); i++) ele_end *= shape_end[i];
    for (size_t i = 0; i < shape_center_of_mass.size(); i++) ele_center_of_mass *= shape_center_of_mass[i];
    for (size_t i = 0; i < shape_box_length.size(); i++) ele_box_length *= shape_box_length[i];
    for (size_t i = 0; i < shape_no_wrap_crd.size(); i++) ele_no_wrap_crd *= shape_no_wrap_crd[i];
    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto start = GetDeviceAddress<T1>(inputs, 0);
    auto end = GetDeviceAddress<T1>(inputs, 1);
    auto center_of_mass = GetDeviceAddress<T>(inputs, 2);
    auto box_length = GetDeviceAddress<T>(inputs, 3);
    auto no_wrap_crd = GetDeviceAddress<T>(inputs, 4);
    auto crd = GetDeviceAddress<T>(inputs, 5);
    auto scaler = GetDeviceAddress<T>(inputs, 6);

    MapCenterOfMass(residue_numbers, start, end, center_of_mass, box_length, no_wrap_crd, crd, scaler,
                    reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_start * sizeof(T1));
    input_size_list_.push_back(ele_end * sizeof(T1));
    input_size_list_.push_back(ele_center_of_mass * sizeof(T));
    input_size_list_.push_back(ele_box_length * sizeof(T));
    input_size_list_.push_back(ele_no_wrap_crd * sizeof(T));
    input_size_list_.push_back(ele_crd * sizeof(T));
    input_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(sizeof(T));
  }

 private:
  size_t ele_start = 1;
  size_t ele_end = 1;
  size_t ele_center_of_mass = 1;
  size_t ele_box_length = 1;
  size_t ele_no_wrap_crd = 1;
  size_t ele_crd = 1;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int residue_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_MAP_CENTER_OF_MASS_KERNEL_H_
