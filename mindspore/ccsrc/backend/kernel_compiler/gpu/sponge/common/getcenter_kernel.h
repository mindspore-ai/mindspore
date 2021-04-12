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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_GETCENTER_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_GETCENTER_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/getcenter_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class GetCenterOfGeometryGpuKernel : public GpuKernel {
 public:
  GetCenterOfGeometryGpuKernel() : ele_center_atoms(1) {}
  ~GetCenterOfGeometryGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    center_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "center_numbers"));
    center_numbers_inverse = static_cast<int>(GetAttr<float>(kernel_node, "center_numbers_inverse"));

    auto shape_center_atoms = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);

    for (size_t i = 0; i < shape_center_atoms.size(); i++) ele_center_atoms *= shape_center_atoms[i];
    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto center_atoms = GetDeviceAddress<const T1>(inputs, 0);
    auto crd = GetDeviceAddress<const T>(inputs, 1);

    auto center_of_geometry = GetDeviceAddress<T>(outputs, 0);

    GetCenterOfGeometry(center_numbers, center_numbers_inverse, center_atoms, crd, center_of_geometry,
                        reinterpret_cast<cudaStream_t>(stream_ptr));

    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_center_atoms * sizeof(T1));
    input_size_list_.push_back(ele_crd * sizeof(T));

    output_size_list_.push_back(3 * sizeof(T));
  }

 private:
  size_t ele_center_atoms = 1;
  size_t ele_crd = 1;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int center_numbers;
  float center_numbers_inverse;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_GETCENTER_KERNEL_H_
