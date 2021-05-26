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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_ATOMCRDTOCV_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_ATOMCRDTOCV_KERNEL_H_

#include "backend/kernel_compiler/gpu/cuda_impl/sponge/common/atomcrdtocv_impl.cuh"

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
class AtomCrdToCVGpuKernel : public GpuKernel {
 public:
  AtomCrdToCVGpuKernel() : ele_crd(1) {}
  ~AtomCrdToCVGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    start_serial = static_cast<int>(GetAttr<int64_t>(kernel_node, "start_serial"));
    end_serial = static_cast<int>(GetAttr<int64_t>(kernel_node, "end_serial"));
    number = static_cast<int>(GetAttr<int64_t>(kernel_node, "number"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    auto shape_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto shape_old_crd = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto shape_box = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);

    for (size_t i = 0; i < shape_crd.size(); i++) ele_crd *= shape_crd[i];
    for (size_t i = 0; i < shape_old_crd.size(); i++) ele_old_crd *= shape_old_crd[i];
    for (size_t i = 0; i < shape_box.size(); i++) ele_box *= shape_box[i];

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto crd = GetDeviceAddress<const T>(inputs, 0);
    auto old_crd = GetDeviceAddress<const T>(inputs, 1);
    auto box = GetDeviceAddress<T>(inputs, 2);

    auto g_radial = GetDeviceAddress<T>(outputs, 0);
    auto g_angular = GetDeviceAddress<T>(outputs, 1);

    auto nowarp_crd = GetDeviceAddress<T>(outputs, 2);
    auto box_map_times = GetDeviceAddress<T1>(outputs, 3);

    AtomCrdToCV(atom_numbers, start_serial, end_serial, number, crd, old_crd, nowarp_crd, box_map_times, box, g_radial,
                g_angular, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(ele_crd * sizeof(T));
    input_size_list_.push_back(ele_old_crd * sizeof(T));
    input_size_list_.push_back(ele_box * sizeof(T));

    output_size_list_.push_back(number * sizeof(T));
    output_size_list_.push_back(number * sizeof(T));

    output_size_list_.push_back(3 * atom_numbers * sizeof(T));
    output_size_list_.push_back(3 * atom_numbers * sizeof(T1));
  }

 private:
  size_t ele_crd = 1;
  size_t ele_old_crd = 1;
  size_t ele_box = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int end_serial;
  int start_serial;
  int number;
  int atom_numbers;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONG_COMMON_ATOMCRDTOCV_KERNEL_H_
