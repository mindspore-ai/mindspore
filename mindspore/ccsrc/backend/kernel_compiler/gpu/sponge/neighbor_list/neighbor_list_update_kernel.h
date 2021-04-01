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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_NEIGHBOR_LIST_UPDATE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_NEIGHBOR_LIST_UPDATE_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/neighbor_list/neighbor_list_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class NeighborListUpdateGpuKernel : public GpuKernel {
 public:
  NeighborListUpdateGpuKernel() : skin(2.0), cutoff(10.0), max_atom_in_grid_numbers(64), max_neighbor_numbers(800) {}
  ~NeighborListUpdateGpuKernel() override = default;
  bool Init(const CNodePtr &kernel_node) override {
    grid_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "grid_numbers"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    refresh_interval = static_cast<int>(GetAttr<int64_t>(kernel_node, "refresh_interval"));
    not_first_time = static_cast<int>(GetAttr<int64_t>(kernel_node, "not_first_time"));
    Nxy = static_cast<int>(GetAttr<int64_t>(kernel_node, "Nxy"));
    excluded_atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "excluded_atom_numbers"));

    cutoff_square = static_cast<float>(GetAttr<float>(kernel_node, "cutoff_square"));
    half_skin_square = static_cast<float>(GetAttr<float>(kernel_node, "half_skin_square"));
    cutoff_with_skin = static_cast<float>(GetAttr<float>(kernel_node, "cutoff_with_skin"));
    half_cutoff_with_skin = static_cast<float>(GetAttr<float>(kernel_node, "half_cutoff_with_skin"));
    cutoff_with_skin_square = static_cast<float>(GetAttr<float>(kernel_node, "cutoff_with_skin_square"));
    h_bucket.resize(grid_numbers);
    h_gpointer.resize(grid_numbers);
    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto atom_numbers_in_grid_bucket = GetDeviceAddress<int>(inputs, 0);
    auto bucket = GetDeviceAddress<int>(inputs, 1);
    auto crd = GetDeviceAddress<float>(inputs, 2);
    auto box_length = GetDeviceAddress<float>(inputs, 3);
    auto grid_N = GetDeviceAddress<int>(inputs, 4);
    auto grid_length_inverse = GetDeviceAddress<float>(inputs, 5);
    auto atom_in_grid_serial = GetDeviceAddress<int>(inputs, 6);
    auto old_crd = GetDeviceAddress<float>(inputs, 7);
    auto crd_to_uint_crd_cof = GetDeviceAddress<float>(inputs, 8);
    auto uint_crd = GetDeviceAddress<unsigned int>(inputs, 9);
    auto gpointer = GetDeviceAddress<int>(inputs, 10);
    auto nl_atom_numbers = GetDeviceAddress<int>(inputs, 11);
    auto nl_atom_serial = GetDeviceAddress<int>(inputs, 12);
    auto uint_dr_to_dr_cof = GetDeviceAddress<float>(inputs, 13);
    auto excluded_list_start = GetDeviceAddress<int>(inputs, 14);
    auto excluded_list = GetDeviceAddress<int>(inputs, 15);
    auto excluded_numbers = GetDeviceAddress<int>(inputs, 16);
    auto need_refresh_flag = GetDeviceAddress<int>(inputs, 17);
    auto d_refresh_count = GetDeviceAddress<int>(inputs, 18);

    GRID_BUCKET *d_bucket = reinterpret_cast<GRID_BUCKET *>(GetDeviceAddress<int>(workspaces, 0));
    GRID_POINTER *d_gpointer = reinterpret_cast<GRID_POINTER *>(GetDeviceAddress<int>(workspaces, 1));
    NEIGHBOR_LIST *nl = GetDeviceAddress<NEIGHBOR_LIST>(workspaces, 2);
    float *half_crd_to_uint_crd_cof = GetDeviceAddress<float>(workspaces, 3);

    // std::vector<GRID_BUCKET> h_bucket(grid_numbers);
    for (size_t i = 0; i < h_bucket.size(); i += 1) {
      h_bucket[i].atom_serial = bucket + i * max_atom_in_grid_numbers;
    }
    // std::vector<GRID_POINTER> h_gpointer(grid_numbers);
    for (size_t i = 0; i < h_gpointer.size(); i += 1) {
      h_gpointer[i].grid_serial = gpointer + i * 125;
    }

    cudaMemcpyAsync(d_bucket, h_bucket.data(), sizeof(GRID_BUCKET) * grid_numbers, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    cudaMemcpyAsync(d_gpointer, h_gpointer.data(), sizeof(GRID_POINTER) * grid_numbers, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    Construct_Neighbor_List(atom_numbers, max_neighbor_numbers, nl_atom_numbers, nl_atom_serial, nl,
                            reinterpret_cast<cudaStream_t>(stream_ptr));

    Neighbor_List_Update(grid_numbers, atom_numbers, d_refresh_count, refresh_interval, not_first_time, skin, Nxy,
                         cutoff_square, cutoff_with_skin_square, grid_N, box_length, atom_numbers_in_grid_bucket,
                         grid_length_inverse, atom_in_grid_serial, d_bucket, crd, old_crd, crd_to_uint_crd_cof,
                         half_crd_to_uint_crd_cof, uint_crd, uint_dr_to_dr_cof, d_gpointer, nl, excluded_list_start,
                         excluded_list, excluded_numbers, half_skin_square, need_refresh_flag,
                         reinterpret_cast<cudaStream_t>(stream_ptr));
    CopyNeighborListAtomNumber(atom_numbers, nl, nl_atom_numbers, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(int) * grid_numbers);
    input_size_list_.push_back(sizeof(int) * max_atom_in_grid_numbers * grid_numbers);
    input_size_list_.push_back(sizeof(VECTOR) * atom_numbers);
    input_size_list_.push_back(sizeof(VECTOR));

    input_size_list_.push_back(sizeof(INT_VECTOR));
    input_size_list_.push_back(sizeof(VECTOR));
    input_size_list_.push_back(sizeof(int) * atom_numbers);

    input_size_list_.push_back(sizeof(VECTOR) * atom_numbers);
    input_size_list_.push_back(sizeof(VECTOR));
    input_size_list_.push_back(sizeof(UNSIGNED_INT_VECTOR) * atom_numbers);

    input_size_list_.push_back(sizeof(int) * grid_numbers * 125);
    input_size_list_.push_back(sizeof(int) * atom_numbers);
    input_size_list_.push_back(sizeof(int) * atom_numbers * max_neighbor_numbers);
    input_size_list_.push_back(sizeof(VECTOR));

    input_size_list_.push_back(sizeof(int) * atom_numbers);
    input_size_list_.push_back(sizeof(int) * excluded_atom_numbers);
    input_size_list_.push_back(sizeof(int) * atom_numbers);

    input_size_list_.push_back(sizeof(int));
    input_size_list_.push_back(sizeof(int));

    workspace_size_list_.push_back(sizeof(GRID_BUCKET) * grid_numbers);
    workspace_size_list_.push_back(sizeof(GRID_POINTER) * grid_numbers);
    workspace_size_list_.push_back(sizeof(NEIGHBOR_LIST) * atom_numbers);
    workspace_size_list_.push_back(sizeof(float) * 3);

    output_size_list_.push_back(sizeof(float));
  }

 private:
  float skin;
  float cutoff;
  int not_first_time;
  int atom_numbers;
  int grid_numbers;
  int refresh_interval;
  int Nxy;
  int max_atom_in_grid_numbers;
  int max_neighbor_numbers;
  int excluded_atom_numbers;
  float half_skin_square;
  float cutoff_square;
  float cutoff_with_skin;
  float half_cutoff_with_skin;
  float cutoff_with_skin_square;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  std::vector<GRID_BUCKET> h_bucket;
  std::vector<GRID_POINTER> h_gpointer;
};
}  // namespace kernel
}  // namespace mindspore

#endif
