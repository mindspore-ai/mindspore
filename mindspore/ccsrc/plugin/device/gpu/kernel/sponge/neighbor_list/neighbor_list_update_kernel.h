/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
 *  NeighborListUpdate. This is an experimental interface that is subject to change and/or deletion.
 */

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_NEIGHBOR_LIST_UPDATE_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_NEIGHBOR_LIST_UPDATE_KERNEL_H_

#include <cuda_runtime_api.h>
#include <vector>
#include <string>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"
#include "plugin/device/gpu/kernel/cuda_impl/sponge/neighbor_list/neighbor_list_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr float kSkinDef = 2.0;
constexpr float kCutoffDef = 10.0;
constexpr int kMaxAtomInGridNumDef = 64;
constexpr int kMaxNbNumDef = 800;

constexpr size_t kIdx3 = 3;
constexpr size_t kIdx5 = 5;
constexpr size_t kIdx6 = 6;
constexpr size_t kIdx8 = 8;
constexpr size_t kIdx2 = 2;
constexpr size_t kIdx4 = 4;
constexpr size_t kIdx9 = 9;
constexpr size_t kIdx16 = 16;
constexpr size_t kIdx10 = 10;
constexpr size_t kIdx7 = 7;
constexpr size_t kIdx14 = 14;
constexpr size_t kIdx12 = 12;
constexpr size_t kIdx11 = 11;
constexpr size_t kIdx13 = 13;
constexpr size_t kIdx18 = 18;
constexpr size_t kIdx15 = 15;
constexpr size_t kIdx17 = 17;

template <typename T, typename T1>
class NeighborListUpdateGpuKernelMod : public NativeGpuKernelMod {
 public:
  NeighborListUpdateGpuKernelMod()
      : skin(kSkinDef),
        cutoff(kCutoffDef),
        max_atom_in_grid_numbers(kMaxAtomInGridNumDef),
        max_neighbor_numbers(kMaxNbNumDef) {}
  ~NeighborListUpdateGpuKernelMod() override = default;
  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    grid_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "grid_numbers"));
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    refresh_interval = static_cast<int>(GetAttr<int64_t>(kernel_node, "refresh_interval"));
    not_first_time = static_cast<int>(GetAttr<int64_t>(kernel_node, "not_first_time"));
    nxy = static_cast<int>(GetAttr<int64_t>(kernel_node, "nxy"));
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

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspaces,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto atom_numbers_in_grid_bucket = GetDeviceAddress<int>(inputs, 0);
    auto bucket = GetDeviceAddress<int>(inputs, 1);
    auto crd = GetDeviceAddress<float>(inputs, kIdx2);
    auto box_length = GetDeviceAddress<float>(inputs, kIdx3);
    auto grid_n = GetDeviceAddress<int>(inputs, kIdx4);
    auto grid_length_inverse = GetDeviceAddress<float>(inputs, kIdx5);
    auto atom_in_grid_serial = GetDeviceAddress<int>(inputs, kIdx6);
    auto old_crd = GetDeviceAddress<float>(inputs, kIdx7);
    auto crd_to_uint_crd_cof = GetDeviceAddress<float>(inputs, kIdx8);
    auto uint_crd = GetDeviceAddress<unsigned int>(inputs, kIdx9);
    auto gpointer = GetDeviceAddress<int>(inputs, kIdx10);
    auto nl_atom_numbers = GetDeviceAddress<int>(inputs, kIdx11);
    auto nl_atom_serial = GetDeviceAddress<int>(inputs, kIdx12);
    auto uint_dr_to_dr_cof = GetDeviceAddress<float>(inputs, kIdx13);
    auto excluded_list_start = GetDeviceAddress<int>(inputs, kIdx14);
    auto excluded_list = GetDeviceAddress<int>(inputs, kIdx15);
    auto excluded_numbers = GetDeviceAddress<int>(inputs, kIdx16);
    auto need_refresh_flag = GetDeviceAddress<int>(inputs, kIdx17);
    auto d_refresh_count = GetDeviceAddress<int>(inputs, kIdx18);

    GRID_BUCKET *d_bucket = reinterpret_cast<GRID_BUCKET *>(GetDeviceAddress<int>(workspaces, 0));
    GRID_POINTER *d_gpointer = reinterpret_cast<GRID_POINTER *>(GetDeviceAddress<int>(workspaces, 1));
    NEIGHBOR_LIST *nl = GetDeviceAddress<NEIGHBOR_LIST>(workspaces, kIdx2);
    float *half_crd_to_uint_crd_cof = GetDeviceAddress<float>(workspaces, kIdx3);

    for (size_t i = 0; i < h_bucket.size(); i += 1) {
      h_bucket[i].atom_serial = bucket + i * max_atom_in_grid_numbers;
    }
    const size_t kGridSize = 125;
    for (size_t i = 0; i < h_gpointer.size(); i += 1) {
      h_gpointer[i].grid_serial = gpointer + i * kGridSize;
    }

    cudaMemcpyAsync(d_bucket, h_bucket.data(), sizeof(GRID_BUCKET) * grid_numbers, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    cudaMemcpyAsync(d_gpointer, h_gpointer.data(), sizeof(GRID_POINTER) * grid_numbers, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    ConstructNeighborListHalf(atom_numbers, max_neighbor_numbers, nl_atom_numbers, nl_atom_serial, nl,
                              reinterpret_cast<cudaStream_t>(stream_ptr));
    NeighborListUpdate(grid_numbers, atom_numbers, d_refresh_count, refresh_interval, not_first_time, skin, nxy,
                       cutoff_square, cutoff_with_skin_square, grid_n, box_length, atom_numbers_in_grid_bucket,
                       grid_length_inverse, atom_in_grid_serial, d_bucket, crd, old_crd, crd_to_uint_crd_cof,
                       half_crd_to_uint_crd_cof, uint_crd, uint_dr_to_dr_cof, d_gpointer, nl, excluded_list_start,
                       excluded_list, excluded_numbers, half_skin_square, need_refresh_flag,
                       reinterpret_cast<cudaStream_t>(stream_ptr));
    CopyNeighborListHalf(atom_numbers, nl, nl_atom_numbers, reinterpret_cast<cudaStream_t>(stream_ptr));
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

    const size_t kGridSize = 125;
    input_size_list_.push_back(sizeof(int) * grid_numbers * kGridSize);
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
    const size_t kCrdSize = 3;
    workspace_size_list_.push_back(sizeof(float) * kCrdSize);

    output_size_list_.push_back(sizeof(float));
  }

 private:
  float skin;
  float cutoff;
  int not_first_time;
  int grid_numbers;
  int refresh_interval;
  int atom_numbers;
  int nxy;
  int max_atom_in_grid_numbers;
  int max_neighbor_numbers;
  int excluded_atom_numbers;
  float half_skin_square;
  float cutoff_square;
  float cutoff_with_skin;
  float half_cutoff_with_skin;
  float cutoff_with_skin_square;

  std::vector<GRID_BUCKET> h_bucket;
  std::vector<GRID_POINTER> h_gpointer;
};
}  // namespace kernel
}  // namespace mindspore

#endif
