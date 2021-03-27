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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_PME_ENERGY_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPONGE_PME_PME_ENERGY_KERNEL_H_
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <vector>
#include <string>
#include <map>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "runtime/device/gpu/cuda_common.h"
#include "backend/kernel_compiler/gpu/cuda_impl/sponge/pme/pme_energy_impl.cuh"
namespace mindspore {
namespace kernel {
template <typename T, typename T1>
class PMEEnergyGpuKernel : public GpuKernel {
 public:
  PMEEnergyGpuKernel() : ele_uint_crd(1) {}
  ~PMEEnergyGpuKernel() override = default;

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    atom_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "atom_numbers"));
    excluded_numbers = static_cast<int>(GetAttr<int64_t>(kernel_node, "excluded_numbers"));
    beta = static_cast<float>(GetAttr<float_t>(kernel_node, "beta"));
    fftx = static_cast<int>(GetAttr<int64_t>(kernel_node, "fftx"));
    ffty = static_cast<int>(GetAttr<int64_t>(kernel_node, "ffty"));
    fftz = static_cast<int>(GetAttr<int64_t>(kernel_node, "fftz"));
    PME_Nall = fftx * ffty * fftz;
    PME_Nfft = fftx * ffty * (fftz / 2 + 1);

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto boxlength = GetDeviceAddress<T>(inputs, 0);
    auto uint_crd = GetDeviceAddress<T1>(inputs, 1);
    auto charge = GetDeviceAddress<T>(inputs, 2);
    auto nl_numbers = GetDeviceAddress<T1>(inputs, 3);
    auto nl_serial = GetDeviceAddress<T1>(inputs, 4);
    auto scaler = GetDeviceAddress<T>(inputs, 5);
    auto excluded_list_start = GetDeviceAddress<int>(inputs, 6);
    auto excluded_list = GetDeviceAddress<int>(inputs, 7);
    auto excluded_atom_numbers = GetDeviceAddress<int>(inputs, 8);

    auto pme_uxyz = GetDeviceAddress<int>(workspace, 0);       // workspace
    auto pme_frxyz = GetDeviceAddress<float>(workspace, 1);    // workspace
    auto pme_q = GetDeviceAddress<T>(workspace, 2);            // workspace
    auto pme_fq = GetDeviceAddress<float>(workspace, 3);       // workspace
    auto pme_atom_near = GetDeviceAddress<int>(workspace, 4);  // workspace
    auto pme_bc = GetDeviceAddress<float>(workspace, 5);       // workspace
    auto pme_kxyz = GetDeviceAddress<int>(workspace, 6);       // workspace
    auto nl = GetDeviceAddress<T1>(workspace, 7);

    auto reciprocal_ene = GetDeviceAddress<T>(outputs, 0);
    auto self_ene = GetDeviceAddress<T>(outputs, 1);
    auto direct_ene = GetDeviceAddress<T>(outputs, 2);
    auto correction_ene = GetDeviceAddress<T>(outputs, 3);

    PMEEnergy(fftx, ffty, fftz, atom_numbers, beta, boxlength, pme_bc, pme_uxyz, pme_frxyz, pme_q, pme_fq,
              pme_atom_near, pme_kxyz, uint_crd, charge, nl_numbers, nl_serial, nl, scaler, excluded_list_start,
              excluded_list, excluded_atom_numbers, reciprocal_ene, self_ene, direct_ene, correction_ene,
              reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
    input_size_list_.push_back(sizeof(VECTOR));
    input_size_list_.push_back(atom_numbers * sizeof(UNSIGNED_INT_VECTOR));
    input_size_list_.push_back(atom_numbers * sizeof(VECTOR));
    input_size_list_.push_back(atom_numbers * sizeof(T1));
    input_size_list_.push_back(max_nl_numbers * sizeof(T1));
    input_size_list_.push_back(sizeof(VECTOR));

    input_size_list_.push_back(atom_numbers * sizeof(T1));
    input_size_list_.push_back(excluded_numbers * sizeof(T1));
    input_size_list_.push_back(atom_numbers * sizeof(T1));

    workspace_size_list_.push_back(atom_numbers * sizeof(UNSIGNED_INT_VECTOR));
    workspace_size_list_.push_back(atom_numbers * sizeof(VECTOR));
    workspace_size_list_.push_back(PME_Nall * sizeof(T));
    workspace_size_list_.push_back(PME_Nfft * sizeof(cufftComplex));
    workspace_size_list_.push_back(atom_numbers * 64 * sizeof(int));
    workspace_size_list_.push_back(PME_Nfft * sizeof(float));
    workspace_size_list_.push_back(64 * sizeof(UNSIGNED_INT_VECTOR));
    workspace_size_list_.push_back(atom_numbers * max_nl_numbers * sizeof(T1));

    output_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(sizeof(T));
    output_size_list_.push_back(sizeof(T));
  }

 private:
  size_t ele_uint_crd = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  int atom_numbers;
  int excluded_numbers;
  int max_nl_numbers = 800;
  int fftx;
  int ffty;
  int fftz;
  float beta;
  int PME_Nall;
  int PME_Nfft;
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

  struct NEIGHBOR_LIST {
    int atom_numbers;
    int *atom_serial;
  };
};
}  // namespace kernel
}  // namespace mindspore
#endif
