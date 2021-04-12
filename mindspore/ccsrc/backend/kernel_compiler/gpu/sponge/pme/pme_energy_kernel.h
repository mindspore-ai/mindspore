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

    float box_length_0 = static_cast<float>(GetAttr<float_t>(kernel_node, "box_length_0"));
    float box_length_1 = static_cast<float>(GetAttr<float_t>(kernel_node, "box_length_1"));
    float box_length_2 = static_cast<float>(GetAttr<float_t>(kernel_node, "box_length_2"));
    std::vector<float> h_box_length{box_length_0, box_length_1, box_length_2};
    VECTOR *box_length = reinterpret_cast<VECTOR *>(h_box_length.data());
    cufftPlan3d(&PME_plan_r2c, fftx, ffty, fftz, CUFFT_R2C);
    cufftPlan3d(&PME_plan_c2r, fftx, ffty, fftz, CUFFT_C2R);
    _thread_PME.x = 8;
    _thread_PME.y = 8;
    PME_Nin = ffty * fftz;
    PME_Nfft = fftx * ffty * (fftz / 2 + 1);
    PME_Nall = fftx * ffty * fftz;
    PME_kxyz_cpu.resize(64);
    volume = box_length[0].x * box_length[0].y * box_length[0].z;
    int kx, ky, kz, kxrp, kyrp, kzrp, index;
    for (kx = 0; kx < 4; kx++) {
      for (ky = 0; ky < 4; ky++) {
        for (kz = 0; kz < 4; kz++) {
          index = kx * 16 + ky * 4 + kz;
          PME_kxyz_cpu[index].uint_x = kx;
          PME_kxyz_cpu[index].uint_y = ky;
          PME_kxyz_cpu[index].uint_z = kz;
        }
      }
    }

    B1.resize(fftx);
    B2.resize(ffty);
    B3.resize(fftz);
    PME_BC0.resize(PME_Nfft);
    for (kx = 0; kx < fftx; kx++) {
      B1[kx] = getb(kx, fftx, 4);
    }

    for (ky = 0; ky < ffty; ky++) {
      B2[ky] = getb(ky, ffty, 4);
    }

    for (kz = 0; kz < fftz; kz++) {
      B3[kz] = getb(kz, fftz, 4);
    }
    float mprefactor = PI * PI / -beta / beta;

    float msq;
    for (kx = 0; kx < fftx; kx++) {
      kxrp = kx;
      if (kx > fftx / 2) kxrp = fftx - kx;
      for (ky = 0; ky < ffty; ky++) {
        kyrp = ky;
        if (ky > ffty / 2) kyrp = ffty - ky;
        for (kz = 0; kz <= fftz / 2; kz++) {
          kzrp = kz;

          msq = kxrp * kxrp / box_length[0].x / box_length[0].x + kyrp * kyrp / box_length[0].y / box_length[0].y +
                kzrp * kzrp / box_length[0].z / box_length[0].z;
          index = kx * ffty * (fftz / 2 + 1) + ky * (fftz / 2 + 1) + kz;
          if ((kx + ky + kz) == 0) {
            PME_BC0[index] = 0;
          } else {
            PME_BC0[index] = 1.0 / PI / msq * exp(mprefactor * msq) / volume;
          }

          PME_BC0[index] *= B1[kx] * B2[ky] * B3[kz];
        }
      }
    }

    InitSizeLists();
    return true;
  }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    auto uint_crd = GetDeviceAddress<T1>(inputs, 0);
    auto charge = GetDeviceAddress<T>(inputs, 1);
    auto nl_numbers = GetDeviceAddress<T1>(inputs, 2);
    auto nl_serial = GetDeviceAddress<T1>(inputs, 3);
    auto scaler = GetDeviceAddress<T>(inputs, 4);
    auto excluded_list_start = GetDeviceAddress<int>(inputs, 5);
    auto excluded_list = GetDeviceAddress<int>(inputs, 6);
    auto excluded_atom_numbers = GetDeviceAddress<int>(inputs, 7);

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

    cufftSetStream(PME_plan_r2c, reinterpret_cast<cudaStream_t>(stream_ptr));
    cufftSetStream(PME_plan_c2r, reinterpret_cast<cudaStream_t>(stream_ptr));
    cudaMemcpyAsync(pme_kxyz, PME_kxyz_cpu.data(), sizeof(UNSIGNED_INT_VECTOR) * 64, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));
    cudaMemcpyAsync(pme_bc, PME_BC0.data(), sizeof(float) * PME_Nfft, cudaMemcpyHostToDevice,
                    reinterpret_cast<cudaStream_t>(stream_ptr));

    PMEEnergy(fftx, ffty, fftz, atom_numbers, beta, pme_bc, pme_uxyz, pme_frxyz, pme_q, pme_fq, pme_atom_near, pme_kxyz,
              uint_crd, charge, nl_numbers, nl_serial, nl, scaler, excluded_list_start, excluded_list,
              excluded_atom_numbers, reciprocal_ene, self_ene, direct_ene, correction_ene, _thread_PME, PME_Nin,
              PME_Nfft, PME_Nall, PME_plan_r2c, PME_plan_c2r, reinterpret_cast<cudaStream_t>(stream_ptr));
    return true;
  }

 protected:
  void InitSizeLists() override {
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

  cufftComplex expc(cufftComplex z) {
    cufftComplex res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
  }

  float M_(float u, int n) {
    if (n == 2) {
      if (u > 2 || u < 0) return 0;
      return 1 - abs(u - 1);
    } else {
      return u / (n - 1) * M_(u, n - 1) + (n - u) / (n - 1) * M_(u - 1, n - 1);
    }
  }

  float getb(int k, int NFFT, int B_order) {
    cufftComplex tempc, tempc2, res;
    float tempf;
    tempc2.x = 0;
    tempc2.y = 0;

    tempc.x = 0;
    tempc.y = 2 * (B_order - 1) * PI * k / NFFT;
    res = expc(tempc);

    for (int kk = 0; kk < (B_order - 1); kk++) {
      tempc.x = 0;
      if (NFFT == 0) {
        MS_LOG(ERROR) << "Divide by zero.";
        break;
      } else {
        tempc.y = 2 * PI * k / NFFT * kk;
      }
      tempc = expc(tempc);
      tempf = M_(kk + 1, B_order);
      tempc2.x += tempf * tempc.x;
      tempc2.y += tempf * tempc.y;
    }
    res = cuCdivf(res, tempc2);
    return res.x * res.x + res.y * res.y;
  }

 private:
  size_t ele_uint_crd = 1;

  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  std::vector<float> B1;
  std::vector<float> B2;
  std::vector<float> B3;
  std::vector<float> PME_BC0;

  int atom_numbers;
  int excluded_numbers;
  int max_nl_numbers = 800;
  int fftx;
  int ffty;
  int fftz;
  float beta;
  int PME_Nin;
  int PME_Nall;
  int PME_Nfft;
  float volume;
  float PI = 3.1415926;
  cufftHandle PME_plan_r2c;
  cufftHandle PME_plan_c2r;

  dim3 _thread_PME;

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
  std::vector<UNSIGNED_INT_VECTOR> PME_kxyz_cpu;
  struct NEIGHBOR_LIST {
    int atom_numbers;
    int *atom_serial;
  };
};
}  // namespace kernel
}  // namespace mindspore
#endif
