/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LSTSQV2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LSTSQV2_CPU_KERNEL_H_

#include <complex>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/nnacl/base/broadcast_to.h"
#include "plugin/device/cpu/kernel/nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
class LstsqV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  LstsqV2CpuKernelMod() = default;
  ~LstsqV2CpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  template <typename T>
  static void BroadCastInput(T *mat_addr, T *broadcasted_mat_addr, BroadcastShapeInfo mat_shape_info,
                             std::string kernel_name) {
    int status = BroadcastToSize64(mat_addr, &mat_shape_info, broadcasted_mat_addr);
    if (status != static_cast<int>(NNACL_OK)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << ", broadcast failed. Error code: " << status;
    }
  }

  static void BroadCastInput(float *mat_addr, float *broadcasted_mat_addr, BroadcastShapeInfo mat_shape_info,
                             std::string kernel_name) {
    int status = BroadcastToSize32(mat_addr, &mat_shape_info, broadcasted_mat_addr);
    if (status != static_cast<int>(NNACL_OK)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << ", broadcast failed. Error code: " << status;
    }
  }

  static void BroadCastInput(std::complex<double> *mat_addr, std::complex<double> *broadcasted_mat_addr,
                             BroadcastShapeInfo mat_shape_info, std::string kernel_name) {
    int status = BroadcastToSize128(mat_addr, &mat_shape_info, broadcasted_mat_addr);
    if (status != static_cast<int>(NNACL_OK)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << ", broadcast failed. Error code: " << status;
    }
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void LstsqV2Check(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using LstsqV2Func = std::function<bool(LstsqV2CpuKernelMod *, const std::vector<KernelTensor *> &,
                                         const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, LstsqV2Func>> func_list_;
  LstsqV2Func kernel_func_;

  size_t m_;
  size_t n_;
  size_t k_;
  size_t a_mat_size_;
  size_t b_mat_size_;
  size_t solution_mat_size_;
  size_t res_vec_size_;
  size_t singular_value_vec_size_;
  size_t batch_;
  size_t a_batch_;
  size_t data_unit_size_;
  DriverName driver_;
  ShapeVector a_batch_shape_;
  ShapeVector b_batch_shape_;
  ShapeVector broadcast_batch_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LSTSQV2_CPU_KERNEL_H_
