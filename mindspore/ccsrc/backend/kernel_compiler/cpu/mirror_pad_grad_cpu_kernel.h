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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIRROR_PAD_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIRROR_PAD_GRAD_CPU_KERNEL_H_
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

// preset size of paddings
#define MAX_PADDINGS 4
#define PADDING_SIZE 2

// define constants for kernel indexing use
#define BATCH 0 * PADDING_SIZE
#define CHANNEL 1 * PADDING_SIZE
#define HEIGHT 2 * PADDING_SIZE
#define WIDTH 3 * PADDING_SIZE
#define TOP 0
#define BOTTOM 1
#define LEFT 0
#define RIGHT 1

namespace mindspore {
namespace kernel {
class MirrorPadGradCPUKernel : public CPUKernel {
 public:
  MirrorPadGradCPUKernel() = default;
  ~MirrorPadGradCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;
  void InitInputOutputSize(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  template <typename T>
  void InitWorkspaceSize();

  template <typename T>
  void LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  template <typename T>
  void MirrorPadGrad_Width_Height(const size_t size, const T *dy, const T *interim_dy, const int dx_batches,
                                  const int dx_channels, const int dx_height, const int dx_width, const int dy_height,
                                  const int dy_width, const int padd_dim, const int64_t *paddings_arg, int mode, T *dx);

  template <typename T>
  void MirrorPadGradBatchChannel(const size_t size, T *dy, T *interim_dy, const int dx_batches, const int dx_channels,
                                 const int dx_height, const int dx_width, const int dy_height, const int dy_width,
                                 const int padd_dim, const int64_t *paddings_arg, int mode, T *const dx);

 private:
  void CheckParam(const CNodePtr &kernel_node);
  TypeId dtype_{kTypeUnknown};
  size_t tensor_size_ = 1;
  size_t shape_size_;
  size_t output_size_ = 1;
  size_t workspace_size_ = 1;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  int mode_;
  int num_paddings_;
};

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
  MirrorPadGradCPUKernel);

MS_REG_CPU_KERNEL(
  MirrorPadGrad,
  KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
  MirrorPadGradCPUKernel);

}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIRROR_PAD_CPU_KERNEL_H_
