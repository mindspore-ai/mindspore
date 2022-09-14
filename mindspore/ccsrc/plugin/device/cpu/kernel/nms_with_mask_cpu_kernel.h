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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NMS_WITH_MASK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NMS_WITH_MASK_CPU_KERNEL_H_
#include <map>
#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
constexpr int X0 = 0;
constexpr int Y0 = 1;
constexpr int X1 = 2;
constexpr int Y1 = 3;
constexpr int SCORE = 4;
constexpr size_t INPUT_NUM = 1;
constexpr size_t OUTPUT_NUM = 3;
class NMSWithMaskCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<NMSWithMaskCpuKernelMod> {
 public:
  NMSWithMaskCpuKernelMod() = default;
  ~NMSWithMaskCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  void NmsBitonicSortByKeyKernel(const int inner, const size_t ceil_power2, const T *input, T *data_buff,
                                 int *index_buff, int box_size);

  void MaskInit(size_t numSq, bool *row_mask);

  template <typename T>
  void PopulateOutput(const T *data_in, T *data_out, const int *index_buff, const int num, int box_size,
                      bool flip_mode);

  void Preprocess(const int num, int *sel_idx, bool *sel_boxes);

  template <typename T>
  bool IouDecision(const T *output, int box_A_start, int box_B_start, float IOU_value) const;

  template <typename T>
  void NmsPass(const int num, const float IOU_value, const T *output, int box_size, bool *row_mask);

  void ReducePass(const int num, bool *sel_boxes, const bool *row_mask);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  int num_input_{0};
  float iou_value_{0.0};
  size_t ceil_power_2_{0};
  static const int box_size_ = 5;  //  pre_defined box width
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NMS_WITH_MASK_CPU_KERNEL_H_
