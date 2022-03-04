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
#include <vector>
#include <algorithm>
#include <limits>
#include <tuple>
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
class NMSWithMaskCpuKernelMod : public NativeCpuKernelMod {
 public:
  NMSWithMaskCpuKernelMod() = default;
  ~NMSWithMaskCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  void InitInputOutputSize(const CNodePtr &kernel_node) override { Init_func_(this, kernel_node); }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

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
  bool IouDecision(const T *output, int box_A_start, int box_B_start, float IOU_value);
  template <typename T>
  void NmsPass(const int num, const float IOU_value, const T *output, int box_size, bool *row_mask);
  void ReducePass(const int num, bool *sel_boxes, const bool *row_mask);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  template <typename T>
  void InitIOSize(const CNodePtr &kernel_node);
  using NMSWithMaskLFunc =
    std::function<bool(NMSWithMaskCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  using NMSWithMaskIFunc = std::function<void(NMSWithMaskCpuKernelMod *, const CNodePtr &)>;
  static std::vector<std::tuple<KernelAttr, NMSWithMaskLFunc, NMSWithMaskIFunc>> func_list_;
  NMSWithMaskLFunc kernel_func_;
  NMSWithMaskIFunc Init_func_;

  int num_input_{0};
  float iou_value_{0.0};
  size_t ceil_power_2{0};
  static const int box_size_ = 5;  //  pre_defined box width
  enum workspace_list_ { DATA_BUFF, INDEX_BUFF, ROW_MASK };
  enum output_list_ { OUTPUT, SEL_IDX, SEL_BOXES };
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NMS_WITH_MASK_CPU_KERNEL_H_
