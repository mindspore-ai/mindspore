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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NMS_WITH_MASK_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NMS_WITH_MASK_CPU_KERNEL_H_
#include <vector>
#include <algorithm>
#include <limits>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr int X0 = 0;
constexpr int Y0 = 1;
constexpr int X1 = 2;
constexpr int Y1 = 3;
constexpr int SCORE = 4;
constexpr size_t INPUT_NUM = 1;
constexpr size_t OUTPUT_NUM = 3;
template <typename T>
class NMSWithMaskCPUKernel : public CPUKernel {
 public:
  NMSWithMaskCPUKernel() = default;
  ~NMSWithMaskCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void InitInputOutputSize(const CNodePtr &kernel_node) override;

 private:
  void NmsBitonicSortByKeyKernel(const int inner, const size_t ceil_power2, const T *input, T *data_buff,
                                 int *index_buff, int box_size);

  void MaskInit(size_t numSq, bool *row_mask);

  void PopulateOutput(const T *data_in, T *data_out, const int *index_buff, const int num, int box_size,
                      bool flip_mode);

  void Preprocess(const int num, int *sel_idx, bool *sel_boxes);

  bool IouDecision(const T *output, int box_A_start, int box_B_start, float IOU_value);

  void NmsPass(const int num, const float IOU_value, const T *output, int box_size, bool *row_mask);

  void ReducePass(const int num, bool *sel_boxes, const bool *row_mask);

  int num_input_{0};
  float iou_value_{0.0};
  size_t ceil_power_2{0};
  static const int box_size_ = 5;  //  pre_defined box width
  enum workspace_list_ { DATA_BUFF, INDEX_BUFF, ROW_MASK };
  enum output_list_ { OUTPUT, SEL_IDX, SEL_BOXES };
};

MS_REG_CPU_KERNEL_T(NMSWithMask,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeBool),
                    NMSWithMaskCPUKernel, float);

MS_REG_CPU_KERNEL_T(NMSWithMask,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat16)
                      .AddOutputAttr(kNumberTypeFloat16)
                      .AddOutputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeBool),
                    NMSWithMaskCPUKernel, float16);

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NMS_WITH_MASK_CPU_KERNEL_H_
