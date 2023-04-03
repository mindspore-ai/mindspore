/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_COMBINED_NON_MAX_SUPPRESSION_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_COMBINED_NON_MAX_SUPPRESSION_CPU_KERNEL_H_
#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <queue>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace non_max_suppression_local {
struct score_index {
  int box_index{0};
  float score{0};
  int suppress_begin_index{0};
  score_index() {}
  score_index(int bi, float s, int sbi) : box_index(bi), score(s), suppress_begin_index(sbi) {}
  bool operator<(const score_index &b) const {
    return (score < b.score) || (mindspore::common::IsFloatEqual(score, b.score) && (box_index > b.box_index));
  }
};
struct result_para {
  int box_index;
  float score;
  int class_idx;
  float box_coord[4];
};

bool result_cmp(const result_para &a, const result_para &b) { return a.score > b.score; }
}  // namespace non_max_suppression_local

namespace mindspore {
namespace kernel {
class CombinedNonMaxSuppressionCpuKernelMod : public NativeCpuKernelMod {
 public:
  CombinedNonMaxSuppressionCpuKernelMod() = default;
  ~CombinedNonMaxSuppressionCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  size_t nms_perbath(float *, float *, float *, float *, float *, int *);
  void regular_input2buffer(std::vector<std::vector<float>> *const, const float *, const int) const;
  float IOU(std::vector<std::vector<float>> *const, const int, const int) const;
  void non_max_suppression(std::vector<std::vector<float>> *const, std::vector<float> *const, std::vector<int> &) const;
  void nms_perclass(float *, float *, std::vector<non_max_suppression_local::result_para> &, int &) const;
  void CheckInput();
  void CheckOutput();

  int num_bath_ = 0;
  int num_boxes_ = 0;
  int q_ = 0;
  int num_class_ = 0;
  int num_detection_ = 0;
  int max_total_size_ = 0;
  // The length of each type of selection defined by the user
  int max_output_size_per_class_ = 0;
  // Calculation num_detection length
  int size_per_class_ = 0;
  // When lower than a score_threshold, delete the relevant box
  float score_threshold_ = 0.0;
  // When it is higher than the threshold value, according to the soft_nms_sigma determines deletion or decay
  float iou_threshold_ = 0.0;
  float soft_nms_sigma_ = 0.0;
  bool pad_per_class_ = 0;
  bool clip_boxes_ = 1;

  std::vector<int64_t> input0_shape_;
  std::vector<int64_t> input1_shape_;
  std::vector<int64_t> input2_shape_;
  std::vector<int64_t> input3_shape_;
  std::vector<int64_t> input4_shape_;
  std::vector<int64_t> input5_shape_;
  std::vector<int64_t> output0_shape_;
  std::vector<int64_t> output1_shape_;
  std::vector<int64_t> output2_shape_;
  std::vector<int64_t> output3_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_COMBINED_NON_MAX_SUPPRESSION_CPU_KERNEL_H_
