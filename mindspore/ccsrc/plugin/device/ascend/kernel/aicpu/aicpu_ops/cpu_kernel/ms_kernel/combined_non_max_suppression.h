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
#ifndef AICPU_KERNELS_NORMALIZED_COMBINEDNONMAXSUPPRESSION_H_
#define AICPU_KERNELS_NORMALIZED_COMBINEDNONMAXSUPPRESSION_H_

#include <vector>

#include "cpu_kernel/inc/cpu_ops_kernel.h"

#include "utils/bcast.h"

namespace non_max_suppression_local {
struct score_index {
  int box_index;
  float score;
  int suppress_begin_index;
  score_index() {}
  score_index(int bi, float s, int sbi) : box_index(bi), score(s), suppress_begin_index(sbi) {}
  bool operator<(const score_index &b) const {
    return (score < b.score) || ((score == b.score) && (box_index > b.box_index));
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

namespace aicpu {

class CombinedNonMaxSuppressionCpuKernel : public CpuKernel {
 public:
  CombinedNonMaxSuppressionCpuKernel() = default;
  ~CombinedNonMaxSuppressionCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CombinedNonMaxSuppressionCheck(const CpuKernelContext &ctx);

  uint32_t CombinedNonMaxSuppressionCompute(const CpuKernelContext &ctx);
  uint32_t nms_perbath(const CpuKernelContext &, float *, float *, float *, float *, float *, int *);
  void regular_input2buffer(float **, float *, const int);
  float IOU(float **, int, int);
  void non_max_suppression(float **, float *, std::vector<int> &);
  void nms_perclass(float *, float *, std::vector<non_max_suppression_local::result_para> &, int &);
  int num_bath;
  int num_boxes;
  int q;
  int num_class;
  // per batch size
  int num_detection;
  int max_total_size;
  // The length of each type of selection defined by the user
  int max_output_size_per_class;
  // Calculation num_detection length
  int size_per_class;
  // When lower than a score_threshold, delete the relevant box
  float score_threshold;
  // When it is higher than the threshold value, according to the soft_nms_sigma determines deletion or decay
  float iou_threshold;
  float soft_nms_sigma;
  bool pad_per_class;
  bool clip_boxes;
};
}  // namespace aicpu
#endif
