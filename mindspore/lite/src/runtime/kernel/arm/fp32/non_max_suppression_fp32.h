/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_NON_MAX_SUPPRESSION_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_NON_MAX_SUPPRESSION_H_

#include <vector>
#include <algorithm>
#include "src/lite_kernel.h"
#include "mindspore/lite/nnacl/non_max_suppression_parameter.h"

using mindspore::lite::RET_OK;

namespace mindspore::kernel {
class NonMaxSuppressionCPUKernel : public LiteKernel {
 public:
  NonMaxSuppressionCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}

  ~NonMaxSuppressionCPUKernel() override = default;

  int Init() override;
  int ReSize() override { return RET_OK; };
  int PreProcess() override;
  int Run() override;

 private:
  int GetParams();

 private:
  int center_point_box_;
  float iou_threshold_;
  float score_threshold_;
  int32_t max_output_per_class_;
  NMSParameter *param_ = nullptr;
};

typedef struct NMSIndex {
  int32_t batch_index_;
  int32_t class_index_;
  int32_t box_index_;
} NMSIndex;

class NMSBox {
 public:
  NMSBox() = default;
  ~NMSBox() = default;
  explicit NMSBox(float score, int box_index, int center_point_box, float y_a, float x_a, float y_b, float x_b)
      : score_(score), index_(box_index) {
    if (0 == center_point_box) {
      y1_ = std::min(y_a, y_b);
      y2_ = std::max(y_a, y_b);
      x1_ = std::min(x_a, x_b);
      x2_ = std::max(x_a, x_b);
    } else {
      // x_center, y_center, width, height
      float half_wid = x_b / 2;
      x1_ = x_a - half_wid;
      x2_ = x_a + half_wid;
      float half_height = y_b / 2;
      y1_ = y_a - half_height;
      y2_ = y_a + half_height;
    }
    area_ = (y2_ - y1_) * (x2_ - x1_);
  }
  inline bool operator<(const NMSBox &box) const { return score_ < box.score_; }

 public:
  float score_;
  int index_;
  float y1_;  // y1 x1 y2 x2 ascending order
  float y2_;
  float x1_;
  float x2_;
  float area_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_FP32_NON_MAX_SUPPRESSION_H_
