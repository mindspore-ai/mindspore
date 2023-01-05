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
#include "non_max_suppression_with_overlaps.h"

#include <algorithm>
#include <queue>

#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "cpu_attr_value.h"
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "kernel_log.h"
#include "status.h"
#include "utils/allocator_utils.h"
#include "utils/kernel_util.h"

namespace {
const char *kNonMaxSuppressionWithOverlaps = "NonMaxSuppressionWithOverlaps";
const uint32_t kOutputNum = 1;
const uint32_t kInputNum = 5;
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kThirdInputIndex = 2;
const uint32_t kforthInputIndex = 3;
const uint32_t kfifthInputIndex = 4;
const uint32_t kFirstOutputIndex = 0;
const uint32_t kOverlapsRank = 2;
}  // namespace

namespace aicpu {
uint32_t NonMaxSuppressionWithOverlapsCpuKernel::GetInputAndCheck(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "NonMaxSuppressionWithOverlaps check input and output number failed.");
  overlaps_ = ctx.Input(kFirstInputIndex);
  scores_ = ctx.Input(kSecondInputIndex);
  Tensor *max_output_size_tensor = ctx.Input(kThirdInputIndex);
  max_output_size_ = *static_cast<int32_t *>(max_output_size_tensor->GetData());
  KERNEL_CHECK_FALSE((max_output_size_ >= 0), KERNEL_STATUS_PARAM_INVALID,
                     "The input max_output_size must be non-negative");
  overlap_threshold_tensor_ = ctx.Input(kforthInputIndex);
  score_threshold_tensor_ = ctx.Input(kfifthInputIndex);
  output_indices_ = ctx.Output(kFirstOutputIndex);

  std::shared_ptr<TensorShape> overlaps_shape = overlaps_->GetTensorShape();
  int32_t overlaps_rank = overlaps_shape->GetDims();
  if (overlaps_rank != kOverlapsRank || overlaps_shape->GetDimSize(0) != overlaps_shape->GetDimSize(1)) {
    KERNEL_LOG_ERROR(
      "The input dim size of overlaps must be 2-D and must be square, "
      "while %d, %lld",
      overlaps_rank, overlaps_shape->GetDimSize(1));
    return KERNEL_STATUS_PARAM_INVALID;
  }
  num_boxes_ = overlaps_shape->GetDimSize(0);

  std::shared_ptr<TensorShape> scores_shape = scores_->GetTensorShape();
  int32_t scores_rank = scores_shape->GetDims();
  KERNEL_CHECK_FALSE((scores_rank == 1), KERNEL_STATUS_PARAM_INVALID,
                     "The input dim size of scores must be 1-D, while %d.", scores_rank);
  KERNEL_CHECK_FALSE((scores_shape->GetDimSize(0) == num_boxes_), KERNEL_STATUS_PARAM_INVALID,
                     "The len of scores must be equal to the number of boxes, "
                     "while dims[%lld], num_boxes_[%d].",
                     scores_shape->GetDimSize(0), num_boxes_);

  overlaps_dtype_ = static_cast<DataType>(overlaps_->GetDataType());
  if (overlaps_dtype_ != DT_FLOAT) {
    KERNEL_LOG_ERROR("The dtype of input[0] overlaps must be float.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  scores_dtype_ = static_cast<DataType>(scores_->GetDataType());
  if (scores_dtype_ != DT_FLOAT) {
    KERNEL_LOG_ERROR("The dtype of input[1] scores must be float.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  overlap_threshold_dtype_ = static_cast<DataType>(overlap_threshold_tensor_->GetDataType());
  if (overlap_threshold_dtype_ != DT_FLOAT) {
    KERNEL_LOG_ERROR("The dtype of input[3] overlap_threshold must be float.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  score_threshold_dtype_ = static_cast<DataType>(score_threshold_tensor_->GetDataType());
  if (score_threshold_dtype_ != DT_FLOAT) {
    KERNEL_LOG_ERROR("The dtype of input[4] score_threshold must be float.");
    return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename T_threshold>
uint32_t NonMaxSuppressionWithOverlapsCpuKernel::DoNonMaxSuppressionWithOverlapsOp() {
  KERNEL_LOG_INFO("DoNonMaxSuppressionWithOverlapsOp start!!");
  Eigen::TensorMap<Eigen::Tensor<T, kOverlapsRank, Eigen::RowMajor>> overlaps_map(
    reinterpret_cast<T *>(overlaps_->GetData()), num_boxes_, num_boxes_);
  std::vector<T> scores_data(num_boxes_);
  std::copy_n(reinterpret_cast<T *>(scores_->GetData()), num_boxes_, scores_data.begin());
  auto overlap_threshold = static_cast<T>(*(static_cast<T_threshold *>(overlap_threshold_tensor_->GetData())));
  auto score_threshold = static_cast<T>(*(static_cast<T_threshold *>(score_threshold_tensor_->GetData())));
  std::unique_ptr<int32_t[]> indices_data(new int32_t[max_output_size_]);
  if (indices_data == nullptr) {
    KERNEL_LOG_ERROR("DoNonMaxSuppressionWithOverlapsOp: new indices_data failed");
    return KERNEL_STATUS_INNER_ERROR;
  }
  struct Candidate {
    int box_index;
    T score;
    int suppress_begin_index;
  };
  auto cmp = [](const Candidate boxes_i, const Candidate boxes_j) { return boxes_i.score < boxes_j.score; };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)> candidate_priority_queue(cmp);
  for (uint32_t i = 0; i < scores_data.size(); ++i) {
    if (scores_data[i] > score_threshold) {
      candidate_priority_queue.emplace(Candidate({(int)i, scores_data[i]}));
    }
  }
  T similarity = static_cast<T>(0.0);
  Candidate next_candidate = {.box_index = 0, .score = static_cast<T>(0.0), .suppress_begin_index = 0};
  int32_t cnt = 0;
  while (cnt < max_output_size_ && !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    candidate_priority_queue.pop();
    bool should_suppress = false;
    for (int j = cnt - 1; j >= next_candidate.suppress_begin_index; --j) {
      similarity = overlaps_map(next_candidate.box_index, indices_data[j]);
      if (similarity >= overlap_threshold) {
        should_suppress = true;
        break;
      }
    }
    next_candidate.suppress_begin_index = cnt;
    if (!should_suppress) {
      indices_data[cnt] = next_candidate.box_index;
      cnt += 1;
    }
  }
  auto value = reinterpret_cast<int32_t *>(output_indices_->GetData());
  for (int j = 0; j <= std::min(cnt, max_output_size_) - 1; j++) {
    *(value + j) = indices_data[j];
  }
  output_indices_->GetTensorShape()->SetDimSizes({std::min(cnt, max_output_size_)});
  KERNEL_LOG_INFO("DoNonMaxSuppressionWithOverlapsOp end!!");
  return KERNEL_STATUS_OK;
}

uint32_t NonMaxSuppressionWithOverlapsCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_LOG_INFO("NonMaxSuppressionWithOverlaps kernel in.");
  uint32_t res = GetInputAndCheck(ctx);
  if (res != KERNEL_STATUS_OK) {
    return res;
  }
  res = DoNonMaxSuppressionWithOverlapsOp<float, float>();
  KERNEL_CHECK_FALSE((res == KERNEL_STATUS_OK), res, "Compute failed.");
  KERNEL_LOG_INFO("Compute end!!");
  return KERNEL_STATUS_OK;
}
REGISTER_CPU_KERNEL(kNonMaxSuppressionWithOverlaps, NonMaxSuppressionWithOverlapsCpuKernel);
}  // namespace aicpu
