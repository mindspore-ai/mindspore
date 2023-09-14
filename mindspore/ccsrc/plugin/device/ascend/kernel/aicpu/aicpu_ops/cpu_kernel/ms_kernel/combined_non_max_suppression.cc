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
#include "cpu_kernel/ms_kernel/combined_non_max_suppression.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <vector>

#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
const uint32_t kInputNum = 6;
const uint32_t kOutputNum = 4;
const int kDim2 = 2;
const char *kCombinedNonMaxSuppression = "CombinedNonMaxSuppression";

void alloc_zeros(float *arr, int arr_len) {
  for (int i = 0; i < arr_len; i++) {
    arr[i] = 0.0;
  }
}

void alloc_zeros(int *arr, int arr_len) {
  for (int i = 0; i < arr_len; i++) {
    arr[i] = 0;
  }
}
}  // namespace

namespace aicpu {
// Normalize the diagonal to the input mode of bottom left and top right
void CombinedNonMaxSuppressionCpuKernel::regular_input2buffer(float **boxes_buffer, float *box_src,
                                                              const int class_idx) {
  /**
   * shape of box_src
   * box_src[num_boxes*q*4]
   * ways to visit box_src[i][class_idx][k] which stored by 1-dimension
   * box_src[i][class_idx][k]=box_src[i*q*4+class_idx*4+k]
   */
  int sub_box_len1 = q * 4;
  int box_len2 = (class_idx << 2);
  for (int i = 0; i < num_boxes; i++) {
    int box_len1 = i * sub_box_len1 + box_len2;
    if (box_src[box_len1] > box_src[box_len1 + 2]) {
      boxes_buffer[i][0] = box_src[box_len1 + 2];
      boxes_buffer[i][2] = box_src[box_len1 + 0];
    } else {
      boxes_buffer[i][0] = box_src[box_len1 + 0];
      boxes_buffer[i][2] = box_src[box_len1 + 2];
    }
    if (box_src[box_len1 + 1] > box_src[box_len1 + 3]) {
      boxes_buffer[i][1] = box_src[box_len1 + 3];
      boxes_buffer[i][3] = box_src[box_len1 + 1];
    } else {
      boxes_buffer[i][1] = box_src[box_len1 + 1];
      boxes_buffer[i][3] = box_src[box_len1 + 3];
    }
  }
}

// Calculate the area ratio of the intersection of two squares
float CombinedNonMaxSuppressionCpuKernel::IOU(float **boxes_buffer, int i, int j) {
  const float *box_a = boxes_buffer[i];
  const float *box_b = boxes_buffer[j];
  float area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
  float area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
  if (area_a <= 0 || area_b <= 0) {
    return 0.0;
  }
  float lx = box_a[0] > box_b[0] ? box_a[0] : box_b[0];
  float ly = box_a[1] > box_b[1] ? box_a[1] : box_b[1];
  float rx = box_a[2] < box_b[2] ? box_a[2] : box_b[2];
  float ry = box_a[3] < box_b[3] ? box_a[3] : box_b[3];
  float w = rx > lx ? (rx - lx) : 0;
  float h = ry > ly ? (ry - ly) : 0;
  float area = w * h;
  return area / (area_a + area_b - area);
}

/**
 * if soft_nms_sigma > 0.0, soft_nms is used, means update by score=score*exp(scale*iou^2)
 * if soft_nms_sigma <= 0.0, nms is used, means delete it when iou > iou_threshold
 * run non max suppression per bath per class
 */
void CombinedNonMaxSuppressionCpuKernel::non_max_suppression(float **boxes_buffer, float *scores_buffer,
                                                             std::vector<int> &selected) {
  std::priority_queue<non_max_suppression_local::score_index> pq;
  for (int i = 0; i < num_boxes; i++) {
    if (scores_buffer[i] > score_threshold) {
      pq.push(non_max_suppression_local::score_index(i, scores_buffer[i], 0));
    }
  }

  float scale = static_cast<float>(0.0);
  bool is_soft_nms = soft_nms_sigma > static_cast<float>(0.0);
  if (is_soft_nms) {
    scale = static_cast<float>(-0.5) / soft_nms_sigma;
  }

  float similarity;
  non_max_suppression_local::score_index next_si;

  while ((static_cast<int>(selected.size()) < size_per_class) && (!pq.empty())) {
    next_si = pq.top();
    float original_score = next_si.score;
    pq.pop();
    bool should_hard_suppress = false;

    for (int j = static_cast<int>(selected.size()) - 1; j >= next_si.suppress_begin_index; j--) {
      similarity = IOU(boxes_buffer, next_si.box_index, selected[j]);
      if (is_soft_nms) {
        next_si.score *=
          similarity <= iou_threshold ? std::exp(scale * similarity * similarity) : static_cast<float>(0.0);
      }
      if (!is_soft_nms && similarity > iou_threshold) {
        should_hard_suppress = true;
        break;
      }
      if (next_si.score <= score_threshold) break;
    }

    next_si.suppress_begin_index = selected.size();
    if (!should_hard_suppress) {
      if (next_si.score == original_score) {
        selected.push_back(next_si.box_index);
        continue;
      }
      if (next_si.score > score_threshold) {
        pq.push(next_si);
      }
    }
  }
}

void CombinedNonMaxSuppressionCpuKernel::nms_perclass(
  float *boxes, float *scores, std::vector<non_max_suppression_local::result_para> &sub_result_vec, int &result_size) {
  int k = 0;
  int box_idx;
  int boxe_len1;
  int sub_box_len1 = q * 4;
  int box_len2 = 0;
  float **boxes_buffer = new float *[num_boxes]();
  float *scores_buffer = new float[num_boxes]();
  for (int i = 0; i < num_boxes; i++) {
    boxes_buffer[i] = new float[4];
  }
  /**
   * shape of score and boxes
   * score[num_boxes*num_class]
   * boxes[num_boxes*q*4]
   */
  if (q == 1) {
    regular_input2buffer(boxes_buffer, boxes, 0);
  }
  for (int j = 0; j < num_class; j++) {
    for (int i = 0; i < num_boxes; i++) {
      scores_buffer[i] = scores[i * num_class + j];
    }
    if (q > 1) {
      regular_input2buffer(boxes_buffer, boxes, j);
      box_len2 = j * 4;
    }
    std::vector<int> selected;
    non_max_suppression(boxes_buffer, scores_buffer, selected);
    for (int i = 0; i < static_cast<int>(selected.size()); i++) {
      box_idx = selected[i];
      boxe_len1 = box_idx * sub_box_len1 + box_len2;
      sub_result_vec[k++] = {box_idx,
                             scores_buffer[box_idx],
                             j,
                             {boxes[boxe_len1 + 0], boxes[boxe_len1 + 1], boxes[boxe_len1 + 2], boxes[boxe_len1 + 3]}};
    }
    result_size += selected.size();
  }
  for (int i = 0; i < num_boxes; i++) {
    delete[] boxes_buffer[i];
  }
  delete[] boxes_buffer;
  delete[] scores_buffer;
  return;
}

uint32_t CombinedNonMaxSuppressionCpuKernel::nms_perbath(const CpuKernelContext &ctx, float *boxes, float *scores,
                                                         float *nmsed_boxes, float *nmsed_scores, float *nmsed_class,
                                                         int *valid_detection) {
  alloc_zeros(nmsed_boxes, num_bath * num_detection * 4);
  alloc_zeros(nmsed_scores, num_bath * num_detection);
  alloc_zeros(nmsed_class, num_bath * num_detection);
  alloc_zeros(valid_detection, num_bath);
  const float box_min = 0.0;
  const float box_max = 1.0;
  /**
   * shape of scores and boxes:
   * scores[num_bath*num_boxes*num_class]
   * boxes[num_bath*num_boxes*q*4]
   */
  int score_len2 = num_boxes * num_class;
  int boxes_len2 = num_boxes * q * 4;
  auto shard_nms = [&](size_t start, size_t end) {
    for (int i = start; i < static_cast<int>(end); i++) {
      int per_detections = 0;
      int scores_index = 0;
      int result_size = 0;
      std::vector<non_max_suppression_local::result_para> result_vec(size_per_class * num_class,
                                                                     {0, 0.0, 0, {0.0, 0.0, 0.0, 0.0}});
      nms_perclass(boxes + i * boxes_len2, scores + i * score_len2, result_vec, result_size);
      if (!pad_per_class) {
        per_detections = std::min(result_size, max_total_size);
      } else {
        per_detections = std::min(result_size, num_detection);
      }
      std::sort(result_vec.begin(), result_vec.begin() + result_size, non_max_suppression_local::result_cmp);
      scores_index = i * num_detection;
      for (int k = 0; k < per_detections; k++) {
        if (clip_boxes) {
          nmsed_boxes[(scores_index << 2) + 0] = std::max(std::min(result_vec[k].box_coord[0], box_max), box_min);
          nmsed_boxes[(scores_index << 2) + 1] = std::max(std::min(result_vec[k].box_coord[1], box_max), box_min);
          nmsed_boxes[(scores_index << 2) + 2] = std::max(std::min(result_vec[k].box_coord[2], box_max), box_min);
          nmsed_boxes[(scores_index << 2) + 3] = std::max(std::min(result_vec[k].box_coord[3], box_max), box_min);
          nmsed_scores[scores_index] = result_vec[k].score;
          nmsed_class[scores_index] = static_cast<float>(result_vec[k].class_idx);
        } else {
          nmsed_boxes[(scores_index << 2) + 0] = result_vec[k].box_coord[0];
          nmsed_boxes[(scores_index << 2) + 1] = result_vec[k].box_coord[1];
          nmsed_boxes[(scores_index << 2) + 2] = result_vec[k].box_coord[2];
          nmsed_boxes[(scores_index << 2) + 3] = result_vec[k].box_coord[3];
          nmsed_scores[scores_index] = result_vec[k].score;
          nmsed_class[scores_index] = static_cast<float>(result_vec[k].class_idx);
        }
        scores_index++;
      }
      valid_detection[i] = per_detections;
    }
  };
  uint32_t min_core_num = 1;
  int64_t max_core_num = std::max(min_core_num, aicpu::CpuKernelUtils::GetCPUNum(ctx) - 2);
  if (max_core_num > num_bath) {
    max_core_num = num_bath;
  }
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, num_bath, num_bath / max_core_num, shard_nms),
                      "CombinedNonMaxSuppression Compute failed in nms_perbath stage.");
  return KERNEL_STATUS_OK;
}

uint32_t CombinedNonMaxSuppressionCpuKernel::Compute(CpuKernelContext &ctx) {
  // check params
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                      "CombinedNonMaxSuppression check input and output number failed.");
  KERNEL_HANDLE_ERROR(CombinedNonMaxSuppressionCheck(ctx), "CombinedNonMaxSuppression check params failed.");
  CombinedNonMaxSuppressionCompute(ctx);
  return KERNEL_STATUS_OK;
}

uint32_t CombinedNonMaxSuppressionCpuKernel::CombinedNonMaxSuppressionCheck(const CpuKernelContext &ctx) {
  KERNEL_CHECK_NULLPTR(ctx.Input(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 0 data failed.");
  KERNEL_CHECK_NULLPTR(ctx.Input(1)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 1 data failed.");
  KERNEL_CHECK_NULLPTR(ctx.Input(2)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 2 data failed.");
  KERNEL_CHECK_NULLPTR(ctx.Input(3)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 3 data failed.");
  if (ctx.Input(4) != nullptr) {
    KERNEL_CHECK_NULLPTR(ctx.Input(4)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 4 data failed.");
  }
  KERNEL_CHECK_NULLPTR(ctx.Input(5)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get input 5 data failed.");
  KERNEL_CHECK_NULLPTR(ctx.Output(0)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 0 data failed.");
  KERNEL_CHECK_NULLPTR(ctx.Output(1)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 1 data failed.");
  KERNEL_CHECK_NULLPTR(ctx.Output(2)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 2 data failed.");
  KERNEL_CHECK_NULLPTR(ctx.Output(3)->GetData(), KERNEL_STATUS_PARAM_INVALID, "Get output 3 data failed.");
  KERNEL_CHECK_FALSE((ctx.Input(0)->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input0 [%s] must be [DT_FLOAT].", DTypeStr(ctx.Input(0)->GetDataType()).c_str());
  KERNEL_CHECK_FALSE((ctx.Input(1)->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input1 [%s] must be [DT_FLOAT].", DTypeStr(ctx.Input(1)->GetDataType()).c_str());
  KERNEL_CHECK_FALSE((ctx.Input(2)->GetDataType() == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input2 [%s] must be [DT_INT32].", DTypeStr(ctx.Input(2)->GetDataType()).c_str());
  KERNEL_CHECK_FALSE((ctx.Input(3)->GetDataType() == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input3 [%s] must be [DT_INT32].", DTypeStr(ctx.Input(3)->GetDataType()).c_str());
  if (ctx.Input(4) != NULL) {
    KERNEL_CHECK_FALSE((ctx.Input(4)->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                       "The data type of input4 [%s] must be [DT_FLOAT].",
                       DTypeStr(ctx.Input(4)->GetDataType()).c_str());
  }
  KERNEL_CHECK_FALSE((ctx.Input(5)->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of input5 [%s] must be [DT_FLOAT].", DTypeStr(ctx.Input(5)->GetDataType()).c_str());
  KERNEL_CHECK_FALSE((ctx.Output(0)->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output0 [%s] must be [DT_FLOAT].",
                     DTypeStr(ctx.Output(0)->GetDataType()).c_str());
  KERNEL_CHECK_FALSE((ctx.Output(1)->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output1 [%s] must be [DT_FLOAT].",
                     DTypeStr(ctx.Output(1)->GetDataType()).c_str());
  KERNEL_CHECK_FALSE((ctx.Output(2)->GetDataType() == DT_FLOAT), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output2 [%s] must be [DT_FLOAT].",
                     DTypeStr(ctx.Output(2)->GetDataType()).c_str());
  KERNEL_CHECK_FALSE((ctx.Output(3)->GetDataType() == DT_INT32), KERNEL_STATUS_PARAM_INVALID,
                     "The data type of output3 [%s] must be [DT_INT32].",
                     DTypeStr(ctx.Output(3)->GetDataType()).c_str());
  auto input0_shape = ctx.Input(0)->GetTensorShape();
  auto input1_shape = ctx.Input(1)->GetTensorShape();
  auto input2_shape = ctx.Input(2)->GetTensorShape();
  auto input3_shape = ctx.Input(3)->GetTensorShape();
  auto input5_shape = ctx.Input(5)->GetTensorShape();
  KERNEL_CHECK_FALSE((input0_shape->GetDims() == 4), KERNEL_STATUS_PARAM_INVALID, "The input0's dims [%d] must be 4",
                     input0_shape->GetDims());
  KERNEL_CHECK_FALSE((input1_shape->GetDims() == 3), KERNEL_STATUS_PARAM_INVALID, "The input1's dims [%d] must be 3",
                     input1_shape->GetDims());
  KERNEL_CHECK_FALSE(
    (input2_shape->GetDims() == 0 || (input2_shape->GetDims() == 1 && input2_shape->GetDimSize(0) == 1)),
    KERNEL_STATUS_PARAM_INVALID, "The input2's dims [%d] must be 0 or 1x1", input2_shape->GetDims());
  KERNEL_CHECK_FALSE(
    (input3_shape->GetDims() == 0 || (input3_shape->GetDims() == 1 && input3_shape->GetDimSize(0) == 1)),
    KERNEL_STATUS_PARAM_INVALID, "The input3's dims [%d] must be 0 or 1x1", input3_shape->GetDims());
  if (ctx.Input(4) != nullptr) {
    auto input4_shape = ctx.Input(4)->GetTensorShape();
    KERNEL_CHECK_FALSE(
      (input4_shape->GetDims() == 0 || (input4_shape->GetDims() == 1 && input4_shape->GetDimSize(0) == 1)),
      KERNEL_STATUS_PARAM_INVALID, "The input4's dims [%d] must be 0 or 1x1", input4_shape->GetDims());
  }
  KERNEL_CHECK_FALSE(
    (input5_shape->GetDims() == 0 || (input5_shape->GetDims() == 1 && input5_shape->GetDimSize(0) == 1)),
    KERNEL_STATUS_PARAM_INVALID, "The input5's dims [%d] must be 0 or 1x1", input5_shape->GetDims());
  auto output0_shape = ctx.Output(0)->GetTensorShape();
  auto output1_shape = ctx.Output(1)->GetTensorShape();
  auto output2_shape = ctx.Output(2)->GetTensorShape();
  auto output3_shape = ctx.Output(3)->GetTensorShape();
  KERNEL_CHECK_FALSE((output0_shape->GetDims() == 3), KERNEL_STATUS_PARAM_INVALID, "The output0's dims [%d] must be 3",
                     output0_shape->GetDims());
  KERNEL_CHECK_FALSE((output1_shape->GetDims() == 2), KERNEL_STATUS_PARAM_INVALID, "The output1's dims [%d] must be 2",
                     output1_shape->GetDims());
  KERNEL_CHECK_FALSE((output2_shape->GetDims() == 2), KERNEL_STATUS_PARAM_INVALID, "The output2's dims [%d] must be 2",
                     output2_shape->GetDims());
  KERNEL_CHECK_FALSE((output3_shape->GetDims() == 1), KERNEL_STATUS_PARAM_INVALID, "The output3's dims [%d] must be 1",
                     output3_shape->GetDims());
  KERNEL_CHECK_FALSE((input0_shape->GetDimSize(0) == input1_shape->GetDimSize(0)), KERNEL_STATUS_PARAM_INVALID,
                     "The input0's 1st dims [%d] need be same with the input1's 1st dims[%d]",
                     input0_shape->GetDimSize(0), input1_shape->GetDimSize(0));
  KERNEL_CHECK_FALSE((input0_shape->GetDimSize(1) == input1_shape->GetDimSize(1)), KERNEL_STATUS_PARAM_INVALID,
                     "The input0's 2nd dims [%d] need be same with the input1's 2nd dims[%d]",
                     input0_shape->GetDimSize(1), input1_shape->GetDimSize(1));
  KERNEL_CHECK_FALSE((input0_shape->GetDimSize(2) == input1_shape->GetDimSize(2) || input0_shape->GetDimSize(2) == 1),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The input0's 3th dims [%d] need be same with the input1's 3th dims [%d] or 1",
                     input0_shape->GetDimSize(2), output1_shape->GetDimSize(2));
  KERNEL_CHECK_FALSE((input0_shape->GetDimSize(3) == 4), KERNEL_STATUS_PARAM_INVALID,
                     "The input0's 4th dims [%d] need be same with 4", input0_shape->GetDimSize(1));
  KERNEL_CHECK_FALSE((output0_shape->GetDimSize(0) == output1_shape->GetDimSize(0) &&
                      output0_shape->GetDimSize(0) == output2_shape->GetDimSize(0) &&
                      output0_shape->GetDimSize(0) == output3_shape->GetDimSize(0)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The input0's 1st dims [%d], input1's 1st dims [%d],"
                     " input2's 1st dims [%d], input3's 1st dims [%d], need be same with each other",
                     output0_shape->GetDimSize(0), output1_shape->GetDimSize(0), output2_shape->GetDimSize(0),
                     output3_shape->GetDimSize(0));
  KERNEL_CHECK_FALSE((output0_shape->GetDimSize(1) == output1_shape->GetDimSize(1) &&
                      output0_shape->GetDimSize(1) == output2_shape->GetDimSize(1)),
                     KERNEL_STATUS_PARAM_INVALID,
                     "The input0's 2nd dims [%d], input1's 2nd dims [%d], input2's 2nd dims [%d],"
                     " need be same with each other",
                     output0_shape->GetDimSize(1), output1_shape->GetDimSize(1), output2_shape->GetDimSize(1));
  KERNEL_LOG_INFO(
    " CombinedNonMaxSuppressionCpuKernel[%s], input0: size[%llu], "
    " input1: size[%llu]",
    ctx.GetOpType().c_str(), ctx.Input(0)->GetDataSize(), ctx.Input(1)->GetDataSize());
  KERNEL_LOG_INFO(
    " output0: size[%llu], output1: size[%llu],"
    " output2: size[%llu], output3: size[%llu].",
    ctx.Output(0)->GetDataSize(), ctx.Output(1)->GetDataSize(), ctx.Output(2)->GetDataSize(),
    ctx.Output(3)->GetDataSize());

  return KERNEL_STATUS_OK;
}

uint32_t CombinedNonMaxSuppressionCpuKernel::CombinedNonMaxSuppressionCompute(const CpuKernelContext &ctx) {
  float *boxes = reinterpret_cast<float *>(ctx.Input(0)->GetData());
  float *scores = reinterpret_cast<float *>(ctx.Input(1)->GetData());
  max_output_size_per_class = *(reinterpret_cast<int *>(ctx.Input(2)->GetData()));
  max_total_size = *(reinterpret_cast<int *>(ctx.Input(3)->GetData()));
  iou_threshold = *(reinterpret_cast<float *>(ctx.Input(4)->GetData()));
  score_threshold = *(reinterpret_cast<float *>(ctx.Input(5)->GetData()));
  num_bath = static_cast<int>(ctx.Input(0)->GetTensorShape()->GetDimSize(0));
  num_boxes = static_cast<int>(ctx.Input(0)->GetTensorShape()->GetDimSize(1));
  q = static_cast<int>(ctx.Input(0)->GetTensorShape()->GetDimSize(kDim2));
  num_class = static_cast<int>(ctx.Input(1)->GetTensorShape()->GetDimSize(kDim2));
  pad_per_class = false;
  clip_boxes = true;
  if (ctx.GetAttr("pad_per_class") != nullptr) {
    pad_per_class = static_cast<bool>(ctx.GetAttr("pad_per_class")->GetBool());
  }
  if (ctx.GetAttr("clip_boxes") != nullptr) {
    clip_boxes = static_cast<bool>(ctx.GetAttr("clip_boxes")->GetBool());
  }
  float *nmsed_boxes = reinterpret_cast<float *>(ctx.Output(0)->GetData());
  float *nmsed_scores = reinterpret_cast<float *>(ctx.Output(1)->GetData());
  float *nmsed_class = reinterpret_cast<float *>(ctx.Output(2)->GetData());
  int *valid_detection = reinterpret_cast<int *>(ctx.Output(3)->GetData());
  auto output0_shape = ctx.Output(0)->GetTensorShape();
  auto output1_shape = ctx.Output(1)->GetTensorShape();
  auto output2_shape = ctx.Output(2)->GetTensorShape();
  auto output3_shape = ctx.Output(3)->GetTensorShape();
  size_per_class = max_output_size_per_class < num_boxes ? max_output_size_per_class : num_boxes;
  soft_nms_sigma = 0.0;
  if (pad_per_class) {
    num_detection = std::min(max_total_size, max_output_size_per_class * num_class);
  } else {
    num_detection = max_total_size;
  }
  KERNEL_CHECK_FALSE((output0_shape->GetDimSize(0) == num_bath), KERNEL_STATUS_PARAM_INVALID,
                     "The output0's 1nd dims [%d] must be [%d]", output0_shape->GetDimSize(0), num_bath);
  KERNEL_CHECK_FALSE((output1_shape->GetDimSize(0) == num_bath), KERNEL_STATUS_PARAM_INVALID,
                     "The output0's 1nd dims [%d] must be [%d]", output1_shape->GetDimSize(0), num_bath);
  KERNEL_CHECK_FALSE((output2_shape->GetDimSize(0) == num_bath), KERNEL_STATUS_PARAM_INVALID,
                     "The output0's 1nd dims [%d] must be [%d]", output2_shape->GetDimSize(0), num_bath);
  KERNEL_CHECK_FALSE((output3_shape->GetDimSize(0) == num_bath), KERNEL_STATUS_PARAM_INVALID,
                     "The output0's 1nd dims [%d] must be [%d]", output3_shape->GetDimSize(0), num_bath);
  KERNEL_CHECK_FALSE((max_output_size_per_class > 0), KERNEL_STATUS_PARAM_INVALID,
                     "max_output_size_per_class [%d] must be > 0", max_output_size_per_class);
  KERNEL_CHECK_FALSE((max_total_size > 0), KERNEL_STATUS_PARAM_INVALID, "max_total_size [%d] must be > 0",
                     max_total_size);
  KERNEL_CHECK_FALSE((iou_threshold >= 0 && iou_threshold <= 1), KERNEL_STATUS_PARAM_INVALID,
                     "iou_threshold [%f] must be in [0,1]", iou_threshold);
  KERNEL_CHECK_FALSE((static_cast<int>(output0_shape->GetDimSize(1)) == num_detection), KERNEL_STATUS_PARAM_INVALID,
                     "The output0's 2nd dims [%d] need be same with %d", output0_shape->GetDimSize(1), num_detection);
  KERNEL_CHECK_FALSE((static_cast<int>(output1_shape->GetDimSize(1)) == num_detection), KERNEL_STATUS_PARAM_INVALID,
                     "The output1's 2nd dims [%d] need be same with %d", output1_shape->GetDimSize(1), num_detection);
  KERNEL_CHECK_FALSE((static_cast<int>(output2_shape->GetDimSize(1)) == num_detection), KERNEL_STATUS_PARAM_INVALID,
                     "The output2's 2nd dims [%d] need be same with %d", output2_shape->GetDimSize(1), num_detection);
  nms_perbath(ctx, boxes, scores, nmsed_boxes, nmsed_scores, nmsed_class, valid_detection);
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kCombinedNonMaxSuppression, CombinedNonMaxSuppressionCpuKernel);
}  // namespace aicpu
