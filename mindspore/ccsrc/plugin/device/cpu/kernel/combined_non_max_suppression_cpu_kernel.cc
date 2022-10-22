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
#include "plugin/device/cpu/kernel/combined_non_max_suppression_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr char kKernelName[] = "CombinedNonMaxSuppression";
constexpr size_t kCombinedNonMaxSuppressionInputsNum = 6;
constexpr size_t kCombinedNonMaxSuppressionOutputsNum = 4;
constexpr size_t KIndex0 = 0;
constexpr size_t KIndex1 = 1;
constexpr size_t KIndex2 = 2;
constexpr size_t KIndex3 = 3;
constexpr size_t KIndex4 = 4;
constexpr size_t KIndex5 = 5;
constexpr size_t KIndex6 = 6;
constexpr size_t KIndex7 = 7;
constexpr size_t KIndex8 = 8;
constexpr size_t KIndex9 = 9;
constexpr size_t KIndex10 = 10;
constexpr int64_t DimSize4 = 4;
constexpr float k_5 = 0.5;
constexpr int multiplier = 4;
}  // namespace

void CombinedNonMaxSuppressionCpuKernelMod::regular_input2buffer(std::vector<std::vector<float>> *boxes_buffer,
                                                                 float *box_src, int class_idx) {
  /**
   * shape of box_src
   * box_src[num_boxes_*q_*4]
   * ways to visit box_src[i][class_idx][k] which stored by 1-dimension
   * box_src[i][class_idx][k]=box_src[i*q_*4+class_idx*4+k]
   */
  int sub_box_len1 = q_ * multiplier;
  int box_len2 = (class_idx << KIndex2);
  for (size_t i = 0; i < IntToSize(num_boxes_); i++) {
    size_t box_len1 = IntToSize(i * sub_box_len1 + box_len2);
    if (box_src[box_len1] > box_src[box_len1 + KIndex2]) {
      (*boxes_buffer)[i][0] = box_src[box_len1 + KIndex2];
      (*boxes_buffer)[i][KIndex2] = box_src[box_len1 + 0];
    } else {
      (*boxes_buffer)[i][0] = box_src[box_len1 + 0];
      (*boxes_buffer)[i][KIndex2] = box_src[box_len1 + KIndex2];
    }
    if (box_src[box_len1 + KIndex1] > box_src[box_len1 + KIndex3]) {
      (*boxes_buffer)[i][KIndex1] = box_src[box_len1 + KIndex3];
      (*boxes_buffer)[i][KIndex3] = box_src[box_len1 + KIndex1];
    } else {
      (*boxes_buffer)[i][KIndex1] = box_src[box_len1 + KIndex1];
      (*boxes_buffer)[i][KIndex3] = box_src[box_len1 + KIndex3];
    }
  }
}

// Calculate the area ratio of the intersection of two squares
float CombinedNonMaxSuppressionCpuKernelMod::IOU(std::vector<std::vector<float>> *boxes_buffer, int i, int j) const {
  std::vector<float> box_a = (*boxes_buffer)[i];
  std::vector<float> box_b = (*boxes_buffer)[j];
  float lx, ly, rx, ry;
  float w, h;
  float area;
  float area_a = (box_a[KIndex2] - box_a[0]) * (box_a[KIndex3] - box_a[KIndex1]);
  float area_b = (box_b[KIndex2] - box_b[0]) * (box_b[KIndex3] - box_b[KIndex1]);
  if (area_a <= 0 || area_b <= 0) {
    return 0.0;
  }
  lx = box_a[0] > box_b[0] ? box_a[0] : box_b[0];
  ly = box_a[KIndex1] > box_b[KIndex1] ? box_a[KIndex1] : box_b[KIndex1];
  rx = box_a[KIndex2] < box_b[KIndex2] ? box_a[KIndex2] : box_b[KIndex2];
  ry = box_a[KIndex3] < box_b[KIndex3] ? box_a[KIndex3] : box_b[KIndex3];
  w = rx > lx ? (rx - lx) : 0;
  h = ry > ly ? (ry - ly) : 0;
  area = w * h;
  return area / (area_a + area_b - area);
}

/**
 * if soft_nms_sigma_ > 0.0, soft_nms is used, means update by score=score*exp(scale*iou^2)
 * if soft_nms_sigma_ <= 0.0, nms is used, means delete it when iou > iou_threshold_
 * run non max suppression per bath per class
 */
void CombinedNonMaxSuppressionCpuKernelMod::non_max_suppression(std::vector<std::vector<float>> *boxes_buffer,
                                                                std::vector<float> *scores_buffer,
                                                                std::vector<int> &selected) {
  std::priority_queue<non_max_suppression_local::score_index> pq;
  for (size_t i = 0; i < IntToSize(num_boxes_); i++) {
    if ((*scores_buffer)[i] > score_threshold_) {
      pq.push(non_max_suppression_local::score_index(static_cast<int>(i), (*scores_buffer)[i], 0));
    }
  }

  float scale = static_cast<float>(0.0);
  bool is_soft_nms = soft_nms_sigma_ > static_cast<float>(0.0);
  if (is_soft_nms) {
    scale = static_cast<float>(-k_5) / soft_nms_sigma_;
  }

  float similarity;
  non_max_suppression_local::score_index next_si;
  while (static_cast<int>(selected.size()) < size_per_class_ && !pq.empty()) {
    next_si = pq.top();
    float original_score = next_si.score;
    pq.pop();
    bool should_hard_suppress = false;
    for (int j = SizeToInt(selected.size()) - 1; j >= next_si.suppress_begin_index; j--) {
      similarity = IOU(boxes_buffer, next_si.box_index, selected[IntToSize(j)]);
      if (is_soft_nms) {
        next_si.score *=
          similarity <= iou_threshold_ ? std::exp(scale * similarity * similarity) : static_cast<float>(0.0);
      }
      if (!is_soft_nms && similarity > iou_threshold_) {
        should_hard_suppress = true;
        break;
      }
      if (next_si.score <= score_threshold_) break;
    }
    next_si.suppress_begin_index = static_cast<int>(selected.size());
    if (!should_hard_suppress) {
      if (mindspore::common::IsFloatEqual(next_si.score, original_score)) {
        selected.push_back(next_si.box_index);
        continue;
      }
      if (next_si.score > score_threshold_) {
        pq.push(next_si);
      }
    }
  }
}

void CombinedNonMaxSuppressionCpuKernelMod::nms_perclass(
  float *boxes, float *scores, std::vector<non_max_suppression_local::result_para> &sub_result_vec, int &result_size) {
  size_t k = 0;
  int box_idx;
  size_t boxe_len1;
  int sub_box_len1 = q_ * multiplier;
  int box_len2 = 0;
  std::vector<std::vector<float>> boxes_buffer(num_boxes_, std::vector<float>(KIndex4));
  std::vector<float> scores_buffer(num_boxes_);
  /**
   * shape of score and boxes
   * score[num_boxes_*num_class_]
   * boxes[num_boxes_*q_*4]
   */
  if (q_ == 1) {
    regular_input2buffer(&boxes_buffer, boxes, 0);
  }
  for (int j = 0; j < num_class_; j++) {
    for (int i = 0; i < num_boxes_; i++) {
      scores_buffer[IntToSize(i)] = scores[IntToSize(i * num_class_ + j)];
    }
    if (q_ > 1) {
      regular_input2buffer(&boxes_buffer, boxes, j);
      box_len2 = j * multiplier;
    }
    std::vector<int> selected;
    non_max_suppression(&boxes_buffer, &scores_buffer, selected);
    for (size_t i = 0; i < selected.size(); i++) {
      box_idx = selected[i];
      boxe_len1 = IntToSize(box_idx * sub_box_len1 + box_len2);
      sub_result_vec[k++] = {
        box_idx,
        scores_buffer[IntToSize(box_idx)],
        j,
        {boxes[boxe_len1 + 0], boxes[boxe_len1 + 1], boxes[boxe_len1 + KIndex2], boxes[boxe_len1 + KIndex3]}};
    }
    result_size += SizeToInt(selected.size());
  }
}

size_t CombinedNonMaxSuppressionCpuKernelMod::nms_perbath(float *boxes, float *scores, float *nmsed_boxes,
                                                          float *nmsed_scores, float *nmsed_class,
                                                          int *valid_detection) {
  int box_size = num_bath_ * num_detection_ * sizeof(float) * multiplier;
  int score_size = num_bath_ * num_detection_ * sizeof(float);
  void(memset_s(nmsed_boxes, box_size, 0.0, box_size));
  void(memset_s(nmsed_scores, score_size, 0.0, score_size));
  void(memset_s(nmsed_class, score_size, 0.0, score_size));
  void(memset_s(valid_detection, sizeof(int) * num_bath_, 0, sizeof(int) * num_bath_));
  const float box_min = 0.0;
  const float box_max = 1.0;
  /**
   * shape of scores and boxes:
   * scores[num_bath_*num_boxes_*num_class_]
   * boxes[num_bath_*num_boxes_*q_*4]
   */
  int score_len2 = num_boxes_ * num_class_;
  int boxes_len2 = num_boxes_ * q_ * multiplier;
  auto shard_nms = [this, &boxes, &scores, score_len2, boxes_len2, &nmsed_boxes, &nmsed_scores, &nmsed_class,
                    &valid_detection, box_max, box_min](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      int tmp_i = static_cast<int>(i);
      int per_detections = 0;
      size_t scores_KIndex = 0;
      int result_size = 0;
      std::vector<non_max_suppression_local::result_para> result_vec(size_per_class_ * num_class_,
                                                                     {0, 0.0, 0, {0.0, 0.0, 0.0, 0.0}});
      nms_perclass(boxes + tmp_i * boxes_len2, scores + tmp_i * score_len2, result_vec, result_size);
      if (!pad_per_class_) {
        per_detections = std::min(result_size, max_total_size_);
      } else {
        per_detections = std::min(result_size, num_detection_);
      }
      std::sort(result_vec.begin(), result_vec.begin() + result_size, non_max_suppression_local::result_cmp);
      scores_KIndex = IntToSize(tmp_i * num_detection_);
      for (size_t k = 0; k < IntToSize(per_detections); k++) {
        if (clip_boxes_) {
          nmsed_boxes[(scores_KIndex << KIndex2) + 0] =
            std::max(std::min(result_vec[k].box_coord[0], box_max), box_min);
          nmsed_boxes[(scores_KIndex << KIndex2) + KIndex1] =
            std::max(std::min(result_vec[k].box_coord[KIndex1], box_max), box_min);
          nmsed_boxes[(scores_KIndex << KIndex2) + KIndex2] =
            std::max(std::min(result_vec[k].box_coord[KIndex2], box_max), box_min);
          nmsed_boxes[(scores_KIndex << KIndex2) + KIndex3] =
            std::max(std::min(result_vec[k].box_coord[KIndex3], box_max), box_min);
          nmsed_scores[scores_KIndex] = result_vec[k].score;
          nmsed_class[scores_KIndex] = static_cast<float>(result_vec[k].class_idx);
        } else {
          nmsed_boxes[(scores_KIndex << KIndex2) + 0] = result_vec[k].box_coord[0];
          nmsed_boxes[(scores_KIndex << KIndex2) + KIndex1] = result_vec[k].box_coord[KIndex1];
          nmsed_boxes[(scores_KIndex << KIndex2) + KIndex2] = result_vec[k].box_coord[KIndex2];
          nmsed_boxes[(scores_KIndex << KIndex2) + KIndex3] = result_vec[k].box_coord[KIndex3];
          nmsed_scores[scores_KIndex] = result_vec[k].score;
          nmsed_class[scores_KIndex] = static_cast<float>(result_vec[k].class_idx);
        }
        scores_KIndex++;
      }
      valid_detection[i] = per_detections;
    }
  };
  ParallelLaunchAutoSearch(shard_nms, num_bath_, this, &parallel_search_info_);
  return true;
}

void CombinedNonMaxSuppressionCpuKernelMod::CheckInput() {
  constexpr int kInputDimension0 = 4;
  if (input0_shape_.size() != kInputDimension0) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the boxes's dims must be 4, but got " << input0_shape_.size()
                      << " .";
  }
  constexpr int kInputDimension1 = 3;
  if (input1_shape_.size() != kInputDimension1) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the scores's dims must be 3, but got " << input1_shape_.size()
                      << " .";
  }
  if (input2_shape_.size() != 0) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the max_output_size_per_class's dims must be 0, but got "
                      << input1_shape_.size() << " .";
  }
  if (input3_shape_.size() != 0) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the max_total_size's dims must be 0, but got "
                      << input1_shape_.size() << " .";
  }
  if (input4_shape_.size() != 0) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the iou_threshold's dims must be, 0 but got "
                      << input1_shape_.size() << " .";
  }
  if (input5_shape_.size() != 0) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the score_threshold's dims must be 0, but got "
                      << input1_shape_.size() << ".";
  }
  if (input0_shape_[0] != input1_shape_[0]) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the boxes's 1st dim need to be with the scores's 1st dim, but got "
                      << input0_shape_[0] << " and " << input1_shape_[0] << ".";
  }
  if (input0_shape_[KIndex1] != input1_shape_[KIndex1]) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the boxes's 2nd dim need to be same with the scores's 2nd dim,"
                      << " but got " << input0_shape_[KIndex1] << " and " << input1_shape_[KIndex1] << ".";
  }
  if (input0_shape_[KIndex2] != input1_shape_[KIndex2] && input0_shape_[KIndex2] != 1) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the boxes's 3rd dim need to be same with the scores's 3rd dim or 1"
                      << ", but got " << input0_shape_[KIndex2] << ".";
  }
  if (input0_shape_[KIndex3] != DimSize4) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the boxes's 4th dim need to be equal to 4, but got "
                      << input0_shape_[KIndex3] << ".";
  }
}

void CombinedNonMaxSuppressionCpuKernelMod::CheckOutput() {
  constexpr size_t kOutputDimension0 = 3;
  constexpr size_t kOutputDimension1 = 2;
  constexpr size_t kOutputDimension2 = 2;
  constexpr size_t kOutputDimension3 = 1;
  if (output0_shape_.size() != kOutputDimension0) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_boxes's dims must be 3, but got "
                      << output0_shape_.size() << ".";
  }
  if (output1_shape_.size() != kOutputDimension1) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_scores's dims must be 2, but got "
                      << output1_shape_.size() << ".";
  }
  if (output2_shape_.size() != kOutputDimension2) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_classes's dims must be 2, but got "
                      << output2_shape_.size() << ".";
  }
  if (output3_shape_.size() != kOutputDimension3) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the valid_detection's dims must be 1, but got "
                      << output3_shape_.size() << ".";
  }
  if ((output0_shape_[0] != output1_shape_[0] || output0_shape_[0] != output2_shape_[0]) ||
      output0_shape_[0] != output3_shape_[0]) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_boxes's 1st dim, nmsed_scores's 1st dim,"
                      << " nmsed_classes's 1st dim, valid_detection's 1st dim, must be same with each other, but got"
                      << " four as follows: " << output0_shape_[0] << " and " << output1_shape_[0] << " and "
                      << output2_shape_[0] << " and " << output3_shape_[0] << ".";
  }
  if (output0_shape_[1] != output1_shape_[1] || output0_shape_[1] != output2_shape_[1]) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_boxes's 2nd dim, nmsed_scores's 2nd dim, nmsed_classes's"
                      << " 2nd dim bust be same with each other, but got the three as follows: " << output0_shape_[1]
                      << " and " << output1_shape_[1] << " and " << output2_shape_[1] << ".";
  }
  if (static_cast<int>(output0_shape_[0]) != num_bath_) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_boxes's 1st dim must be same with the boxes's 1st dim,"
                      << " but got " << output0_shape_[0] << ".";
  }
  if (static_cast<int>(output1_shape_[0]) != num_bath_) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_scores's 1st dim must be same with the boxes's 1st dim,"
                      << " but got " << output1_shape_[0] << ".";
  }
  if (static_cast<int>(output2_shape_[0]) != num_bath_) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the nmsed_classes's 1st dim must be same with the boxes's 1st dim,"
                      << " but got " << output2_shape_[0] << ".";
  }
  if (static_cast<int>(output3_shape_[0]) != num_bath_) {
    MS_LOG(EXCEPTION) << "For " << kKernelName << ", the valid_detection's 1st dim must be same with the boxes's 1st"
                      << " dim, but got " << output3_shape_[0] << ".";
  }
}

void CombinedNonMaxSuppressionCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  node_wpt_ = kernel_node;
  input0_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  input1_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, KIndex1);
  input2_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, KIndex2);
  input3_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, KIndex3);
  input4_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, KIndex4);
  input5_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, KIndex5);
  soft_nms_sigma_ = 0.0;
  num_bath_ = static_cast<int>(input0_shape_[0]);
  num_boxes_ = static_cast<int>(input0_shape_[KIndex1]);
  q_ = static_cast<int>(input0_shape_[KIndex2]);
  num_class_ = static_cast<int>((input1_shape_[KIndex2]));
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);

  pad_per_class_ = false;
  clip_boxes_ = true;
  auto prim = common::AnfAlgo::GetCNodePrimitive(kernel_node);
  auto pad_per_class = prim->GetAttr("pad_per_class");
  auto clip_boxes = prim->GetAttr("clip_boxes");
  if (pad_per_class != nullptr) {
    pad_per_class_ = GetValue<bool>(pad_per_class);
  }
  if (clip_boxes != nullptr) {
    clip_boxes_ = GetValue<bool>(clip_boxes);
  }
  CHECK_KERNEL_INPUTS_NUM(input_num, kCombinedNonMaxSuppressionInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kCombinedNonMaxSuppressionOutputsNum, kernel_name_);
}

bool CombinedNonMaxSuppressionCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                                   const std::vector<kernel::AddressPtr> &,
                                                   const std::vector<kernel::AddressPtr> &outputs) {
  float *boxes = static_cast<float *>(inputs[0]->addr);
  float *scores = static_cast<float *>(inputs[KIndex1]->addr);
  max_output_size_per_class_ = *(static_cast<int *>(inputs[KIndex2]->addr));
  max_total_size_ = *(static_cast<int *>(inputs[KIndex3]->addr));
  iou_threshold_ = *(static_cast<float *>(inputs[KIndex4]->addr));
  score_threshold_ = *(static_cast<float *>(inputs[KIndex5]->addr));
  float *nmsed_boxes = static_cast<float *>(outputs[KIndex0]->addr);
  float *nmsed_scores = static_cast<float *>(outputs[KIndex1]->addr);
  float *nmsed_class = static_cast<float *>(outputs[KIndex2]->addr);
  int *valid_detection = static_cast<int *>(outputs[KIndex3]->addr);
  if (pad_per_class_) {
    num_detection_ = std::min(max_total_size_, max_output_size_per_class_ * num_class_);
  } else {
    num_detection_ = max_total_size_;
  }
  auto node_ = node_wpt_.lock();
  if (!node_) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', node_wpt_(kernel_node) is expired. Error no: " << node_ << ".";
  }
  ShapeVector shape0 = {input0_shape_[0], static_cast<int64_t>(num_detection_), DimSize4};
  ShapeVector shape1 = {input0_shape_[0], static_cast<int64_t>(num_detection_)};
  ShapeVector shape2 = {input0_shape_[0], static_cast<int64_t>(num_detection_)};
  ShapeVector shape3 = {input0_shape_[0]};
  common::AnfAlgo::SetOutputInferTypeAndShape(
    {kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeFloat32, kNumberTypeInt32}, {shape0, shape1, shape2, shape3},
    node_.get());
  output0_shape_ = AnfAlgo::GetOutputDeviceShape(node_, KIndex0);
  output1_shape_ = AnfAlgo::GetOutputDeviceShape(node_, KIndex1);
  output2_shape_ = AnfAlgo::GetOutputDeviceShape(node_, KIndex2);
  output3_shape_ = AnfAlgo::GetOutputDeviceShape(node_, KIndex3);
  size_per_class_ = max_output_size_per_class_ < num_boxes_ ? max_output_size_per_class_ : num_boxes_;
  CheckInput();
  CheckOutput();
  if (max_total_size_ <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " max_total_size must be > 0, but got " << max_total_size_ << ".";
  }
  if (max_output_size_per_class_ <= 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " max_output_size_per_class must be > 0, but got "
                      << max_output_size_per_class_ << ".";
  }
  if (iou_threshold_ < 0 || iou_threshold_ > 1) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " iou_threshold must be in [0,1], but got " << iou_threshold_ << ".";
  }
  if (static_cast<int>(output0_shape_[KIndex1]) != num_detection_) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " The nmsed_boxes's 2nd dims must be same with " << num_detection_
                      << "but got " << output0_shape_[KIndex1] << ".";
  }
  if (static_cast<int>(output1_shape_[KIndex1]) != num_detection_) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " The nmsed_scores's 2nd dims must be same with " << num_detection_
                      << "but got " << output1_shape_[KIndex1] << ".";
  }
  if (static_cast<int>(output2_shape_[KIndex1]) != num_detection_) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " The nmsed_classes's 2nd dims must be same with " << num_detection_
                      << "but got " << output2_shape_[KIndex1] << ".";
  }
  (void)nms_perbath(boxes, scores, nmsed_boxes, nmsed_scores, nmsed_class, valid_detection);
  return true;
}
std::vector<KernelAttr> CombinedNonMaxSuppressionCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> kernel_attr_list = {
    KernelAttr()
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeFloat32)
      .AddOutputAttr(kNumberTypeInt32),
  };

  return kernel_attr_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, CombinedNonMaxSuppression, CombinedNonMaxSuppressionCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
