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

#include "src/runtime/kernel/arm/fp32/non_max_suppression_fp32.h"
#include <queue>
#include <functional>
#include <utility>
#include "nnacl/non_max_suppression_parameter.h"
#include "schema/model_generated.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::schema::PrimitiveType_NonMaxSuppression;

namespace mindspore::kernel {
namespace {
constexpr size_t kMinInputsSize = 2;
constexpr size_t kMaxInputsSize = 5;
constexpr size_t kOutputNum = 1;
constexpr size_t kBoxTensorIndex = 0;
constexpr size_t kScoreTensorIndex = 1;
constexpr size_t kMaxOutputNumTensorIndex = 2;
constexpr size_t kIoUThresholdTensorIndex = 3;
constexpr size_t kScoreThresholdTensorIndex = 4;
constexpr int kBoxPointNum = 4;
}  // namespace

int NonMaxSuppressionCPUKernel::Init() {
  // boxes, scores, max_output_boxes, iou_threshold, score_threshold
  if (in_tensors_.size() < kMinInputsSize || in_tensors_.size() > kMaxInputsSize || out_tensors_.size() != kOutputNum) {
    MS_LOG(ERROR) << "NonMaxSuppression input size should be in [" << kMinInputsSize << ", " << kMaxInputsSize << "]"
                  << ", got " << in_tensors_.size() << ", output size should be" << kOutputNum << ", got "
                  << out_tensors_.size();
    return RET_ERROR;
  }

  param_ = reinterpret_cast<NMSParameter *>(op_parameter_);
  if (param_ == nullptr) {
    MS_LOG(ERROR) << "cast to NMSParameter pointer got nullptr";
    return RET_NULL_PTR;
  }
  if (param_->center_point_box_ != 0 && param_->center_point_box_ != 1) {
    MS_LOG(ERROR) << "NonMaxSuppression center_point_box should be 0 or 1, got " << param_->center_point_box_;
    return RET_ERROR;
  }
  center_point_box_ = param_->center_point_box_;

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int NonMaxSuppressionCPUKernel::GetParams() {
  // optional input order: max_output_per_class, iou_threshold, score_threshold
  max_output_per_class_ = 0;
  if (in_tensors_.size() >= 3) {
    auto max_output_tensor = in_tensors_.at(kMaxOutputNumTensorIndex);
    if (max_output_tensor != nullptr && reinterpret_cast<int32_t *>(max_output_tensor->data_c()) != nullptr) {
      max_output_per_class_ = *(reinterpret_cast<int32_t *>(max_output_tensor->data_c()));
    }
  }
  iou_threshold_ = 0.0f;
  if (in_tensors_.size() >= 4) {
    auto iou_threshold_tensor = in_tensors_.at(kIoUThresholdTensorIndex);
    if (iou_threshold_tensor != nullptr && reinterpret_cast<float *>(iou_threshold_tensor->data_c() != nullptr)) {
      iou_threshold_ = *(reinterpret_cast<float *>(iou_threshold_tensor->data_c()));
    }
  }
  score_threshold_ = 0.0f;
  if (in_tensors_.size() >= 5) {
    auto score_threshold_tensor = in_tensors_.at(kScoreThresholdTensorIndex);
    if (score_threshold_tensor != nullptr && reinterpret_cast<float *>(score_threshold_tensor->data_c()) != nullptr) {
      score_threshold_ = *(reinterpret_cast<float *>(score_threshold_tensor->data_c()));
    }
  }
  return RET_OK;
}

int NonMaxSuppressionCPUKernel::PreProcess() { return GetParams(); }

void ExpandDims(std::vector<int> *shape, size_t size) {
  for (size_t i = 0; i < size; i++) {
    shape->insert(shape->begin(), 1);
  }
}

int NonMaxSuppressionCPUKernel::Run() {
  auto box_tensor = in_tensors_.at(kBoxTensorIndex);
  if (box_tensor == nullptr) {
    return RET_ERROR;
  }
  bool simple_out = false;
  auto box_dims = box_tensor->shape();  // batch, box_num, 4
  constexpr size_t kBoxTensorDims = 3;
  if (box_dims.size() != kBoxTensorDims) {
    ExpandDims(&box_dims, kBoxTensorDims - box_dims.size());
    simple_out = true;
  }
  constexpr size_t kBoxCoordIndex = 2;
  if (box_dims[kBoxCoordIndex] != kBoxPointNum) {
    return RET_ERROR;
  }

  auto score_tensor = in_tensors_.at(kScoreTensorIndex);
  if (score_tensor == nullptr) {
    return RET_ERROR;
  }
  auto score_dims = score_tensor->shape();  // batch, class, box_num
  constexpr size_t kScoreTensorDims = 3;
  if (score_dims.size() != kScoreTensorDims) {
    ExpandDims(&score_dims, kScoreTensorDims - score_dims.size());
  }
  constexpr size_t kBatchIndex = 0;
  if (score_dims.at(kBatchIndex) != box_dims.at(kBatchIndex)) {
    MS_LOG(ERROR) << "Boxes tensor batch num should be equal to scores tensor's batch num.";
    return RET_ERROR;
  }
  constexpr size_t kScoreDimsBoxNumIndex = 2;
  constexpr size_t kBoxDimsBoxNumIndex = 1;
  if (score_dims.at(kScoreDimsBoxNumIndex) != box_dims.at(kBoxDimsBoxNumIndex)) {
    MS_LOG(ERROR) << "Boxes tensor spatial dimension should be equal to scores tensor's spatial dimension.";
    return RET_ERROR;
  }
  const float *scores = reinterpret_cast<const float *>(score_tensor->data_c());  // batch, class, num
  if (scores == nullptr) {
    MS_LOG(ERROR) << "score tensor data nullptr";
    return RET_ERROR;
  }

  int batch_num = score_dims.at(kBatchIndex);
  constexpr size_t kClassIndex = 1;
  int class_num = score_dims.at(kClassIndex);
  int box_num = score_dims.at(kScoreDimsBoxNumIndex);
  float *scores_data = reinterpret_cast<float *>(score_tensor->data_c());
  if (scores_data == nullptr) {
    MS_LOG(ERROR) << "score tensor data nullptr";
    return RET_ERROR;
  }
  float *box_data = reinterpret_cast<float *>(box_tensor->data_c());
  if (box_data == nullptr) {
    MS_LOG(ERROR) << "box tensor data nullptr";
    return RET_ERROR;
  }

  std::vector<NMSBox> selected_box_per_class;
  selected_box_per_class.reserve(std::min(static_cast<int32_t>(box_num), max_output_per_class_));
  std::vector<NMSIndex> selected_index;

  for (auto i = 0; i < batch_num; ++i) {
    int batch_offset = i * class_num * box_num;
    for (auto j = 0; j < class_num; ++j) {
      // per batch per class filter
      float *per_class_scores = scores_data + batch_offset + j * box_num;
      float *box = box_data + i * box_num * kBoxPointNum;
      std::vector<NMSBox> above_score_candidates;
      above_score_candidates.reserve(box_num);
      for (auto k = 0; k < box_num; ++k) {
        if (per_class_scores[k] > score_threshold_) {
          above_score_candidates.emplace_back(per_class_scores[k], k, center_point_box_, box[0], box[1], box[2],
                                              box[3]);
        }
        box += kBoxPointNum;
      }
      std::priority_queue<NMSBox, std::vector<NMSBox>, std::less<NMSBox>> sorted_candidates(
        std::less<NMSBox>(), std::move(above_score_candidates));

      selected_box_per_class.clear();
      while (!sorted_candidates.empty() && static_cast<int32_t>(selected_index.size()) < max_output_per_class_) {
        auto cand = sorted_candidates.top();
        bool selected = true;
        auto IoUSuppressed = [this, &cand](const NMSBox &box) {
          float intersec_x1 = std::max(cand.x1_, box.x1_);
          float intersec_x2 = std::min(cand.x2_, box.x2_);
          float intersec_y1 = std::max(cand.y1_, box.y1_);
          float intersec_y2 = std::min(cand.y2_, box.y2_);
          const float intersec_area =
            std::max(intersec_x2 - intersec_x1, 0.0f) * std::max(intersec_y2 - intersec_y1, 0.0f);
          if (intersec_area <= 0.0f) {
            return false;
          }
          const float intersec_over_union = intersec_area / (cand.area_ + box.area_ - intersec_area);
          return intersec_over_union > this->iou_threshold_;
        };
        if (std::any_of(selected_box_per_class.begin(), selected_box_per_class.end(), IoUSuppressed)) {
          selected = false;
        }
        if (selected) {
          selected_box_per_class.push_back(cand);
          selected_index.emplace_back(
            NMSIndex{static_cast<int32_t>(i), static_cast<int32_t>(j), static_cast<int32_t>(cand.index_)});
        }
        sorted_candidates.pop();
      }
    }
  }
  auto output = out_tensors_.at(0);
  int selected_num = static_cast<int>(selected_index.size());
  if (!simple_out) {
    const int output_last_dim = 3;
    output->set_shape({selected_num, output_last_dim});
    MS_ASSERT(output_last_dim * sizeof(int32_t) == sizeof(NMSIndex));
    auto *out_data = reinterpret_cast<int32_t *>(output->MutableData());
    if (out_data == nullptr) {
      MS_LOG(ERROR) << "out_data is nullptr.";
      return RET_ERROR;
    }
    memcpy(out_data, selected_index.data(), selected_index.size() * sizeof(NMSIndex));
  } else {
    output->set_shape({selected_num});
    std::vector<int> result;
    for (size_t i = 0; i < selected_index.size(); i++) {
      result.push_back(selected_index[i].box_index_);
    }
    auto *out_data = reinterpret_cast<int32_t *>(output->MutableData());
    if (out_data == nullptr) {
      MS_LOG(ERROR) << "out_data is nullptr.";
      return RET_ERROR;
    }
    memcpy(out_data, result.data(), result.size() * sizeof(int));
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NonMaxSuppression, LiteKernelCreator<NonMaxSuppressionCPUKernel>)
}  // namespace mindspore::kernel
