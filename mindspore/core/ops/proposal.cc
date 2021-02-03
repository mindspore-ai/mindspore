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

#include <vector>

#include "ops/proposal.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
void Proposal::set_feat_stride(const float feat_stride) { this->AddAttr(kFeatStride, MakeValue(feat_stride)); }

float Proposal::get_feat_stride() const {
  auto value_ptr = GetAttr(kFeatStride);
  return GetValue<float>(value_ptr);
}

void Proposal::set_base_size(const float base_size) { this->AddAttr(kBaseSize, MakeValue(base_size)); }

float Proposal::get_base_size() const {
  auto value_ptr = GetAttr(kBaseSize);
  return GetValue<float>(value_ptr);
}

void Proposal::set_min_size(const float min_size) { this->AddAttr(kMinSize, MakeValue(min_size)); }

float Proposal::get_min_size() const {
  auto value_ptr = GetAttr(kMinSize);
  return GetValue<float>(value_ptr);
}

void Proposal::set_ratio(const std::vector<float> &ratio) { this->AddAttr(kRatio, MakeValue(ratio)); }

std::vector<float> Proposal::get_ratio() const {
  auto value_ptr = GetAttr(kRatio);
  return GetValue<std::vector<float>>(value_ptr);
}

void Proposal::set_scale(const std::vector<float> &scale) { this->AddAttr(kScale, MakeValue(scale)); }

std::vector<float> Proposal::get_scale() const {
  auto value_ptr = GetAttr(kScale);
  return GetValue<std::vector<float>>(value_ptr);
}

void Proposal::set_pre_nms_topn(const int64_t pre_nms_topn) { this->AddAttr(kPreNmsTopn, MakeValue(pre_nms_topn)); }

int64_t Proposal::get_pre_nms_topn() const {
  auto value_ptr = GetAttr(kPreNmsTopn);
  return GetValue<int64_t>(value_ptr);
}

void Proposal::set_post_nms_topn(const int64_t post_nms_topn) { this->AddAttr(kPostNmsTopn, MakeValue(post_nms_topn)); }

int64_t Proposal::get_post_nms_topn() const {
  auto value_ptr = GetAttr(kPostNmsTopn);
  return GetValue<int64_t>(value_ptr);
}

void Proposal::set_nms_thresh(const float nms_thresh) { this->AddAttr(kNmsThresh, MakeValue(nms_thresh)); }

float Proposal::get_nms_thresh() const {
  auto value_ptr = GetAttr(kNmsThresh);
  return GetValue<float>(value_ptr);
}

void Proposal::Init(const float feat_stride, const float base_size, const float min_size,
                    const std::vector<float> &ratio, const std::vector<float> &scale, const int64_t pre_nms_topn,
                    const int64_t post_nms_topn, const float nms_thresh) {
  this->set_feat_stride(feat_stride);
  this->set_base_size(base_size);
  this->set_min_size(min_size);
  this->set_ratio(ratio);
  this->set_scale(scale);
  this->set_pre_nms_topn(pre_nms_topn);
  this->set_post_nms_topn(post_nms_topn);
  this->set_nms_thresh(nms_thresh);
}
REGISTER_PRIMITIVE_C(kNameProposal, Proposal);
}  // namespace ops
}  // namespace mindspore
