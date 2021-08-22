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

#ifndef MINDSPORE_CORE_OPS_PROPOSAL_H_
#define MINDSPORE_CORE_OPS_PROPOSAL_H_
#include <vector>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameProposal = "Proposal";
class Proposal : public PrimitiveC {
 public:
  Proposal() : PrimitiveC(kNameProposal) {}
  ~Proposal() = default;
  MS_DECLARE_PARENT(Proposal, PrimitiveC);

  void Init(const float feat_stride, const float base_size, const float min_size, const std::vector<float> &ratio,
            const std::vector<float> &scale, const int64_t pre_nms_topn, const int64_t post_nms_topn,
            const float nms_thresh);
  void set_feat_stride(const float feat_stride);
  void set_base_size(const float base_size);
  void set_min_size(const float min_size);
  void set_ratio(const std::vector<float> &ratio);
  void set_scale(const std::vector<float> &scale);
  void set_pre_nms_topn(const int64_t pre_nms_topn);
  void set_post_nms_topn(const int64_t post_nms_topn);
  void set_nms_thresh(const float nms_thresh);
  float get_feat_stride() const;
  float get_base_size() const;
  float get_min_size() const;
  std::vector<float> get_ratio() const;
  std::vector<float> get_scale() const;
  int64_t get_pre_nms_topn() const;
  int64_t get_post_nms_topn() const;
  float get_nms_thresh() const;
};
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PROPOSAL_H_
