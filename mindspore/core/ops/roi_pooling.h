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

#ifndef MINDSPORE_CORE_OPS_ROI_POOLING_H_
#define MINDSPORE_CORE_OPS_ROI_POOLING_H_
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameROIPooling = "ROIPooling";
class ROIPooling : public PrimitiveC {
 public:
  ROIPooling() : PrimitiveC(kNameROIPooling) {}
  ~ROIPooling() = default;
  MS_DECLARE_PARENT(ROIPooling, PrimitiveC);
  void Init(const int64_t pooled_h, const int64_t pooled_w, const float scale);
  void set_pooled_h(const int64_t pooled_h);
  void set_pooled_w(const int64_t pooled_w);
  void set_scale(const float scale);
  int64_t get_pooled_h() const;
  int64_t get_pooled_w() const;
  float get_scale() const;
};
AbstractBasePtr ROIPoolingInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args);
using PrimROIPoolingPtr = std::shared_ptr<ROIPooling>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_ROI_POOLING_H_
