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

#ifndef MINDSPORE_CORE_OPS_PRIOR_BOX_H_
#define MINDSPORE_CORE_OPS_PRIOR_BOX_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNamePriorBox = "PriorBox";
class PriorBox : public PrimitiveC {
 public:
  PriorBox() : PrimitiveC(kNamePriorBox) {}
  ~PriorBox() = default;
  MS_DECLARE_PARENT(PriorBox, PrimitiveC);

  void Init(const std::vector<int64_t> &min_sizes, const std::vector<int64_t> &max_sizes,
            const std::vector<float> &aspect_ratios, const std::vector<float> &variances, const int64_t image_size_w,
            const int64_t image_size_h, const float step_w, const float step_h, const bool clip, const bool flip,
            const float offset);
  void set_min_sizes(const std::vector<int64_t> &min_sizes);
  void set_max_sizes(const std::vector<int64_t> &max_sizes);
  void set_aspect_ratios(const std::vector<float> &aspect_ratios);
  void set_variances(const std::vector<float> &variances);
  void set_image_size_w(const int64_t image_size_w);
  void set_image_size_h(const int64_t image_size_h);
  void set_step_w(const float step_w);
  void set_step_h(const float step_h);
  void set_clip(const bool clip);
  void set_flip(const bool flip);
  void set_offset(const float offset);
  std::vector<int64_t> get_min_sizes() const;
  std::vector<int64_t> get_max_sizes() const;
  std::vector<float> get_aspect_ratios() const;
  std::vector<float> get_variances() const;
  int64_t get_image_size_w() const;
  int64_t get_image_size_h() const;
  float get_step_w() const;
  float get_step_h() const;
  bool get_flip() const;
  bool get_clip() const;
  float get_offset() const;
};

AbstractBasePtr PriorBoxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const std::vector<AbstractBasePtr> &input_args);
using PrimPriorBoxPtr = std::shared_ptr<PriorBox>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_PRIOR_BOX_H_
