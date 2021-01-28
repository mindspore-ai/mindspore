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
#ifndef MINDSPORE_CORE_OPS_CLIP_H_
#define MINDSPORE_CORE_OPS_CLIP_H_
#include <memory>

#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameClip = "Clip";
class Clip : public PrimitiveC {
 public:
  Clip() : PrimitiveC(kNameClip) {}
  ~Clip() = default;
  MS_DECLARE_PARENT(Clip, PrimitiveC);
  void Init(const float max, const float min);
  void set_max(const float max);
  void set_min(const float min);
  float get_max() const;
  float get_min() const;
};

using PrimClipPtr = std::shared_ptr<Clip>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_CLIP_H_
