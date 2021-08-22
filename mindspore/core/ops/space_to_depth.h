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

#ifndef MINDSPORE_CORE_OPS_SPACE_TO_DEPTH_H_
#define MINDSPORE_CORE_OPS_SPACE_TO_DEPTH_H_
#include <vector>
#include <memory>

#include "ops/primitive_c.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSpaceToDepth = "SpaceToDepth";
class SpaceToDepth : public PrimitiveC {
 public:
  SpaceToDepth() : PrimitiveC(kNameSpaceToDepth) { InitIOName({"x"}, {"y"}); }
  ~SpaceToDepth() = default;
  MS_DECLARE_PARENT(SpaceToDepth, PrimitiveC);
  void Init(const int64_t block_size, const Format &format = NCHW);
  void set_block_size(const int64_t block_size);
  int64_t get_block_size() const;
  void set_format(const Format &format);
  Format get_format() const;
};
AbstractBasePtr SpaceToDepthInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args);
using PrimSpaceToDepthPtr = std::shared_ptr<SpaceToDepth>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_SpaceToDepth_H_
