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

#ifndef MINDSPORE_CORE_OPS_SPACE_TO_BATCH_ND_H_
#define MINDSPORE_CORE_OPS_SPACE_TO_BATCH_ND_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameSpaceToBatchND = "SpaceToBatchND";
class SpaceToBatchND : public PrimitiveC {
 public:
  SpaceToBatchND() : PrimitiveC(kNameSpaceToBatchND) {}
  ~SpaceToBatchND() = default;
  MS_DECLARE_PARENT(SpaceToBatchND, PrimitiveC);
  void Init(std::vector<int64_t> block_shape, const std::vector<std::vector<int64_t>> paddings);
  void set_paddings(const std::vector<std::vector<int64_t>> paddings);
  void set_block_shape(std::vector<int64_t> block_shape);
  std::vector<int64_t> get_block_shape() const;
  std::vector<std::vector<int64_t>> get_paddings() const;
};
AbstractBasePtr SpaceToBatchNDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args);
using PrimSpaceToBatchNDPtr = std::shared_ptr<SpaceToBatchND>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_SPACE_TO_BATCH_ND_H_
