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

#ifndef MINDSPORE_CORE_OPS_BATCH_TO_SPACE_ND_H_
#define MINDSPORE_CORE_OPS_BATCH_TO_SPACE_ND_H_

#include <map>
#include <vector>
#include <string>
#include <memory>
#include "ops/primitive_c.h"
#include "abstract/abstract_value.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBatchToSpaceND = "BatchToSpaceND";
class BatchToSpaceND : public PrimitiveC {
 public:
  BatchToSpaceND() : PrimitiveC(kNameBatchToSpaceND) {}
  ~BatchToSpaceND() = default;
  MS_DECLARE_PARENT(BatchToSpaceND, PrimitiveC);
  void Init(std::vector<int64_t> block_shape, std::vector<std::vector<int64_t>> crops);
  void set_crops(std::vector<std::vector<int64_t>> crops);
  void set_block_shape(std::vector<int64_t> block_shape);
  std::vector<int64_t> get_block_shape() const;
  std::vector<std::vector<int64_t>> get_crops() const;
};
AbstractBasePtr BatchToSpaceNDInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args);
using PrimBatchToSpaceNDPtr = std::shared_ptr<BatchToSpaceND>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_BATCH_TO_SPACE_ND_H_
