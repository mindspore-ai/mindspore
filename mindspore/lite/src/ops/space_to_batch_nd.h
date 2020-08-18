/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
#include <set>
#include <cmath>
#include "ir/dtype/type_id.h"
#include "src/ops/primitive_c.h"
#ifdef PRIMITIVE_WRITEABLE
#include "schema/inner/model_generated.h"
#else
#include "schema/model_generated.h"
#endif

#ifndef LITE_MINDSPORE_LITE_C_OPS_SPACE_TO_BATCH_N_D_H_
#define LITE_MINDSPORE_LITE_C_OPS_SPACE_TO_BATCH_N_D_H_

namespace mindspore {
namespace lite {
class SpaceToBatchND : public PrimitiveC {
 public:
  explicit SpaceToBatchND(OriginPrimitive *primitive) : PrimitiveC(primitive) {}

  std::vector<int> GetBlockShape() const;
  std::vector<int> GetPaddings() const;
  void SetBlockShape(const std::vector<int> &block_shape);
  void SetPaddings(const std::vector<int> &paddings);
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_SPACE_TO_BATCH_N_D_H_
