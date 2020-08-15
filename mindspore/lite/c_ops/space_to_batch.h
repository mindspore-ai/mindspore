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
#include "mindspore/lite/c_ops/primitive_c.h"
#ifdef PRIMITIVE_WRITEABLE
#include "schema/inner/model_generated.h"
#else
#include "schema/model_generated.h"
#endif

#ifndef LITE_MINDSPORE_LITE_C_OPS_SPACE_TO_BATCH_H_
#define LITE_MINDSPORE_LITE_C_OPS_SPACE_TO_BATCH_H_

namespace mindspore {
class SpaceToBatch : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  explicit SpaceToBatch(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#else
  explicit SpaceToBatch(schema::Primitive *primitive) : PrimitiveC(primitive) {}
#endif
  int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) override;
  std::vector<int> GetBlockShape() const;
  std::vector<int> GetPaddings() const;
  void SetBlockShape(const std::vector<int> &block_shape);
  void SetPaddings(const std::vector<int> &paddings);

  std::vector<int> BlockSizes() { return block_sizes_; }
  std::vector<int> Paddings() { return block_sizes_; }
  std::vector<int> InShape() { return block_sizes_; }
  std::vector<int> PaddedInShape() { return block_sizes_; }

 private:
  std::vector<int> block_sizes_;
  std::vector<int> paddings_;
  std::vector<int> in_shape_;
  std::vector<int> padded_in_shape_;
};
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_SPACE_TO_BATCH_H_
