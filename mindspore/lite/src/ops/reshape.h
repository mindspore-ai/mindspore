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

#ifndef MINDSPORE_LITE_SRC_OPS_RESHAPE_H_
#define MINDSPORE_LITE_SRC_OPS_RESHAPE_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>

#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Reshape : public PrimitiveC {
 public:
  Reshape() = default;
  ~Reshape() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(Reshape, PrimitiveC);
  explicit Reshape(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  int UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) override;
  void SetFormat(int format);
  void SetShape(const std::vector<int64_t> &shape);
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  int GetFormat() const;
  std::vector<int64_t> GetShape() const;

 private:
  int CalNewShape(const lite::Tensor *in_tensor, std::vector<int> *out_shape) const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_OPS_RESHAPE_H_
