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

#ifndef LITE_MINDSPORE_LITE_C_OPS_PAD_H_
#define LITE_MINDSPORE_LITE_C_OPS_PAD_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>

#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Pad : public PrimitiveC {
 public:
  Pad() = default;
  ~Pad() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(Pad, PrimitiveC);
  explicit Pad(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetPaddings(const std::vector<int> &paddings);
  void SetPaddingMode(int padding_mode);
  void SetConstantValue(float constant_value);
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  std::vector<int> GetPaddings() const;
  int GetPaddingMode() const;
  float GetConstantValue() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_PAD_H_
