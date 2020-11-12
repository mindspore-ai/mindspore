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

#ifndef MINDSPORE_LITE_SRC_OPS_SLICE_H_
#define MINDSPORE_LITE_SRC_OPS_SLICE_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>

#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class Slice : public PrimitiveC {
 public:
  Slice() = default;
  ~Slice() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(Slice, PrimitiveC);
  explicit Slice(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetFormat(int format);
  void SetBegin(const std::vector<int> &begin);
  void SetSize(const std::vector<int> &size);
  int UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) override;
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  int GetFormat() const;
  std::vector<int> GetBegin() const;
  std::vector<int> GetSize() const;
  std::vector<int> GetAxes() const;
  // due to difference between tflite and onnx, when inferring shape, construct new parameters of begin and size.
  // when running graph, we need to obtain new begins and sizes using the two function as below.
  std::vector<int> GetPostProcessBegin() const;
  std::vector<int> GetPostProcessSize() const;

 protected:
  std::vector<int> begin = {0};
  std::vector<int> size = {-1};
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_OPS_SLICE_H_
