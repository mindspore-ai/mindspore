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
#include <vector>
#include <functional>
#include "src/ops/primitive_c.h"
#include "src/tensorlist.h"
#include "ir/dtype/type_id.h"

#ifndef LITE_MINDSPORE_LITE_C_OPS_TENSORLISTSTACK_H_
#define LITE_MINDSPORE_LITE_C_OPS_TENSORLISTSTACK_H_
namespace mindspore {
namespace lite {
class TensorListStack : public PrimitiveC {
 public:
  // tensor:input, element_dtype, num_elements(default=-1:reprent any tensor dim0), element_shape
  TensorListStack() = default;
  ~TensorListStack() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(TensorListStack, PrimitiveC);
  void SetElementDType(int type);
  void SetNumElements(int num_elements);
  explicit TensorListStack(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  int UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs) override;
#else
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  TypeId GetElementDType() const;
  int GetNumElements() const;
  bool IsFullyDefined(const std::vector<int> &shape) const;
  int MergeShape(const std::vector<int> &shape);
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;

 private:
  size_t unKnownRank_ = 255;
  std::vector<int> output_shape_;
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_TENSORLISTSTACK_H_
