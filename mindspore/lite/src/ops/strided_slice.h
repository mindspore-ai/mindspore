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

#ifndef LITE_MINDSPORE_LITE_C_OPS_STRIDED_SLICE_H_
#define LITE_MINDSPORE_LITE_C_OPS_STRIDED_SLICE_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>

#include "src/ops/primitive_c.h"

namespace mindspore {
namespace lite {
class StridedSlice : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(StridedSlice, PrimitiveC);
  StridedSlice() = default;
  explicit StridedSlice(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
  void SetBeginMask(int begin_mask);
  void SetEndMask(int end_mask);
  void SetEllipsisMask(int ellipsis_mask);
  void SetNewAxisMask(int new_axis_mask);
  void SetShrinkAxisMask(int shrink_axis_mask);
  void SetBegin(const std::vector<int> &begin);
  void SetEnd(const std::vector<int> &end);
  void SetStride(const std::vector<int> &stride);
  void SetIsScale(const std::vector<int> &is_scale);
  int UnPackAttr(const Primitive &prim, const std::vector<AnfNodePtr> &inputs);
#else
  StridedSlice() = default;

  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  int GetBeginMask() const;
  int GetEndMask() const;
  int GetEllipsisMask() const;
  int GetNewAxisMask() const;
  int GetShrinkAxisMask() const;
  std::vector<int> GetBegin() const;
  std::vector<int> GetEnd() const;
  std::vector<int> GetStride() const;
  std::vector<int> GetIsScale() const;

  int NDims() { return this->ndim_; }
  void ApplyNewAxisMask();
  std::vector<int> ApplyShrinkMask(std::vector<int> out_shape);
  void ApplyBeginMask();
  void ApplyEndMask();
  void ApplyEllipsisMask();
  std::vector<int> GetInShape() { return this->in_shape_; }
  std::vector<int> GetBegins() { return this->begins_; }
  std::vector<int> GetEnds() { return this->ends_; }
  std::vector<int> GetStrides() { return this->strides_; }

 protected:
  int ndim_ = 0;
  std::vector<int> in_shape_;
  std::vector<int> begins_;
  std::vector<int> ends_;
  std::vector<int> strides_;
  std::vector<bool> begins_mask_;
  std::vector<bool> ends_mask_;
  std::vector<bool> ellipsis_mask_;
  std::vector<bool> new_axis_mask_;
  std::vector<bool> shrink_axis_mask_;
  void TransIndexToPositive();
};
}  // namespace lite
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_STRIDED_SLICE_H_
