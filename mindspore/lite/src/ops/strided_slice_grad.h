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

#ifndef MINDSPORE_LITE_SRC_OPS_STRIDED_SLICE_GRAD_H_
#define MINDSPORE_LITE_SRC_OPS_STRIDED_SLICE_GRAD_H_

#include <vector>
#include <set>
#include <cmath>
#include <memory>

#include "src/ops/strided_slice.h"

namespace mindspore {
namespace lite {
class StridedSliceGrad : public StridedSlice {
 public:
  StridedSliceGrad() = default;
  ~StridedSliceGrad() = default;
#ifdef PRIMITIVE_WRITEABLE
  MS_DECLARE_PARENT(StridedSliceGrad, StridedSlice);
  explicit StridedSliceGrad(schema::PrimitiveT *primitive) : StridedSlice(primitive) {}
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
  int UnPackToFlatBuilder(const schema::Primitive *primitive, flatbuffers::FlatBufferBuilder *fbb) override;
#endif
  int InferShape(std::vector<lite::Tensor *> inputs_, std::vector<lite::Tensor *> outputs_) override;
  // bool CheckInputs(std::vector<lite::Tensor *> inputs_);
  int GetBeginMask() const;
  int GetEndMask() const;
  int GetEllipsisMask() const;
  int GetNewAxisMask() const;
  int GetShrinkAxisMask() const;
  std::vector<int> GetBegin() const;
  std::vector<int> GetEnd() const;
  std::vector<int> GetStride() const;
  std::vector<int> GetIsScale() const;
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_OPS_STRIDED_SLICE_GRAD_H_
