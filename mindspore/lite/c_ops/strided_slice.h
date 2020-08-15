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

#ifndef LITE_MINDSPORE_LITE_C_OPS_STRIDED_SLICE_H_
#define LITE_MINDSPORE_LITE_C_OPS_STRIDED_SLICE_H_

namespace mindspore {
class StridedSlice : public PrimitiveC {
 public:
#ifdef PRIMITIVE_WRITEABLE
  explicit StridedSlice(schema::PrimitiveT *primitive) : PrimitiveC(primitive) {}
#else
  explicit StridedSlice(schema::Primitive *primitive) : PrimitiveC(primitive) {}
#endif
  int InferShape(std::vector<lite::tensor::Tensor *> inputs_, std::vector<lite::tensor::Tensor *> outputs_) override;
  int GetBeginMask() const;
  int GetEndMask() const;
  int GetEllipsisMask() const;
  int GetNewAxisMask() const;
  int GetShrinkAxisMask() const;
  std::vector<int> GetBegin() const;
  std::vector<int> GetEnd() const;
  std::vector<int> GetStride() const;
  std::vector<int> GetIsScale() const;
  void SetBeginMask(int begin_mask);
  void SetEndMask(int end_mask);
  void SetEllipsisMask(int ellipsis_mask);
  void SetNewAxisMask(int new_axis_mask);
  void SetShrinkAxisMask(int shrink_axis_mask);
  void SetBegin(const std::vector<int> &begin);
  void SetEnd(const std::vector<int> &end);
  void SetStride(const std::vector<int> &stride);
  void SetIsScale(const std::vector<int> &is_scale);

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
  int ndim_;
  std::vector<int> in_shape_;
  std::vector<int> begins_;
  std::vector<int> ends_;
  std::vector<int> strides_;
  std::vector<bool> begins_mask_;
  std::vector<bool> ends_mask_;
  std::vector<bool> ellipsis_mask_;
  std::vector<bool> new_axis_mask_;
  std::vector<bool> shrink_axis_mask_;
};
}  // namespace mindspore

#endif  // LITE_MINDSPORE_LITE_C_OPS_STRIDED_SLICE_H_
