/**
 * Copyright 2019-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_LAYOUT_TRANSFER_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_LAYOUT_TRANSFER_H_

#include <string>
#include <map>
#include "frontend/parallel/status.h"
#include "frontend/parallel/tensor_layout/tensor_layout.h"

namespace mindspore {
namespace parallel {
using ReplacementMemo = std::map<size_t, int64_t>;

class LayoutTransfer {
 public:
  LayoutTransfer() = default;
  virtual ~LayoutTransfer() = 0;
  std::string ToString() const;
  Status Init(const TensorLayout &from_in, const TensorLayout &to_in, bool keep_state = false);
  TensorLayout from_in() const { return from_in_; }
  TensorLayout to_in() const { return to_in_; }
  bool IsDynamicShape() const;
  bool IsAssembledStaticShape() const;
  Status RollbackToDynamicShape();
  ReplacementMemo FromLayoutDimsReplacementMemo() const;
  ReplacementMemo ToLayoutDimsReplacementMemo() const;

 protected:
  bool IsSameTensorShape() const { return from_in_.IsSameTensorShape(to_in_); }
  bool IsSameDeviceArrangement() const { return from_in_.IsSameDeviceArrangement(to_in_); }

  TensorLayout from_in_;
  TensorLayout origin_from_in_;
  TensorLayout to_in_;
  TensorLayout origin_to_in_;

 private:
  virtual Status CheckValidTransfer() = 0;
  Status CalculateFromTensorShape(Shape *from_shape, const Array &from_factors, const Shape &to_shape,
                                  const Array &to_factors);
  Status CalculateToTensorShape(const Shape &from_shape, const Shape &origin_to_shape, const Array &to_in_factors,
                                Shape *to_shape);
  Status CalculateToTensorShapeUsingEnumeration(const Shape &from_tsr_shape, Shape *to_tsr_shape, const Array &factors);
  Status AssembleStaticTensorShape(const TensorLayout &from_in, const TensorLayout &to_in,
                                   TensorLayout *new_from_layout, TensorLayout *new_to_layout);
  bool is_dynamic_shape_ = false;
  bool assembled_static_shape_ = false;
  ReplacementMemo from_dims_replace_memo_;
  ReplacementMemo to_dims_replace_memo_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_TENSOR_LAYOUT_LAYOUT_TRANSFER_H_
