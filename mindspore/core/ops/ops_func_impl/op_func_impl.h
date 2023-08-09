/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_OP_FUNC_IMPL_H
#define MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_OP_FUNC_IMPL_H

#include <vector>
#include <set>
#include <memory>
#include "ir/primitive.h"
#include "base/op_arg_base.h"

namespace mindspore {
// The operator input shape and value check status.
constexpr int32_t OP_CHECK_SUCCESS = 0;
constexpr int32_t OP_CHECK_RETRY = -1;

namespace ops {
/// \brief This class is a collection of functions related to operator, such as InferShape, InferType, Check, etc.
class OpFuncImpl {
 public:
  OpFuncImpl() = default;
  virtual ~OpFuncImpl() = default;

  /// \brief Infer the output shape for target operator.
  /// The Infer function of the OpArgBase input type is a higher level abstraction of AbstractBase and KernelTensor with
  /// better performance.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's input arguments pointer list.
  ///
  /// \return The inferred output shape.
  /// For Tensor type output, return its shape. For example, the shape of output Tensor is (8, 16), return
  /// std::vector<ShapeVector>{{8, 16}}.
  ///
  /// For Scalar type output, return an std::vector<ShapeVector> containing an empty
  /// ShapeVector, i.e. std::vector<ShapeVector>{{}}.
  ///
  /// For Tuple/List (all elements must be Tensor and Scalar) type output, return output shape
  /// consists of the shape of all elements in Tuple/List. For example, if return a Tuple of the structure ((8,16),
  /// (8,16)) contains two Tensors of shape (8, 16), then return std::vector<ShapeVector>{{8, 16}, {8, 16}}. If return a
  /// Tuple type with a structure such as ((), ()) that contains two Scalar, then return std::vector<ShapeVector>{{},
  /// {}}.
  virtual std::vector<ShapeVector> InferShape(const Primitive *primitive,
                                              const std::vector<OpArgBase *> &input_args) const = 0;

  /// \brief Infer the output type for target operator.
  /// The Infer function of the OpArgBase input type is a higher level abstraction of AbstractBase and KernelTensor
  /// with better performance.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's input arguments pointer list.
  ///
  /// \return The inferred object type, such as TensorType, Tuple, List.
  virtual TypePtr InferType(const Primitive *primitive, const std::vector<OpArgBase *> &input_args) const = 0;

  /// \brief The operator input shape and value check, the function only carries the check of the
  /// value of InferShape unrelated parameters.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's input arguments pointer list.
  ///
  /// \return OP_CHECK_SUCCESS if success, else OP_CHECK_RETRY.
  virtual int32_t CheckValidation(const Primitive *primitive, const std::vector<OpArgBase *> &input_args) const {
    return OP_CHECK_SUCCESS;
  }

  /// \brief Get the indices of infer-depend value.
  ///
  /// \return Set with indices of infer-depend value.
  virtual std::set<int64_t> GetValueDependArgIndices() const { return {}; }
};

using OpFuncImplPtr = std::shared_ptr<OpFuncImpl>;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_OP_FUNC_IMPL_H
