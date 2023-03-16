/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CORE_ABSTRACT_OPS_OP_INFER_H
#define MINDSPORE_CORE_ABSTRACT_OPS_OP_INFER_H

#include <vector>
#include <set>
#include <memory>
#include "ir/primitive.h"
#include "mindspore/core/ops/core_ops.h"
#include "abstract/abstract_value.h"
#include "ir/anf.h"

namespace mindspore {
namespace abstract {
class OpInferBase {
 public:
  OpInferBase() = default;
  virtual ~OpInferBase() = default;

  /// \brief Infer the output shape for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's inputs.
  ///
  /// \return The inferred shape.
  virtual BaseShapePtr InferShape(const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) const = 0;

  /// \brief Infer the output type for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's inputs.
  ///
  /// \return The inferred type.
  virtual TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const = 0;

  /// \brief Infer the output value for target operator. Only override when needed.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's inputs.
  ///
  /// \return Inferred Value based on given inputs.
  virtual ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    return kValueAny;
  }

  /// \brief Get the indices of infer-depend value.
  ///
  /// \return Set with indices of infer-depend value.
  virtual std::set<int64_t> GetValueDependArgIndices() const { return {}; }

  /// \brief Infer the related shape and type for target operator.
  ///
  /// \param[in] engine
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's inputs.
  ///
  /// \return AbstractBasePtr with inferred shape and inferred type.
  virtual AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
    auto type = InferType(primitive, input_args);
    auto shape = InferShape(primitive, input_args);
    return MakeAbstract(shape, type);
  }
};

using OpInferBasePtr = std::shared_ptr<OpInferBase>;
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_OPS_OP_INFER_H
