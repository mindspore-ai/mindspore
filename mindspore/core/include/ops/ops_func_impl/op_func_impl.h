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
#include "abstract/abstract_value.h"
#include "ops/infer_info/infer_info.h"

namespace mindspore {
// The operator input shape and value check status.
constexpr int32_t OP_CHECK_SUCCESS = 0;
constexpr int32_t OP_CHECK_RETRY = -1;

namespace ops {
using abstract::AbstractBasePtr;
using TypeIdList = std::vector<TypeId>;

/// \brief This class is a collection of functions related to operator, such as InferShape, InferType, Check, etc.
class MS_CORE_API OpFuncImpl {
 public:
  OpFuncImpl() = default;
  virtual ~OpFuncImpl() = default;

  virtual BaseShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  /// \brief Infer the output shape for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_values Operator's input value pointer list.
  ///
  /// \return The inferred output shape array.
  virtual ShapeArray InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  /// \brief Infer the output shape for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_infos Operator's input InferInfo list.
  ///
  /// \return The inferred output shape array.
  virtual ShapeArray InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  /// \brief Infer the output type for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's input arguments pointer list.
  ///
  /// \return The inferred object type, such as TensorType, Tuple, List.
  virtual TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  /// \brief Infer the output type for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_values Operator's input value pointer list.
  ///
  /// \return The inferred object type list, such as TensorType, Tuple, List.
  virtual TypePtrList InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  /// \brief Infer the output shape for target operator.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_infos Operator's input InferInfo list.
  ///
  /// \return The inferred output type id.
  virtual TypeIdList InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
    MS_LOG(EXCEPTION) << "Not implement exception";
  }

  /// \brief The operator input shape and value check, the function only carries the check of the
  /// value of InferShape unrelated parameters.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's input arguments pointer list.
  ///
  /// \return OP_CHECK_SUCCESS if success, else OP_CHECK_RETRY.
  virtual int32_t CheckValidation(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    return OP_CHECK_SUCCESS;
  }

  /// \brief The operator input shape and value check, the function only carries the check of the
  /// value of InferShape unrelated parameters.
  ///
  /// \param[in] primitive Operator's primitive.
  /// \param[in] input_args Operator's input arguments pointer list.
  ///
  /// \return OP_CHECK_SUCCESS if success, else OP_CHECK_RETRY.
  virtual int32_t CheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
    return OP_CHECK_SUCCESS;
  }

  /// \brief Get the indices of infer-depend value.
  ///
  /// \return Set with indices of infer-depend value.
  virtual std::set<int64_t> GetValueDependArgIndices() const { return {}; }

  /// \brief If the general infer is implemented
  ///
  /// \return Boolean value indicating if the general infer is implemented
  virtual bool GeneralInferRegistered() const { return false; }
};

using OpFuncImplPtr = std::shared_ptr<OpFuncImpl>;
using OpFuncImplRawPtr = OpFuncImpl *;
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_OPS_FUNC_IMPL_OP_FUNC_IMPL_H
