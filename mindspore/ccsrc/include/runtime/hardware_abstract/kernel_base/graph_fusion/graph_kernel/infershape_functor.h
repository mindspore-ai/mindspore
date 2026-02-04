/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_INFERSHAPE_FUNCTOR_H_
#define MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_INFERSHAPE_FUNCTOR_H_

#include <string>
#include <memory>

#include "ir/anf.h"
#include "ir/functor.h"
#include "base/base.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore::opt::dynamic_shape {
/// \brief The class to implement an InferShape function, which is decoupled from the mindspore/core.
class RUNTIME_HARDWARE_EXPORT InferShapeFunctor : public Functor {
 public:
  /// \brief Constructor of InferShapeFunctor.
  explicit InferShapeFunctor(const std::string &name) : Functor(name) {}

  /// \brief Destructor of InferShapeFunctor.
  ~InferShapeFunctor() override = default;
  MS_DECLARE_PARENT(InferShapeFunctor, Functor)

  /// \brief Infer output shape.
  /// \param[in] args AbstractBasePtrList of the inputs.
  /// \return Result BaseShapePtr.
  virtual BaseShapePtr InferShape(const AbstractBasePtrList &args) = 0;

  /// \brief Pack functor name to a Value
  /// \return The name of this infershape functor.
  ValuePtr ToValue() const override { return MakeValue(name_); };

  /// \brief Rename the functor.
  void FromValue(const ValuePtr &value) override { name_ = GetValue<std::string>(value); };
};
using InferShapeFunctorPtr = std::shared_ptr<InferShapeFunctor>;
constexpr auto kAttrInferShapeFunctor = "infer_shape_functor";
}  // namespace mindspore::opt::dynamic_shape
#endif  // MINDSPORE_CCSRC_INCLUDE_RUNTIMR_HARDWARE_ABSTRACT_KERNEL_BASE_GRAPH_FUSION_GRAPH_KERNEL_INFERSHAPE_FUNCTOR_H_
