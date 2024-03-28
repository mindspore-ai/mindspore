/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_PI_JIT_NATIVE_BACKWARD_FUNCTION_H_
#define MINDSPORE_PI_JIT_NATIVE_BACKWARD_FUNCTION_H_

#include <memory>
#include <string>
#include "pipeline/jit/pi/auto_grad/backward_function.h"
#include "pipeline/pynative/grad/function/func_builder.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace pijit {
namespace grad {
using FuncBuilderPtr = pynative::autograd::FuncBuilderPtr;
using BpropHandlePtr = const expander::bprop::BpropHandle *;

class NativeBackwardFunc;
using NativeBackwardFuncPtr = std::shared_ptr<NativeBackwardFunc>;

/// \brief NativeBackwardFunc is a class, which represent a function to calculate the gradient.
class NativeBackwardFunc : public BackwardFunc {
 public:
  /// \brief The constructor of NativeBackwardFunc.
  ///
  /// \param[in] name The name of this backward function.
  ///
  /// \return The instance of NativeBackwardFunc.
  explicit NativeBackwardFunc(const PrimitivePtr &prim, const FuncBuilderPtr &ir_builder, const BpropHandlePtr handle)
      : BackwardFunc(prim->name()), prim_(prim), ir_builder_(ir_builder), handle_(handle) {}

  /// \brief Destructor.
  virtual ~NativeBackwardFunc() = default;

  /// \brief Create a instance of native backward function.
  ///
  /// \param[in] prim The primitive of the forward execution.
  ///
  /// \return The instance of native backward function.
  static NativeBackwardFuncPtr GetInstance(const PrimitivePtr &prim);

  /// \brief Start calculate the gradient of the backward function.
  ///
  /// \param[in] inputs The arguments of the forward execution.
  /// \param[in] out The output of the forward execution.
  /// \param[in] dout The dout of the output.
  ///
  /// \return The gradients of the inputs of forward execution.
  ValuePtrList Run(const ValuePtrList &inputs, const ValuePtr &out, const ValuePtr &dout) override;

  /// \brief Postprocess gradients from func to align with next_edges.
  ///
  /// \param[in] gradient_value Gradients value is gradients result from func which need postprocess.
  ///
  /// \return Real gradients after postprocess, the size is same as next edges size.
  ValuePtrList PostProcess(const ValuePtrList &gradient_value) override;

  /// \brief Get the primitive of the forward.
  ///
  /// \return The primitive of the forward.
  const PrimitivePtr &GetPrim() const { return prim_; }

  /// \brief Create the value filled with one, shape like the input.
  ///
  /// \param[in] value The input value.
  ///
  /// \return The value filled with one.
  ValuePtr Ones(const ValuePtr &value) const override { return ir_builder_->Ones(value); }

  /// \brief Create the value filled with zero, shape like the input.
  ///
  /// \param[in] value The input value.
  ///
  /// \return The value filled with zero.
  ValuePtr Zeros(const ValuePtr &value) const override { return ir_builder_->Zeros(value); }

  /// \brief Calculate the sum of inputs.
  ///
  /// \param[in] input The first input value.
  /// \param[in] other The second input value.
  ///
  /// \return The sum of inputs.
  ValuePtr Add(const ValuePtr &input, const ValuePtr &other) const override { return ir_builder_->Add(input, other); }

  /// \brief Convert the inputs, output and dout of forward execution into the inputs of function builder.
  ///
  /// \param[in] inputs The arguments of the forward execution.
  /// \param[in] out The output of the forward execution.
  /// \param[in] dout The dout of the output.
  ///
  /// \return The inputs of the function builder.
  expander::NodePtrList PreProcess(const ValuePtrList &inputs, const ValuePtr &out, const ValuePtr &dout) const;

 private:
  /// \brief The primitive of forward execution.
  PrimitivePtr prim_;
  /// \brief The function builder of this backward function.
  FuncBuilderPtr ir_builder_;
  /// \brief The bprop handle of the primitive.
  const BpropHandlePtr handle_;
};
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_NATIVE_BACKWARD_FUNCTION_H_
