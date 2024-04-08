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

#ifndef MINDSPORE_PI_JIT_BACKWARD_FUNCTION_H_
#define MINDSPORE_PI_JIT_BACKWARD_FUNCTION_H_

#include <memory>
#include <string>
#include <vector>
#include "ir/anf.h"

namespace mindspore {
namespace pijit {
namespace grad {
/// \brief BackwardFunc is a class, which represent a function to calculate the gradient.
class BackwardFunc {
 public:
  /// \brief The constructor of BackwardFunc.
  ///
  /// \param[in] name The name of this backward function.
  ///
  /// \return The instance of BackwardFunc.
  explicit BackwardFunc(const std::string &name) : name_(name) {}

  /// \brief Destructor.
  virtual ~BackwardFunc() = default;

  /// \brief Get the name of the backward function.
  ///
  /// \return The name of this backward function.
  const std::string &GetName() const { return name_; }

  /// \brief Start calculate the gradient of the backward function.
  ///
  /// \param[in] inputs The arguments of the forward execution.
  /// \param[in] out The output of the forward execution.
  /// \param[in] dout The dout of the output.
  ///
  /// \return The gradients of the inputs of forward execution.
  virtual ValuePtrList Run(const ValuePtrList &inputs, const ValuePtr &out, const ValuePtr &dout) = 0;

  /// \brief Postprocess gradients from func to align with next_edges.
  ///
  /// \param[in] gradient_value Gradients value is gradients result from func which need postprocess.
  ///
  /// \return Real gradients after postprocess, the size is same as next edges size.
  virtual ValuePtrList PostProcess(const ValuePtrList &gradient_value) { return gradient_value; }

  /// \brief Get indexes of inputs required to calculate the gradient.
  ///
  /// \return The indexes of inputs required to calculate the gradient.
  const std::vector<size_t> &GetGradientIndexes() const { return gradient_indexes_; }

  /// \brief Set the indexes of forward's inputs required to calculate the gradient.
  ///
  /// \param[in] indexes The indexes of inputs required to calculate the gradient.
  void SetGradientIndexes(const std::vector<size_t> &indexes) { gradient_indexes_ = indexes; }

  /// \brief Add a index of forward's input required to calculate the gradient.
  ///
  /// \param[in] index The index of forward's input required to calculate the gradient.
  void AddGradientIndex(size_t index) { gradient_indexes_.push_back(index); }

  /// \brief Create the value filled with one, shape like the input.
  ///
  /// \param[in] value The input value.
  ///
  /// \return The value filled with one.
  virtual ValuePtr Ones(const ValuePtr &value) const = 0;

  /// \brief Create the value filled with zero, shape like the input.
  ///
  /// \param[in] value The input value.
  ///
  /// \return The value filled with zero.
  virtual ValuePtr Zeros(const ValuePtr &value) const = 0;

  /// \brief Calculate the sum of inputs.
  ///
  /// \param[in] input The first input value.
  /// \param[in] other The second input value.
  ///
  /// \return The sum of inputs.
  virtual ValuePtr Add(const ValuePtr &input, const ValuePtr &other) const = 0;

 private:
  /// \brief The name of this backward function.
  std::string name_;
  /// \brief The index set of the inputs required to calculate the gradient.
  std::vector<size_t> gradient_indexes_;
};

using BackwardFuncPtr = std::shared_ptr<BackwardFunc>;
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
#endif  // MINDSPORE_PI_JIT_BACKWARD_FUNCTION_H_
