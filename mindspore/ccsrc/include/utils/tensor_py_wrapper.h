/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_WRAPPER_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_WRAPPER_H_

#include <memory>
#include <vector>
#include <functional>
#include <string>

#include "pybind11/pybind11.h"

#include "ir/tensor_py_wrapperbase.h"
#include "include/utils/visible.h"

namespace py = pybind11;
namespace mindspore {
namespace tensor {
// TensorPyWrapperBase: An entity class
class COMMON_EXPORT TensorPyWrapper : public TensorPyWrapperBase {
 public:
  /// \brief Create tensorpy from another tensorpy, data is shared.
  /// \param[in] input [TensorPy] The input tensorpy.
  explicit TensorPyWrapper(const py::object &input);
  bool operator==(const Value &other) const override {
    if (other.isa<TensorPyWrapper>()) {
      auto &other_ = static_cast<const TensorPyWrapper &>(other);
      return *this == other_;
    }
    return false;
  }
  MS_DECLARE_PARENT(TensorPyWrapper, TensorPyWrapperBase);
  ~TensorPyWrapper() = default;
  /// \brief Create Abstract for Tensor.
  /// \return Abstract of Tensor.
  abstract::AbstractBasePtr ToAbstract() override;

  const py::object &GetTensorWrapper() const { return tensor_py; }

  void SetTensorPyObj(const py::object &base_tensor) { tensor_py = base_tensor; }

 private:
  py::object tensor_py;
};
using TensorPyWrapperPtr = std::shared_ptr<TensorPyWrapper>;
COMMON_EXPORT TensorPyWrapperPtr ConvertToTensorPyWrapper(const py::handle &obj);

/// \brief Make default_parameter of Parameter to TensorPy, and return to Pybind.
/// \param[in] value [ValuePtr] The given input parameter.
/// \return A TensorPy.
COMMON_EXPORT py::object GetTensorPyFromValue(const ValuePtr &value);

COMMON_EXPORT py::object GetTensorFromTensorPyWrapper(const TensorPyWrapperPtr &self);
/// \brief Make default_parameter of Parameter to MetaTensor.
/// \param[in] value [ValuePtr] The given input parameter.
/// \return A MetaTensor.
COMMON_EXPORT const MetaTensorPtr GetMetaTensorFromValue(const ValuePtr &value);
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_WRAPPER_H_
