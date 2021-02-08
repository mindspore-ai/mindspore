/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_PY_FUNC_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_PY_FUNC_OP_H_

#include <memory>
#include <vector>
#include <utility>
#include <string>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
class PyFuncOp : public TensorOp {
 public:
  explicit PyFuncOp(py::function func) : py_func_ptr_(std::move(func)) { output_type_ = DataType::DE_UNKNOWN; }
  explicit PyFuncOp(py::function func, DataType::Type output_type)
      : py_func_ptr_(std::move(func)), output_type_(output_type) {}

  ~PyFuncOp() override = default;

  uint32_t NumInput() override { return 0; }
  uint32_t NumOutput() override { return 0; }

  // Compute function for n-n mapping.
  Status Compute(const TensorRow &input, TensorRow *output) override;

  /// \brief Function to convert a primitive type py::object to a TensorRow
  /// \notes Changes the py::object to a tensor with corresponding C++ DataType based on output_type_ and adds it to a
  ///    TensorRow. This function is used inside Compute.
  /// \param[in] ret_py_obj The python object we want to cast
  /// \param[output] The TensorRow output
  /// \return Status
  Status CastOutput(const py::object &ret_py_obj, TensorRow *output);
  std::string Name() const override { return kPyFuncOp; }
  Status to_json(nlohmann::json *out_json) override;

  /// \brief Check whether this pyfunc op is deterministic
  /// \return True if this pyfunc op is random
  bool IsRandom();

 private:
  py::function py_func_ptr_;
  DataType::Type output_type_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_KERNELS_PY_FUNC_OP_H_
