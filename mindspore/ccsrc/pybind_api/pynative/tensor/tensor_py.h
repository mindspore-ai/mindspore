/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_
#define MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_

#include <memory>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind_api/pynative/tensor/dlpack_utils.h"

#include "ir/tensor.h"
#include "include/utils/tensor_py.h"
#include "include/utils/visible.h"
#include "include/utils/np_dtypes.h"

namespace py = pybind11;
// brief mindspore namespace.
//
// mindspore namespace is the top level namespace of Mindsporeession project.
// Other namespace should be a sub namespace of mindspore namespace in the ME project.
namespace mindspore {
// brief mindspore::tensor namespace
//
// A sub namespace in ME to support tensor related definition.
namespace tensor {

// Tensor python wrapper and adapter class.
class PYBIND_EXPORT TensorPybind {
 public:
  static bool IsPinned(const TensorPy &tensor);

  static bool IsShared(const TensorPy &tensor);

  static TensorPtr MakePinMemoryTensor(const TensorPy &tensor);

  static py::bytes GetBytes(const Tensor &tensor);

  static TensorPtr ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims,
                                        const TypePtr &type_ptr = nullptr);

  static py::object ToList(const TensorPtr &tensor);

  static py::object Item(const TensorPtr &tensor);

  static py::array SyncAsNumpy(const Tensor &tensor);

  static TensorPtr FromDLPack(const py::object &dlpack_capsule);

  static py::object ToDLPack(const py::object &tensor);
  static py::tuple GetPyTupleShape(const Tensor &tensor);

  static py::tuple GetPyTupleStrides(const Tensor &tensor);

  static py::int_ GetPyItemSize(const Tensor &tensor);

  static py::int_ GetPyNBytes(const Tensor &tensor);

  static void FlushFromCache(const Tensor &tensor);

  static void Offload(const TensorPtr &tensor, bool release);

  static void Load(const Tensor &tensor);

  static bool SharedMemory(const TensorPtr &tensor);

  // move tensor from device to host, or host to device asynchronously
  static TensorPtr MoveTo(const Tensor &self, const std::string &to, bool blocking = True);

  static void SetDeviceAddress(const TensorPtr &tensor, uintptr_t addr, const ShapeVector &shape,
                               const TypePtr type_ptr);

  static uintptr_t DataPtr(const TensorPtr &tensor);

  static std::string GetDevice(const TensorPtr &tensor);

  static void SetUserData(const TensorPtr &tensor, const py::str &key, const py::object &value);

  static py::object GetUserData(const TensorPtr &tensor, const py::str &key);
};

// CSRTensor python wrapper and adapter class.
class CSRTensorPy {
 public:
  static py::tuple GetPyTupleShape(const CSRTensor &csr_tensor);
  static py::object GetIndptr(const CSRTensorPtr &csr_tensor);
  static py::object GetIndices(const CSRTensorPtr &csr_tensor);
  static py::object GetValues(const CSRTensorPtr &csr_tensor);
};

// COOTensor python wrapper and adapter class.
class COOTensorPy {
 public:
  static py::tuple GetPyTupleShape(const COOTensor &coo_tensor);
  static py::object GetIndices(const COOTensorPtr &coo_tensor);
  static py::object GetValues(const COOTensorPtr &coo_tensor);
};

// RowTensor python wrapper and adapter class.
class RowTensorPy {
 public:
  static py::tuple GetPyTupleShape(const RowTensor &row_tensor);
  static py::object GetIndices(const RowTensorPtr &row_tensor);
  static py::object GetValues(const RowTensorPtr &row_tensor);
};

class PYBIND_EXPORT TensorPyImpl {
 public:
  /// \brief Create a C++ Tensor.
  ///
  /// \param[in] input [py::dict] The input form python, as like: {"input_data": input_data, "dtype": dtype,
  /// "init": init, "const_arg": const_arg, "device": device, "symbolic_shape": symbolic_shape}.
  ///
  /// \return A C++ Tensor.
  static TensorPtr InitTensor(const py::dict &input);

  /// \brief Get the initialization form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The initialization.
  static py::object GetInitializerFromPython(const py::dict &input);

  /// \brief Get the constant argument form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The constant argument.
  static bool GetConstArgFromPython(const py::dict &input);

  /// \brief Get the device info form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The device info.
  static std::string GetDeviceFromPython(const py::dict &input);

  /// \brief Get the dynamically optimize shape form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The dynamically optimize shape.
  static py::object GetSymbolicShapeFromPython(const py::dict &input);

  /// \brief Get the type of Tensor form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ////
  /// \return The type of Tensor.
  static const TypePtr GetDtypeFromPython(const py::dict &input);

  /// \brief Get the shape of Tensor form python.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return The shape of Tensor.
  static const ShapeVector GetShapeFromPython(const py::dict &input);

  /// \brief Create a TensorPy.
  ///
  /// \param[in] input [py::dict] The input form python.
  ///
  /// \return A TensorPy.
  static const TensorPyPtr InitTensorPy(const py::dict &input);

  /// \brief Convert python object to Tensor.
  ///
  /// \param[in] bytes_obj [py::bytes] Python object.
  /// \param[in] dims [py::tuple] The dimensions.
  /// \param[in] type_ptr [TypePtr] The data type.
  ///
  /// \return A created TensorPy.
  static TensorPyPtr ConvertBytesToTensor(const py::bytes &bytes_obj, const py::tuple &dims, const TypePtr &type_ptr);

  static TensorPyPtr FromDLPack(const py::object &dlpack_capsule);
  static py::object ToDLPack(const py::object &tensor);
  static py::object Item(const TensorPyPtr &tensorpy);
  static void RemoveTensorBackwardHook(uint64_t handle_id);
  static ShapeVector GetShapeFromTuple(const py::tuple &tuple);

 private:
  static TensorPtr InitTensorByInputDta(const py::dict &input, const TypePtr &dtype);
  static TensorPtr InitTensorByShape(const py::dict &input, const TypePtr &dtype);
};
}  // namespace tensor
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_UTILS_TENSOR_PY_H_
