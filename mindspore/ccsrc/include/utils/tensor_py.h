/**
 * Copyright 2024-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_H_

#include <memory>
#include <utility>
#include <vector>
#include <functional>
#include <string>
#include <tuple>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "ir/tensor.h"
#include "include/utils/visible.h"
#include "include/utils/stub_tensor.h"
#include "include/utils/np_dtypes.h"

namespace py = pybind11;

namespace mindspore {
namespace tensor {

struct TensorPyUserData {
  py::object obj;
  ~TensorPyUserData() {
    // cppcheck-suppress unreadVariable
    py::gil_scoped_acquire acquire_gil;
    obj = py::object();
  }
};

// TensorPyBase: An entity class
class COMMON_EXPORT TensorPy {
 public:
  TensorPy() = default;
  /// \brief Create tensorpy from another tensorpy, data is shared.
  /// \param[in] input [TensorPy] The input tensorpy.
  explicit TensorPy(const TensorPy &input);

  /// \brief Create tensorpy with base tensor.
  /// \param[in] input [TensorPtr] The input base tensor.
  explicit TensorPy(const TensorPtr &input);

  /// \brief Create TensorPy with StubNode.
  /// \param[in] input [StubNodePtr] stub_node.
  explicit TensorPy(const stub::StubNodePtr &stub_node);

  /// \brief Create 0 dimension tensorpy from an int64_t scalar.
  /// \param[in] input [int64_t] The data for tensorpy.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(int64_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensorpy from an int32_t scalar.
  /// \param[in] input [int32_t] The data for tensorpy.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(int32_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensorpy from an int16_t scalar.
  /// \param[in] input [int16_t] The data for tensorpy.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(int16_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 0 dimension tensorpy from an int8_t scalar.
  /// \param[in] input [int8_t] The data for tensorpy.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(int8_t input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensorpy from an int vector.
  /// \param[in] input [std::vector<int64_t>] The data for tensorpy.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(const std::vector<int64_t> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensorpy from an int vector.
  /// \param[in] input [std::vector<int32_t>] The data for tensorpy.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(const std::vector<int32_t> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensorpy from a float vector.
  /// \param[in] input [std::vector<double>] The data for tensor.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(const std::vector<double> &input, const TypePtr &data_type = nullptr);

  /// \brief Create 1 dimension tensorpy from a float vector.
  /// \param[in] input [std::vector<float>] The data for tensor.
  /// \param[in] data_type [TypePtr] Data type.
  explicit TensorPy(const std::vector<float> &input, const TypePtr &data_type = nullptr);

  /// \brief Create a lazy allocated tensorpy.
  /// \param[in] data_type [TypeId] Data type of the tensorpy.
  /// \param[in] shape [ShapeVector] The shape represented by ShapeVector of the tensorpy.
  TensorPy(TypeId data_type, const ShapeVector &shape);

  /// Destructor of TensorPy.
  ~TensorPy() = default;

  /// \brief Indicates whether the tensor is initialized.
  /// \return True or False, initialize or not.
  bool IsInitFinished();

  /// \brief Set the tensor initialization state.
  /// \param[in] flag [bool] The tensor initialization state.
  void SetInitFinished(bool flag);

  /// \brief Whether the tensor is a constant when it is used for the argument of a network.
  /// \return True or False, the tensor is constant or not.
  bool IsConstArg();

  /// \brief Set the tensor to constant.
  /// \param[in] flag [bool] The tensor constant flag.
  void SetConstArg(bool flag);

  /// \brief Used to mark whether the tensor is virtual.
  /// \return True or False, the tensor is virtual or not.
  bool IsVirtual();

  /// \brief Set the tensor to virtual.
  /// \param[in] flag [bool] The tensor is virtual or not.
  void SetVirtualFlag(bool flag);

  /// \brief Used for delayed initialization in parallel mode, when using init, `dtype` and `shape` must be set.
  /// \return The information of init data.
  const py::object GetInitializer() const;

  /// \brief Set the tensor to delay initialization.
  /// \param[in] init [py::object] The information of init data.
  void SetInitializer(const py::object &init);

  /// \brief Get the device type.
  /// \return The device name and type.
  const std::string GetDevice() const;

  /// \brief Set the tensor device information.
  /// \param[in] dev [std::string] The device information.
  void SetDevice(const std::string &dev);

  /// \brief Get the C++ Tensor.
  /// \return The created C++ Tensor.
  TensorPtr GetTensor() const;

  /// \brief Set C++ Tensor.
  ///
  /// \param[in] tensor [TensorPtr] The C++ Tensor.
  void SetTensor(const TensorPtr &tensor);

  /// \brief Get parent Tensor.
  /// \return Parent Tensor.
  const py::object GetParentTensor();

  /// \brief Set the tensor to parent Tensor.
  /// \param[in] parent [py::object] Parent Tensor.
  void SetParentTensor(const py::object &parent);

  /// \brief Get an index value of another Tensor.
  /// \return An index value of another Tensor.
  const py::object GetIndexOfParent();

  /// \brief Set to the index parent tensor.
  /// \param[in] index [py::object] The index parent tensor.
  void SetIndexOfParent(const py::object &index);

  /// \brief Get tensor's shape.
  /// \return py::tuple which represents the shape of the tensor.
  py::tuple GetPyTupleShape();

  /// \brief Get the tensor's length of one element in bytes.
  /// \return Length of one element in bytes.
  py::int_ GetPyItemSize();

  /// \brief Get the tensor's total number of bytes.
  /// \return Total number of bytes taken by the tensor.
  py::int_ GetPyNBytes();

  /// \brief Get the tensor's tuple of bytes to step in each dimension when traversing an array.
  /// \return The strides of the tensor.
  py::tuple GetPyTupleStrides();

  /// \brief Get the data type of the tensor in this TensorPy.
  /// All the types are defined in "include/ir/dtype.h".
  /// \return The data type of the tensor in this TensorPy.
  TypePtr GetDtype() const;

  /// \brief Set the dtype of a tensor in this TensorPy.
  /// \param[in] type [TypePtr] The dtype of the tensor to be set.
  TypePtr SetDtype(const TypePtr type);

  /// \brief Get the data type of a tensor in this TensorPy.
  /// \return The data type.
  TypeId GetDataType() const;

  /// \brief Get tensor's shape.
  /// \return A const vector<int> which represents the shape of the tensor.
  const ShapeVector &GetShape() const;

  /// \brief Check if this Tensor is initialized.
  /// \return Whether this Tensor is initialized.
  bool IsInit() const;

  /// \brief Set the initialization flag of this Tensor.
  /// \param[in] flag Whether this Tensor is initialized.
  void SetInitFlag(bool flag);

  /// \brief Set the shape of the tensor in this TensorPy.
  /// \param[in] shape [ShapeVector] The shape of the tensor.
  void SetShape(const ShapeVector &shape);

  /// \brief Gets tensor's dimension.
  /// \return The number of dimensions of the tensor data.
  int DataDim() const;

  /// \brief Assign value to this tensorpy.
  /// \param[in] tensorpy [TensorPy] The input tensorpy.
  /// \return TensorPy with new value.
  TensorPy &AssignValue(const TensorPy &tensorpy);

  /// \brief Offload tensor data to file.
  /// \param[in] file_path [std::string] file path to save tensor data.
  /// \return Whether the tensor offload success.
  bool Offload(const std::string &file_path);

  /// \brief Get tensor offload file path.
  /// \return Offload file path, or empty string if tensor has not offload.
  const std::string GetOffloadFilePath() const;

  /// \brief Set the cast dtype of this TensorPy.
  /// \param[in] dtype [TypePtr] The input cast dtype.
  void SetCastDtype(const TypePtr &dtype = nullptr);

  /// \brief To synchronize data with the device, you need to wait for the data to be valid.
  void DataSync(bool need_wait = true) const;

  /// \brief Execute lazy task.
  void ExecuteLazyTask() const;

  /// \brief Determines whether the memory of tensor is contiguous.
  /// \return True if tensor memory is contiguous, false otherwise.
  bool IsContiguous() const;

  /// \brief Get tensor storage stride.
  /// \return Storage stride.
  std::vector<int64_t> GetStride() const;

  /// \brief Get tensor storage offset.
  /// \return Storage offset.
  const int64_t GetStorageOffset() const;

  /// \brief Get display information of this TensorPy.
  /// \return The display information of this TensorPy.
  std::string ToString() const;

  /// \brief Get display information in repr form.
  /// \return The display information in repr form.
  std::string ToStringRepr() const;

  /// \brief Get tensors stub flag.
  /// \return If compile with backend, return false, else return true.
  static bool CheckStub();

  /// \brief Get tensor's param_info info.
  /// \return The tensor's param_info info.
  ParamInfoPtr GetParamInfo() const;

  /// \brief Set tensor's param_info info.
  /// \param[in] param_info [ParamInfoPtr] The input param_info.
  void SetParamInfo(const ParamInfoPtr &param_info);

  /// \brief Used for dynamically optimize shape.
  /// \return The symbolic shape.
  const py::object GetSymbolicShape() const;

  /// \brief Set the shape of tensor to symbolic.
  /// \param[in] symbolic [py::object] The symbolic shape.
  void SetSymbolicShape(const py::object &symbolic);

  /// \brief Getting tensor data size.
  /// \return The total number of elements of the tensor data.
  const size_t GetDataSize() const;

  /// \brief Get Tensor data pointer for c++ type.
  /// \return The pointer to the object.
  void *GetTensorDataObject() const;

  /// \brief Get the device address.
  /// \return The device address.
  const DeviceAddressPtr GetDeviceAddress() const;

  /// \brief Whether the tensor is parameter output.
  /// \return True or False, is parameter output or not.
  bool IsMSParameterOutput() const;

  /// \brief Set the tensorpy to parameter output.
  /// \param[in] flag [bool] Is parameter output or not.
  void SetMSParameterOutput(bool flag);

  /// \brief Whether the tensor is used by a inplace operator.
  /// \return true
  bool has_side_effect() const { return has_side_effect_; }

  /// \brief Set side effect flag to py tensor.
  /// \param side_effect
  void set_has_side_effect(bool side_effect) { has_side_effect_ = side_effect; }

  /// \brief Check whether the type of tensor is complex.
  /// \return Boolean indicate whether the type of tensor is complex.
  bool IsComplex() const;

  /// \brief Check whether the type of tensor is signed.
  /// \return Boolean indicate whether the type of tensor is signed.
  bool IsSigned() const;

  /// \brief Check whether the tensor is used in auto grad.
  /// \return Boolean indicate whether the tensor is used in auto grad.
  bool HasAutoGrad() const;

  /// \brief Check whether the memory of tensor is contiguous.
  /// \return True if tensor memory is contiguous, false otherwise.
  bool NeedContiguous() const;

  void UpdateStub(const TensorPtr &tensor);

  /// \brief Get storage of tensor.
  /// \return The storage of tensor.
  const py::object GetStorage() const;

  /// \brief Get storage of tensor.
  /// \return The storage of tensor.
  void SetStorage(py::object storage);

  bool has_stub() const { return stub_ != nullptr; }
  const stub::StubNodePtr &stub() const { return stub_; }
  const stub::StubNodePtr &MakeStub() {
    stub_ = std::make_shared<stub::StubNode>();
    return stub_;
  }

 private:
  bool init_finished_flag_{false};
  bool const_arg_flag_{false};
  bool virtual_flag_{false};
  bool ms_parameter_output_{false};
  bool has_side_effect_{false};
  py::object initializer_;
  py::object parent_tensor_;
  py::object index_of_parent_;
  py::object symbolic_shape_;
  py::object storage_{py::none()};
  std::string device_;
  TensorPtr tensor_{nullptr};
  py::object flatten_tensor_;
  stub::StubNodePtr stub_{nullptr};
};

using TensorPyPtr = std::shared_ptr<TensorPy>;
using TensorPyPtrList = std::vector<std::shared_ptr<TensorPy>>;

/// \brief Check whether the object is TensorPy.
/// \param[in] obj [py::handle] The python object.
/// \return Is TensorPy or not.
COMMON_EXPORT bool IsTensorPy(const py::handle &obj);

/// \brief Check whether the object is TensorPy or subclass of TensorPy.
/// \param[in] obj [PyObject] The python object.
/// \return Is TensorPy or not.
COMMON_EXPORT bool IsPyObjectTensorPy(PyObject *obj);

/// \brief Check whether the object is Python Tensor.
/// \param[in] obj [PyObject] The python object.
/// \return Is TensorPy or not.
COMMON_EXPORT bool IsPyObjectPythonTensorStrict(PyObject *obj);

/// \brief Convert the python object to TensorPy.
/// \param[in] obj [py::handle] The python object.
/// \return A pointer address of TensorPy.
COMMON_EXPORT const py::handle ConvertToTensorPy(const py::handle &obj);

/// \brief Convert the python object to C++ Tensor.
/// \param[in] obj [py::handle] The python object.
/// \return A pointer address of C++ Tensor.
COMMON_EXPORT TensorPtr ConvertToTensor(const py::handle &obj);

/// \brief Set Tensor value for TensorPy.
///
/// \param[in] obj [py::handle] The python object.
/// \param[in] tensor_value [TensorPtr] C++ Tensor.
COMMON_EXPORT void SetTensorValue(const py::handle &obj, const TensorPtr &tensor_value);

/// \brief Convert the PyObject to C++ Tensor
/// \param[in] obj [PyObject] The python object.
/// \return A pointer address of C++ Tensor.
COMMON_EXPORT TensorPtr ConvertPyObjectToTensor(PyObject *obj);

COMMON_EXPORT const ValuePtr ConvertToValue(const py::handle &obj);
COMMON_EXPORT const ValuePtr ConvertPyObjectToValue(PyObject *obj);

/// \brief Create Tensor from a numpy array object.
///
/// \param[in] input [py::array] Data value of the tensor.
/// \param[in] type_ptr [TypePtr] Data type of the tensor.
/// \return A pointer address of C++ Tensor.
COMMON_EXPORT TensorPtr MakeTensor(const py::array &input, const TypePtr &type_ptr = nullptr);

/// \brief Create Tensor from a numpy array without copy.
///
/// \param[in] input [py::array] Data value of the tensor.
/// \return A pointer address of C++ Tensor.
COMMON_EXPORT TensorPtr MakeTensorOfNumpy(const py::array &input);

/// \brief Create a numpy array from tensor nonblocking.
///
/// \param[in] tensor_value [TensorPtr] C++ Tensor.
/// \return [py::array] Data value of the tensor.
COMMON_EXPORT py::array NumpyNonBlocking(const Tensor &tensor);

/// \brief Create a numpy array from tensor.
///
/// \param[in] tensor_value [TensorPtr] C++ Tensor.
/// \return [py::array] Data value of the tensor.
COMMON_EXPORT py::array AsNumpy(const Tensor &tensor);

/// \brief Get buffer info from a py array.
///
/// \param[in] input [py::array] Data value of the tensor.
/// \return [py::buffer_info] Pybind buffer info.
COMMON_EXPORT py::buffer_info GetPyBufferFromPyArray(const py::array &input);

template <typename T>
struct PyType {
  PyObject_HEAD T value;
};
/// \brief Convert the python object to C++ TensorPy.
/// \param[in] obj [py::handle] The python object.
/// \return A pointer address of C++ TensorPy.
COMMON_EXPORT PyType<TensorPy> *ConvertPyObject2TensorPyType(const py::object obj);
/// \brief get TensorPy Type.
/// \param[in] none.
/// \return PyTypeObject of TensorPy Type.
COMMON_EXPORT PyTypeObject *GetTensorPyType();
/// \brief set TensorPy Type for global use.
/// \param[in] TensorPyType [PyTypeObject *] The python type.
/// \return none.
COMMON_EXPORT void SetTensorPyType(PyTypeObject *TensorPyType);
/// \brief alloc Python Tensor from C++ Tensor.
/// \param[in] tensor [TensorPtr] C++ Tensor.
/// \return A PyObject address of Python Tensor.
COMMON_EXPORT PyObject *TensorPythonInit(const TensorPtr &tensor);
COMMON_EXPORT PyObject *TensorPythonInitFromTensor(TensorPtr tensor);

COMMON_EXPORT py::object PackTensorToPyObject(const TensorPtr &tensor);

/// \brief Get the Python Tensor Object.
/// \return The python Tensor.
COMMON_EXPORT py::object GetPythonTensor();

COMMON_EXPORT PyObject *PackTensor(const TensorPtr &tensor, bool has_side_effect = false);
COMMON_EXPORT PyObject *PackStubTensor(const stub::StubNodePtr &stub_node);
COMMON_EXPORT PyObject *Wrap(const TensorPtr &tensor);
COMMON_EXPORT PyObject *Wrap(const std::vector<TensorPtr> &tensors);
COMMON_EXPORT PyObject *Wrap(const ValuePtrList &values);
template <typename... Args>
PyObject *Wrap(const std::tuple<Args...> &tuple) {
  constexpr size_t size = std::tuple_size<std::tuple<Args...>>::value;
  PyObject *output = PyTuple_New(size);
  std::apply(
    [&output](const auto &...args) {
      size_t index = 0;
      ((PyTuple_SET_ITEM(output, index++, Wrap(args))), ...);
    },
    tuple);
  return output;
}

COMMON_EXPORT PyObject *Wrap(const ValuePtr &value);
}  // namespace tensor
}  // namespace mindspore

namespace pybind11 {
namespace detail {
// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;

template <typename T>
struct npy_scalar_caster {
  PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
  using Array = array_t<T>;

  bool load(handle src, bool convert) const {
    // Taken from Eigen casters. Permits either scalar dtype or scalar array.
    handle type = dtype::of<T>().attr("type");
    if (!convert && !isinstance<Array>(src) && !isinstance(src, type)) {
      return false;
    }

    Array tmp = Array::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }

    return false;
  }

  static handle cast(T src, return_value_policy, handle) {
    Array tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});

    // You could also just return the array if you want a scalar array.
    object scalar = tmp[tuple()];
    return scalar.release();
  }
};

template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = "float16";
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  virtual ~npy_format_descriptor<float16>() {}
};

template <>
struct type_caster<float16> : public npy_scalar_caster<float16> {
  static constexpr auto name = "float16";
};

template <>
struct npy_format_descriptor<bfloat16> {
  static constexpr auto name = "bfloat16";
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(mindspore::GetBFloat16NpDType());
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  virtual ~npy_format_descriptor<bfloat16>() {}
};

template <>
struct type_caster<bfloat16> : public npy_scalar_caster<bfloat16> {
  static constexpr auto name = "bfloat16";
};

template <>
struct type_caster<mindspore::tensor::TensorPtr> {
  PYBIND11_TYPE_CASTER(mindspore::tensor::TensorPtr, _("Tensor"));
  bool load(handle src, bool) {
    if (mindspore::tensor::IsTensorPy(src)) {
      value = mindspore::tensor::ConvertToTensor(src);
      return true;
    }
    return false;
  }
  static handle cast(const mindspore::tensor::TensorPtr &src, return_value_policy, handle) {
    return handle(mindspore::tensor::Wrap(src));
  }
};
}  // namespace detail
}  // namespace pybind11

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_TENSOR_PY_H_
