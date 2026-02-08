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
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "include/securec.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "pybind_api/pynative/tensor/tensor_py.h"
#include "include/utils/tensor_py.h"
#include "common/mockcpp.h"

using mindspore::tensor::TensorPybind;
using mindspore::tensor::TensorPyImpl;

namespace mindspore {
namespace tensor {

class TestTensorPy : public UT::Common {
 public:
  TestTensorPy() {}
  virtual void SetUp() { UT::InitPythonPath(); }
  py::object GetDtypeObj() {
    constexpr auto common_module = "mindspore._c_expression.typing";
    py::module mod = python_adapter::GetPyModule(common_module);
    auto py_dtype = python_adapter::CallPyModFn(mod, "type_id_to_type", TypeId::kNumberTypeFloat64);
    return py_dtype;
  }
};

/// Feature: Test InitByShapeTest.
/// Description: Test tensor initialization based on shape and dtype.
/// Expectation: Successfully init tensor.
TEST_F(TestTensorPy, InitByShapeTest) {
    py::tuple tuple = py::make_tuple(2, 3);
    py::dict input = py::dict();
    input[py::str("shape")] = tuple;
    input[py::str("dtype")] = GetDtypeObj();

    TensorPtr tensor = TensorPyImpl::InitTensor(input);
    std::vector<int64_t> dimensions = tensor->shape();

    ASSERT_EQ(dimensions.size(), 2);

    ASSERT_EQ(tensor->data_type(), TypeId::kNumberTypeFloat64);
}

/// Feature: Test GetDeviceFromPythonTest.
/// Description: Test get device info of tensor.
/// Expectation: Successfully get device info.
TEST_F(TestTensorPy, GetDeviceFromPythonTest) {
    py::dict input = py::dict();
    std::string res = TensorPyImpl::GetDeviceFromPython(input);
    ASSERT_EQ(res, "");

    input = py::dict();
    std::string device = "CPU";
    PyObject *devicePyObj = PyUnicode_FromString(device.c_str());
    py::object deviceObj = py::reinterpret_steal<py::object>(devicePyObj);
    input[py::str("device")] = deviceObj;
    res = TensorPyImpl::GetDeviceFromPython(input);
    ASSERT_EQ(res, "CPU");

    try {
        input = py::dict();
        input[py::str("device")] = py::reinterpret_steal<py::object>(PyLong_FromLong(123));
        res = TensorPyImpl::GetDeviceFromPython(input);
    } catch (const std::exception &ex) {
        const std::string &exception_str = ex.what();
        ASSERT_TRUE(exception_str.find("the device should be string") != std::string::npos);
    }

    try {
        input = py::dict();
        std::string device = "Ascend";
        py::object deviceObj = py::reinterpret_steal<py::object>(PyUnicode_FromString(device.c_str()));
        input[py::str("device")] = deviceObj;
        std::string res = TensorPyImpl::GetDeviceFromPython(input);
    } catch (const std::exception &ex) {
        const std::string &exception_str = ex.what();
        ASSERT_TRUE(exception_str.find("Only 'CPU' is supported for device") != std::string::npos);
    }
}

/// Feature: Test GetConstArgFromPythonTest.
/// Description: Test get const arg of tensor.
/// Expectation: Successfully get const arg.
TEST_F(TestTensorPy, GetConstArgFromPythonTest) {
    py::dict input = py::dict();
    bool const_arg_res = TensorPyImpl::GetConstArgFromPython(input);
    ASSERT_EQ(const_arg_res, false);

    input = py::dict();
    py::bool_ const_arg_py = true;
    input[py::str("const_arg")] = const_arg_py;
    const_arg_res = TensorPyImpl::GetConstArgFromPython(input);
    ASSERT_EQ(const_arg_res, true);

    try {
        input = py::dict();
        py::object const_arg_py = py::cast(123);
        input[py::str("const_arg")] = const_arg_py;
        const_arg_res = TensorPyImpl::GetConstArgFromPython(input);
    } catch (const std::exception &ex) {
        const std::string &exception_str = ex.what();
        ASSERT_TRUE(exception_str.find("For 'Tensor', the type of 'const_arg' should be 'bool'") != std::string::npos);
    }
}

/// Feature: Test GetPyItemSizeTest.
/// Description: Test get item size of tensor.
/// Expectation: Successfully get item size.
TEST_F(TestTensorPy, GetPyItemSizeTest) {
    py::tuple tensor_data_tuple = py::make_tuple(1, 2, 3, 4, 5, 6);
    TensorPtr tensor_int32 = tensor::MakeTensor(py::array(tensor_data_tuple), kInt32);
    py::int_ item_size_res = TensorPybind::GetPyItemSize(*tensor_int32);
    ASSERT_EQ(item_size_res, 4);
}

/// Feature: Test GetPyNBytesTest.
/// Description: Test get nbytes of tensor.
/// Expectation: Successfully get nbytes.
TEST_F(TestTensorPy, GetPyNBytesTest) {
    py::tuple tensor_data_tuple = py::make_tuple(1, 2, 3, 4, 5, 6);
    TensorPtr tensor_int32 = tensor::MakeTensor(py::array(tensor_data_tuple), kInt32);
    py::int_ nbytes_res = TensorPybind::GetPyNBytes(*tensor_int32);
    ASSERT_EQ(nbytes_res, 24);
}

/// Feature: Test SyncAsNumpyTest.
/// Description: Test synchronously convert tensor to numpy array.
/// Expectation: Throw exception when tensor's dtype is bfloat16.
TEST_F(TestTensorPy, SyncAsNumpyTest) {
    MOCKER(IsCustomNumpyTypeValid).stubs().will(returnValue(false));
    try {
        py::tuple tensor_data_tuple = py::make_tuple(1, 2, 3, 4, 5, 6);
        TensorPtr tensor_int32 = tensor::MakeTensor(py::array(tensor_data_tuple), kBFloat16);
        py::array sync_as_numpy_res = TensorPybind::SyncAsNumpy(*tensor_int32);
        ASSERT_FALSE(sync_as_numpy_res.is_none());
    } catch (const std::exception &ex) {
        const std::string &exception_str = ex.what();
        ASSERT_TRUE(exception_str.find("The Numpy bfloat16 data type is not supported now") != std::string::npos);
    }
}

}  // namespace tensor
}  // namespace mindspore
