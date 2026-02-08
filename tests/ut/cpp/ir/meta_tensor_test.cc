/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "mindspore/ccsrc/include/utils/tensor_py.h"

namespace mindspore {
namespace tensor {
class TestMetaTensor : public UT::Common {
 public:
  TestMetaTensor() {}
  virtual void SetUp() {
    std::vector<int64_t> dimensions({2, 3});
    meta_tensor_ = MetaTensor(TypeId::kNumberTypeFloat64, dimensions);
  }

 protected:
  MetaTensor meta_tensor_;
};

TEST_F(TestMetaTensor, InitTest) {
  std::vector<int64_t> dimensions({2, 3});
  MetaTensor meta_tensor(TypeId::kNumberTypeFloat64, dimensions);

  // Test type
  ASSERT_EQ(TypeId::kNumberTypeFloat64, meta_tensor.data_type());

  // Test dimensions
  ASSERT_EQ(2, meta_tensor.DimensionSize(0));
  ASSERT_EQ(3, meta_tensor.DimensionSize(1));
  ASSERT_EQ(-1, meta_tensor.DimensionSize(2));

  // Test number of elements
  ASSERT_EQ(6, meta_tensor.ElementsNum());
}

// Test type
TEST_F(TestMetaTensor, TypeTest) {
  meta_tensor_.set_data_type(TypeId::kNumberTypeInt32);
  ASSERT_EQ(TypeId::kNumberTypeInt32, meta_tensor_.data_type());
}

// Test shape
TEST_F(TestMetaTensor, ShapeTest) {
  std::vector<int64_t> dimensions({5, 6, 7});
  meta_tensor_.set_shape(dimensions);

  ASSERT_EQ(5, meta_tensor_.DimensionSize(0));
  ASSERT_EQ(6, meta_tensor_.DimensionSize(1));
  ASSERT_EQ(7, meta_tensor_.DimensionSize(2));

  // Test number of elements
  ASSERT_EQ(210, meta_tensor_.ElementsNum());

  std::vector<int64_t> dimensions1({100000, 100000, 100000});
  meta_tensor_.set_shape(dimensions1);
  ASSERT_EQ(1e15, meta_tensor_.ElementsNum());
}

TEST_F(TestMetaTensor, EqualTest) {
  std::vector<int64_t> dimensions({2, 3});
  MetaTensor meta_tensor_x(TypeId::kNumberTypeFloat64, dimensions);
  MetaTensor meta_tensor_y(meta_tensor_x);

  ASSERT_TRUE(meta_tensor_x == meta_tensor_y);

  MetaTensor meta_tensor_z(TypeId::kNumberTypeFloat32, dimensions);
  ASSERT_FALSE(meta_tensor_x == meta_tensor_z);

  meta_tensor_z = meta_tensor_x;
  ASSERT_TRUE(meta_tensor_x == meta_tensor_z);
}

class TestTensor : public UT::Common {
 public:
  TestTensor() {}
  virtual void SetUp() { UT::InitPythonPath(); }
};

py::array_t<float, py::array::c_style> BuildInputTensor() {
  // Init tensor data by py::array_t<float>
  py::array_t<float, py::array::c_style> input = py::array_t<float, py::array::c_style>({2, 3});
  auto array = input.mutable_unchecked();
  float start = 0;
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      array(i, j) = start++;
    }
  }
  return input;
}

TEST_F(TestTensor, PyArrayScalarTest) {
  std::vector<int64_t> dimensions;
  py::array data = py::array_t<int64_t, py::array::c_style>(dimensions);
  uint8_t *data_buf = reinterpret_cast<uint8_t *>(data.request(true).ptr);

  int64_t num = 1;
  errno_t ret = memcpy_s(data_buf, sizeof(int64_t), &num, sizeof(int64_t));

  ASSERT_EQ(0, ret);

  ASSERT_EQ(num, *data_buf);
}

TEST_F(TestTensor, InitScalarTest) {
  std::vector<int64_t> dimensions;
  TensorPtr tensor = tensor::from_spec(TypeId::kNumberTypeInt64, dimensions, device::DeviceType::kCPU);
  uint8_t *data_buf = reinterpret_cast<uint8_t *>(tensor->data_c());

  int64_t num = 1;
  errno_t ret = memcpy_s(data_buf, sizeof(int64_t), &num, sizeof(int64_t));

  ASSERT_EQ(0, ret);

  ASSERT_EQ(num, *data_buf);

  // Test type
  ASSERT_EQ(TypeId::kNumberTypeInt64, tensor->data_type());

  // Test dimensions
  ASSERT_EQ(0, tensor->DataDim());

  // Test shape
  ASSERT_EQ(0, tensor->shape().size());
  std::vector<int64_t> empty_shape;
  ASSERT_EQ(empty_shape, tensor->shape());

  // Test number of elements
  ASSERT_EQ(1, tensor->ElementsNum());
  ASSERT_EQ(1, tensor->DataSize());
}

TEST_F(TestTensor, InitTensorPtrTest) {
  std::vector<int64_t> dimensions;
  Tensor tensor(TypeId::kNumberTypeInt64, dimensions);

  std::shared_ptr<Tensor> tensor_ptr = std::make_shared<Tensor>(tensor);

  // Test type
  ASSERT_EQ(TypeId::kNumberTypeInt64, tensor_ptr->data_type());

  // Test dimensions
  ASSERT_EQ(0, tensor_ptr->DataDim());

  // Test shape
  ASSERT_EQ(0, tensor_ptr->shape().size());
  std::vector<int64_t> empty_shape;
  ASSERT_EQ(empty_shape, tensor_ptr->shape());

  // Test number of elements
  ASSERT_EQ(1, tensor_ptr->ElementsNum());
  ASSERT_EQ(1, tensor_ptr->DataSize());
}

TEST_F(TestTensor, InitByTupleTest) {
  const std::vector<int64_t> shape = {2, 3, 4};
  TypePtr data_type = kFloat32;
  Tensor tuple_tensor(data_type->type_id(), shape);
  ASSERT_EQ(2, tuple_tensor.DimensionSize(0));
  ASSERT_EQ(3, tuple_tensor.DimensionSize(1));
  ASSERT_EQ(4, tuple_tensor.DimensionSize(2));

  // Test number of elements
  ASSERT_EQ(24, tuple_tensor.ElementsNum());
  ASSERT_EQ(TypeId::kNumberTypeFloat32, tuple_tensor.data_type());

  py::tuple tuple = py::make_tuple(1.0, 2.0, 3, 4, 5, 6);
  TensorPtr tensor = tensor::MakeTensor(py::array(tuple), kFloat64);
  py::array array = tensor::AsNumpy(*tensor);

  std::cout << "Dim: " << array.ndim() << std::endl;
  ASSERT_EQ(1, array.ndim());

  std::cout << "Num of Elements: " << array.size() << std::endl;
  ASSERT_EQ(6, array.size());

  std::cout << "Elements: " << std::endl;
  // Must be double, or the result is not right
  double *tensor_data = reinterpret_cast<double *>(tensor->data_c());
  for (int i = 0; i < array.size(); i++) {
    std::cout << tensor_data[i] << std::endl;
  }
}

TEST_F(TestTensor, EqualTest) {
  py::tuple tuple = py::make_tuple(1, 2, 3, 4, 5, 6);
  TensorPtr tensor_int8 = tensor::MakeTensor(py::array(tuple), kInt8);
  ASSERT_TRUE(*tensor_int8 == *tensor_int8);

  ASSERT_EQ(TypeId::kNumberTypeInt8, tensor_int8->data_type_c());

  TensorPtr tensor_int16 = tensor::MakeTensor(py::array(tuple), kInt16);
  ASSERT_EQ(TypeId::kNumberTypeInt16, tensor_int16->data_type_c());

  TensorPtr tensor_int32 = tensor::MakeTensor(py::array(tuple), kInt32);
  ASSERT_EQ(TypeId::kNumberTypeInt32, tensor_int32->data_type_c());

  TensorPtr tensor_float16 = tensor::MakeTensor(py::array(tuple), kFloat16);
  ASSERT_EQ(TypeId::kNumberTypeFloat16, tensor_float16->data_type_c());

  TensorPtr tensor_float32 = tensor::MakeTensor(py::array(tuple), kFloat32);
  ASSERT_EQ(TypeId::kNumberTypeFloat32, tensor_float32->data_type_c());

  TensorPtr tensor_float64 = tensor::MakeTensor(py::array(tuple), kFloat64);
  ASSERT_EQ(TypeId::kNumberTypeFloat64, tensor_float64->data_type_c());
}

TEST_F(TestTensor, ValueEqualTest) {
  py::tuple tuple = py::make_tuple(1, 2, 3, 4, 5, 6);
  TensorPtr t1 = tensor::MakeTensor(py::array(tuple), kInt32);
  TensorPtr t2 = tensor::MakeTensor(py::array(tuple), kInt32);
  ASSERT_TRUE(t1->ValueEqual(*t1));
  ASSERT_TRUE(t1->ValueEqual(*t2));

  std::vector<int64_t> shape = {6};
  TensorPtr t3 = tensor::from_spec(kInt32->type_id(), shape, device::DeviceType::kCPU);
  TensorPtr t4 = tensor::from_spec(kInt32->type_id(), shape, device::DeviceType::kCPU);
  ASSERT_TRUE(t3->ValueEqual(*t3));
  ASSERT_TRUE(t3->ValueEqual(*t4));
  ASSERT_FALSE(t3->ValueEqual(*t1));
  ASSERT_FALSE(t1->ValueEqual(*t3));

  memcpy_s(t3->data_c(), t3->DataNBytes(), t1->data_c(), t1->DataNBytes());
  ASSERT_TRUE(t1->ValueEqual(*t3));
  ASSERT_FALSE(t3->ValueEqual(*t4));
  ASSERT_FALSE(t4->ValueEqual(*t3));
}

TEST_F(TestTensor, PyArrayTest) {
  py::array_t<float, py::array::c_style> input({2, 3});
  auto array = input.mutable_unchecked();
  float sum = 0;
  std::cout << "sum"
            << " = " << std::endl;

  float start = 0;
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      array(i, j) = start++;
      sum += array(i, j);
      std::cout << "sum + "
                << "array[" << i << ", " << j << "]"
                << " = " << sum << std::endl;
    }
  }

  ASSERT_EQ(15, sum);
}

TEST_F(TestTensor, InitByFloatArrayDataCTest) {
  // Init tensor data by py::array_t<float>
  auto tensor = tensor::MakeTensor(BuildInputTensor());

  // Print some information of the tensor
  std::cout << "Datatype: " << tensor->data_type() << std::endl;
  ASSERT_EQ(TypeId::kNumberTypeFloat32, tensor->data_type());

  std::cout << "Dim: " << tensor->DataDim() << std::endl;
  ASSERT_EQ(2, tensor->DataDim());

  std::cout << "Num of Elements: " << tensor->ElementsNum() << std::endl;
  ASSERT_EQ(6, tensor->ElementsNum());

  // Print each elements
  std::cout << "Elements: " << std::endl;
  float *tensor_data = reinterpret_cast<float *>(tensor->data_c());
  for (int64_t i = 0; i < tensor->ElementsNum(); i++) {
    std::cout << tensor_data[i] << std::endl;
  }
}

TEST_F(TestTensor, InitByFloatArrayDataTest) {
  // Init tensor data by py::array_t<float>
  TensorPtr tensor = tensor::MakeTensor(BuildInputTensor());

  // Print some information of the tensor
  std::cout << "Datatype: " << tensor->data_type() << std::endl;
  ASSERT_EQ(TypeId::kNumberTypeFloat32, tensor->data_type());

  std::cout << "Dim: " << tensor->DataDim() << std::endl;
  ASSERT_EQ(2, tensor->DataDim());

  std::vector<int64_t> dimensions = tensor->shape();
  ASSERT_GT(dimensions.size(), 1);
  std::cout << "Dim0: " << dimensions[0] << std::endl;
  ASSERT_EQ(2, dimensions[0]);

  std::cout << "Dim1: " << dimensions[1] << std::endl;
  ASSERT_EQ(3, dimensions[1]);

  std::cout << "Num of Elements: " << tensor->ElementsNum() << std::endl;
  ASSERT_EQ(6, tensor->ElementsNum());

  // Print each elements
  std::cout << "Elements: " << std::endl;
  py::array_t<float> data = py::cast<py::array_t<float>>(tensor::AsNumpy(*tensor));
  auto array = data.unchecked<2>();
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      std::cout << array(i, j) << std::endl;
    }
  }
}

TEST_F(TestTensor, PyArrayDataTest) {
  py::array_t<float, py::array::c_style> input({2, 3});
  float *data = reinterpret_cast<float *>(input.request().ptr);
  float ge_tensor_data[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
  errno_t ret = memcpy_s(data, input.nbytes(), ge_tensor_data, sizeof(ge_tensor_data));
  ASSERT_EQ(0, ret);
  auto array = input.mutable_unchecked();
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      ASSERT_EQ(array(i, j), ge_tensor_data[3 * i + j]);
    }
  }
}

TEST_F(TestTensor, TensorDataTest) {
  // Init a data buffer
  float ge_tensor_data[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};

  // Create a Tensor with wanted data type and shape
  TensorPtr tensor = tensor::from_spec(TypeId::kNumberTypeFloat32, std::vector<int64_t>({2, 3}), device::DeviceType::kCPU);

  // Get the writable data pointer from the tensor
  float *me_tensor_data = reinterpret_cast<float *>(tensor->data_c());

  // Copy data from buffer to tensor's data
  errno_t ret = memcpy_s(me_tensor_data, tensor->DataNBytes(), ge_tensor_data, sizeof(ge_tensor_data));
  ASSERT_EQ(0, ret);

  // Testify if the data has been copied to the tensor data
  py::array_t<float> data = py::cast<py::array_t<float>>(tensor::AsNumpy(*tensor));
  auto array = data.mutable_unchecked();
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      std::cout << "array[" << i << ", " << j << "]"
                << " = " << array(i, j) << std::endl;
      ASSERT_EQ(array(i, j), ge_tensor_data[3 * i + j]);
    }
  }
}

TEST_F(TestTensor, TensorPyCast) {
  std::vector<int64_t> shape{2, 3, 4, 5};
  auto tensor = tensor::from_spec(kNumberTypeFloat32, shape, device::DeviceType::kCPU);
  PyObject *tensor_py = TensorPythonInit(tensor);
  py::object tensor_obj = py::reinterpret_borrow<py::object>(tensor_py);
  py::tuple py_tuple = py::make_tuple(tensor_obj);
  auto shape1 = ConvertToTensor(py_tuple[0])->shape();
  const py::tuple &t = py_tuple;
  auto shape2 = ConvertToTensor(t[0])->shape();
  ASSERT_EQ(shape, shape1);
  ASSERT_EQ(shape, shape2);
}

/// Feature: Tensor
/// Description: Test user data for Tensor.
/// Expectation: user data works as expected.
TEST_F(TestTensor, TensorWithUserData) {
  auto tensor = tensor::from_scalar(3.14f);
  auto mydata = std::make_shared<std::string>("mydata");

  // Set user data.
  tensor->set_user_data("mykey", mydata);
  ASSERT_TRUE(tensor->has_user_data("mykey"));
  ASSERT_EQ(tensor->user_data<std::string>("mykey"), mydata);

  // Copy with user data.
  auto tensor1 = std::make_shared<Tensor>(*tensor);
  ASSERT_TRUE(tensor1->has_user_data("mykey"));
  ASSERT_EQ(tensor1->user_data<std::string>("mykey"), mydata);

  // Erase user data.
  tensor->set_user_data<std::string>("mykey", nullptr);
  ASSERT_FALSE(tensor->has_user_data("mykey"));
  ASSERT_EQ(tensor->user_data<std::string>("mykey"), nullptr);

  // user data in tensor1 is not removed.
  ASSERT_TRUE(tensor1->has_user_data("mykey"));
  ASSERT_EQ(tensor1->user_data<std::string>("mykey"), mydata);
}

/// Feature: Tensor
/// Description: Test set new shape for tensor.
/// Expectation: Tensor's shape will be changed and data will be reinitialized.
TEST_F(TestTensor, TensorSetShapeDataTest) {
  // Create a Tensor with wanted data type and shape
  std::vector<int64_t> old_shape({2, 3});
  TensorPtr tensor = tensor::from_spec(TypeId::kNumberTypeInt64, std::vector<int64_t>(old_shape), device::DeviceType::kCPU);
  tensor->set_shape(old_shape);
  ASSERT_EQ(6, tensor->DataSize());
  ASSERT_NE(nullptr, tensor->unsafe_data());

  // Init a data buffer
  int64_t ge_tensor_data[] = {1, 2, 3, 4, 5, 6};
  // Get the writable data pointer from the tensor
  int64_t *tensor_data = reinterpret_cast<int64_t *>(tensor->data_c());
  // Copy data from buffer to tensor's data
  errno_t ret = memcpy_s(tensor_data, tensor->DataNBytes(), ge_tensor_data, sizeof(ge_tensor_data));
  ASSERT_EQ(0, ret);
  ASSERT_NE(nullptr, tensor->unsafe_data());

  // Shape change litter
  std::vector<int64_t> little_shape({1, 2});
  tensor->set_shape(little_shape);
  ASSERT_EQ(2, tensor->DataSize());
  ASSERT_NE(nullptr, tensor->unsafe_data());
}

/// Feature: Tensor offload
/// Description: Test offload tensor to file.
/// Expectation: Offload and Load file success.
TEST_F(TestTensor, TensorOffloadTest) {
  // Create a Tensor with wanted data type and shape
  std::vector<int64_t> tensor_shape({2, 3});
  TensorPtr tensor = tensor::from_spec(TypeId::kNumberTypeInt64, tensor_shape, device::DeviceType::kCPU);
  ASSERT_EQ(6, tensor->DataSize());
  ASSERT_NE(nullptr, tensor->unsafe_data());

  // Init a data buffer
  int64_t init_data[] = {1, 2, 3, 4, 5, 6};
  errno_t ret = memcpy_s(tensor->data_c(), tensor->DataNBytes(), init_data, sizeof(init_data));
  ASSERT_EQ(0, ret);
  ASSERT_NE(nullptr, tensor->unsafe_data());
  auto const kTmpFilePath = "./test_file_path";
  tensor->Offload(kTmpFilePath);
  ASSERT_EQ(tensor->GetOffloadFilePath(), kTmpFilePath);

  // Check tensor data
  int64_t load_data[] = {0, 0, 0, 0, 0, 0};
  ret = memcpy_s(load_data, tensor->DataNBytes(), tensor->data_c(), sizeof(load_data));
  ASSERT_EQ(0, ret);
  const size_t kElemNum = 6;
  for (size_t i = 0; i < kElemNum; ++i) {
    ASSERT_EQ(load_data[i], init_data[i]);
  }
}

/// Feature: SparseTensor
/// Description: test AbstractSparseTensor/SparseTensorType API.
/// Expectation: AbstractSparseTensor/SparseTensorType work as expected.
TEST_F(TestTensor, SparseTensor) {
  AbstractBasePtrList abs_list;
  ShapeVector indices_shape{4, 2};
  ShapeVector values_shape{4};
  abs_list.emplace_back(std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeInt32), indices_shape));
  abs_list.emplace_back(std::make_shared<abstract::AbstractTensor>(TypeIdToType(kNumberTypeFloat32), values_shape));
  AbstractBasePtrList shape_list;
  shape_list.emplace_back(std::make_shared<abstract::AbstractScalar>(5));
  shape_list.emplace_back(std::make_shared<abstract::AbstractScalar>(5));
  abs_list.emplace_back(std::make_shared<abstract::AbstractTuple>(shape_list));
  auto abs_sparse_tensor = std::make_shared<abstract::AbstractSparseTensor>(abs_list);
  ASSERT_EQ(abs_sparse_tensor->elements().size(), abs_list.size());
  ASSERT_TRUE(abs_sparse_tensor->BuildType()->isa<SparseTensorType>());
  ASSERT_TRUE(abs_sparse_tensor->shape()->isa<abstract::AbstractTuple>());
  ASSERT_TRUE(abs_sparse_tensor->elements()[0]->isa<abstract::AbstractTensor>());
  ASSERT_TRUE(abs_sparse_tensor->elements()[1]->isa<abstract::AbstractTensor>());

  // SparseTensorType
  TypePtrList elements{TypeIdToType(kNumberTypeInt32), TypeIdToType(kNumberTypeInt32), TypeIdToType(kNumberTypeFloat32),
                       TypeIdToType(kNumberTypeInt64), TypeIdToType(kNumberTypeInt64)};
  TypePtr sparse_tensor_type = std::make_shared<SparseTensorType>(elements);
  ASSERT_TRUE(sparse_tensor_type->isa<SparseTensorType>());
  ASSERT_EQ(sparse_tensor_type->cast<SparseTensorTypePtr>()->elements()[0]->type_id(), kNumberTypeInt32);
  ASSERT_EQ(sparse_tensor_type->cast<SparseTensorTypePtr>()->elements()[1]->type_id(), kNumberTypeInt32);
  ASSERT_EQ(sparse_tensor_type->cast<SparseTensorTypePtr>()->elements()[2]->type_id(), kNumberTypeFloat32);
  ASSERT_EQ(sparse_tensor_type->cast<SparseTensorTypePtr>()->elements()[3]->type_id(), kNumberTypeInt64);
  ASSERT_EQ(sparse_tensor_type->cast<SparseTensorTypePtr>()->elements()[4]->type_id(), kNumberTypeInt64);
}
}  // namespace tensor
}  // namespace mindspore
