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
#include <iostream>
#include <memory>
#include <vector>

#include "common/common_test.h"
#include "common/py_func_graph_fetcher.h"

#include "securec/include/securec.h"
#include "ir/tensor.h"

namespace mindspore {
namespace tensor {

class TestMetaTensor : public UT::Common {
 public:
  TestMetaTensor() {}
  virtual void SetUp() {
    std::vector<int> dimensions({2, 3});
    meta_tensor_ = MetaTensor(TypeId::kNumberTypeFloat64, dimensions);
  }

 protected:
  MetaTensor meta_tensor_;
};

TEST_F(TestMetaTensor, InitTest) {
  std::vector<int> dimensions({2, 3});
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
  std::vector<int> dimensions({5, 6, 7});
  meta_tensor_.set_shape(dimensions);

  ASSERT_EQ(5, meta_tensor_.DimensionSize(0));
  ASSERT_EQ(6, meta_tensor_.DimensionSize(1));
  ASSERT_EQ(7, meta_tensor_.DimensionSize(2));

  // Test number of elements
  ASSERT_EQ(210, meta_tensor_.ElementsNum());
}

TEST_F(TestMetaTensor, EqualTest) {
  std::vector<int> dimensions({2, 3});
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
  virtual void SetUp() {
    UT::InitPythonPath();
  }
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
  std::vector<int> dimensions;
  py::array data = py::array_t<int64_t, py::array::c_style>(dimensions);
  uint8_t *data_buf = reinterpret_cast<uint8_t *>(data.request(true).ptr);

  int64_t num = 1;
  errno_t ret = memcpy_s(data_buf, sizeof(int64_t), &num, sizeof(int64_t));

  ASSERT_EQ(0, ret);

  ASSERT_EQ(num, *data_buf);
}

TEST_F(TestTensor, InitScalarTest) {
  std::vector<int> dimensions;
  Tensor tensor(TypeId::kNumberTypeInt64, dimensions);
  uint8_t *data_buf = reinterpret_cast<uint8_t *>(tensor.data_c(true));

  int64_t num = 1;
  errno_t ret = memcpy_s(data_buf, sizeof(int64_t), &num, sizeof(int64_t));

  ASSERT_EQ(0, ret);

  ASSERT_EQ(num, *data_buf);

  // Test type
  ASSERT_EQ(TypeId::kNumberTypeInt64, tensor.data_type());

  // Test dimensions
  ASSERT_EQ(0, tensor.DataDim());

  // Test shape
  ASSERT_EQ(0, tensor.shape().size());
  std::vector<int> empty_shape;
  ASSERT_EQ(empty_shape, tensor.shape());

  // Test number of elements
  ASSERT_EQ(1, tensor.ElementsNum());
  ASSERT_EQ(1, tensor.DataSize());
}

TEST_F(TestTensor, InitTensorPtrTest) {
  std::vector<int> dimensions;
  Tensor tensor(TypeId::kNumberTypeInt64, dimensions);

  std::shared_ptr<Tensor> tensor_ptr = std::make_shared<Tensor>(tensor);

  // Test type
  ASSERT_EQ(TypeId::kNumberTypeInt64, tensor_ptr->data_type());

  // Test dimensions
  ASSERT_EQ(0, tensor_ptr->DataDim());

  // Test shape
  ASSERT_EQ(0, tensor_ptr->shape().size());
  std::vector<int> empty_shape;
  ASSERT_EQ(empty_shape, tensor_ptr->shape());

  // Test number of elements
  ASSERT_EQ(1, tensor_ptr->ElementsNum());
  ASSERT_EQ(1, tensor_ptr->DataSize());
}

TEST_F(TestTensor, InitByTupleTest) {
  py::tuple dimensions = py::make_tuple(2, 3, 4);
  TypePtr data_type = kFloat32;
  Tensor tuple_tensor = Tensor(data_type, dimensions);
  ASSERT_EQ(2, tuple_tensor.DimensionSize(0));
  ASSERT_EQ(3, tuple_tensor.DimensionSize(1));
  ASSERT_EQ(4, tuple_tensor.DimensionSize(2));

  // Test number of elements
  ASSERT_EQ(24, tuple_tensor.ElementsNum());
  ASSERT_EQ(TypeId::kNumberTypeFloat32, tuple_tensor.data_type());

  py::tuple tuple = py::make_tuple(1.0, 2.0, 3, 4, 5, 6);
  TensorPtr tensor = std::make_shared<Tensor>(tuple, kFloat64);
  py::array array = tensor->data();

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
  TensorPtr tensor_int8 = std::make_shared<Tensor>(tuple, kInt8);
  ASSERT_TRUE(*tensor_int8 == *tensor_int8);

  ASSERT_EQ(TypeId::kNumberTypeInt8, tensor_int8->data_type_c());

  TensorPtr tensor_int16 = std::make_shared<Tensor>(tuple, kInt16);
  ASSERT_EQ(TypeId::kNumberTypeInt16, tensor_int16->data_type_c());

  TensorPtr tensor_int32 = std::make_shared<Tensor>(tuple, kInt32);
  ASSERT_EQ(TypeId::kNumberTypeInt32, tensor_int32->data_type_c());

  TensorPtr tensor_float16 = std::make_shared<Tensor>(tuple, kFloat16);
  ASSERT_EQ(TypeId::kNumberTypeFloat16, tensor_float16->data_type_c());

  TensorPtr tensor_float32 = std::make_shared<Tensor>(tuple, kFloat32);
  ASSERT_EQ(TypeId::kNumberTypeFloat32, tensor_float32->data_type_c());

  TensorPtr tensor_float64 = std::make_shared<Tensor>(tuple, kFloat64);
  ASSERT_EQ(TypeId::kNumberTypeFloat64, tensor_float64->data_type_c());
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
  auto tensor = std::make_shared<Tensor>(BuildInputTensor());

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
  for (int i = 0; i < tensor->ElementsNum(); i++) {
    std::cout << tensor_data[i] << std::endl;
  }
}

TEST_F(TestTensor, InitByFloatArrayDataTest) {
  // Init tensor data by py::array_t<float>
  TensorPtr tensor = std::make_shared<Tensor>(BuildInputTensor());

  // Print some information of the tensor
  std::cout << "Datatype: " << tensor->data_type() << std::endl;
  ASSERT_EQ(TypeId::kNumberTypeFloat32, tensor->data_type());

  std::cout << "Dim: " << tensor->DataDim() << std::endl;
  ASSERT_EQ(2, tensor->DataDim());

  std::vector<int> dimensions = tensor->shape();
  ASSERT_GT(dimensions.size(), 1);
  std::cout << "Dim0: " << dimensions[0] << std::endl;
  ASSERT_EQ(2, dimensions[0]);

  std::cout << "Dim1: " << dimensions[1] << std::endl;
  ASSERT_EQ(3, dimensions[1]);

  std::cout << "Num of Elements: " << tensor->ElementsNum() << std::endl;
  ASSERT_EQ(6, tensor->ElementsNum());

  // Print each elements
  std::cout << "Elements: " << std::endl;
  py::array_t<float> data = (py::array_t<float>)tensor->data();
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
  Tensor tensor = Tensor(TypeId::kNumberTypeFloat32, std::vector<int>({2, 3}));

  // Get the writable data pointer from the tensor
  float *me_tensor_data = reinterpret_cast<float *>(tensor.data_c(true));

  // Copy data from buffer to tensor's data
  errno_t ret = memcpy_s(me_tensor_data, tensor.data().nbytes(), ge_tensor_data, sizeof(ge_tensor_data));
  ASSERT_EQ(0, ret);

  // Testify if the data has been copied to the tensor data
  py::array_t<float> data = (py::array_t<float>)tensor.data();
  auto array = data.mutable_unchecked();
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      std::cout << "array[" << i << ", " << j << "]"
                << " = " << array(i, j) << std::endl;
      ASSERT_EQ(array(i, j), ge_tensor_data[3 * i + j]);
    }
  }
}

}  // namespace tensor
}  // namespace mindspore
