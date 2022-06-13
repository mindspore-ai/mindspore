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
#include "minddata/dataset/core/client.h"
#include "common/common.h"
#include "gtest/gtest.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"

using namespace mindspore::dataset;

namespace py = pybind11;

class MindDataTestTensorRowDE : public UT::Common {
 public:
  MindDataTestTensorRowDE() {}
  void SetUp() { GlobalInit(); }
};

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow using scalar bool value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertToTensorRowBoolTest) {
  Status s;

  TensorRow bool_output;
  bool bool_value = true;
  s = TensorRow::ConvertToTensorRow(bool_value, &bool_output);
  ASSERT_EQ(s, Status::OK());
  TensorRow expected_bool;
  std::shared_ptr<Tensor> expected_tensor;
  Tensor::CreateScalar(bool_value, &expected_tensor);
  expected_bool.push_back(expected_tensor);
  ASSERT_EQ(*(bool_output.at(0)) == *(expected_bool.at(0)), true);
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow using scalar int value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertToTensorRowIntTest) {
  Status s;
  TensorRow int_output;
  int32_t int_value = 12;
  TensorRow expected_int;
  s = TensorRow::ConvertToTensorRow(int_value, &int_output);
  ASSERT_EQ(s, Status::OK());
  std::shared_ptr<Tensor> expected_tensor;
  Tensor::CreateScalar(int_value, &expected_tensor);
  expected_int.push_back(expected_tensor);
  ASSERT_EQ(*(int_output.at(0)) == *(expected_int.at(0)), true);
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow using scalar float value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertToTensorRowFloatTest) {
  Status s;
  TensorRow expected_bool;
  TensorRow float_output;
  float float_value = 12.57;
  TensorRow expected_float;
  s = TensorRow::ConvertToTensorRow(float_value, &float_output);
  ASSERT_EQ(s, Status::OK());
  std::shared_ptr<Tensor> expected_tensor;
  Tensor::CreateScalar(float_value, &expected_tensor);
  expected_float.push_back(expected_tensor);
  ASSERT_EQ(*(float_output.at(0)) == *(expected_float.at(0)), true);
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow using vector of bool
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertToTensorRowBoolVectorTest) {
  Status s;
  TensorRow bool_output;
  std::vector<bool> bool_value = {true, false};
  s = TensorRow::ConvertToTensorRow(bool_value, &bool_output);
  ASSERT_EQ(s, Status::OK());
  TensorRow expected_bool;
  std::shared_ptr<Tensor> expected_tensor;
  Tensor::CreateFromVector<bool>(bool_value, &expected_tensor);
  expected_bool.push_back(expected_tensor);
  ASSERT_EQ(*(bool_output.at(0)) == *(expected_bool.at(0)), true);
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow using vector of int
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertToTensorRowIntVectorTest) {
  Status s;
  TensorRow int_output;
  std::vector<uint64_t> int_value = {12, 16};
  TensorRow expected_int;
  s = TensorRow::ConvertToTensorRow(int_value, &int_output);
  ASSERT_EQ(s, Status::OK());
  std::shared_ptr<Tensor> expected_tensor;
  Tensor::CreateFromVector(int_value, &expected_tensor);
  expected_int.push_back(expected_tensor);
  ASSERT_EQ(*(int_output.at(0)) == *(expected_int.at(0)), true);
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow using vector of float
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertToTensorRowFloatVectorTest) {
  Status s;
  TensorRow float_output;
  std::vector<double> float_value = {12.57, 0.264};
  TensorRow expected_float;
  s = TensorRow::ConvertToTensorRow(float_value, &float_output);
  ASSERT_EQ(s, Status::OK());
  std::shared_ptr<Tensor> expected_tensor;
  Tensor::CreateFromVector(float_value, &expected_tensor);
  expected_float.push_back(expected_tensor);
  ASSERT_EQ(*(float_output.at(0)) == *(expected_float.at(0)), true);
}

/// Feature: TensorRow
/// Description: Test ConvertFromTensorRow using scalar bool value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowBoolTest) {
  Status s;
  bool bool_value = true;
  bool result;
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateScalar(bool_value, &input_tensor);
  input_tensor_row.push_back(input_tensor);
  s = TensorRow::ConvertFromTensorRow(input_tensor_row, &result);
  ASSERT_EQ(s, Status::OK());
  ASSERT_EQ(bool_value, result);
}

/// Feature: TensorRow
/// Description: Test ConvertFromTensorRow using scalar int value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowIntTest) {
  Status s;
  int32_t int_value = 12;
  int32_t result;
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateScalar(int_value, &input_tensor);
  input_tensor_row.push_back(input_tensor);
  s = TensorRow::ConvertFromTensorRow(input_tensor_row, &result);
  ASSERT_EQ(s, Status::OK());
  ASSERT_EQ(int_value, result);
}

/// Feature: TensorRow
/// Description: Test ConvertFromTensorRow using scalar float value
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowFloatTest) {
  Status s;
  float float_value = 12.57;
  float result;
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateScalar(float_value, &input_tensor);
  input_tensor_row.push_back(input_tensor);
  s = TensorRow::ConvertFromTensorRow(input_tensor_row, &result);
  ASSERT_EQ(s, Status::OK());
  ASSERT_EQ(float_value, result);
}

/// Feature: TensorRow
/// Description: Test ConvertFromTensorRow using vector of bools
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowBoolVectorTest) {
  Status s;
  std::vector<bool> bool_value = {true, false};
  std::vector<bool> result;
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateFromVector<bool>(bool_value, &input_tensor);
  input_tensor_row.push_back(input_tensor);
  s = TensorRow::ConvertFromTensorRow(input_tensor_row, &result);
  ASSERT_EQ(s, Status::OK());
  ASSERT_EQ(result, bool_value);
}

/// Feature: TensorRow
/// Description: Test ConvertFromTensorRow using vector of ints
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowIntVectorTest) {
  Status s;
  std::vector<uint64_t> int_value = {12, 16};
  std::vector<uint64_t> result;
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateFromVector(int_value, &input_tensor);
  input_tensor_row.push_back(input_tensor);
  s = TensorRow::ConvertFromTensorRow(input_tensor_row, &result);
  ASSERT_EQ(s, Status::OK());
  ASSERT_EQ(result, int_value);
}

/// Feature: TensorRow
/// Description: Test ConvertFromTensorRow using vector of floats
/// Expectation: Output is equal to the expected output
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowFloatVectorTest) {
  Status s;
  std::vector<double> float_value = {12.57, 0.264};
  std::vector<double> result;
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor;
  Tensor::CreateFromVector(float_value, &input_tensor);
  input_tensor_row.push_back(input_tensor);
  s = TensorRow::ConvertFromTensorRow(input_tensor_row, &result);
  ASSERT_EQ(s, Status::OK());
  ASSERT_EQ(result, float_value);
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow with invalid data for input
/// Expectation: Throw correct error and message
TEST_F(MindDataTestTensorRowDE, ConvertToTensorRowInvalidDataTest) {
  TensorRow output;
  std::string string_input = "Bye";
  ASSERT_FALSE(TensorRow::ConvertToTensorRow(string_input, &output).IsOk());
  std::vector<std::string> string_vector_input = {"Hello"};
  ASSERT_FALSE(TensorRow::ConvertToTensorRow(string_vector_input, &output).IsOk());
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow with mismatched type input
/// Expectation: Throw correct error and message
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowTypeMismatchTest) {
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor1;
  Tensor::CreateScalar(false, &input_tensor1);
  input_tensor_row.push_back(input_tensor1);
  double output;
  ASSERT_FALSE(TensorRow::ConvertFromTensorRow(input_tensor_row, &output).IsOk());
  std::vector<double> output_vector;
  ASSERT_FALSE(TensorRow::ConvertFromTensorRow(input_tensor_row, &output_vector).IsOk());
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow with invalid shape input
/// Expectation: Throw correct error and message
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowInvalidShapeTest) {
  TensorRow input_tensor_row;
  std::shared_ptr<Tensor> input_tensor1;
  Tensor::CreateEmpty(TensorShape({2, 2}), DataType(DataType::DE_FLOAT64), &input_tensor1);
  input_tensor_row.push_back(input_tensor1);
  std::vector<double> output;
  ASSERT_FALSE(TensorRow::ConvertFromTensorRow(input_tensor_row, &output).IsOk());
  std::vector<double> output_vector;
  ASSERT_FALSE(TensorRow::ConvertFromTensorRow(input_tensor_row, &output_vector).IsOk());
}

/// Feature: TensorRow
/// Description: Test ConvertToTensorRow with an empty input
/// Expectation: Throw correct error and message
TEST_F(MindDataTestTensorRowDE, ConvertFromTensorRowEmptyInputTest) {
  TensorRow input_tensor_row;
  double output;
  ASSERT_FALSE(TensorRow::ConvertFromTensorRow(input_tensor_row, &output).IsOk());
}