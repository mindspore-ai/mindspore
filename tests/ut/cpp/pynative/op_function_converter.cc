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

#include "gtest/gtest.h"
#include "mockcpp/mockcpp.hpp"
#include "common/mockcpp.h"
#include "pynative/common.h"
#include "mindspore/core/ops/auto_generate/gen_ops_def.h"
#include "pipeline/pynative/op_function/converter.h"
#include "pipeline/pynative/pynative_utils.h"
#include "ir/tensor.h"

namespace mindspore {
namespace pynative {
class PyBoostConverterTest : public PyCommon {};

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensor for pyboost input converter.
/// Expectation: Python Tensor to Tensor success.
TEST_F(PyBoostConverterTest, ToTensorTest1) {
  Converter converter(&ops::gSin);

  auto tensor_py = NewPyTensor(std::make_shared<tensor::Tensor>(1));

  py::list list;
  list.append(tensor_py);
  converter.Parse(list);

  auto t = converter.ToTensor(list, kIndex0);
  ASSERT_EQ(t, tensor_py.cast<TensorPtr>());
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensor for pyboost input converter.
/// Expectation: Python Tensor to Tensor success.
TEST_F(PyBoostConverterTest, ToTensorTest2) {
  Converter converter(&ops::gSin);

  auto tensor = std::make_shared<tensor::BaseTensor>(1);
  auto stub_tensor = NewPyStubTensor(tensor);

  py::list list;
  list.append(stub_tensor);
  converter.Parse(list);

  auto t = converter.ToTensor(list, kIndex0);
  ASSERT_NE(t, nullptr);
  ASSERT_EQ(t->isa<stub::StubNode>(), true);

  auto stub_node = t->cast<stub::StubNodePtr>();
  ASSERT_EQ(stub_node->WaitValue(), tensor);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensor for pyboost input converter.
/// Expectation: Python float to Tensor success.
TEST_F(PyBoostConverterTest, ToTensorTest3) {
  Converter converter(&ops::gAdd);

  auto x_obj = NewPyTensor(std::make_shared<tensor::Tensor>(1));
  auto y_obj = py::float_(1.0);

  py::list list;
  list.append(x_obj);
  list.append(y_obj);
  converter.Parse(list);

  auto x_out = converter.ToTensor(list, kIndex0);
  auto y_out = converter.ToTensor(list, kIndex1);
  ASSERT_NE(x_out, nullptr);
  ASSERT_NE(y_out, nullptr);
  ASSERT_EQ(y_out->isa<tensor::Tensor>(), true);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensorOptional for pyboost input input converter.
/// Expectation: ToTensorOptional return none when input is py::none().
TEST_F(PyBoostConverterTest, ToTensorOptionalTest) {
  Converter converter(&ops::gClampTensor);

  auto input = NewPyTensor(std::make_shared<tensor::Tensor>(1));
  auto min = NewPyTensor(std::make_shared<tensor::Tensor>(1));
  auto max = py::none();

  py::list list;
  list.append(input);
  list.append(min);
  list.append(max);
  converter.Parse(list);

  auto min_out = converter.ToTensorOptional(list, kIndex1);
  ASSERT_EQ(min_out.has_value(), true);
  ASSERT_NE(min_out.value(), nullptr);
  ASSERT_EQ(min_out.value()->isa<tensor::Tensor>(), true);

  auto max_out = converter.ToTensorOptional(list, kIndex2);
  ASSERT_EQ(max_out.has_value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToTensorOptional for pyboost input converter.
/// Expectation: To int success.
TEST_F(PyBoostConverterTest, ToIntOptionalTest1) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gArgMaxExt);

  auto input = NewPyTensor(std::make_shared<tensor::Tensor>(1));
  auto dim = py::none();
  auto keep_dim = py::bool_(true);

  py::list list;
  list.append(input);
  list.append(dim);
  list.append(keep_dim);
  converter.Parse(list);

  auto input_out = converter.ToTensor(list, kIndex0);
  ASSERT_NE(input_out, nullptr);
  ASSERT_EQ(input_out->isa<tensor::Tensor>(), true);

  auto dim_out = converter.ToIntOptional(list, kIndex1);
  ASSERT_EQ(dim_out.has_value(), false);

  auto keep_dim_out = converter.ToBool(list, kIndex2);
  ASSERT_EQ(keep_dim_out->value(), true);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToIntOptional for pyboost input converter.
/// Expectation: To in success.
TEST_F(PyBoostConverterTest, ToIntOptionalTest2) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gArgMaxExt);

  auto input = NewPyTensor(std::make_shared<tensor::Tensor>(1));
  auto dim = py::int_(1);
  auto keep_dim = py::bool_(false);

  py::list list;
  list.append(input);
  list.append(dim);
  list.append(keep_dim);
  converter.Parse(list);

  auto dim_out = converter.ToIntOptional(list, kIndex1);
  ASSERT_EQ(dim_out.has_value(), true);
  ASSERT_EQ(dim_out.value()->value(), 1);

  auto keep_dim_out = converter.ToBool(list, kIndex2);
  ASSERT_EQ(keep_dim_out->value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBoolOptional for pyboost input converter.
/// Expectation: To bool and get none.
TEST_F(PyBoostConverterTest, ToBoolOptionalTest1) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::none());
  converter.Parse(list);

  auto t = converter.ToBoolOptional(list, kIndex0);
  ASSERT_EQ(t.has_value(), false);
}

/// Feature: Test Pyboost Converter.
/// Description: Test ToBoolOptional for pyboost input converter.
/// Expectation: To bool success.
TEST_F(PyBoostConverterTest, ToBoolOptionalTest2) {
  py::gil_scoped_acquire gil;
  Converter converter(&ops::gSin);

  py::list list;
  list.append(py::bool_(true));
  converter.Parse(list);

  auto t = converter.ToBoolOptional(list, kIndex0);
  ASSERT_EQ(t.has_value(), true);
  ASSERT_EQ(t.value()->value(), true);
}
}  // namespace pynative
}  // namespace mindspore
