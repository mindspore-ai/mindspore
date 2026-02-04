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

#include <vector>
#include "gtest/gtest.h"
#include "mindapi/base/type_id.h"
#include "include/pynative/utils/pyboost/custom/tensor.h"
#include "ir/tensor.h"
#include "ir/tensor_new.h"
#include "include/utils/tensor_utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "mindspore/ccsrc/frontend/jit/ps/parse/data_converter.h"
#include "mindspore/ccsrc/pybind_api/pynative/tensor/tensor_py.h"
#include "pynative/common.h"
namespace mindspore {
class TensorTest : public PyCommon {
 protected:
  void SetUp() override {
    UT::InitPythonPath();
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    context->set_param<std::string>(MS_CTX_DEVICE_TARGET, kCPUDevice);

    std::vector<float> vec = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<ShapeValueDType> shape = {2, 3};
    mindspore::ValuePtr value =
      std::make_shared<mindspore::tensor::Tensor>(kNumberTypeFloat32, shape, true, vec.data());

    normal_tensor_ = ms::Tensor(value);
    empty_tensor_ = ms::Tensor(kNumberTypeInt64, {});
  }

  ms::Tensor normal_tensor_;
  ms::Tensor empty_tensor_;
};

/// Feature: Test ms_tensor.
/// Description: Test BasicConstruction.
/// Expectation: Success.
TEST_F(TensorTest, BasicConstruction) {
  EXPECT_TRUE(normal_tensor_.is_defined());
  EXPECT_EQ(normal_tensor_.data_type(), kNumberTypeFloat32);
  EXPECT_EQ(normal_tensor_.shape(), (ShapeVector{2, 3}));

  EXPECT_TRUE(empty_tensor_.is_defined());
  EXPECT_EQ(empty_tensor_.numel(), 1);
}

/// Feature: Test ms_tensor.
/// Description: Test GetDataPtr.
/// Expectation: GetDataPtr success.
TEST_F(TensorTest, DataPointerHandling) {
  auto *ptr = normal_tensor_.GetDataPtr();
  EXPECT_NE(ptr, nullptr);
}

/// Feature: Test ms_tensor.
/// Description: Test properties of ms_tensor.
/// Expectation: Properties are correct.
TEST_F(TensorTest, PropertyAccess) {
  EXPECT_EQ(normal_tensor_.format(), "DefaultFormat");
  EXPECT_EQ(normal_tensor_.stride(), (std::vector<int64_t>{3, 1}));
  EXPECT_EQ(normal_tensor_.storage_offset(), 0);
  EXPECT_TRUE(normal_tensor_.is_contiguous());
}

/// Feature: Test ms_tensor.
/// Description: Test ShapeOperations of ms_tensor.
/// Expectation: ShapeOperations success.
TEST_F(TensorTest, ShapeOperations) {
  auto flat_tensor = normal_tensor_.flatten(0, 1);
  EXPECT_EQ(flat_tensor.shape(), (ShapeVector{6}));

  auto reshaped = normal_tensor_.reshape({3, 2});
  EXPECT_EQ(reshaped.shape(), (ShapeVector{3, 2}));
}

/// Feature: Test ms_tensor.
/// Description: Test chunk of ms_tensor.
/// Expectation: Chunk success.
TEST_F(TensorTest, ChunkOperation) {
  auto chunks = normal_tensor_.chunk(2, 0);
  ASSERT_EQ(chunks.size(), 2);
  EXPECT_EQ(chunks[0].shape(), (ShapeVector{1, 3}));
  EXPECT_EQ(chunks[1].shape(), (ShapeVector{1, 3}));
}

/// Feature: Test ms_tensor.
/// Description: Test exception handling of undefined tensor operations.
/// Expectation: Operation on undefined tensor throw appropriate exceptions.
TEST_F(TensorTest, ExceptionHandling) {
  ms::Tensor undefined_tensor(nullptr);
  EXPECT_THROW(undefined_tensor.GetDataPtr(), std::exception);
  EXPECT_THROW(undefined_tensor.data_type(), std::exception);
}

/// Feature: Test ms_tensor.
/// Description: Test assigning one tensor to another.
/// Expectation: Assignment success.
TEST_F(TensorTest, AssignmentOperation) {
  ms::Tensor src(kNumberTypeFloat16, {2, 2});
  ms::Tensor dest(kNumberTypeFloat16, {2, 2});
  ASSERT_NO_THROW(dest.AssignTensor(src));
}

/// Feature: Test ms_tensor.
/// Description: Test tensor edge cases.
/// Expectation: Edge cases tensor behave correctly and return expected results.
TEST_F(TensorTest, EdgeCases) {
  EXPECT_EQ(empty_tensor_.numel(), 1);
  EXPECT_EQ(empty_tensor_.shape(), ShapeVector{});
}

/// Feature: Test ms_tensor.
/// Description: Test integration between ms_tensor and pybind11 for type conversion and handling.
/// Expectation: Type conversions between ms_tensor and pybind11 objects work correctly.
TEST_F(TensorTest, PybindIntegration) {
  pybind11::handle py_handle = pybind11::cast(normal_tensor_);
  EXPECT_FALSE(py_handle.is_none());

  pybind11::none py_none;
  auto null_tensor = py_none.cast<ms::Tensor>();
  EXPECT_FALSE(null_tensor.is_defined());
}
}  // namespace mindspore
