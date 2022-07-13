/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <memory>
#include <string>
#include "common/common.h"
#include "common/cvop_common.h"
#include "minddata/dataset/kernels/image/to_tensor_op.h"
#include "minddata/dataset/kernels/data/type_cast_op.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/core/tensor.h"
#include "gtest/gtest.h"
#include "securec.h"

namespace common = mindspore::common;
using namespace mindspore::dataset;

class MindDataTestToTensorOp : public UT::CVOP::CVOpCommon {
 public:
  MindDataTestToTensorOp() : CVOpCommon() {}
};

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to int8
/// Expectation: Run successfully
TEST_F(MindDataTestToTensorOp, TestToTensorOpInt8) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInt8.";
  auto to_tensor_op = std::make_unique<ToTensorOp>("int8");
  std::shared_ptr<Tensor> output_tensor;
  Status s = to_tensor_op->Compute(input_tensor_, &output_tensor);
  ASSERT_TRUE(DataType("int8") == output_tensor->type());
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to int16
/// Expectation: Run successfully
TEST_F(MindDataTestToTensorOp, TestToTensorOpInt16) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInt16.";
  auto to_tensor_op = std::make_unique<ToTensorOp>("int16");
  std::shared_ptr<Tensor> output_tensor;
  Status s = to_tensor_op->Compute(input_tensor_, &output_tensor);
  ASSERT_TRUE(DataType("int16") == output_tensor->type());
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to int32
/// Expectation: Run successfully
TEST_F(MindDataTestToTensorOp, TestToTensorOpInt32) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInt32.";
  auto to_tensor_op = std::make_unique<ToTensorOp>("int32");
  std::shared_ptr<Tensor> output_tensor;
  Status s = to_tensor_op->Compute(input_tensor_, &output_tensor);
  ASSERT_TRUE(DataType("int32") == output_tensor->type());
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to int64
/// Expectation: Run successfully
TEST_F(MindDataTestToTensorOp, TestToTensorOpInt64) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInt64.";
  auto to_tensor_op = std::make_unique<ToTensorOp>("int64");
  std::shared_ptr<Tensor> output_tensor;
  Status s = to_tensor_op->Compute(input_tensor_, &output_tensor);
  ASSERT_TRUE(DataType("int64") == output_tensor->type());
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to float16
/// Expectation: Run successfully
TEST_F(MindDataTestToTensorOp, TestToTensorOpFloat16) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpFloat16.";
  auto to_tensor_op = std::make_unique<ToTensorOp>("float16");
  std::shared_ptr<Tensor> output_tensor;
  Status s = to_tensor_op->Compute(input_tensor_, &output_tensor);
  ASSERT_TRUE(DataType("float16") == output_tensor->type());
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to float32
/// Expectation: Run successfully
TEST_F(MindDataTestToTensorOp, TestToTensorOpFloat32) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpFloat32.";
  auto to_tensor_op = std::make_unique<ToTensorOp>("float32");
  std::shared_ptr<Tensor> output_tensor;
  Status s = to_tensor_op->Compute(input_tensor_, &output_tensor);
  ASSERT_TRUE(DataType("float32") == output_tensor->type());
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to float64
/// Expectation: Run successfully
TEST_F(MindDataTestToTensorOp, TestToTensorOpFloat64) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpFloat64.";
  auto to_tensor_op = std::make_unique<ToTensorOp>("float64");
  std::shared_ptr<Tensor> output_tensor;
  Status s = to_tensor_op->Compute(input_tensor_, &output_tensor);
  ASSERT_TRUE(DataType("float64") == output_tensor->type());
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to float32 with uint32 type input
/// Expectation: Catch error for invalid type
TEST_F(MindDataTestToTensorOp, TestToTensorOpInputUInt32Invalid) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInputUInt32Invalid.";

  // Cast uint8 input to invalid uint32 type for ToTensor Op
  auto type_cast_op = std::make_unique<TypeCastOp>("uint32");
  std::shared_ptr<Tensor> interim_tensor;
  Status s_cast = type_cast_op->Compute(input_tensor_, &interim_tensor);
  ASSERT_OK(s_cast);

  auto to_tensor_op = std::make_unique<ToTensorOp>("float32");
  std::shared_ptr<Tensor> output_tensor;
  Status s_to_tensor = to_tensor_op->Compute(interim_tensor, &output_tensor);
  ASSERT_ERROR(s_to_tensor);
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to float32 with uint64 type input
/// Expectation: Catch error for invalid type
TEST_F(MindDataTestToTensorOp, TestToTensorOpInputUInt64Invalid) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInputUInt64Invalid.";

  // Cast uint8 input to invalid uint64 type for ToTensor Op
  auto type_cast_op = std::make_unique<TypeCastOp>("uint64");
  std::shared_ptr<Tensor> interim_tensor;
  Status s_cast = type_cast_op->Compute(input_tensor_, &interim_tensor);
  ASSERT_OK(s_cast);

  auto to_tensor_op = std::make_unique<ToTensorOp>("float32");
  std::shared_ptr<Tensor> output_tensor;
  Status s_to_tensor = to_tensor_op->Compute(interim_tensor, &output_tensor);
  ASSERT_ERROR(s_to_tensor);
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to float32 with int64 type input
/// Expectation: Catch error for invalid type
TEST_F(MindDataTestToTensorOp, TestToTensorOpInputInt64Invalid) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInputInt64Invalid.";

  // Cast uint8 input to invalid int64 type for ToTensor Op
  auto type_cast_op = std::make_unique<TypeCastOp>("int64");
  std::shared_ptr<Tensor> interim_tensor;
  Status s_cast = type_cast_op->Compute(input_tensor_, &interim_tensor);
  ASSERT_OK(s_cast);

  auto to_tensor_op = std::make_unique<ToTensorOp>("float32");
  std::shared_ptr<Tensor> output_tensor;
  Status s_to_tensor = to_tensor_op->Compute(interim_tensor, &output_tensor);
  ASSERT_ERROR(s_to_tensor);
}

/// Feature: ToTensor Op
/// Description: Check type changing with ToTensor C op to float32 with string type input
/// Expectation: Catch error for invalid type
TEST_F(MindDataTestToTensorOp, TestToTensorOpInputStringInvalid) {
  MS_LOG(INFO) << "Doing MindDataTestToTensorOp::TestToTensorOpInputStringInvalid.";

  // Create string tensor and shape as a 3D input
  std::vector<std::string> strings{"1", "2", "3", "4", "5", "6"};
  std::shared_ptr<Tensor> string_tensor;
  Tensor::CreateFromVector(strings, TensorShape({1, 2, 3}), &string_tensor);

  auto to_tensor_op = std::make_unique<ToTensorOp>("float32");
  std::shared_ptr<Tensor> output_tensor;
  Status s_to_tensor = to_tensor_op->Compute(string_tensor, &output_tensor);
  ASSERT_ERROR(s_to_tensor);
}
