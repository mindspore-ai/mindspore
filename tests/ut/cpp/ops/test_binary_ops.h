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
#ifndef TESTS_UT_CPP_OPS_TEST_BINARY_OPS_H_
#define TESTS_UT_CPP_OPS_TEST_BINARY_OPS_H_

#include <vector>
#include <string>
#include <memory>
#include "common/common_test.h"
#include "ops/test_ops.h"
#include "ops/test_value_utils.h"
#include "ops/utils/general_infer_utils.h"
#include "ir/tensor_new.h"
#include "abstract/abstract_value.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore::ops {

// Unified params for InferValue testing of binary operators
struct BinaryOpInferValueParams {
  std::string op_name;
  ValuePtr input1;
  ValuePtr input2;
  tensor::TensorPtr out;
};

class TestBinaryOpInferValue : public TestOps, public testing::WithParamInterface<BinaryOpInferValueParams> {};

#define REGISTER_BINARY_OP_INFER_VALUE_TEST(FIXTURE_NAME, TEST_NAME)                       \
  class FIXTURE_NAME : public TestBinaryOpInferValue {};                                   \
  TEST_P(FIXTURE_NAME, TEST_NAME) {                                                        \
    const auto &param = GetParam();                                                        \
    auto primitive = std::make_shared<Primitive>(param.op_name);                           \
    ASSERT_NE(primitive, nullptr);                                                         \
                                                                                           \
    auto input1 = param.input1->ToAbstract();                                              \
    auto input2 = param.input2->ToAbstract();                                              \
    auto input_args = abstract::AbstractBasePtrList{input1, input2};                       \
                                                                                           \
    auto value_opt = abstract::InferValueByFuncImpl(primitive, input_args);                \
    if (!value_opt.has_value()) {                                                          \
      MS_LOG(ERROR) << param.op_name << " has no infer value implementation!";             \
      ASSERT_TRUE(false);                                                                  \
    }                                                                                      \
    auto infer_out = value_opt.value();                                                    \
    if (infer_out == nullptr) {                                                            \
      MS_LOG(ERROR) << param.op_name << " can not infer value with inputs: " << input_args;\
      ASSERT_TRUE(false);                                                                  \
    }                                                                                      \
    auto infer_tensor = infer_out->cast<tensor::TensorPtr>();                              \
    ASSERT_NE(infer_tensor, nullptr);                                                      \
    ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));                                     \
  }

// Unified params for InferValue testing of ternary operators
struct TernaryOpInferValueParams {
  std::string op_name;
  ValuePtr input1;
  ValuePtr input2;
  ValuePtr input3;
  tensor::TensorPtr out;
};

class TestTernaryOpInferValue : public TestOps, public testing::WithParamInterface<TernaryOpInferValueParams> {};

#define REGISTER_TERNARY_OP_INFER_VALUE_TEST(FIXTURE_NAME, TEST_NAME)                      \
  class FIXTURE_NAME : public TestTernaryOpInferValue {};                                  \
  TEST_P(FIXTURE_NAME, TEST_NAME) {                                                        \
    const auto &param = GetParam();                                                        \
    auto primitive = std::make_shared<Primitive>(param.op_name);                           \
    ASSERT_NE(primitive, nullptr);                                                         \
                                                                                           \
    auto input1 = param.input1->ToAbstract();                                              \
    auto input2 = param.input2->ToAbstract();                                              \
    auto input3 = param.input3->ToAbstract();                                              \
    auto input_args = abstract::AbstractBasePtrList{input1, input2, input3};               \
                                                                                           \
    auto value_opt = abstract::InferValueByFuncImpl(primitive, input_args);                \
    if (!value_opt.has_value()) {                                                          \
      MS_LOG(ERROR) << param.op_name << " has no infer value implementation!";             \
      ASSERT_TRUE(false);                                                                  \
    }                                                                                      \
    auto infer_out = value_opt.value();                                                    \
    if (infer_out == nullptr) {                                                            \
      MS_LOG(ERROR) << param.op_name << " can not infer value with inputs: " << input_args;\
      ASSERT_TRUE(false);                                                                  \
    }                                                                                      \
    auto infer_tensor = infer_out->cast<tensor::TensorPtr>();                              \
    ASSERT_NE(infer_tensor, nullptr);                                                      \
    ASSERT_TRUE(infer_tensor->ValueEqual(*param.out));                                     \
  }

// Helper function to create tensor from buffer (similar to CreateTensorPtr in test_ops_muls.cc)
template <typename T>
tensor::TensorPtr CreateBinaryOpTensor(const TypeId &type, const ShapeVector &shape, std::vector<T> value) {
  void *data_ptr = &value[0];
  auto tensor = tensor::from_buffer(type, shape, data_ptr, type);
  return tensor;
}

} // namespace mindspore::ops

#endif // TESTS_UT_CPP_OPS_TEST_BINARY_OPS_H_
