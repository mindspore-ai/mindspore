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
#include "ut/src/runtime/kernel/opencl/common.h"
#include "nnacl/arithmetic_self_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_ArithmeticSelf : public CommonTest {};

namespace {
// PrimitiveType_Abs
// PrimitiveType_Cos
// PrimitiveType_Sin
// PrimitiveType_Log
// PrimitiveType_Neg
// PrimitiveType_NegGrad
// PrimitiveType_LogGrad
// PrimitiveType_Sqrt
// PrimitiveType_Square
// PrimitiveType_Rsqrt
// PrimitiveType_LogicalNot
// PrimitiveType_Floor
// PrimitiveType_Ceil
// PrimitiveType_Round: src/ops/populate/arithmetic_self_populate.cc
OpParameter *CreateParameter(schema::PrimitiveType type) {
  auto *param = test::CreateParameter<ArithmeticSelfParameter>(type);
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_ArithmeticSelf, Round) {
  std::vector<int> shape = {1, 1, 4, 4};
  float input_data[] = {0.75, 0.06, 0.74, 0.30, 0.9, 0.59, 0.03, 0.37, 0.75, 0.06, 0.74, 0.30, 0.9, 0.59, 0.03, 0.37};
  float output_data[] = {1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(schema::PrimitiveType_Round);
    TestMain({{shape, input_data, VAR}}, {shape, output_data}, param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
