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

namespace mindspore::lite::opencl::test {
class TestOpenCL_Shape : public CommonTest {};

namespace {
// PrimitiveType_Shape: src/ops/populate/shape_populate.cc
OpParameter *CreateParameter() {
  auto *param = test::CreateParameter<OpParameter>(schema::PrimitiveType_Shape);
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Shape, test0) {
  std::vector<int> input_shape = {2, 4};
  std::vector<int> output_shape = {2};
  float input_data[] = {-0.4045, -0.0924, -0.617, -0.10114, -0.9893, 0.3342, 2.445, -2.182};
  float output_data[] = {2, 4};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter();
    TestMain({{input_shape, input_data, VAR}}, {output_shape, output_data}, param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
