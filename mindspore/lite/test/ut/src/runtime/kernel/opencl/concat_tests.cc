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
#include "nnacl/concat_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Concat : public CommonTest {};

namespace {
// PrimitiveType_Concat: src/ops/populate/concat_populate.cc
OpParameter *CreateParameter(int axis) {
  auto *param = test::CreateParameter<ConcatParameter>(schema::PrimitiveType_Concat);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Concat, input2_axis0) {
  std::vector<int> input0_shape = {1, 1, 1, 8};
  std::vector<int> input1_shape = {1, 1, 1, 8};
  std::vector<int> output_shape = {2, 1, 1, 8};
  int axis = 0;
  float input0_data[] = {0.75, 0.06, 0.74, 0.30, 0.9, 0.59, 0.03, 0.37};
  float input1_data[] = {0.5, 0.6, 0.74, 0.23, 0.46, 0.69, 0.13, 0.47};
  float output_data[] = {0.75, 0.06, 0.74, 0.30, 0.9, 0.59, 0.03, 0.37, 0.5, 0.6, 0.74, 0.23, 0.46, 0.69, 0.13, 0.47};
  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(axis);
    TestMain({{input0_shape, input0_data, VAR}, {input1_shape, input1_data, VAR}}, {output_shape, output_data}, param,
             fp16_enable, fp16_enable ? 1e-3 : 1e-9);
  }
}

}  // namespace mindspore::lite::opencl::test
