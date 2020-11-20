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
#include "nnacl/stack_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_Stack : public CommonTest {};

namespace {
// PrimitiveType_Stack: src/ops/populate/stack_populate.cc
OpParameter *CreateParameter(int axis) {
  auto *param = test::CreateParameter<StackParameter>(schema::PrimitiveType_Stack);
  param->axis_ = axis;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Stack, input8_ndim3_axis0) {
  constexpr int INPUT_NUM = 8;
  int axis = 0;
  std::vector<int> input_shapes[INPUT_NUM] = {{1, 1, 8}, {1, 1, 8}, {1, 1, 8}, {1, 1, 8},
                                              {1, 1, 8}, {1, 1, 8}, {1, 1, 8}, {1, 1, 8}};
  std::vector<int> output_shape = {8, 1, 1, 8};
  float input_datas[INPUT_NUM][8] = {
    {0.75, 0.06, 0.74, 0.30, 0.9, 0.59, 0.03, 0.37},  {0.5, 0.6, 0.74, 0.23, 0.46, 0.69, 0.13, 0.47},
    {0.31, 0.63, 0.84, 0.43, 0.56, 0.79, 0.12, 0.57}, {0.35, 0.26, 0.17, 0.33, 0.66, 0.89, 0.93, 0.77},
    {0.57, 0.6, 0.84, 0.83, 0.48, 0.78, 0.63, 0.87},  {0.66, 0.56, 0.64, 0.63, 0.56, 0.59, 0.73, 0.37},
    {0.35, 0.26, 0.54, 0.33, 0.76, 0.59, 0.73, 0.34}, {0.15, 0.36, 0.44, 0.73, 0.56, 0.49, 0.93, 0.37}};
  float output_data[] = {0.75, 0.06, 0.74, 0.30, 0.9,  0.59, 0.03, 0.37, 0.5,  0.6,  0.74, 0.23, 0.46,
                         0.69, 0.13, 0.47, 0.31, 0.63, 0.84, 0.43, 0.56, 0.79, 0.12, 0.57, 0.35, 0.26,
                         0.17, 0.33, 0.66, 0.89, 0.93, 0.77, 0.57, 0.6,  0.84, 0.83, 0.48, 0.78, 0.63,
                         0.87, 0.66, 0.56, 0.64, 0.63, 0.56, 0.59, 0.73, 0.37, 0.35, 0.26, 0.54, 0.33,
                         0.76, 0.59, 0.73, 0.34, 0.15, 0.36, 0.44, 0.73, 0.56, 0.49, 0.93, 0.37};

  for (auto fp16_enable : {false}) {
    auto *param = CreateParameter(axis);
    TestMain({{input_shapes[0], input_datas[0], VAR},
              {input_shapes[1], input_datas[1], VAR},
              {input_shapes[2], input_datas[2], VAR},
              {input_shapes[3], input_datas[3], VAR},
              {input_shapes[4], input_datas[4], VAR},
              {input_shapes[5], input_datas[5], VAR},
              {input_shapes[6], input_datas[6], VAR},
              {input_shapes[7], input_datas[7], VAR}},
             {output_shape, output_data}, param, fp16_enable);
  }
}

}  // namespace mindspore::lite::opencl::test
