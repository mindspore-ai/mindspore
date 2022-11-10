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
#include <vector>
#include "ut/src/runtime/kernel/opencl/common.h"
#include "nnacl/crop_parameter.h"

namespace mindspore::lite::opencl::test {
class TestOpenCL_Crop : public CommonTest {};

namespace {
// PrimitiveType_Reshape: src/ops/populate/crop_populate.cc
OpParameter *CreateParameter(int64_t axis, const std::vector<int> &offset) {
  auto *param = test::CreateParameter<CropParameter>(schema::PrimitiveType_Crop);
  for (size_t i = 0; i < offset.size(); i++) {
    param->offset_[i] = offset[i];
  }
  param->axis_ = axis;
  param->offset_size_ = static_cast<int>(offset.size());
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_Crop, 4D_4D_Basic) {
  std::vector<int> in_shape = {1, 2, 3, 4};
  std::vector<int> out_shape = {1, 1, 1, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  float input_shape_data[4] = {0};
  float output_data[] = {4, 5, 6, 7};
  int64_t axis = 0;
  std::vector<int> param_offset = {0, 0, 1, 0};
  for (auto fp16_enable : {false, true}) {
    TestMain({{in_shape, input_data, VAR, kNumberTypeFloat32}, {out_shape, input_shape_data, VAR, kNumberTypeFloat32}},
             {out_shape, output_data}, CreateParameter(axis, param_offset), fp16_enable, 1e-9, 1e-9, true);
  }
}

TEST_F(TestOpenCL_Crop, 4D_4D_AxisOffset) {
  std::vector<int> in_shape = {1, 2, 3, 4};
  std::vector<int> out_shape = {1, 1, 2, 4};
  float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  float input_shape_data[8] = {0};
  float output_data[] = {16, 17, 18, 19, 20, 21, 22, 23};
  int64_t axis = 1;
  std::vector<int> param_offset = {1, 1, 0};
  for (auto fp16_enable : {false, true}) {
    TestMain({{in_shape, input_data, VAR, kNumberTypeFloat32}, {out_shape, input_shape_data, VAR, kNumberTypeFloat32}},
             {out_shape, output_data}, CreateParameter(axis, param_offset), fp16_enable, 1e-9, 1e-9, true);
  }
}
}  // namespace mindspore::lite::opencl::test
