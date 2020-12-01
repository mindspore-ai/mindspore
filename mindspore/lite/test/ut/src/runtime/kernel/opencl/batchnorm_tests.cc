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
#include "nnacl/batchnorm_parameter.h"

namespace mindspore::lite::opencl::test {

class TestOpenCL_BatchNorm : public CommonTest {};

namespace {
// PrimitiveType_BatchNorm: src/ops/populate/batch_norm_populate.cc
OpParameter *CreateParameter(float epsilon) {
  auto *param = test::CreateParameter<BatchNormParameter>(schema::PrimitiveType_BatchNorm);
  param->epsilon_ = epsilon;
  return reinterpret_cast<OpParameter *>(param);
}
}  // namespace

TEST_F(TestOpenCL_BatchNorm, Align) {
  std::vector<int> input_shape = {1, 2, 2, 8};
  std::vector<int> weight_shape = {1, 1, 1, 8};
  std::vector<int> output_shape = {1, 2, 2, 8};
  float input_data[] = {2.471454,   -2.1379554,  -0.0904604, 1.2928944,  -0.19215967, -0.8677279, -0.12759617,
                        1.2242758,  -0.06398406, -0.4041858, 0.20352598, -2.067808,   0.52113044, -1.567617,
                        0.28003863, 0.41367245,  0.77298605, 0.29908583, 1.4015813,   1.330567,   1.760135,
                        0.6320845,  0.6995399,   -1.208123,  -1.9738104, -1.3283046,  1.022744,   0.02741058,
                        0.84505165, -0.89434445, 1.983211,   -0.5485428};
  float scale_data[] = {0.1201471, 0.142174, 0.5683258, 0.86815494, 0.23426804, 0.3634345, 0.0077846, 0.6813278};
  float offset_data[] = {0.58764684, 0.70790595, 0.945536, 0.8817803, 0.78489226, 0.5884778, 0.3441211, 0.5654443};
  float mean_data[] = {0.3016613, -0.89284, 0.63434774, 0.145766, 0.73353934, -0.6744012, 0.7087985, -0.02967937};
  float var_data[] = {2.5604038, 0.84985304, 0.36261332, 1.9083935, 0.4920925, 0.6476224, 0.6269014, 0.8567283};
  float output_data[] = {0.7505676,  0.515882,   0.26147857, 1.6026789,  0.47575232, 0.50116986, 0.33589783,
                         1.4884706,  0.56019205, 0.7832671,  0.53893626, -0.5093127, 0.71395767, 0.18509413,
                         0.33990562, 0.891792,   0.6230367,  0.89172685, 1.6696336,  1.6263539,  1.1277269,
                         1.1784974,  0.34403008, -0.3019984, 0.4167911,  0.6407478,  1.3120956,  0.80740136,
                         0.8221321,  0.4891496,  0.3566509,  0.18351318};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(1e-5);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, scale_data, CONST_TENSOR},
              {weight_shape, offset_data, CONST_TENSOR},
              {weight_shape, mean_data, CONST_TENSOR},
              {weight_shape, var_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-3 : 1e-5);
  }
}

TEST_F(TestOpenCL_BatchNorm, UnAlign) {
  std::vector<int> input_shape = {1, 2, 2, 7};
  std::vector<int> weight_shape = {1, 1, 1, 7};
  std::vector<int> output_shape = {1, 2, 2, 7};
  float input_data[] = {2.471454,    -2.1379554, -0.0904604, 1.2928944,  -0.19215967, -0.8677279,  -0.12759617,
                        -0.06398406, -0.4041858, 0.20352598, -2.067808,  0.52113044,  -1.567617,   0.28003863,
                        0.77298605,  0.29908583, 1.4015813,  1.330567,   1.760135,    0.6320845,   0.6995399,
                        -1.9738104,  -1.3283046, 1.022744,   0.02741058, 0.84505165,  -0.89434445, 1.983211};
  float scale_data[] = {0.1201471, 0.142174, 0.5683258, 0.86815494, 0.23426804, 0.3634345, 0.0077846};
  float offset_data[] = {0.58764684, 0.70790595, 0.945536, 0.8817803, 0.78489226, 0.5884778, 0.3441211};
  float mean_data[] = {0.3016613, -0.89284, 0.63434774, 0.145766, 0.73353934, -0.6744012, 0.7087985};
  float var_data[] = {2.5604038, 0.84985304, 0.36261332, 1.9083935, 0.4920925, 0.6476224, 0.6269014};
  float output_data[] = {0.7505676,  0.515882,   0.26147857, 1.6026789,  0.47575232, 0.50116986, 0.33589783,
                         0.56019205, 0.7832671,  0.53893626, -0.5093127, 0.71395767, 0.18509413, 0.33990562,
                         0.6230367,  0.89172685, 1.6696336,  1.6263539,  1.1277269,  1.1784974,  0.34403008,
                         0.4167911,  0.6407478,  1.3120956,  0.80740136, 0.8221321,  0.4891496,  0.3566509};

  for (auto fp16_enable : {false, true}) {
    auto *param = CreateParameter(1e-5);
    TestMain({{input_shape, input_data, VAR},
              {weight_shape, scale_data, CONST_TENSOR},
              {weight_shape, offset_data, CONST_TENSOR},
              {weight_shape, mean_data, CONST_TENSOR},
              {weight_shape, var_data, CONST_TENSOR}},
             {output_shape, output_data}, param, fp16_enable, fp16_enable ? 1e-3 : 1e-5);
  }
}

}  // namespace mindspore::lite::opencl::test
