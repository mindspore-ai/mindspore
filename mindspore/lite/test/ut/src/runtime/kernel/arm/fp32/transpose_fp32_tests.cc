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

#include <iostream>
#include <memory>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32/transpose_fp32.h"
#include "mindspore/lite/nnacl/transpose.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {

class TestTransposeFp32 : public mindspore::CommonTest {
 public:
  TestTransposeFp32() {}
};

TEST_F(TestTransposeFp32, TransposeFp32_axes4) {
  /* 1x2x3x4 */
  float in[24] = {-0.35779851, -0.4857257,  1.2791597,   -0.36793608, 0.95098744, -0.12716428, 0.17405411,  0.42663834,
                  -1.11871315, 1.02777593,  1.20223761,  0.30183748,  1.39663453, -1.11923312, -1.02032341, 1.91074871,
                  1.52489095,  -1.13020852, -0.66358529, 1.8033383,   0.62647028, 1.03094635,  -1.65733338, 0.3952082};

  float out[24] = {0};

  float correct[24] = {-0.35779851, 1.39663453,  0.95098744,  1.52489095,  -1.11871315, 0.62647028,
                       -0.4857257,  -1.11923312, -0.12716428, -1.13020852, 1.02777593,  1.03094635,
                       1.2791597,   -1.02032341, 0.17405411,  -0.66358529, 1.20223761,  -1.65733338,
                       -0.36793608, 1.91074871,  0.42663834,  1.8033383,   0.30183748,  0.3952082};

  int output_shape[4] = {4, 3, 2, 1};
  int perm[8] = {3, 2, 1, 0, 0, 0, 0, 0};
  int strides[8] = {24, 12, 4, 1, 1, 1, 1, 1};
  int out_strides[8] = {6, 2, 1, 1, 1, 1, 1, 1};

  auto param = new (std::nothrow) TransposeParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New param fails.";
    return;
  }
  param->num_axes_ = 4;
  param->conjugate_ = false;
  param->data_size_ = 24 * sizeof(float);

  for (int i = 0; i < 8; i++) {
    param->perm_[i] = perm[i];
    param->strides_[i] = strides[i];
    param->out_strides_[i] = out_strides[i];
  }

  auto ret = DoTransposeFp32(in, out, output_shape, param);
  ASSERT_EQ(ret, 0);
  delete param;
  ASSERT_EQ(0, CompareOutputData(out, correct, 24, 0.000001));
}

TEST_F(TestTransposeFp32, TransposeFp32_axes3) {
  /* 2x3x4 */
  float in[24] = {1.62434536,  -0.61175641, -0.52817175, -1.07296862, 0.86540763,  -2.3015387,  1.74481176, -0.7612069,
                  0.3190391,   -0.24937038, 1.46210794,  -2.06014071, -0.3224172,  -0.38405435, 1.13376944, -1.09989127,
                  -0.17242821, -0.87785842, 0.04221375,  0.58281521,  -1.10061918, 1.14472371,  0.90159072, 0.50249434};

  float out[24] = {0};

  float correct[24] = {1.62434536,  -0.3224172,  0.86540763, -0.17242821, 0.3190391,   -1.10061918,
                       -0.61175641, -0.38405435, -2.3015387, -0.87785842, -0.24937038, 1.14472371,
                       -0.52817175, 1.13376944,  1.74481176, 0.04221375,  1.46210794,  0.90159072,
                       -1.07296862, -1.09989127, -0.7612069, 0.58281521,  -2.06014071, 0.50249434};

  int output_shape[3] = {4, 3, 2};
  int perm[8] = {2, 1, 0, 0, 0, 0, 0, 0};
  int strides[8] = {12, 4, 1, 1, 1, 1, 1, 1};
  int out_strides[8] = {6, 2, 1, 1, 1, 1, 1, 1};

  auto param = new (std::nothrow) TransposeParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New param fails.";
    return;
  }
  param->num_axes_ = 3;
  param->conjugate_ = false;
  param->data_size_ = 24 * sizeof(float);

  for (int i = 0; i < 8; i++) {
    param->perm_[i] = perm[i];
    param->strides_[i] = strides[i];
    param->out_strides_[i] = out_strides[i];
  }

  auto ret = DoTransposeFp32(in, out, output_shape, param);
  ASSERT_EQ(ret, 0);
  delete param;
  ASSERT_EQ(0, CompareOutputData(out, correct, 24, 0.000001));
}

TEST_F(TestTransposeFp32, TransposeFp32_axes2) {
  /* 6x4 */
  float in[24] = {1.62434536,  -0.61175641, -0.52817175, -1.07296862, 0.86540763,  -2.3015387,  1.74481176, -0.7612069,
                  0.3190391,   -0.24937038, 1.46210794,  -2.06014071, -0.3224172,  -0.38405435, 1.13376944, -1.09989127,
                  -0.17242821, -0.87785842, 0.04221375,  0.58281521,  -1.10061918, 1.14472371,  0.90159072, 0.50249434};

  float out[24] = {0};

  float correct[24] = {1.62434536,  0.86540763, 0.3190391,   -0.3224172,  -0.17242821, -1.10061918,
                       -0.61175641, -2.3015387, -0.24937038, -0.38405435, -0.87785842, 1.14472371,
                       -0.52817175, 1.74481176, 1.46210794,  1.13376944,  0.04221375,  0.90159072,
                       -1.07296862, -0.7612069, -2.06014071, -1.09989127, 0.58281521,  0.50249434};

  int output_shape[2] = {4, 6};
  int perm[8] = {1, 0, 0, 0, 0, 0, 0, 0};
  int strides[8] = {4, 1, 1, 1, 1, 1, 1, 1};
  int out_strides[8] = {6, 1, 1, 1, 1, 1, 1, 1};

  auto param = new (std::nothrow) TransposeParameter();
  if (param == nullptr) {
    MS_LOG(ERROR) << "New param fails.";
    return;
  }

  param->num_axes_ = 2;
  param->conjugate_ = false;
  param->data_size_ = 24 * sizeof(float);

  for (int i = 0; i < 8; i++) {
    param->perm_[i] = perm[i];
    param->strides_[i] = strides[i];
    param->out_strides_[i] = out_strides[i];
  }

  auto ret = DoTransposeFp32(in, out, output_shape, param);
  ASSERT_EQ(ret, 0);
  delete param;
  ASSERT_EQ(0, CompareOutputData(out, correct, 24, 0.000001));
}

TEST_F(TestTransposeFp32, TransposeFp32_test5) {
  /* 1x2x3x2x2 */
  std::vector<float> input = {1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763,  -2.3015387,
                              1.74481176, -0.7612069,  0.3190391,   -0.24937038, 1.46210794,  -2.06014071,
                              -0.3224172, -0.38405435, 1.13376944,  -1.09989127, -0.17242821, -0.87785842,
                              0.04221375, 0.58281521,  -1.10061918, 1.14472371,  0.90159072,  0.50249434};

  float correct[24] = {1.62434536,  -0.3224172,  0.86540763, -0.17242821, 0.3190391,   -1.10061918,
                       -0.52817175, 1.13376944,  1.74481176, 0.04221375,  1.46210794,  0.90159072,
                       -0.61175641, -0.38405435, -2.3015387, -0.87785842, -0.24937038, 1.14472371,
                       -1.07296862, -1.09989127, -0.7612069, 0.58281521,  -2.06014071, 0.50249434};

  std::vector<float> output(24);
  std::vector<int> input_shape = {1, 2, 3, 2, 2};
  std::vector<int> output_shape = {2, 2, 3, 2, 1};
  int perm[8] = {4, 3, 2, 1, 0, 0, 0, 0};
  int strides[8] = {24, 12, 4, 2, 1, 1, 1, 1};
  int out_strides[8] = {12, 6, 2, 1, 1, 1, 1, 1};

  TransposeParameter param;
  param.op_parameter_.type_ = schema::PrimitiveType_Transpose;
  param.num_axes_ = 5;
  param.conjugate_ = false;
  param.data_size_ = 24 * sizeof(float);

  for (int i = 0; i < 8; i++) {
    param.perm_[i] = perm[i];
    param.strides_[i] = strides[i];
    param.out_strides_[i] = out_strides[i];
  }

  lite::Tensor input_tensor;
  input_tensor.set_data(input.data());
  input_tensor.set_shape(input_shape);
  input_tensor.set_format(schema::Format_NHWC);
  input_tensor.set_data_type(kNumberTypeFloat32);
  std::vector<lite::Tensor *> inputs_tensor;
  inputs_tensor.emplace_back(&input_tensor);

  lite::Tensor output_tensor;
  output_tensor.set_data(output.data());
  output_tensor.set_shape(output_shape);
  output_tensor.set_format(schema::Format_NHWC);
  output_tensor.set_data_type(kNumberTypeFloat32);
  std::vector<lite::Tensor *> outputs_tensor;
  outputs_tensor.emplace_back(&output_tensor);

  lite::InnerContext ctx;
  ctx.thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Transpose};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();

  for (int i = 0; i < 24; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  ASSERT_EQ(0, CompareOutputData(output.data(), correct, 24, 0.000001));
  input_tensor.set_data(nullptr);
  output_tensor.set_data(nullptr);
}

}  // namespace mindspore
