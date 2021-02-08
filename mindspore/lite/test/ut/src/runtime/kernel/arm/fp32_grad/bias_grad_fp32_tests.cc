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
#include "src/common/file_utils.h"
#include "src/runtime/kernel/arm/fp32_grad/bias_grad.h"
#include "src/kernel_registry.h"

namespace mindspore {

class TestBiasGradFp32 : public mindspore::CommonTest {
 public:
  TestBiasGradFp32() {}
};

TEST_F(TestBiasGradFp32, BiasGradFp32) {
  // prepare stage
  ArithmeticParameter *bias_param = static_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  ASSERT_NE(bias_param, nullptr);

  size_t input_size;
  std::string input_path = "./test_data/operators/biasgradfp32_1_dy_10_28_28_7.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  std::vector<int> dim_dy({10, 28, 28, 7});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(input_data);

  std::vector<lite::Tensor *> inputs = {&dy_tensor};
  auto output_data = new float[7];
  ASSERT_NE(output_data, nullptr);
  std::vector<int> dim_dw = {7};
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(output_data);
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_BiasAddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(bias_param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < 7; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./test_data/operators/biasgradfp32_1_db_7.bin";
  auto res = CompareRelativeOutput(output_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  // delete bias_param;
  dy_tensor.set_data(nullptr);
  dw_tensor.set_data(nullptr);
  delete kernel_obj;
  MS_LOG(INFO) << "BiasGradFp32 passed";
}

TEST_F(TestBiasGradFp32, BiasGrad2DFp32) {
  // prepare stage
  ArithmeticParameter *bias_param = static_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  ASSERT_NE(bias_param, nullptr);

  size_t input_size;
  std::string input_path = "./test_data/operators/fc_yt.f32";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_dy({2, 20});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(input_data);

  std::vector<lite::Tensor *> inputs = {&dy_tensor};
  auto output_data = new float[20];
  ASSERT_NE(output_data, nullptr);
  std::vector<int> dim_dw = {20};
  lite::Tensor dw_tensor(TypeId::kNumberTypeFloat32, dim_dw);
  dw_tensor.set_data(output_data);
  std::vector<lite::Tensor *> outputs = {&dw_tensor};

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_BiasAddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(bias_param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./test_data/operators/fc_b_grad.f32";
  auto res = CompareRelativeOutput(output_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  // delete bias_param;
  dy_tensor.set_data(nullptr);
  dw_tensor.set_data(nullptr);
  delete kernel_obj;
  MS_LOG(INFO) << "BiasGradFp32 passed";
}

}  // namespace mindspore
