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
#include <vector>
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "src/common/file_utils_ext.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/fp32/reduce.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp32_grad/arithmetic_grad.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {

class TestArithmeticGradFp32 : public mindspore::CommonTest {
 public:
  TestArithmeticGradFp32() {}
};

std::vector<lite::tensor::Tensor *> GenerateTensorsForTest(const char *test, int test_id) {
  size_t input_size;
  std::vector<int> large_dim({4, 6});
  std::vector<int> small_dim({6});
  int large_size = (4 * 6);
  int small_size = (1 * 6);
  char *dx1_file = const_cast<char *>("./test_data/operators/arithmetic_fp32_1_x1_4_6.bin");
  char *dx2_file = const_cast<char *>("./test_data/operators/arithmetic_fp32_1_x2_1_6.bin");

  if (test_id == 7) {
    large_dim = std::vector<int>({4, 5, 6});
    small_dim = std::vector<int>({6});
    large_size = (4 * 5 * 6);
    small_size = (6);
    dx1_file = const_cast<char *>("./test_data/operators/arithmetic_fp32_7_x1_4_5_6.bin");
    dx2_file = const_cast<char *>("./test_data/operators/arithmetic_fp32_7_x2_1_1_6.bin");
  }
  if (test_id >= 8) {
    large_dim = std::vector<int>({5, 4, 6});
    small_dim = std::vector<int>({5, 1, 6});
    large_size = (4 * 5 * 6);
    small_size = (5 * 6);
    dx1_file = const_cast<char *>("./test_data/operators/arithmetic_fp32_8_x1_5_4_6.bin");
    dx2_file = const_cast<char *>("./test_data/operators/arithmetic_fp32_8_x2_5_1_6.bin");
  }

  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(test, &input_size));
  lite::tensor::Tensor *dy_tensor = new lite::tensor::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  dy_tensor->SetData(dy_data);

  auto x1_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dx1_file, &input_size));
  lite::tensor::Tensor *x1_tensor = new lite::tensor::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  x1_tensor->SetData(x1_data);

  auto x2_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dx2_file, &input_size));
  lite::tensor::Tensor *x2_tensor = new lite::tensor::Tensor(TypeId::kNumberTypeFloat32, small_dim);
  x2_tensor->SetData(x2_data);

  auto dx1_data = new float[large_size];
  lite::tensor::Tensor *dx1_tensor = new lite::tensor::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  dx1_tensor->SetData(dx1_data);

  auto dx2_data = new float[small_size];
  lite::tensor::Tensor *dx2_tensor = new lite::tensor::Tensor(TypeId::kNumberTypeFloat32, small_dim);
  dx2_tensor->SetData(dx2_data);

  std::vector<lite::tensor::Tensor *> ret_vector = {dy_tensor, x1_tensor, x2_tensor, dx1_tensor, dx2_tensor};
  return ret_vector;
}

TEST_F(TestArithmeticGradFp32, TestAddGradFp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_AddGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_1_dy_4_6.bin", 1);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_AddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_1_dx1_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_1_dx2_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestAddGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestAddGrad2Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_AddGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_1_dy_4_6.bin", 1);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_AddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_1_dx1_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_1_dx2_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestAddGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestAddGrad3Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_AddGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_8_dy_5_4_6.bin", 8);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_AddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_8_dx2_5_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_8_dx1_5_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestAddGrad3Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestSubGradFp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_SubGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_2_dy_4_6.bin", 2);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SubGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_2_dx1_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_2_dx2_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestSubGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestSubGrad2Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_SubGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_3_dy_4_6.bin", 3);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SubGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_3_dx1_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_3_dx2_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestSubGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGradFp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_MulGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_4_dy_4_6.bin", 4);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);

  int loop_count = 1000;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    kernel_obj->Run();
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  printf("total cost (for %d loops): %lu us\n", loop_count, cost);
  // auto time_avg = cost / loop_count;
  // printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_4_dx1_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_4_dx2_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestMulGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGrad2Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_MulGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_4_dy_4_6.bin", 4);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_4_dx1_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_4_dx2_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestMulGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGrad3Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_MulGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_9_dy_5_4_6.bin", 9);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_9_dx1_5_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_9_dx2_5_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestMulGrad3Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGrad4Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_MulGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_9_dy_5_4_6.bin", 9);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_9_dx1_5_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_9_dx2_5_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestMulGrad4Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestDivGradFp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_DivGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_5_dy_4_6.bin", 5);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_5_dx1_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->Data()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_5_dx2_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, dx2_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestDivGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestDivGrad2Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_DivGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_6_dy_4_6.bin", 6);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_6_dx2_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->Data()), dx2_path));

  std::string output_path = "./test_data/operators/arithmetic_fp32_6_dx1_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, output_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestDivGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestDivGrad3Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_DivGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_10_dy_5_4_6.bin", 10);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string dx1_path = "./test_data/operators/arithmetic_fp32_10_dx1_5_4_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->Data()), dx1_path));

  std::string output_path = "./test_data/operators/arithmetic_fp32_10_dx2_5_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, output_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestDivGrad3Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, Test3DDivGrad2Fp32) {
  auto param = new ArithmeticParameter();
  param->op_parameter_.type_ = PrimitiveType_DivGrad;
  std::vector<lite::tensor::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_7_dy_4_5_6.bin", 7);

  std::vector<lite::tensor::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::tensor::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), NULL, desc, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->Data());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string dx1_path = "./test_data/operators/arithmetic_fp32_7_dx1_4_5_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->Data()), dx1_path));

  std::string output_path = "./test_data/operators/arithmetic_fp32_7_dx2_1_1_6.bin";
  EXPECT_EQ(0, lite::CompareRelativeOutput(output_ptr, output_path));

  for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete param;
  MS_LOG(INFO) << "TestDivGrad2Fp32 passed";
}

}  // namespace mindspore
