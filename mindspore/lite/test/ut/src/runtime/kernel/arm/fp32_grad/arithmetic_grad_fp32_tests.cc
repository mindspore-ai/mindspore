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

#include "schema/inner/model_generated.h"
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "nnacl/fp32/reduce_fp32.h"
#include "src/runtime/kernel/arm/fp32_grad/arithmetic_grad.h"
#include "src/kernel_registry.h"

namespace mindspore {

#ifdef PRIMITIVE_WRITEABLE
ArithmeticParameter *PopulateArithmeticParameter(mindspore::schema::PrimitiveType type,
                                                 std::vector<lite::Tensor *> inputs,
                                                 std::vector<lite::Tensor *> outputs) {
  ArithmeticParameter *arithmetic_param = static_cast<ArithmeticParameter *>(malloc(sizeof(ArithmeticParameter)));
  if (arithmetic_param == nullptr) {
    MS_LOG(ERROR) << "new ArithmeticParameter failed.";
    return nullptr;
  }
  arithmetic_param->op_parameter_.type_ = type;
  schema::PrimitiveT *prim = new schema::PrimitiveT;
  if (prim == nullptr) {
    free(arithmetic_param);
    MS_LOG(ERROR) << "new PrimitiveT failed.";
    return nullptr;
  }

  prim->value.type = type;
  return arithmetic_param;
}

class TestArithmeticGradFp32 : public mindspore::CommonTest {
 public:
  TestArithmeticGradFp32() {}
};

std::vector<lite::Tensor *> GenerateTensorsForTest(const char *test, int test_id) {
  size_t input_size;
  std::vector<lite::Tensor *> ret_vector;
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
  if (dy_data == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    return ret_vector;
  }
  lite::Tensor *dy_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  if (dy_tensor == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    return ret_vector;
  }
  dy_tensor->set_data(dy_data);

  auto x1_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dx1_file, &input_size));
  if (x1_data == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    return ret_vector;
  }
  lite::Tensor *x1_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  if (x1_tensor == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    delete[] x1_data;
    return ret_vector;
  }
  x1_tensor->set_data(x1_data);

  auto x2_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dx2_file, &input_size));
  if (x2_data == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    delete[] x1_data;
    delete x1_tensor;
    return ret_vector;
  }
  lite::Tensor *x2_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, small_dim);
  if (x2_tensor == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    delete[] x1_data;
    delete x1_tensor;
    delete[] x2_data;
    return ret_vector;
  }
  x2_tensor->set_data(x2_data);

  auto dx1_data = new float[large_size];
  if (dx1_data == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    delete[] x1_data;
    delete x1_tensor;
    delete[] x2_data;
    delete x2_tensor;
    return ret_vector;
  }
  lite::Tensor *dx1_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  if (dx1_tensor == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    delete[] x1_data;
    delete x1_tensor;
    delete[] x2_data;
    delete x2_tensor;
    delete[] dx1_data;
    return ret_vector;
  }
  dx1_tensor->set_data(dx1_data);

  auto dx2_data = new float[small_size];
  if (dx2_data == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    delete[] x1_data;
    delete x1_tensor;
    delete[] x2_data;
    delete x2_tensor;
    delete[] dx1_data;
    delete dx1_tensor;
    return ret_vector;
  }
  lite::Tensor *dx2_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, small_dim);
  if (dx2_tensor == nullptr) {
    MS_LOG(ERROR) << "new operator failed";
    delete[] dy_data;
    delete dy_tensor;
    delete[] x1_data;
    delete x1_tensor;
    delete[] x2_data;
    delete x2_tensor;
    delete[] dx1_data;
    delete dx1_tensor;
    delete[] dx2_data;
    return ret_vector;
  }
  dx2_tensor->set_data(dx2_data);

  ret_vector.push_back(dy_tensor);
  ret_vector.push_back(x1_tensor);
  ret_vector.push_back(x2_tensor);
  ret_vector.push_back(dx1_tensor);
  ret_vector.push_back(dx2_tensor);

  return ret_vector;
}

TEST_F(TestArithmeticGradFp32, TestAddGradFp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_1_dy_4_6.bin", 1);
  ASSERT_NE(all_tensors.size(), 0);
  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_AddGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_AddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_1_dx1_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_1_dx2_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  delete kernel_obj;
  MS_LOG(INFO) << "TestAddGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestAddGrad2Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_1_dy_4_6.bin", 1);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_AddGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_AddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_1_dx1_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_1_dx2_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i]; //TODO tensor data is unique pointer
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestAddGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestAddGrad3Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_8_dy_5_4_6.bin", 8);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_AddGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_AddGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_8_dx2_5_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_8_dx1_5_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));

  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestAddGrad3Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestSubGradFp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_2_dy_4_6.bin", 2);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_SubGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SubGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_2_dx1_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_2_dx2_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));

  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestSubGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestSubGrad2Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_3_dy_4_6.bin", 3);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_SubGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_SubGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_3_dx1_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_3_dx2_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));

  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  delete kernel_obj;
  MS_LOG(INFO) << "TestSubGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGradFp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_4_dy_4_6.bin", 4);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_MulGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
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

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_4_dx1_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_4_dx2_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  delete kernel_obj;
  // delete param;
  MS_LOG(INFO) << "TestMulGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGrad2Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_4_dy_4_6.bin", 4);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_MulGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_4_dx1_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_4_dx2_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestMulGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGrad3Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_9_dy_5_4_6.bin", 9);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_MulGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_9_dx1_5_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_9_dx2_5_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestMulGrad3Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMulGrad4Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_9_dy_5_4_6.bin", 9);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_MulGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MulGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_9_dx1_5_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_9_dx2_5_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestMulGrad4Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestDivGradFp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_5_dy_4_6.bin", 5);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_DivGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/operators/arithmetic_fp32_5_dx1_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), output_path));

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_5_dx2_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, dx2_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  delete kernel_obj;
  // delete param;
  MS_LOG(INFO) << "TestDivGradFp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestDivGrad2Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_6_dy_4_6.bin", 6);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[2], all_tensors[1]};
  std::vector<lite::Tensor *> outputs = {all_tensors[4], all_tensors[3]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_DivGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[0]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string dx2_path = "./test_data/operators/arithmetic_fp32_6_dx2_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[1]->MutableData()), dx2_path));

  std::string output_path = "./test_data/operators/arithmetic_fp32_6_dx1_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, output_path));

  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestDivGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestDivGrad3Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_10_dy_5_4_6.bin", 10);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_DivGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string dx1_path = "./test_data/operators/arithmetic_fp32_10_dx1_5_4_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), dx1_path));

  std::string output_path = "./test_data/operators/arithmetic_fp32_10_dx2_5_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, output_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  // for (int i = 0; i < 5; i++) delete all_tensors[i];
  // delete param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestDivGrad3Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, Test3DDivGrad2Fp32) {
  std::vector<lite::Tensor *> all_tensors =
    GenerateTensorsForTest("./test_data/operators/arithmetic_fp32_7_dy_4_5_6.bin", 7);
  ASSERT_NE(all_tensors.size(), 0);

  std::vector<lite::Tensor *> inputs = {all_tensors[0], all_tensors[1], all_tensors[2]};
  std::vector<lite::Tensor *> outputs = {all_tensors[3], all_tensors[4]};
  auto param = PopulateArithmeticParameter(schema::PrimitiveType_DivGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_DivGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string dx1_path = "./test_data/operators/arithmetic_fp32_7_dx1_4_5_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), dx1_path));

  std::string output_path = "./test_data/operators/arithmetic_fp32_7_dx2_1_1_6.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, output_path));
  for (auto tensor : all_tensors) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  delete kernel_obj;
  MS_LOG(INFO) << "TestDivGrad2Fp32 passed";
}

TEST_F(TestArithmeticGradFp32, TestMaximumGradBroadcastFp32) {
  std::vector<int> large_dim({4, 6});
  std::vector<int> small_dim({6});

  large_dim = std::vector<int>({1, 2, 3});
  small_dim = std::vector<int>({1, 3});
  int large_size = (2 * 3);
  int small_size = 3;
  size_t input_size;
  char *dx1_file = const_cast<char *>("./test_data/operators/x1_maximum.bin");
  char *dx2_file = const_cast<char *>("./test_data/operators/x2_maximum.bin");

  std::string yt_path = "./test_data/operators/yt_maximum.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(yt_path.c_str(), &input_size));
  ASSERT_NE(dy_data, nullptr);
  EXPECT_EQ(input_size, large_size * sizeof(float));
  lite::Tensor *dy_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  ASSERT_NE(dy_tensor, nullptr);
  dy_tensor->set_data(dy_data);

  auto x1_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dx1_file, &input_size));
  ASSERT_NE(x1_data, nullptr);
  lite::Tensor *x1_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, small_dim);
  ASSERT_NE(x1_tensor, nullptr);
  x1_tensor->set_data(x1_data);

  auto x2_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dx2_file, &input_size));
  ASSERT_NE(x2_data, nullptr);
  lite::Tensor *x2_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  ASSERT_NE(x2_tensor, nullptr);
  x2_tensor->set_data(x2_data);

  auto dx1_data = new float[small_size];
  ASSERT_NE(dx1_data, nullptr);
  lite::Tensor *dx1_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, small_dim);
  ASSERT_NE(dx1_tensor, nullptr);
  dx1_tensor->set_data(dx1_data);

  auto dx2_data = new float[large_size];
  ASSERT_NE(dx2_data, nullptr);
  lite::Tensor *dx2_tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, large_dim);
  ASSERT_NE(dx2_tensor, nullptr);
  dx2_tensor->set_data(dx2_data);

  std::vector<lite::Tensor *> inputs = {x1_tensor, x2_tensor, dy_tensor};
  std::vector<lite::Tensor *> outputs = {dx1_tensor, dx2_tensor};

  auto param = PopulateArithmeticParameter(schema::PrimitiveType_MaximumGrad, inputs, outputs);
  ASSERT_NE(param, nullptr);

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_MaximumGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  kernel_obj->Run();

  float *output_ptr = reinterpret_cast<float *>(outputs[1]->MutableData());
  printf("==================output data=================\n");
  for (int i = 0; i < 6; i++) {
    std::cout << output_ptr[i] << " ,";
  }
  std::cout << std::endl;

  std::string dx1_path = "./test_data/operators/x1_grad_maximum.bin";
  EXPECT_EQ(0, CompareRelativeOutput(reinterpret_cast<float *>(outputs[0]->MutableData()), dx1_path));

  std::string output_path = "./test_data/operators/x2_grad_maximum.bin";
  EXPECT_EQ(0, CompareRelativeOutput(output_ptr, output_path));
  for (auto tensor : inputs) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  for (auto tensor : outputs) {
    delete[] reinterpret_cast<float *>(tensor->MutableData());
    tensor->set_data(nullptr);
    delete tensor;
  }
  delete kernel_obj;
  MS_LOG(INFO) << "TestMaximumGradBroadcastFp32 passed";
}
#endif
}  // namespace mindspore
