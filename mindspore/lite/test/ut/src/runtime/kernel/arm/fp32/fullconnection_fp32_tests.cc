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
#include <sys/time.h>
#include <iostream>
#include <memory>
#include "common/common_test.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/common/file_utils.h"
#include "src/litert/tensor_category.h"
#include "src/common/log_adapter.h"
#include "src/litert/kernel/cpu/fp32/fullconnection_fp32.h"
#include "src/litert/infer_manager.h"
#include "src/litert/kernel_registry.h"

namespace mindspore {
using mindspore::lite::Tensor;

class TestFcFp32 : public mindspore::CommonTest {
 public:
  TestFcFp32() {}
};

TEST_F(TestFcFp32, FcTest1) {
  std::vector<lite::Tensor *> inputs;
  std::vector<float> in = {-3.2366564, -4.7733846, -7.8329225, 16.146885, 5.060793,  -6.1471,  -1.7680453, -6.5721383,
                           17.87506,   -5.1192183, 10.742863,  1.4536934, 19.693445, 19.45783, 5.063163,   0.5234792};
  inputs.push_back(
    CreateTensor<float>(kNumberTypeFloat32, {2, 2, 2, 2}, in, mindspore::NHWC, lite::Category::CONST_TENSOR));
  std::vector<float> weight = {-0.0024438887, 0.0006738146, -0.008169129, 0.0021510671,  -0.012470592,   -0.0053063435,
                               0.006050155,   0.008656233,  0.012911413,  -0.0028635843, -0.00034080597, -0.0010622552,
                               -0.012254699,  -0.01312836,  0.0025241964, -0.004706142,  0.002451482,    -0.009558459,
                               0.004481974,   0.0033251503, -0.011705584, -0.001720293,  -0.0039410214,  -0.0073637343};
  inputs.push_back(
    CreateTensor<float>(kNumberTypeFloat32, {3, 8}, weight, mindspore::NHWC, lite::Category::CONST_TENSOR));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {3}, {1.6103756, -0.9872417, 0.546849}, mindspore::NHWC,
                                       lite::Category::CONST_TENSOR));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {2, 3}, {}));

  auto param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  memset(param, 0, sizeof(MatMulParameter));
  param->b_transpose_ = true;
  param->a_transpose_ = false;
  param->has_bias_ = true;
  param->act_type_ = ActType_No;
  param->op_parameter_.type_ = 67;
  param->op_parameter_.is_train_session_ = false;
  KernelInferShape(inputs, outputs, reinterpret_cast<OpParameter *>(param));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter_.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_FullConnection};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
#ifdef SUPPORT_TRAIN
  kernel->AllocWorkspace();
#endif
  ASSERT_EQ(kernel->Run(), RET_OK);

  std::vector<float> except_result = {1.6157111, -0.98469573, 0.6098231, 1.1649342, -1.2334653, 0.404779};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestFcFp32, FcTest2) {
  std::vector<lite::Tensor *> inputs;
  size_t buffer_size;
  auto in_data = mindspore::lite::ReadFile("./matmul/FcFp32_input1.bin", &buffer_size);
  std::vector<char> in(in_data, in_data + buffer_size);
  delete[](in_data);
  inputs.push_back(
    CreateTensor<char>(kNumberTypeFloat32, {20, 4, 2, 10}, in, mindspore::NCHW, lite::Category::CONST_TENSOR));
  auto w_data = mindspore::lite::ReadFile("./matmul/FcFp32_weight1.bin", &buffer_size);
  std::vector<char> weight(w_data, w_data + buffer_size);
  delete[](w_data);
  inputs.push_back(
    CreateTensor<char>(kNumberTypeFloat32, {30, 80}, weight, mindspore::NCHW, lite::Category::CONST_TENSOR));
  auto bias_data = mindspore::lite::ReadFile("./matmul/FcFp32_bias1.bin", &buffer_size);
  std::vector<char> bias(bias_data, bias_data + buffer_size);
  delete[](bias_data);
  inputs.push_back(CreateTensor<char>(kNumberTypeFloat32, {30}, bias, mindspore::NCHW, lite::Category::CONST_TENSOR));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {20, 30}, {}));

  auto param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  memset(param, 0, sizeof(MatMulParameter));
  param->b_transpose_ = true;
  param->a_transpose_ = false;
  param->has_bias_ = true;
  param->act_type_ = ActType_No;
  param->op_parameter_.type_ = 67;
  KernelInferShape(inputs, outputs, reinterpret_cast<OpParameter *>(param));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 1;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter_.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_FullConnection};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
#ifdef SUPPORT_TRAIN
  kernel->AllocWorkspace();
#endif
  ASSERT_EQ(kernel->Run(), RET_OK);

  auto out_data = mindspore::lite::ReadFile("./matmul/FcFp32_output1.bin", &buffer_size);
  std::vector<char> except_result(out_data, out_data + buffer_size);
  delete[](out_data);
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()),
                                 reinterpret_cast<float *>(except_result.data()), outputs[0]->ElementsNum(), 0.000001));
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

TEST_F(TestFcFp32, FcTest3) {
  std::vector<lite::Tensor *> inputs;
  std::vector<float> in = {1, 0, 3, 0, 4, 5, 2, 5, 2, 5, 1, 5, 0, 1, 2, 0, 2, 1, 0, 5};
  inputs.push_back(
    CreateTensor<float>(kNumberTypeFloat32, {1, 1, 1, 20}, in, mindspore::NHWC, lite::Category::CONST_TENSOR));
  std::vector<float> weight = {
    0, 5, 5, 3, 0, 5, 3, 1, 0, 1, 3, 0, 5, 5, 2, 4, 0, 1, 1, 2, 3, 0, 5, 5, 4, 4, 1, 4, 1, 1, 5, 3, 3, 1, 0, 3,
    1, 2, 4, 5, 3, 4, 4, 0, 3, 5, 0, 3, 4, 1, 0, 1, 3, 4, 0, 5, 2, 5, 0, 4, 2, 2, 2, 2, 4, 4, 5, 2, 1, 1, 5, 1,
    4, 4, 5, 1, 2, 4, 0, 3, 1, 1, 0, 2, 1, 5, 2, 0, 1, 1, 5, 5, 4, 0, 0, 4, 2, 3, 2, 1, 4, 0, 5, 0, 2, 3, 1, 2,
    1, 2, 1, 4, 2, 3, 5, 5, 4, 5, 2, 0, 3, 0, 2, 0, 1, 3, 0, 4, 1, 5, 2, 5, 4, 2, 5, 1, 4, 5, 3, 1, 0, 4, 4, 4,
    1, 3, 4, 2, 2, 4, 1, 4, 0, 1, 0, 2, 4, 5, 2, 1, 0, 3, 5, 2, 4, 2, 1, 4, 2, 0, 1, 0, 2, 3, 0, 3, 2, 5, 5, 4,
    3, 0, 0, 2, 0, 1, 5, 2, 2, 1, 3, 0, 3, 0, 5, 3, 3, 3, 5, 5, 3, 4, 0, 1, 2, 1, 2, 4, 3, 5, 4, 3, 0, 0, 4, 4,
    2, 3, 5, 4, 3, 5, 1, 2, 1, 5, 0, 5, 1, 1, 5, 5, 0, 0, 1, 3, 2, 2, 2, 3, 4, 2, 2, 3, 2, 4, 3, 0, 2, 0, 3, 2,
    1, 5, 2, 4, 4, 5, 2, 5, 0, 5, 3, 3, 0, 3, 2, 5, 5, 1, 1, 0, 2, 3, 0, 1, 1, 2, 4, 1, 3, 3, 5, 5, 0, 1, 0, 0,
    1, 2, 3, 3, 5, 2, 2, 5, 1, 4, 3, 3, 0, 2, 5, 4, 3, 1, 2, 4, 0, 2, 1, 3, 1, 2, 1, 0, 5, 5, 4, 5};
  inputs.push_back(
    CreateTensor<float>(kNumberTypeFloat32, {16, 20}, weight, mindspore::NHWC, lite::Category::CONST_TENSOR));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1, 16}, {}));

  auto param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  memset(param, 0, sizeof(MatMulParameter));
  param->b_transpose_ = true;
  param->a_transpose_ = false;
  param->has_bias_ = false;
  param->act_type_ = ActType_No;
  param->op_parameter_.type_ = 67;
  KernelInferShape(inputs, outputs, reinterpret_cast<OpParameter *>(param));

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter_.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_FullConnection};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
#ifdef SUPPORT_TRAIN
  kernel->AllocWorkspace();
#endif
  struct timeval start, end;
  gettimeofday(&start, nullptr);
  for (int i = 0; i < 100000; ++i) {
    kernel->Run();
  }
  gettimeofday(&end, nullptr);
  printf("## elapsed: %lu\n", 1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - end.tv_usec);

  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}

void FcTest4_Resize(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs) {
  auto &in_tensor = inputs->at(0);
  in_tensor->FreeData();
  in_tensor->set_shape({2, 4});
  float in[] = {1, 2, 3, 4, 2, 3, 1, 2};
  memcpy(in_tensor->MutableData(), in, in_tensor->Size());

  auto &out_tensor = outputs->at(0);
  out_tensor->FreeData();
  out_tensor->set_shape({2, 10});
  out_tensor->MallocData();
}

TEST_F(TestFcFp32, FcTest4_Vec2Batch) {
  std::vector<lite::Tensor *> inputs;
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1, 4}, {1, 2, 3, 4}));
  std::vector<float> weight = {1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                               6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 1, 2, 3, 4};
  inputs.push_back(
    CreateTensor<float>(kNumberTypeFloat32, {10, 4}, weight, mindspore::NHWC, lite::Category::CONST_TENSOR));
  inputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {10}, {1, 1, 1, 1, 1, 2, 2, 2, 2, 2}, mindspore::NHWC,
                                       lite::Category::CONST_TENSOR));

  std::vector<lite::Tensor *> outputs;
  outputs.push_back(CreateTensor<float>(kNumberTypeFloat32, {1, 10}, {}));

  auto param = static_cast<MatMulParameter *>(malloc(sizeof(MatMulParameter)));
  memset(param, 0, sizeof(MatMulParameter));
#ifdef SUPPORT_TRAIN
  param->op_parameter_.is_train_session_ = true;
#else
  param->op_parameter_.is_train_session_ = false;
#endif
  param->a_transpose_ = false;
  param->b_transpose_ = true;
  param->has_bias_ = true;
  param->act_type_ = ActType_No;

  auto ctx = std::make_shared<lite::InnerContext>();
  ctx->thread_num_ = 2;
  ASSERT_EQ(ctx->Init(), RET_OK);
  param->op_parameter_.thread_num_ = ctx->thread_num_;

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_FullConnection};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto *kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->Prepare(), RET_OK);
#ifdef SUPPORT_TRAIN
  kernel->AllocWorkspace();
#endif
  ASSERT_EQ(kernel->Run(), RET_OK);

  std::vector<float> except_result = {11, 21, 31, 41, 51, 62, 72, 82, 92, 32};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
  FcTest4_Resize(&inputs, &outputs);
  ASSERT_EQ(kernel->ReSize(), RET_OK);
#ifdef SUPPORT_TRAIN
  kernel->FreeWorkspace();
  kernel->AllocWorkspace();
#endif
  ASSERT_EQ(kernel->Run(), RET_OK);
  except_result = {11, 21, 31, 41, 51, 62, 72, 82, 92, 32, 9, 17, 25, 33, 41, 50, 58, 66, 74, 21};
  ASSERT_EQ(0, CompareOutputData(static_cast<float *>(outputs[0]->data()), except_result.data(),
                                 outputs[0]->ElementsNum(), 0.000001));
#ifdef SUPPORT_TRAIN
  kernel->FreeWorkspace();
#endif
  delete kernel;
  DestroyTensors(inputs);
  DestroyTensors(outputs);
}
}  // namespace mindspore
