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
#include "src/common/utils.h"
#include "src/common/file_utils.h"
#include "nnacl/fp32_grad/pooling_grad.h"
#include "nnacl/kernel/pooling.h"
#include "src/litert/kernel/cpu/fp32_grad/pooling_grad.h"
#include "mindspore/lite/src/litert/kernel_registry.h"

namespace mindspore {
class TestPoolingGradFp32 : public mindspore::CommonTest {
 public:
  TestPoolingGradFp32() {}
};

void InitPoolingArgsFp32(PoolingComputeParam *pooling_param) {
  pooling_param->input_batch_ = 1;
  pooling_param->input_h_ = 28;
  pooling_param->input_w_ = 28;
  pooling_param->input_channel_ = 3;
  pooling_param->output_batch_ = 1;
  pooling_param->output_h_ = 28;
  pooling_param->output_w_ = 28;
  pooling_param->output_channel_ = 32;
  pooling_param->window_h_ = 3;
  pooling_param->window_w_ = 3;
}

void InitPoolingParamFP32(PoolingParameter *pooling_param) {
  pooling_param->window_h_ = 3;
  pooling_param->window_w_ = 3;
  pooling_param->stride_h_ = 1;
  pooling_param->stride_w_ = 1;

  pooling_param->pad_u_ = 1;
  pooling_param->pad_d_ = 1;
  pooling_param->pad_l_ = 1;
  pooling_param->pad_r_ = 1;
  pooling_param->global_ = false;
  pooling_param->op_parameter_.thread_num_ = 1;
}

TEST_F(TestPoolingGradFp32, AvgPoolingGradFp32) {
  // prepare stage
  auto pooling_param = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(pooling_param, nullptr);
  InitPoolingParamFP32(pooling_param);

  auto pooling_args = static_cast<PoolingComputeParam *>(malloc(sizeof(PoolingComputeParam)));
  ASSERT_NE(pooling_param, nullptr);
  InitPoolingArgsFp32(pooling_args);

  pooling_args->output_channel_ = 3;
  pooling_param->pool_mode_ = PoolMode_AvgPool;

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    pooling_args->output_batch_ * pooling_args->output_channel_ * pooling_args->output_h_ * pooling_args->output_w_;

  size_t input_size;
  std::string input_path = "./pooling/avgpoolgradfp32_1_dy_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);

  // warm up loop
  for (int i = 0; i < 3; i++) {
    std::fill(output_data, output_data + output_data_size, 0.f);
    AvgPoolingGrad(input_data, output_data, pooling_args->output_batch_, pooling_param, pooling_args);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    std::fill(output_data, output_data + output_data_size, 0.f);
    AvgPoolingGrad(input_data, output_data, pooling_args->output_batch_, pooling_param, pooling_args);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./pooling/avgpoolgradfp32_1_dx_1_28_28_3.bin";
  auto res = CompareOutput(output_data, output_data_size, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] output_data;
  free(pooling_param);
  MS_LOG(INFO) << "TestAvgPoolingGradFp32 passed";
}

TEST_F(TestPoolingGradFp32, AvgPoolingKernelGradFp32) {
  // prepare stage
  auto pooling_param = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(pooling_param, nullptr);
  InitPoolingParamFP32(pooling_param);
  pooling_param->pool_mode_ = PoolMode_AvgPool;

  auto pooling_args = static_cast<PoolingComputeParam *>(malloc(sizeof(PoolingComputeParam)));
  ASSERT_NE(pooling_param, nullptr);
  InitPoolingArgsFp32(pooling_args);
  pooling_args->output_channel_ = 3;

  // runtime part
  printf("Calculating runtime cost...\n");
  // uint64_t time_avg = 0;
  size_t output_data_size =
    pooling_args->output_batch_ * pooling_args->output_channel_ * pooling_args->output_h_ * pooling_args->output_w_;

  size_t input_size;
  std::string input_path = "./pooling/avgpoolgradfp32_1_dy_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_dy({1, 28, 28, 3});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(input_data);

  std::string input1_path = "./pooling/avgpoolgradfp32_1_x_1_28_28_3.bin";
  auto input1_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1_path.c_str(), &input_size));
  ASSERT_NE(input1_data, nullptr);
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input1_data);

  std::vector<lite::Tensor *> inputs = {&x_tensor, &x_tensor, &dy_tensor};

  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.set_data(output_data);
  std::vector<lite::Tensor *> outputs = {&dx_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, NHWC, schema::PrimitiveType_AvgPoolGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(pooling_param), &context, desc);
  ASSERT_NE(kernel_obj, nullptr);

  auto ret = kernel_obj->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel_obj->Run();
  EXPECT_EQ(0, ret);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./pooling/avgpoolgradfp32_1_dx_1_28_28_3.bin";
  auto res = CompareOutput(output_data, output_data_size, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] input1_data;
  delete[] output_data;
  dx_tensor.set_data(nullptr);
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  // delete pooling_param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestAvgPoolingGradFp32 passed";
}

TEST_F(TestPoolingGradFp32, AvgPoolingBatchGradFp32) {
  // prepare stage
  auto pooling_param = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(pooling_param, nullptr);

  InitPoolingParamFP32(pooling_param);
  pooling_param->pool_mode_ = PoolMode_AvgPool;

  // runtime part
  printf("Calculating runtime cost...\n");
  // uint64_t time_avg = 0;
  size_t input_size;
  std::string input_path = "./pooling/avgpoolgradfp32_1_dy_3_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  ASSERT_NE(input_data, nullptr);
  std::vector<int> dim_dy({3, 28, 28, 3});
  lite::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.set_data(input_data);

  std::string input1_path = "./pooling/avgpoolgradfp32_1_x_3_28_28_3.bin";
  auto input1_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1_path.c_str(), &input_size));
  ASSERT_NE(input1_data, nullptr);
  std::vector<int> dim_x({3, 28, 28, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(input1_data);

  std::vector<lite::Tensor *> inputs = {&x_tensor, &x_tensor, &dy_tensor};

  std::vector<int> dim_dx({3, 28, 28, 3});
  lite::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  ASSERT_EQ(dx_tensor.MallocData(), 0);
  auto output_data = reinterpret_cast<float *>(dx_tensor.MutableData());
  std::vector<lite::Tensor *> outputs = {&dx_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, NHWC, schema::PrimitiveType_AvgPoolGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(pooling_param), &context, desc);
  ASSERT_NE(kernel_obj, nullptr);

  auto ret = kernel_obj->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel_obj->Run();
  EXPECT_EQ(0, ret);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./pooling/avgpoolgradfp32_1_dx_3_28_28_3.bin";
  size_t output_data_size = dx_tensor.ElementsNum();
  auto res = CompareOutput(output_data, output_data_size, output_path);
  EXPECT_EQ(res, 0);

  delete[] input_data;
  delete[] input1_data;
  x_tensor.set_data(nullptr);
  dy_tensor.set_data(nullptr);
  // delete pooling_param;
  delete kernel_obj;
  MS_LOG(INFO) << "TestAvgPoolingGradBatchFp32 passed";
}

TEST_F(TestPoolingGradFp32, AvgPoolGradStride2Fp32) {
  // prepare stage
  // input size will be equal to the original size of x, output size will be the output size as in forward
  auto pool = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(pool, nullptr);
  InitPoolingParamFP32(pool);
  pool->pool_mode_ = PoolMode_AvgPool;
  pool->stride_h_ = 2;
  pool->stride_w_ = 2;

  auto args = static_cast<PoolingComputeParam *>(malloc(sizeof(PoolingComputeParam)));
  ASSERT_NE(args, nullptr);
  InitPoolingArgsFp32(args);
  args->output_channel_ = 3;
  args->input_batch_ = 3;
  args->output_batch_ = 3;
  args->output_h_ = 14;
  args->output_w_ = 14;

  size_t input_size;

  auto x_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/avgpoolgradfp32_s2_x_3_28_28_3.bin", &input_size));
  ASSERT_NE(x_data, nullptr);
  std::vector<int> dim_x({args->output_batch_, args->input_h_, args->input_w_, args->input_channel_});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(x_data);

  auto yt_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/avgpoolgradfp32_s2_dy_3_28_28_3.bin", &input_size));
  ASSERT_NE(yt_data, nullptr);
  std::vector<int> dim_y({args->output_batch_, args->output_h_, args->output_w_, args->output_channel_});
  lite::Tensor yt_tensor(TypeId::kNumberTypeFloat32, dim_y);
  yt_tensor.set_data(yt_data);
  lite::Tensor out_tensor(TypeId::kNumberTypeFloat32, dim_x);
  ASSERT_EQ(out_tensor.MallocData(), 0);
  float *out_data = static_cast<float *>(out_tensor.MutableData());
  std::vector<lite::Tensor *> inputs = {&x_tensor, &yt_tensor, &yt_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey pool_desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, NHWC, schema::PrimitiveType_AvgPoolGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(pool_desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(pool), &context, pool_desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::string output_path = "./pooling/avgpoolgradfp32_s2_dx_3_28_28_3.bin";
  auto res = CompareRelativeOutput(out_data, output_path);
  EXPECT_EQ(res, 0);

  delete[] x_data;
  delete[] yt_data;
  x_tensor.set_data(nullptr);
  yt_tensor.set_data(nullptr);
  delete kernel;
  MS_LOG(INFO) << "AvgPoolGradStride2Fp32 Filter Grad passed";
}

TEST_F(TestPoolingGradFp32, AvgPoolGradStride3Fp32) {
  // prepare stage
  // input size will be equal to the original size of x, output size will be the output size as in forward
  auto pool = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(pool, nullptr);
  InitPoolingParamFP32(pool);
  pool->pool_mode_ = PoolMode_AvgPool;
  pool->stride_h_ = 3;
  pool->stride_w_ = 3;

  auto args = static_cast<PoolingComputeParam *>(malloc(sizeof(PoolingComputeParam)));
  ASSERT_NE(args, nullptr);
  InitPoolingArgsFp32(args);
  args->output_channel_ = 3;
  args->input_batch_ = 3;
  args->output_batch_ = 3;
  args->output_h_ = 10;
  args->output_w_ = 10;

  size_t input_size;

  auto x_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/avgpoolgradfp32_s3_x_3_28_28_3.bin", &input_size));
  ASSERT_NE(x_data, nullptr);
  std::vector<int> dim_x({args->output_batch_, args->input_h_, args->input_w_, args->input_channel_});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(x_data);

  auto yt_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/avgpoolgradfp32_s3_dy_3_28_28_3.bin", &input_size));
  ASSERT_NE(yt_data, nullptr);
  std::vector<int> dim_y({args->output_batch_, args->output_h_, args->output_w_, args->output_channel_});
  lite::Tensor yt_tensor(TypeId::kNumberTypeFloat32, dim_y);
  yt_tensor.set_data(yt_data);

  lite::Tensor out_tensor(TypeId::kNumberTypeFloat32, dim_x);
  ASSERT_EQ(out_tensor.MallocData(), 0);
  auto out_data = static_cast<float *>(out_tensor.MutableData());

  std::vector<lite::Tensor *> inputs = {&x_tensor, &yt_tensor, &yt_tensor};
  std::vector<lite::Tensor *> outputs = {&out_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey pool_desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, NHWC, schema::PrimitiveType_AvgPoolGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(pool_desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(inputs, outputs, reinterpret_cast<OpParameter *>(pool), &context, pool_desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::string output_path = "./pooling/avgpoolgradfp32_s3_dx_3_28_28_3.bin";
  auto res = CompareRelativeOutput(out_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] x_data;
  delete[] yt_data;
  x_tensor.set_data(nullptr);
  yt_tensor.set_data(nullptr);
  delete kernel;
  MS_LOG(INFO) << "AvgPoolGradStride3Fp32 Filter Grad passed";
}

TEST_F(TestPoolingGradFp32, MaxPoolingGradFp32) {
  // prepare stage
  auto pooling_param = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(pooling_param, nullptr);
  InitPoolingParamFP32(pooling_param);
  pooling_param->pool_mode_ = PoolMode_MaxPool;

  auto args = static_cast<PoolingComputeParam *>(malloc(sizeof(PoolingComputeParam)));
  ASSERT_NE(args, nullptr);
  InitPoolingArgsFp32(args);
  args->output_channel_ = 3;

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size = args->output_batch_ * args->output_channel_ * args->output_h_ * args->output_w_;

  size_t input_size;
  std::string i_path = "./pooling/maxpoolgradfp32_1_x_1_28_28_3.bin";
  auto in_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(i_path.c_str(), &input_size));
  ASSERT_NE(in_data, nullptr);

  std::string dy_path = "./pooling/maxpoolgradfp32_1_dy_1_28_28_3.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &input_size));
  ASSERT_NE(dy_data, nullptr);

  std::string dx_path = "./pooling/maxpoolgradfp32_1_dx_1_28_28_3.bin";
  auto dx_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dx_path.c_str(), &input_size));
  ASSERT_NE(dx_data, nullptr);
  int in_batch_size = args->input_h_ * args->input_w_ * args->input_channel_ * args->input_batch_;
  auto output_data = new float[output_data_size];
  ASSERT_NE(output_data, nullptr);
  // warm up loop
  for (int i = 0; i < 3; i++) {
    std::fill(output_data, output_data + in_batch_size, 0.f);
    MaxPoolingGrad(in_data, dy_data, output_data, args->output_batch_, pooling_param, args);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    std::fill(output_data, output_data + in_batch_size, 0.f);
    MaxPoolingGrad(in_data, dy_data, output_data, args->output_batch_, pooling_param, args);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./pooling/maxpoolgradfp32_1_xgrad_1_28_28_3.bin";
  auto res = CompareOutput(output_data, output_data_size, output_path);
  EXPECT_EQ(res, 0);

  free(pooling_param);
  delete[] in_data;
  delete[] dy_data;
  delete[] dx_data;
  delete[] output_data;
  MS_LOG(INFO) << "TestMaxPoolingGradFp32 passed";
}

TEST_F(TestPoolingGradFp32, MaxPoolGradBatchFp32) {
  // prepare stage
  // input size will be equal to the original size of x, output size will be the output size as in forward
  auto maxpool = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(maxpool, nullptr);
  InitPoolingParamFP32(maxpool);
  maxpool->pool_mode_ = PoolMode_MaxPool;

  size_t input_size;
  auto x_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_1_x_3_28_28_3.bin", &input_size));
  ASSERT_NE(x_data, nullptr);
  std::vector<int> dim_x({3, 28, 28, 3});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(x_data);

  auto y_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_1_dx_3_28_28_3.bin", &input_size));
  ASSERT_NE(y_data, nullptr);
  std::vector<int> dim_y({3, 28, 28, 3});
  lite::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
  y_tensor.set_data(y_data);

  auto yt_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_1_dy_3_28_28_3.bin", &input_size));
  ASSERT_NE(yt_data, nullptr);
  lite::Tensor yt_tensor(TypeId::kNumberTypeFloat32, dim_y);
  yt_tensor.set_data(yt_data);

  lite::Tensor out_tensor(TypeId::kNumberTypeFloat32, dim_x);
  ASSERT_EQ(out_tensor.MallocData(), 0);
  auto out_data = static_cast<float *>(out_tensor.MutableData());
  std::vector<lite::Tensor *> maxpool_inputs = {&x_tensor, &y_tensor, &yt_tensor};
  std::vector<lite::Tensor *> maxpool_outputs = {&out_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, NHWC, schema::PrimitiveType_MaxPoolGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(maxpool_inputs, maxpool_outputs, reinterpret_cast<OpParameter *>(maxpool), &context, desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::string output_path = "./pooling/maxpoolgradfp32_1_xgrad_3_28_28_3.bin";
  auto res = CompareRelativeOutput(out_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] x_data;
  delete[] y_data;
  delete[] yt_data;
  x_tensor.set_data(nullptr);
  y_tensor.set_data(nullptr);
  yt_tensor.set_data(nullptr);
  delete kernel;
  MS_LOG(INFO) << "MaxPoolGradBatchFp32 Filter Grad passed";
}

TEST_F(TestPoolingGradFp32, MaxPoolGradStride2Fp32) {
  // prepare stage
  // input size will be equal to the original size of x, output size will be the output size as in forward
  auto maxpool = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(maxpool, nullptr);
  InitPoolingParamFP32(maxpool);
  maxpool->pool_mode_ = PoolMode_MaxPool;
  maxpool->stride_h_ = 2;
  maxpool->stride_w_ = 2;

  auto args = static_cast<PoolingComputeParam *>(malloc(sizeof(PoolingComputeParam)));
  ASSERT_NE(args, nullptr);
  InitPoolingArgsFp32(args);
  args->output_channel_ = 3;
  args->input_channel_ = 3;
  args->input_batch_ = 3;
  args->output_batch_ = 3;
  args->output_h_ = 14;
  args->output_w_ = 14;

  size_t input_size;

  auto x_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_s2_x_3_28_28_3.bin", &input_size));
  ASSERT_NE(x_data, nullptr);
  std::vector<int> dim_x({args->output_batch_, args->input_h_, args->input_w_, args->input_channel_});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(x_data);

  auto y_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_s2_dx_3_28_28_3.bin", &input_size));
  ASSERT_NE(y_data, nullptr);
  std::vector<int> dim_y({args->output_batch_, args->output_h_, args->output_w_, args->output_channel_});
  lite::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
  y_tensor.set_data(y_data);

  auto yt_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_s2_dy_3_28_28_3.bin", &input_size));
  ASSERT_NE(yt_data, nullptr);
  lite::Tensor yt_tensor(TypeId::kNumberTypeFloat32, dim_y);
  yt_tensor.set_data(yt_data);

  lite::Tensor out_tensor(TypeId::kNumberTypeFloat32, dim_x);
  ASSERT_EQ(out_tensor.MallocData(), 0);
  auto out_data = static_cast<float *>(out_tensor.MutableData());

  std::vector<lite::Tensor *> maxpool_inputs = {&x_tensor, &y_tensor, &yt_tensor};
  std::vector<lite::Tensor *> maxpool_outputs = {&out_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, NHWC, schema::PrimitiveType_MaxPoolGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(maxpool_inputs, maxpool_outputs, reinterpret_cast<OpParameter *>(maxpool), &context, desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::string output_path = "./pooling/maxpoolgradfp32_s2_xgrad_3_28_28_3.bin";
  auto res = CompareRelativeOutput(out_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] x_data;
  delete[] y_data;
  delete[] yt_data;
  x_tensor.set_data(nullptr);
  y_tensor.set_data(nullptr);
  yt_tensor.set_data(nullptr);
  delete kernel;
  MS_LOG(INFO) << "MaxPoolGradStride2Fp32 Filter Grad passed";
}

TEST_F(TestPoolingGradFp32, MaxPoolGradStride3Fp32) {
  // prepare stage
  // input size will be equal to the original size of x, output size will be the output size as in forward
  auto maxpool = static_cast<PoolingParameter *>(malloc(sizeof(PoolingParameter)));
  ASSERT_NE(maxpool, nullptr);
  InitPoolingParamFP32(maxpool);
  maxpool->pool_mode_ = PoolMode_MaxPool;
  maxpool->stride_h_ = 3;
  maxpool->stride_w_ = 3;

  auto args = static_cast<PoolingComputeParam *>(malloc(sizeof(PoolingComputeParam)));
  ASSERT_NE(args, nullptr);
  InitPoolingArgsFp32(args);
  args->output_channel_ = 3;
  args->input_channel_ = 3;
  args->input_batch_ = 3;
  args->output_batch_ = 3;
  args->output_h_ = 10;
  args->output_w_ = 10;

  size_t input_size;

  auto x_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_s3_x_3_28_28_3.bin", &input_size));
  ASSERT_NE(x_data, nullptr);
  std::vector<int> dim_x({args->output_batch_, args->input_h_, args->input_w_, args->input_channel_});
  lite::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.set_data(x_data);

  auto y_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_s3_dx_3_28_28_3.bin", &input_size));
  ASSERT_NE(y_data, nullptr);
  std::vector<int> dim_y({args->output_batch_, args->output_h_, args->output_w_, args->output_channel_});
  lite::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
  y_tensor.set_data(y_data);

  auto yt_data =
    reinterpret_cast<float *>(mindspore::lite::ReadFile("./pooling/maxpoolgradfp32_s3_dy_3_28_28_3.bin", &input_size));
  ASSERT_NE(yt_data, nullptr);
  lite::Tensor yt_tensor(TypeId::kNumberTypeFloat32, dim_y);
  yt_tensor.set_data(yt_data);

  lite::Tensor out_tensor(TypeId::kNumberTypeFloat32, dim_x);
  ASSERT_EQ(out_tensor.MallocData(), 0);
  auto out_data = static_cast<float *>(out_tensor.MutableData());

  std::vector<lite::Tensor *> maxpool_inputs = {&x_tensor, &y_tensor, &yt_tensor};
  std::vector<lite::Tensor *> maxpool_outputs = {&out_tensor};

  lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, NHWC, schema::PrimitiveType_MaxPoolGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel = creator(maxpool_inputs, maxpool_outputs, reinterpret_cast<OpParameter *>(maxpool), &context, desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::string output_path = "./pooling/maxpoolgradfp32_s3_xgrad_3_28_28_3.bin";
  auto res = CompareRelativeOutput(out_data, output_path);

  EXPECT_EQ(res, 0);

  delete[] x_data;
  delete[] y_data;
  delete[] yt_data;
  x_tensor.set_data(nullptr);
  y_tensor.set_data(nullptr);
  yt_tensor.set_data(nullptr);
  delete kernel;
  MS_LOG(INFO) << "MaxPoolGradStride3Fp32 Filter Grad passed";
}
}  // namespace mindspore
