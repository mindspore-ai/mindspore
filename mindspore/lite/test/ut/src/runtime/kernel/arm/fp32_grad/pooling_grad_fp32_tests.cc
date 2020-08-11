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
#include "mindspore/lite/include/context.h"
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "src/common/utils.h"
#include "src/common/file_utils.h"
#include "src/runtime/kernel/arm/fp32_grad/pooling_grad.h"
#include "src/runtime/kernel/arm/nnacl/fp32_grad/pooling_grad.h"

namespace mindspore {
class TestPoolingGradFp32 :  public mindspore::CommonTest {
 public:
  TestPoolingGradFp32() {}
};

void InitPoolingParamFP32(PoolingParameter *pooling_param) {
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

  pooling_param->stride_h_ = 1;
  pooling_param->stride_w_ = 1;

  pooling_param->pad_u_ = 1;
  pooling_param->pad_d_ = 1;
  pooling_param->pad_l_ = 1;
  pooling_param->pad_r_ = 1;
  pooling_param->thread_num_ = 1;
}

TEST_F(TestPoolingGradFp32, AvgPoolingGradFp32) {
  // prepare stage
  auto pooling_param = new PoolingParameter();
  InitPoolingParamFP32(pooling_param);
  pooling_param->output_channel_ = 3;

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    pooling_param->output_batch_ * pooling_param->output_channel_ * pooling_param->output_h_ * pooling_param->output_w_;

  size_t input_size;
  std::string input_path = "./test_data/pooling/avgpoolgradfp32_1_dy_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));

  auto output_data = new float[output_data_size];
  // warm up loop
  for (int i = 0; i < 3; i++) {
    AvgPoolingGrad(input_data, output_data, pooling_param);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    AvgPoolingGrad(input_data, output_data, pooling_param);
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
  std::string output_path = "./test_data/pooling/avgpoolgradfp32_1_dx_1_28_28_3.bin";
  lite::CompareOutput(output_data, output_path);

  delete input_data;
  delete[] output_data;
  delete pooling_param;
  MS_LOG(INFO) << "TestAvgPoolingGradFp32 passed";
}

TEST_F(TestPoolingGradFp32, AvgPoolingKernelGradFp32) {
  // prepare stage
  auto pooling_param = new PoolingParameter();
  InitPoolingParamFP32(pooling_param);

  pooling_param->output_channel_ = 3;

  // runtime part
  printf("Calculating runtime cost...\n");
  // uint64_t time_avg = 0;
  size_t output_data_size =
    pooling_param->output_batch_ * pooling_param->output_channel_ * pooling_param->output_h_ * pooling_param->output_w_;

  size_t input_size;
  std::string input_path = "./test_data/pooling/avgpoolgradfp32_1_dy_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::vector<int> dim_dy({1, 28, 28, 3});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(input_data);

  std::string input1_path = "./test_data/pooling/avgpoolgradfp32_1_x_1_28_28_3.bin";
  input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input1_path.c_str(), &input_size));
  std::vector<int> dim_x({1, 28, 28, 3});
  lite::tensor::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.SetData(input_data);

  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &x_tensor};

  auto output_data = new float[output_data_size];
  std::vector<int> dim_dx({1, 28, 28, 3});
  lite::tensor::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.SetData(output_data);
  std::vector<lite::tensor::Tensor *> outputs = {&dx_tensor};

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_PoolingGrad};

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(pooling_param), NULL, desc, nullptr);

  kernel_obj->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./test_data/pooling/avgpoolgradfp32_1_dx_1_28_28_3.bin";
  lite::CompareOutput(output_data, output_path);

  // delete input_data;
  // delete[] output_data;
  delete pooling_param;
  MS_LOG(INFO) << "TestAvgPoolingGradFp32 passed";
}

TEST_F(TestPoolingGradFp32, MaxPoolingGradFp32) {
  // prepare stage
  auto pooling_param = new PoolingParameter();
  InitPoolingParamFP32(pooling_param);
  pooling_param->output_channel_ = 3;
  pooling_param->avg_pooling_ = false;
  pooling_param->max_pooling_ = true;
  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  size_t output_data_size =
    pooling_param->output_batch_ * pooling_param->output_channel_ * pooling_param->output_h_ * pooling_param->output_w_;

  size_t input_size;
  std::string i_path = "./test_data/pooling/maxpoolgradfp32_1_i_1_28_28_3.bin";
  auto ill_data = reinterpret_cast<int64_t *>(mindspore::lite::ReadFile(i_path.c_str(), &input_size));
  auto i_data = new int[output_data_size];
  for (uint32_t i = 0; i < output_data_size; i++) {
    i_data[i] = static_cast<int>(ill_data[i]);
  }

  std::string dy_path = "./test_data/pooling/maxpoolgradfp32_1_dy_1_28_28_3.bin";
  auto dy_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(dy_path.c_str(), &input_size));

  auto output_data = new float[output_data_size];
  // warm up loop
  for (int i = 0; i < 3; i++) {
    MaxPoolingGrad(dy_data, i_data, output_data, pooling_param);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    MaxPoolingGrad(dy_data, i_data, output_data, pooling_param);
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
  std::string output_path = "./test_data/pooling/maxpoolgradfp32_1_dx_1_28_28_3.bin";
  lite::CompareOutput(output_data, output_path);

  // delete input_data;
  delete pooling_param;
  delete[] output_data;
  MS_LOG(INFO) << "TestMaxPoolingGradFp32 passed";
}

#if 0
TEST_F(TestPoolingGradFp32, MaxPoolingKernelGradFp32) {
  // prepare stage
  auto maxpool = new PoolingParameter();
  InitPoolingParamFP32(maxpool);
  maxpool->avg_pooling_ = false;
  maxpool->max_pooling_ = true;
  maxpool->input_h_ = 30;
  maxpool->input_w_ = 30;
  maxpool->input_channel_ = 3;

  maxpool->output_batch_ = 1;
  maxpool->output_h_ = 10;
  maxpool->output_w_ = 10;
  maxpool->output_channel_ = 3;
  maxpool->stride_h_ = 3;
  maxpool->stride_w_ = 3;

  maxpool->pad_u_ = 0;
  maxpool->pad_d_ = 0;
  maxpool->pad_l_ = 0;
  maxpool->pad_r_ = 0;

  size_t input_size;
  size_t y_data_size = maxpool->output_batch_ * maxpool->output_channel_ * maxpool->output_h_ * maxpool->output_w_;

  auto x_data = reinterpret_cast<float *>(
    mindspore::lite::ReadFile("./test_data/pooling/maxpoolgradfp32_2_x_1_30_30_3.bin", &input_size));
  std::vector<int> dim_x({1, 30, 30, 3});
  lite::tensor::Tensor x_tensor(TypeId::kNumberTypeFloat32, dim_x);
  x_tensor.SetData(x_data);
  std::vector<lite::tensor::Tensor *> maxpool_inputs = {&x_tensor};

  auto y_data = new float[y_data_size];
  std::vector<int> dim_y({1, 10, 10, 3});
  lite::tensor::Tensor y_tensor(TypeId::kNumberTypeFloat32, dim_y);
  y_tensor.SetData(y_data);

  auto ind_data = new int[y_data_size];
  lite::tensor::Tensor ind_tensor(TypeId::kNumberTypeInt32, dim_y);
  ind_tensor.SetData(ind_data);

  std::vector<lite::tensor::Tensor *> maxpool_outputs = {&y_tensor, &ind_tensor};

  kernel::KernelKey maxpool_desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_Pooling};
  auto maxpool_creator = lite::KernelRegistry::GetInstance()->GetCreator(maxpool_desc);
  auto maxpoolobj = maxpool_creator(maxpool_inputs, maxpool_outputs, reinterpret_cast<OpParameter *>(maxpool),
                                    NULL, maxpool_desc);
  maxpoolobj->Run();

  printf("==================indices data=================\n");
  for (int i = 0; i < 10; i++) {
    std::cout << ind_data[i] << " ,";
  }
  std::cout << std::endl;

  auto pooling_param = new PoolingParameter();
  InitPoolingParamFP32(pooling_param);
  pooling_param->avg_pooling_ = false;
  pooling_param->max_pooling_ = true;
  pooling_param->input_h_ = 10;
  pooling_param->input_w_ = 10;
  pooling_param->input_channel_ = 3;

  pooling_param->output_batch_ = 1;
  pooling_param->output_h_ = 30;
  pooling_param->output_w_ = 30;
  pooling_param->output_channel_ = 3;

  // runtime part
  printf("Calculating runtime cost...\n");
  // uint64_t time_avg = 0;
  size_t output_data_size =
    pooling_param->output_batch_ * pooling_param->output_channel_ * pooling_param->output_h_ * pooling_param->output_w_;

  auto dy_data = reinterpret_cast<float *>(
    mindspore::lite::ReadFile("./test_data/pooling/maxpoolgradfp32_2_dy_1_10_10_3.bin", &input_size));
  std::vector<int> dim_dy({1, 3, 10, 10});
  lite::tensor::Tensor dy_tensor(TypeId::kNumberTypeFloat32, dim_dy);
  dy_tensor.SetData(dy_data);

#if 0
  std::string i_path = "./test_data/pooling/maxpoolgradfp32_2_i_1_3_10_10.bin";
  auto ill_data = reinterpret_cast<int64_t*>(mindspore::lite::ReadFile(i_path.c_str(), &input_size));
  auto i_data = new int[output_data_size];
  for (int i=0; i < output_data_size; i++)
    i_data[i] = static_cast<int>(ill_data[i]);
  std::vector<int> dim_ind({1, 3, 10, 10});
  lite::tensor::Tensor ind_tensor(TypeId::kNumberTypeInt32, dim_ind);
  ind_tensor.SetData(i_data);
#endif

  std::vector<lite::tensor::Tensor *> inputs = {&dy_tensor, &ind_tensor};

  auto output_data = new float[output_data_size];
  std::vector<int> dim_dx({1, 3, 30, 30});
  lite::tensor::Tensor dx_tensor(TypeId::kNumberTypeFloat32, dim_dx);
  dx_tensor.SetData(output_data);
  std::vector<lite::tensor::Tensor *> outputs = {&dx_tensor};

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_PoolingGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(pooling_param), NULL, desc);
  kernel_obj->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << " ,";
  }
  std::cout << std::endl;
  std::string output_path = "./test_data/pooling/maxpoolgradfp32_2_dx_1_30_30_3.bin";
  lite::CompareOutput(output_data, output_path);

  // delete input_data;
  // delete[] output_data;
  delete pooling_param;
  MS_LOG(INFO) << "TestMaxPoolingKernelGradFp32 passed";
}
#endif  // if 0 before MaxPoolingKernelGradFp32
}  // namespace mindspore
