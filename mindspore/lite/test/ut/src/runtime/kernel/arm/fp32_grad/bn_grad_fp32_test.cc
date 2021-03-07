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
#include "src/runtime/kernel/arm/fp32_grad/bn_grad.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "nnacl/fp32/batchnorm_fp32.h"
#include "src/kernel_registry.h"

namespace mindspore {

class TestBNGradFp32 : public mindspore::CommonTest {
 public:
  TestBNGradFp32() {}
  lite::Tensor *CreateInTensor(std::string file_name, std::vector<int> dim);
};

lite::Tensor *TestBNGradFp32::CreateInTensor(std::string file_name, std::vector<int> dim) {
  size_t input_size = 0;
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_name.c_str(), &input_size));
  auto tensor = new lite::Tensor(TypeId::kNumberTypeFloat32, dim);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "new tensor failed";
    return nullptr;
  }
  tensor->set_data(input_data);
  EXPECT_EQ(input_size, tensor->Size());
  return tensor;
}

TEST_F(TestBNGradFp32, BNGradFp32) {
  // prepare stage
  auto bn_param = static_cast<BNGradParameter *>(malloc(sizeof(BNGradParameter)));
  ASSERT_NE(bn_param, nullptr);

  bn_param->epsilon_ = 1e-2;
  const int batch = 2;
  const int channels = 3;
  const int height = 4;
  const int width = 5;

  auto dy_tensor = CreateInTensor("./test_data/bngrad/dy_2_4_5_3.bin", {batch, height, width, channels});
  ASSERT_NE(dy_tensor, nullptr);
  auto x_tensor = CreateInTensor("./test_data/bngrad/input_x_2_4_5_3.bin", {batch, height, width, channels});
  ASSERT_NE(x_tensor, nullptr);
  auto scale_tensor = CreateInTensor("./test_data/bngrad/scale_3.bin", {1, 1, 1, channels});
  ASSERT_NE(scale_tensor, nullptr);
  auto mean_tensor = CreateInTensor("./test_data/bngrad/save_mean_3.bin", {1, 1, 1, channels});
  ASSERT_NE(mean_tensor, nullptr);
  auto var_tensor = CreateInTensor("././test_data/bngrad/save_var_3.bin", {1, 1, 1, channels});
  ASSERT_NE(var_tensor, nullptr);

  // prepare output tensors
  lite::Tensor dx_tensor(TypeId::kNumberTypeFloat32, {batch, height, width, channels});
  ASSERT_EQ(dx_tensor.MallocData(), 0);
  lite::Tensor dscale_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(dscale_tensor.MallocData(), 0);
  lite::Tensor dbias_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(dbias_tensor.MallocData(), 0);

  std::vector<lite::Tensor *> inputs = {dy_tensor, x_tensor, scale_tensor, mean_tensor, var_tensor};
  std::vector<lite::Tensor *> outputs = {&dx_tensor, &dscale_tensor, &dbias_tensor};

  lite::InnerContext ctx;
  ctx.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx.Init());

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_BatchNormGrad};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(bn_param), &ctx, desc);
  ASSERT_NE(kernel_obj, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel_obj->workspace_size());

  kernel_obj->Run();
  std::cout << "==========dx==========\n";
  auto dx = reinterpret_cast<float *>(outputs[0]->MutableData());
  for (int i = 0; i < 7; i++) std::cout << dx[i] << " ";
  std::cout << "\n";
  auto res = CompareRelativeOutput(dx, "./test_data/bngrad/output_dx_2_4_5_3.bin");
  EXPECT_EQ(res, 0);
  std::cout << "\n=======dscale=======\n";
  auto dscale = reinterpret_cast<float *>(outputs[1]->MutableData());
  for (int i = 0; i < channels; i++) std::cout << dscale[i] << " ";
  std::cout << "\n";
  res = CompareRelativeOutput(dscale, "./test_data/bngrad/output_dscale_3.bin");
  EXPECT_EQ(res, 0);
  std::cout << "==========dbias==========\n";
  auto dbias = reinterpret_cast<float *>(outputs[2]->MutableData());
  for (int i = 0; i < 3; i++) std::cout << dbias[i] << " ";
  std::cout << "\n";
  res = CompareRelativeOutput(dbias, "./test_data/bngrad/output_dbias_3.bin");
  for (auto v : inputs) {
    delete[] reinterpret_cast<float *>(v->MutableData());
    v->set_data(nullptr);
    delete v;
  }
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel_obj;
  MS_LOG(INFO) << "BNGradFp32 passed";
}

TEST_F(TestBNGradFp32, BNTtrainFp32) {
  auto bn_param = static_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  ASSERT_NE(bn_param, nullptr);

  bn_param->epsilon_ = 1e-2;
  bn_param->momentum_ = 0.1;
  const int batch = 2;
  const int channels = 3;
  const int height = 4;
  const int width = 5;
  bn_param->channel_ = channels;
  auto x_tensor = CreateInTensor("./test_data/bngrad/input_x_2_4_5_3.bin", {batch, height, width, channels});

  lite::Tensor scale_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(scale_tensor.MallocData(), 0);
  auto scale = reinterpret_cast<float *>(scale_tensor.MutableData());
  std::fill(scale, scale + channels, 1.0f);

  lite::Tensor bias_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(bias_tensor.MallocData(), 0);
  auto bias = reinterpret_cast<float *>(bias_tensor.MutableData());
  std::fill(bias, bias + channels, 1.0f);

  lite::Tensor mean_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(mean_tensor.MallocData(), 0);
  auto mean = reinterpret_cast<float *>(mean_tensor.MutableData());
  std::fill(mean, mean + channels, 0.0f);

  lite::Tensor var_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(var_tensor.MallocData(), 0);
  auto var = reinterpret_cast<float *>(var_tensor.MutableData());
  std::fill(var, var + channels, 1.0f);

  std::vector<lite::Tensor *> inputs = {x_tensor, &scale_tensor, &bias_tensor, &mean_tensor, &var_tensor};

  lite::Tensor out_tensor(TypeId::kNumberTypeFloat32, {batch, height, width, channels});
  ASSERT_EQ(out_tensor.MallocData(), 0);

  lite::Tensor save_scale_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(save_scale_tensor.MallocData(), 0);

  lite::Tensor save_bias_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(save_bias_tensor.MallocData(), 0);

  lite::Tensor save_mean_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(save_mean_tensor.MallocData(), 0);

  lite::Tensor save_var_tensor(TypeId::kNumberTypeFloat32, {1, 1, 1, channels});
  ASSERT_EQ(save_var_tensor.MallocData(), 0);

  std::vector<lite::Tensor *> outputs = {&out_tensor, &save_scale_tensor, &save_bias_tensor, &save_mean_tensor,
                                         &save_var_tensor};

  kernel::KernelKey desc = {kernel::kCPU, TypeId::kNumberTypeFloat32, schema::PrimitiveType_FusedBatchNorm};

  mindspore::lite::InnerContext context;
  context.thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, context.Init());

  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  auto kernel_obj = creator(inputs, outputs, reinterpret_cast<OpParameter *>(bn_param), &context, desc);
  ASSERT_NE(kernel_obj, nullptr);
  mindspore::kernel::LiteKernel::AllocWorkspace(kernel_obj->workspace_size());
  float *save_mean = reinterpret_cast<float *>(save_mean_tensor.MutableData());
  float *save_var = reinterpret_cast<float *>(save_var_tensor.MutableData());
  for (int i = 0; i < channels; i++) {
    save_var[i] = 1.f;
    save_mean[i] = 0.f;
  }
  float *curr_mean = reinterpret_cast<float *>(mean_tensor.MutableData());
  float *curr_var = reinterpret_cast<float *>(var_tensor.MutableData());

  kernel_obj->Train();
  kernel_obj->set_trainable(true);
  kernel_obj->Run();

  std::cout << "================save_mean==============================\n";
  for (int i = 0; i < channels; i++) std::cout << curr_mean[i] << " ";
  std::cout << "\n";
  std::cout << "===============save_var==============================\n";
  for (int i = 0; i < channels; i++) std::cout << curr_var[i] << " ";
  std::cout << "\n";
  delete[] reinterpret_cast<float *>(x_tensor->MutableData());
  auto res = CompareRelativeOutput(curr_mean, "./test_data/bngrad/running_mean_3.bin");
  EXPECT_EQ(res, 0);
  res = CompareRelativeOutput(curr_var, "./test_data/bngrad/running_var_3.bin");
  EXPECT_EQ(res, 0);

  x_tensor->set_data(nullptr);
  delete x_tensor;
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete kernel_obj;
}
}  // namespace mindspore
