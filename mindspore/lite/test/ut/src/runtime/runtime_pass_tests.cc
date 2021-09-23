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
#include "common/common_test.h"
#include "src/lite_kernel.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_pass.h"
#include "nnacl/conv_parameter.h"

namespace mindspore {
namespace lite {
extern void Nc4hw4PassAct(std::vector<kernel::LiteKernel *> *kernels, std::vector<Tensor *> *tensors, int i);
extern void ConvNormC4PassAct(std::vector<kernel::LiteKernel *> *kernels);
}  // namespace lite

class RuntimePass : public mindspore::CommonTest {
 public:
  RuntimePass() = default;
};

void Nc4hw4PassConstruct(std::vector<kernel::LiteKernel *> *kernels, std::vector<lite::Tensor *> *tensors,
                         lite::InnerContext *ctx) {
  auto reg = lite::KernelRegistry::GetInstance();

  lite::Tensor *conv_in_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv_in_tensor);
  lite::Tensor *conv_weight = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv_weight);
  lite::Tensor *conv_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv_out_tensor);
  std::vector<lite::Tensor *> conv_in = {conv_in_tensor, conv_weight};
  std::vector<lite::Tensor *> conv_out = {conv_out_tensor};
  ConvParameter *conv_param = new ConvParameter();
  conv_param->op_parameter_.type_ = schema::PrimitiveType_Conv2DFusion;
  conv_param->group_ = 1;
  OpParameter *op_conv = reinterpret_cast<OpParameter *>(conv_param);
  kernel::KernelKey conv_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Conv2DFusion};
  kernel::LiteKernel *conv_kernel = nullptr;
  reg->GetKernel(conv_in, conv_out, ctx, nullptr, conv_desc, op_conv, &conv_kernel, nullptr);
  kernels->push_back(conv_kernel);

  lite::Tensor *trans_param_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(trans_param_tensor);
  lite::Tensor *trans_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(trans_param_tensor);
  OpParameter *trans_param = new OpParameter();
  trans_param->type_ = schema::PrimitiveType_Transpose;
  kernel::KernelKey trans_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Transpose};
  kernel::LiteKernel *trans_kernel = nullptr;
  std::vector<lite::Tensor *> trans_in = {conv_out_tensor, trans_param_tensor};
  std::vector<lite::Tensor *> trans_out = {trans_out_tensor};
  reg->GetKernel(trans_in, trans_out, ctx, nullptr, trans_desc, trans_param, &trans_kernel, nullptr);
  kernels->push_back(trans_kernel);

  lite::Tensor *in_param_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(in_param_tensor);
  lite::Tensor *in_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(in_out_tensor);
  OpParameter *in_param = new OpParameter();
  in_param->type_ = schema::PrimitiveType_InstanceNorm;
  kernel::KernelKey in_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_InstanceNorm};
  kernel::LiteKernel *in_kernel = nullptr;
  std::vector<lite::Tensor *> in_in = {trans_out_tensor, in_param_tensor};
  std::vector<lite::Tensor *> in_out = {in_out_tensor};
  reg->GetKernel(in_in, in_out, ctx, nullptr, in_desc, in_param, &in_kernel, nullptr);
  kernels->push_back(in_kernel);

  lite::Tensor *trans2_param_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(trans_param_tensor);
  lite::Tensor *trans2_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(trans_param_tensor);
  OpParameter *trans2_param = new OpParameter();
  trans2_param->type_ = schema::PrimitiveType_Transpose;
  kernel::KernelKey trans2_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Transpose};
  kernel::LiteKernel *trans2_kernel = nullptr;
  std::vector<lite::Tensor *> trans2_in = {in_out_tensor, trans2_param_tensor};
  std::vector<lite::Tensor *> trans2_out = {trans2_out_tensor};
  reg->GetKernel(trans2_in, trans2_out, ctx, nullptr, trans2_desc, trans2_param, &trans2_kernel, nullptr);
  kernels->push_back(trans2_kernel);

  lite::Tensor *conv2_weight = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv2_weight);
  lite::Tensor *conv2_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv2_out_tensor);
  std::vector<lite::Tensor *> conv2_in = {trans2_out_tensor, conv_weight};
  std::vector<lite::Tensor *> conv2_out = {conv2_out_tensor};
  ConvParameter *conv2_param = new ConvParameter();
  conv2_param->op_parameter_.type_ = schema::PrimitiveType_Conv2DFusion;
  conv2_param->group_ = 1;
  OpParameter *op2_conv = reinterpret_cast<OpParameter *>(conv2_param);
  kernel::KernelKey conv2_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Conv2DFusion};
  kernel::LiteKernel *conv2_kernel = nullptr;
  reg->GetKernel(conv2_in, conv2_out, ctx, nullptr, conv2_desc, op2_conv, &conv2_kernel, nullptr);
  kernels->push_back(conv2_kernel);

  conv_kernel->set_out_kernels({trans_kernel});
  trans_kernel->set_in_kernels({conv_kernel});
  trans_kernel->set_out_kernels({in_kernel});
  in_kernel->set_in_kernels({trans_kernel});
  in_kernel->set_out_kernels({trans2_kernel});
  trans2_kernel->set_in_kernels({in_kernel});
  trans2_kernel->set_out_kernels({conv2_kernel});
  conv2_kernel->set_in_kernels({trans2_kernel});
  return;
}

void ConvNormC4PassConstruct(std::vector<kernel::LiteKernel *> *kernels, std::vector<lite::Tensor *> *tensors,
                             lite::InnerContext *ctx) {
  auto reg = lite::KernelRegistry::GetInstance();

  lite::Tensor *conv1_in_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv1_in_tensor);
  lite::Tensor *conv1_weight = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv1_weight);
  lite::Tensor *conv1_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv1_out_tensor);
  std::vector<lite::Tensor *> conv1_in = {conv1_in_tensor, conv1_weight};
  std::vector<lite::Tensor *> conv1_out = {conv1_out_tensor};
  ConvParameter *conv1_param = new ConvParameter();
  conv1_param->op_parameter_.type_ = schema::PrimitiveType_Conv2DFusion;
  conv1_param->group_ = 1;
  OpParameter *op1_conv = reinterpret_cast<OpParameter *>(conv1_param);
  kernel::KernelKey conv1_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Conv2DFusion};
  kernel::LiteKernel *conv1_kernel = nullptr;
  reg->GetKernel(conv1_in, conv1_out, ctx, nullptr, conv1_desc, op1_conv, &conv1_kernel, nullptr);
  kernels->push_back(conv1_kernel);

  lite::Tensor *in_param_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(in_param_tensor);
  lite::Tensor *in_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(in_out_tensor);
  OpParameter *in_param = new OpParameter();
  in_param->type_ = schema::PrimitiveType_InstanceNorm;
  kernel::KernelKey in_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_InstanceNorm};
  kernel::LiteKernel *in_kernel = nullptr;
  std::vector<lite::Tensor *> in_in = {conv1_out_tensor, in_param_tensor};
  std::vector<lite::Tensor *> in_out = {in_out_tensor};
  reg->GetKernel(in_in, in_out, ctx, nullptr, in_desc, in_param, &in_kernel, nullptr);
  kernels->push_back(in_kernel);

  lite::Tensor *conv2_weight = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv2_weight);
  lite::Tensor *conv2_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(conv2_out_tensor);

  std::vector<lite::Tensor *> conv2_in = {in_out_tensor, conv2_weight};
  std::vector<lite::Tensor *> conv2_out = {conv2_out_tensor};
  ConvParameter *conv2_param = new ConvParameter();
  conv2_param->op_parameter_.type_ = schema::PrimitiveType_Conv2DFusion;
  conv2_param->group_ = 1;
  OpParameter *op2_conv = reinterpret_cast<OpParameter *>(conv2_param);
  kernel::KernelKey conv2_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Conv2DFusion};
  kernel::LiteKernel *conv2_kernel = nullptr;
  reg->GetKernel(conv2_in, conv2_out, ctx, nullptr, conv2_desc, op2_conv, &conv2_kernel, nullptr);
  kernels->push_back(conv2_kernel);

  lite::Tensor *act_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(act_out_tensor);
  std::vector<lite::Tensor *> act_in = {conv2_out_tensor};
  std::vector<lite::Tensor *> act_out = {act_out_tensor};
  OpParameter *act_param = new OpParameter();
  act_param->type_ = schema::PrimitiveType_Activation;
  kernel::KernelKey act_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Activation};
  kernel::LiteKernel *act_kernel = nullptr;
  reg->GetKernel(act_in, act_out, ctx, nullptr, act_desc, act_param, &act_kernel, nullptr);
  kernels->push_back(act_kernel);

  lite::Tensor *in2_param_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(in2_param_tensor);
  lite::Tensor *in2_out_tensor = new lite::Tensor(kNumberTypeFloat32, {1, 1, 1, 1}, NHWC);
  tensors->push_back(in2_out_tensor);
  OpParameter *in2_param = new OpParameter();
  in2_param->type_ = schema::PrimitiveType_InstanceNorm;
  kernel::KernelKey in2_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_InstanceNorm};
  kernel::LiteKernel *in2_kernel = nullptr;
  std::vector<lite::Tensor *> in2_in = {act_out_tensor, in2_param_tensor};
  std::vector<lite::Tensor *> in2_out = {in2_out_tensor};
  reg->GetKernel(in2_in, in2_out, ctx, nullptr, in2_desc, in2_param, &in2_kernel, nullptr);
  kernels->push_back(in2_kernel);

  conv1_kernel->set_out_kernels({in_kernel});
  in_kernel->set_in_kernels({conv1_kernel});
  in_kernel->set_out_kernels({conv2_kernel});
  conv2_kernel->set_in_kernels({in_kernel});
  conv2_kernel->set_out_kernels({act_kernel});
  act_kernel->set_in_kernels({conv2_kernel});
  act_kernel->set_out_kernels({in2_kernel});
  in2_kernel->set_in_kernels({act_kernel});
  return;
}

TEST_F(RuntimePass, Nc4hw4Pass1) {
  auto ctx = std::make_shared<lite::InnerContext>();
  std::vector<kernel::LiteKernel *> kernels;
  std::vector<lite::Tensor *> tensors;
  Nc4hw4PassConstruct(&kernels, &tensors, ctx.get());

  ASSERT_EQ(kernels.size(), 5);

  /* runtime pass */
  int i = 0;
  Nc4hw4PassAct(&kernels, &tensors, i);

  ASSERT_EQ(kernels.size(), 3);

  for (auto tensor : tensors) {
    delete tensor;
    tensor = nullptr;
  }
  for (auto kernel : kernels) {
    delete kernel;
    kernel = nullptr;
  }
}

TEST_F(RuntimePass, ConvNormC4Pass1) {
  auto ctx = std::make_shared<lite::InnerContext>();
  std::vector<kernel::LiteKernel *> kernels;
  std::vector<lite::Tensor *> tensors;
  ConvNormC4PassConstruct(&kernels, &tensors, ctx.get());

  ASSERT_EQ(kernels.size(), 5);
  ASSERT_EQ(tensors.size(), 10);

  /* runtime pass */
  lite::ConvNormC4PassAct(&kernels);

  ASSERT_EQ(kernels.size(), 5);
  ASSERT_EQ(tensors.size(), 10);
  ASSERT_EQ(tensors[0]->format(), NHWC);   /* conv1_in */
  ASSERT_EQ(tensors[1]->format(), NHWC);   /* conv1_weight */
  ASSERT_EQ(tensors[2]->format(), NC4HW4); /* conv1_out */
  ASSERT_EQ(tensors[3]->format(), NHWC);   /* instance_param */
  ASSERT_EQ(tensors[4]->format(), NHWC);   /* instance_out */
  ASSERT_EQ(tensors[5]->format(), NHWC);   /* conv2_weight */
  ASSERT_EQ(tensors[6]->format(), NC4HW4); /* conv2_out */
  ASSERT_EQ(tensors[7]->format(), NC4HW4); /* act_out */
  ASSERT_EQ(tensors[8]->format(), NHWC);   /* instance2_param */
  ASSERT_EQ(tensors[9]->format(), NHWC);   /* instance2_out */

  for (auto tensor : tensors) {
    delete tensor;
    tensor = nullptr;
  }
  for (auto kernel : kernels) {
    delete kernel;
    kernel = nullptr;
  }
}
}  // namespace mindspore
