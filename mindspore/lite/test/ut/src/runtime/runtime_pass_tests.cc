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
extern void Nc4hw4PassReplace(std::vector<kernel::LiteKernel *> *kernels, std::vector<Tensor *> *tensors, size_t index);
}

class RuntimePass : public mindspore::CommonTest {
 public:
  RuntimePass() = default;
};

void Nc4hw4PassConstruct(std::vector<kernel::LiteKernel *> *kernels, std::vector<lite::Tensor *> *tensors,
                         lite::InnerContext *ctx) {
  lite::Tensor *conv_in_tensor = new lite::Tensor();
  tensors->push_back(conv_in_tensor);
  lite::Tensor *conv_weight = new lite::Tensor();
  tensors->push_back(conv_weight);
  lite::Tensor *conv_out_tensor = new lite::Tensor();
  tensors->push_back(conv_out_tensor);
  std::vector<lite::Tensor *> conv_in = {conv_in_tensor, conv_weight};
  std::vector<lite::Tensor *> conv_out = {conv_out_tensor};
  OpParameter *conv_param = new OpParameter();
  kernel::KernelKey conv_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Conv2DFusion};
  kernel::LiteKernel *conv_kernel = nullptr;
  lite::KernelRegistry::GetInstance()->GetKernel(conv_in, conv_out, ctx, nullptr, conv_desc, conv_param, &conv_kernel,
                                                 nullptr);
  kernels->push_back(conv_kernel);

  lite::Tensor *transpose_param_tensor = new lite::Tensor();
  tensors->push_back(transpose_param_tensor);
  lite::Tensor *transpose_out_tensor = new lite::Tensor();
  tensors->push_back(transpose_param_tensor);
  OpParameter *transpose_param = new OpParameter();
  kernel::KernelKey transpose_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Transpose};
  kernel::LiteKernel *transpose_kernel = nullptr;
  std::vector<lite::Tensor *> transpose_in = {conv_out_tensor, transpose_param_tensor};
  std::vector<lite::Tensor *> transpose_out = {transpose_out_tensor};
  lite::KernelRegistry::GetInstance()->GetKernel(transpose_in, transpose_out, ctx, nullptr, transpose_desc,
                                                 transpose_param, &transpose_kernel, nullptr);
  kernels->push_back(transpose_kernel);

  lite::Tensor *in_param_tensor = new lite::Tensor();
  tensors->push_back(in_param_tensor);
  lite::Tensor *in_out_tensor = new lite::Tensor();
  tensors->push_back(in_out_tensor);
  OpParameter *in_param = new OpParameter();
  kernel::KernelKey in_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_InstanceNorm};
  kernel::LiteKernel *in_kernel = nullptr;
  std::vector<lite::Tensor *> in_in = {transpose_out_tensor, in_param_tensor};
  std::vector<lite::Tensor *> in_out = {in_out_tensor};
  lite::KernelRegistry::GetInstance()->GetKernel(in_in, in_out, ctx, nullptr, in_desc, in_param, &in_kernel, nullptr);
  kernels->push_back(in_kernel);

  lite::Tensor *transpose2_param_tensor = new lite::Tensor();
  tensors->push_back(transpose_param_tensor);
  lite::Tensor *transpose2_out_tensor = new lite::Tensor();
  tensors->push_back(transpose_param_tensor);
  OpParameter *transpose2_param = new OpParameter();
  kernel::KernelKey transpose2_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Transpose};
  kernel::LiteKernel *transpose2_kernel = nullptr;
  std::vector<lite::Tensor *> transpose2_in = {in_out_tensor, transpose2_param_tensor};
  std::vector<lite::Tensor *> transpose2_out = {transpose2_out_tensor};
  lite::KernelRegistry::GetInstance()->GetKernel(transpose2_in, transpose2_out, ctx, nullptr, transpose2_desc,
                                                 transpose2_param, &transpose2_kernel, nullptr);
  kernels->push_back(transpose2_kernel);

  lite::Tensor *conv2_weight = new lite::Tensor();
  tensors->push_back(conv2_weight);
  lite::Tensor *conv2_out_tensor = new lite::Tensor();
  tensors->push_back(conv2_out_tensor);
  std::vector<lite::Tensor *> conv2_in = {transpose2_out_tensor, conv_weight};
  std::vector<lite::Tensor *> conv2_out = {conv2_out_tensor};
  OpParameter *conv2_param = new OpParameter();
  kernel::KernelKey conv2_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_Conv2DFusion};
  kernel::LiteKernel *conv2_kernel = nullptr;
  lite::KernelRegistry::GetInstance()->GetKernel(conv2_in, conv2_out, ctx, nullptr, conv2_desc, conv2_param,
                                                 &conv2_kernel, nullptr);
  kernels->push_back(conv2_kernel);

  conv_kernel->set_out_kernels({transpose_kernel});
  transpose_kernel->set_in_kernels({conv_kernel});
  transpose_kernel->set_out_kernels({in_kernel});
  in_kernel->set_in_kernels({transpose_kernel});
  in_kernel->set_out_kernels({transpose2_kernel});
  transpose2_kernel->set_in_kernels({in_kernel});
  transpose2_kernel->set_out_kernels({conv2_kernel});
  conv2_kernel->set_in_kernels({transpose2_kernel});
  return;
}

TEST_F(RuntimePass, Nc4hw4Pass1) {
  auto ctx = std::make_shared<lite::InnerContext>();
  std::vector<kernel::LiteKernel *> kernels;
  std::vector<lite::Tensor *> tensors;
  Nc4hw4PassConstruct(&kernels, &tensors, ctx.get());

  ASSERT_EQ(kernels.size(), 5);

  /* runtime pass */
  lite::Nc4hw4PassReplace(&kernels, &tensors, 0);

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
}  // namespace mindspore
