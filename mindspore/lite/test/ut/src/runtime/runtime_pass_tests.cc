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

  lite::Tensor *pad_param_tensor = new lite::Tensor();
  tensors->push_back(pad_param_tensor);
  lite::Tensor *pad_out_tensor = new lite::Tensor();
  tensors->push_back(pad_out_tensor);
  OpParameter *pad_param = new OpParameter();
  kernel::KernelKey pad_desc{kernel::kCPU, kNumberTypeFloat32, schema::PrimitiveType_PadFusion};
  kernel::LiteKernel *pad_kernel = nullptr;
  std::vector<lite::Tensor *> pad_in = {transpose_out_tensor, pad_param_tensor};
  std::vector<lite::Tensor *> pad_out = {pad_out_tensor};
  lite::KernelRegistry::GetInstance()->GetKernel(pad_in, pad_out, ctx, nullptr, pad_desc, pad_param, &pad_kernel,
                                                 nullptr);
  kernels->push_back(pad_kernel);

  conv_kernel->set_out_kernels({transpose_kernel});
  transpose_kernel->set_in_kernels({conv_kernel});
  transpose_kernel->set_out_kernels({pad_kernel});
  pad_kernel->set_in_kernels({transpose_kernel});
  return;
}

TEST_F(RuntimePass, Nc4hw4Pass1) {
  auto ctx = std::make_shared<lite::InnerContext>();
  std::vector<kernel::LiteKernel *> kernels;
  std::vector<lite::Tensor *> tensors;
  Nc4hw4PassConstruct(&kernels, &tensors, ctx.get());

  /* runtime pass */
  lite::Nc4hw4PassReplace(&kernels, &tensors, 0);

  ASSERT_EQ(kernels.size(), 2);
  ASSERT_EQ(tensors.size(), 5);

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
