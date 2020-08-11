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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/squeeze_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/ir/tensor.h"

namespace mindspore {

class TestSqueezeInt8 : public mindspore::CommonTest {
 public:
  TestSqueezeInt8() {}
};

TEST_F(TestSqueezeInt8, Squeeze_1d_axis0_offset0_quant0_thread2) {
  std::vector<int8_t> input1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> shape1 = {8, 1};
  std::vector<int8_t *> input(1, nullptr);
  input[0] = input1.data();

  const int output_size = 8;
  int8_t output[8];
  std::vector<int> output_shape = {8};
  lite::tensor::QuantArg input_quant_arg;
  input_quant_arg.scale = 1.0;
  input_quant_arg.zeroPoint = 0;
  lite::tensor::QuantArg output_quant_arg;
  output_quant_arg.scale = 1.0;
  output_quant_arg.zeroPoint = 0;

  lite::tensor::Tensor *input_tensor1 = new lite::tensor::Tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  input_tensor1->SetData(input1.data());
  input_tensor1->set_shape(shape1);
  input_tensor1->AddQuantParam(input_quant_arg);
  input_tensor1->set_data_type(tid_int8);

  std::vector<lite::tensor::Tensor *> inputs_tensor(1);
  inputs_tensor[0] = input_tensor1;

  std::vector<lite::tensor::Tensor *> outputs_tensor(1);
  lite::tensor::Tensor *output0_tensor = new lite::tensor::Tensor;
  output0_tensor->SetData(output);
  output0_tensor->set_shape(output_shape);
  output0_tensor->AddQuantParam(output_quant_arg);
  output0_tensor->set_data_type(tid_int8);
  outputs_tensor[0] = output0_tensor;

  SqueezeParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Squeeze;
  lite::Context *ctx = new lite::Context;
  ctx->thread_num_ = 2;
  op_param.axis_ = 0;
  op_param.offset_[0] = 1;
  op_param.offset_size_ = 1;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Squeeze};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor->shape();
  ASSERT_EQ(output_tensor_shape, output_shape);
  kernel->Run();

  std::vector<int8_t> except_result = {1, 2, 3, 4, 5, 6, 7, 8};
  PrintData("output data", output, output_size);
  PrintData("output data shape", output_tensor_shape.data(), output_tensor_shape.size());
  CompareOutputData(output, except_result.data(), output_size, 0.000001);

  input_tensor1->SetData(nullptr);
  output0_tensor->SetData(nullptr);
  delete input_tensor1;
  delete output0_tensor;
  delete ctx;
}

}  // namespace mindspore
