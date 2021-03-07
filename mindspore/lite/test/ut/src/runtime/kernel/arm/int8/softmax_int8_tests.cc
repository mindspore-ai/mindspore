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
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/int8/softmax_int8.h"
#include "mindspore/lite/nnacl/softmax_parameter.h"
#include "mindspore/lite/src/kernel_registry.h"

namespace mindspore {

class TestSoftmaxInt8 : public mindspore::CommonTest {
 public:
  TestSoftmaxInt8() {}
};

TEST_F(TestSoftmaxInt8, SoftmaxInt8) {
  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  SoftmaxParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_Softmax;
  op_param.axis_ = 2;
  op_param.element_size_ = 24;
  op_param.input_shape_[0] = 1;
  op_param.input_shape_[1] = 2;
  op_param.input_shape_[2] = 3;
  op_param.input_shape_[3] = 4;

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 0.0352941;
  input_quant_arg.zeroPoint = -128;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 0.00392157;
  output_quant_arg.zeroPoint = -128;

  std::vector<int8_t> input = {-71,  -43, -15, 14,  -43, -15, 14, 42, 70, 99, 99, 127,
                               -100, -71, -43, -15, 14,  42,  70, 99, 42, 70, 99, 127};
  std::vector<int> in_shape = {1, 2, 3, 4};

  lite::Tensor input0_tensor;
  TypeId tid_int8 = kNumberTypeInt8;
  inputs_tensor.push_back(&input0_tensor);
  input0_tensor.set_data(input.data());
  input0_tensor.set_shape(in_shape);
  input0_tensor.AddQuantParam(input_quant_arg);
  input0_tensor.set_data_type(tid_int8);

  std::vector<int8_t> output(24);
  std::vector<int> output_shape = {1, 2, 3, 4};

  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.AddQuantParam(output_quant_arg);
  output0_tensor.set_data_type(tid_int8);

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_Softmax};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  std::vector<int8_t> except_result = {-126, -126, -124, -124, -123, -124, -116, -116, 122, 122, 112, 112,
                                       -127, -127, -127, -127, -59,  -59,  -61,  -59,  58,  58,  59,  58};
  ASSERT_EQ(0, CompareOutputData(output.data(), except_result.data(), input.size(), 0.000001));

  input0_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
}

}  // namespace mindspore
