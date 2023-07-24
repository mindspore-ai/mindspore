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
#include "mindspore/lite/src/litert/kernel/cpu/int8/power_int8.h"
#include "nnacl/pow_parameter.h"
#include "mindspore/lite/src/litert/kernel_registry.h"

namespace mindspore {

class TestPowerInt8 : public mindspore::CommonTest {
 public:
  TestPowerInt8() {}
};

TEST_F(TestPowerInt8, normal) {
  PowParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_PowFusion;
  op_param.scale_ = 1;
  op_param.shift_ = 0;

  lite::LiteQuantParam input_quant_arg;
  input_quant_arg.scale = 0.0156863;
  input_quant_arg.zeroPoint = -128;

  lite::LiteQuantParam exp_quant_arg;
  exp_quant_arg.scale = 0.0156863;
  exp_quant_arg.zeroPoint = -128;

  lite::LiteQuantParam output_quant_arg;
  output_quant_arg.scale = 0.0352941;
  output_quant_arg.zeroPoint = -128;

  int8_t in0_data[] = {-64, -1, 63, 127};
  int8_t in1_data[] = {127, 63, -1, -64};
  lite::Tensor input0(kNumberTypeInt8, {1, 1, 1, 4});
  lite::Tensor input1(kNumberTypeInt8, {1, 1, 1, 4});
  memcpy(input0.MutableData(), in0_data, input0.Size());
  memcpy(input1.MutableData(), in1_data, input1.Size());
  input0.AddQuantParam(input_quant_arg);
  input1.AddQuantParam(exp_quant_arg);
  std::vector<lite::Tensor *> inputs_tensor = {&input0, &input1};

  lite::Tensor output(kNumberTypeInt8, {1, 1, 1, 4});
  output.MallocData();
  output.AddQuantParam(output_quant_arg);
  std::vector<lite::Tensor *> outputs_tensor = {&output};

  auto ctx = std::make_shared<lite::InnerContext>();
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, NHWC, schema::PrimitiveType_PowFusion};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);

  auto *kernel = creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), ctx.get(), desc);
  ASSERT_NE(kernel, nullptr);

  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::vector<int8_t> except_result = {-99, 95, 124, -14};
  int8_t *output_data = reinterpret_cast<int8_t *>(output.data());
  ASSERT_EQ(0, CompareOutputData(output_data, except_result.data(), output.ElementsNum(), 0.000001));

  kernel->set_parameter(nullptr);
  delete kernel;
}
}  // namespace mindspore
