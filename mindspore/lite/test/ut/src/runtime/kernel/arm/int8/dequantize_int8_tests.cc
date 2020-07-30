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
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/int8/dequantize.h"
#include "mindspore/lite/src/runtime/kernel/arm/opclib/int8/dequantize.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {

class DequantizeTestFp32 : public mindspore::Common {
 public:
  DequantizeTestFp32() {}
};

TEST_F(DequantizeTestFp32, DequantizeTest1) {
  const lite::tensor::QuantArg quant_arg{0.21176, 5};
  // quant_arg.scale = 100.0;
  // quant_arg.zeroPoint = 20;
  DequantizeParameter param;
  param.op_parameter_.type_ = schema::PrimitiveType_OnnxInt8Dequantize;

  std::vector<int8_t> input = {10, 14, 29, 33, 52, 99, 19, 43, 90, 52, 19, 24, 57, 127, 76, 123};
  // int8_t input0[] = {1, 2, 10};
  // int32_t a = input0[0] + 2;
  std::vector<int> in_shape = {1, 4, 4, 1};
  lite::tensor::Tensor input_tensor;
  input_tensor.SetData(input.data());
  input_tensor.set_shape(in_shape);
  input_tensor.set_data_type(kNumberTypeInt8);
  input_tensor.SetFormat(schema::Format_NHWC);

  input_tensor.AddQuantParam(quant_arg);
  std::vector<lite::tensor::Tensor *> inputs_tensor;
  inputs_tensor.emplace_back(&input_tensor);

  const int out_size = 16;
  float expect_out[16] = {3.1764,  4.02344,  7.19984,  8.04688, 12.07032, 22.02304, 5.08224,  10.16448,
                          20.1172, 12.07032, 5.082240, 6.14104, 13.12912, 27.95232, 17.15256, 27.10528};
  std::vector<float> output(16);
  std::vector<int> out_shape = {1, 4, 4, 1};
  lite::tensor::Tensor output_tensor;
  output_tensor.SetData(output.data());
  output_tensor.set_shape(out_shape);
  output_tensor.set_data_type(kNumberTypeFloat32);
  output_tensor.SetFormat(schema::Format_NHWC);
  std::vector<lite::tensor::Tensor *> outputs_tensor;
  outputs_tensor.emplace_back(&output_tensor);

  lite::Context ctx;
  ctx.threadNum = 3;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_OnnxInt8Dequantize};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();

  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  CompareOutputData(output.data(), expect_out, out_size, 0.000001);
}

}  // namespace mindspore
