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
#include "mindspore/lite/src/runtime/kernel/arm/fp32/quantize.h"
#include "mindspore/lite/src/runtime/kernel/arm/opclib/fp32/quantize.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {

class QuantizeTestFp32 : public mindspore::Common {
 public:
  QuantizeTestFp32() {}
};

TEST_F(QuantizeTestFp32, QuantizeTest1) {
  const lite::tensor::QuantArg quant_arg = {0.3515625, -57};
  QuantizeParameter param;
  param.op_parameter_.type_ = schema::PrimitiveType_OnnxInt8Quantize;

  std::vector<float> input = {1, 2, 5, 6, 10, -20, 3, 8, 18, 10, 3, 4, 11, 16, 15, 25};
  std::vector<int> in_shape = {1, 4, 4, 1};
  lite::tensor::Tensor input_tensor;
  input_tensor.SetData(input.data());
  input_tensor.set_shape(in_shape);
  input_tensor.SetFormat(schema::Format_NHWC);
  input_tensor.set_data_type(kNumberTypeFloat32);
  input_tensor.AddQuantParam(quant_arg);
  std::vector<lite::tensor::Tensor *> inputs_tensor;
  inputs_tensor.emplace_back(&input_tensor);

  const int out_size = 16;
  int8_t expect_out[16] = {-54, -51, -43, -40, -29, -114, -48, -34, -6, -29, -48, -46, -26, -11, -14, 14};
  std::vector<int8_t> output(16);
  std::vector<int> out_shape = {1, 4, 4, 1};
  lite::tensor::Tensor output_tensor;
  output_tensor.SetData(output.data());
  output_tensor.set_shape(out_shape);
  output_tensor.SetFormat(schema::Format_NHWC);
  output_tensor.set_data_type(kNumberTypeInt8);
  std::vector<lite::tensor::Tensor *> outputs_tensor;
  outputs_tensor.emplace_back(&output_tensor);

  lite::Context ctx;
  ctx.threadNum = 3;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_OnnxInt8Quantize};
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
