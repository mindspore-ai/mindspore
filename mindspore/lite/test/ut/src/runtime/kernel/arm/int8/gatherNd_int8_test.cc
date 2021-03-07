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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/nnacl/fp32/gatherNd_fp32.h"
#include "mindspore/lite/nnacl/int8/gatherNd_int8.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {
class TestGatherNdInt8 : public mindspore::CommonTest {
 public:
  TestGatherNdInt8() {}
};

TEST_F(TestGatherNdInt8, GatherNdTest) {
  std::vector<int8_t> in_data = {3, 5, 7, 9, 11, 13, 15, 17, 19, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 1};
  std::vector<int8_t> in_data1 = {2, 4, 4, 2, 2, 4, 2, 4, 2};
  // std::vector<int8_t> in_data1 = {2, 2, 2, 4};

  std::vector<lite::Tensor *> inputs_tensor;
  std::vector<lite::Tensor *> outputs_tensor;

  GatherNdParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_GatherNd;
  std::vector<int> shape = {1, 2, 2, 5};
  std::vector<int> out_shape = {1, 3, 5};

  lite::QuantArg input_quant_arg;
  input_quant_arg.scale = 0.5;
  input_quant_arg.zeroPoint = 1;
  lite::QuantArg input_quant_arg_1;
  input_quant_arg_1.scale = 0.5;
  input_quant_arg_1.zeroPoint = 2;
  lite::QuantArg output_quant_arg;
  output_quant_arg.scale = 1;
  output_quant_arg.zeroPoint = 0;

  lite::Tensor input0_tensor;
  lite::Tensor input1_tensor;

  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);

  input0_tensor.set_data(in_data.data());
  input1_tensor.set_data(in_data1.data());

  input0_tensor.set_shape(shape);
  input1_tensor.set_shape({3, 3});

  input0_tensor.AddQuantParam(input_quant_arg);
  input1_tensor.AddQuantParam(input_quant_arg_1);

  std::vector<int8_t> output(15);
  // std::vector<int8_t> corr_out = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0};
  std::vector<int8_t> corr_out = {6, 7, 8, 9, 0, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};
  lite::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.set_data(output.data());
  output0_tensor.set_shape(out_shape);
  output0_tensor.AddQuantParam(output_quant_arg);

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeInt8, schema::PrimitiveType_GatherNd};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::InnerContext ctx;
  ctx.thread_num_ = 3;
  ASSERT_EQ(lite::RET_OK, ctx.Init());
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  printf("==================output data=================\n");
  for (int i = 0; i < output0_tensor.ElementsNum(); i++) {
    printf("%d, ", output[i]);
  }
  std::cout << std::endl;
  ASSERT_EQ(0, CompareOutputData(output.data(), corr_out.data(), output0_tensor.ElementsNum(), 0.001));

  input0_tensor.set_data(nullptr);
  input1_tensor.set_data(nullptr);
  output0_tensor.set_data(nullptr);
  MS_LOG(INFO) << "TestGatherNd accuracy passed";
}
}  // namespace mindspore
