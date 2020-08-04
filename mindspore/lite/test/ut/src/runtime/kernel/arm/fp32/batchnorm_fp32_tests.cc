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
#include "mindspore/core/utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/runtime/kernel/arm/opclib/fp32/batchnorm.h"
#include "mindspore/lite/src/runtime/kernel/arm/opclib/fused_batchnorm.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"
#include "mindspore/lite/src/common/file_utils.h"

namespace mindspore {

class TestBatchnormFp32 : public mindspore::Common {
 public:
  TestBatchnormFp32() {}
};

TEST_F(TestBatchnormFp32, BNTest) {
  std::vector<float> in_data = {0.0669681, 0.959215, 0.252686,  0.613594,  0.811776,  0.139469,  0.322848,  0.118354,
                                0.082978,  0.399467, 0.961267,  0.0247456, 0.0714259, 0.0791484, 0.0648625, 0.561612,
                                0.412069,  0.311492, 0.46109,   0.377125,  0.369283,  0.0332446, 0.696142,  0.715973,
                                0.525524,  0.477265, 0.0336351, 0.751577,  0.377548,  0.964603,  0.0196834, 0.174865};
  std::vector<float> in_data1 = {0.855446, 0.821765, 0.281008, 0.0798653, 0.22294,  0.793782, 0.963222, 0.17851,
                                 0.667549, 0.274381, 0.592842, 0.216552,  0.190274, 0.237873, 0.610063, 0.307559,
                                 0.830007, 0.760957, 0.583265, 0.763793,  0.456372, 0.391378, 0.547915, 0.862198,
                                 0.510794, 0.826776, 0.515894, 0.30071,   0.404987, 0.184773};
  std::vector<float> in_data2 = {0.712438, 0.4927,   0.078419, 0.310429, 0.546871, 0.0667141, 0.874321, 0.0265647,
                                 0.685165, 0.732586, 0.952889, 0.506402, 0.540784, 0.131119,  0.357713, 0.678992,
                                 0.960839, 0.340706, 0.697678, 0.398146, 0.313321, 0.6485,    0.739153, 0.00190134,
                                 0.536842, 0.996873, 0.445276, 0.371212, 0.420397, 0.0930115};
  std::vector<float> in_data3(32, 1);
  std::vector<float> in_data4(32, 0);
  std::vector<lite::tensor::Tensor *> inputs_tensor;
  std::vector<lite::tensor::Tensor *> outputs_tensor;

  BatchNormParameter op_param;
  op_param.op_parameter_.type_ = schema::PrimitiveType_BatchNorm;
  op_param.epsilon_ = 0.001f;

  std::vector<int> in_shape = {1, 2, 4, 4};

  lite::tensor::Tensor input0_tensor;
  lite::tensor::Tensor input1_tensor;
  lite::tensor::Tensor input2_tensor;
  inputs_tensor.push_back(&input0_tensor);
  inputs_tensor.push_back(&input1_tensor);
  inputs_tensor.push_back(&input2_tensor);
  input0_tensor.SetData(in_data.data());
  input1_tensor.SetData(in_data1.data());
  input2_tensor.SetData(in_data2.data());
  input0_tensor.set_shape(in_shape);

  std::vector<float> output(32);
  std::vector<float> corr_out(32);
  std::vector<int> output_shape = {1, 2, 4, 4};

  lite::tensor::Tensor output0_tensor;
  outputs_tensor.push_back(&output0_tensor);
  output0_tensor.SetData(output.data());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_BatchNorm};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  lite::Context ctx;
  ctx.thread_num_ = 7;
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&op_param), &ctx, desc);
  ASSERT_NE(kernel, nullptr);
  auto output_tensor_shape = output0_tensor.shape();
  kernel->Run();

  FusedBatchNorm(in_data.data(), in_data3.data(), in_data4.data(), in_data1.data(), in_data2.data(), in_shape.data(),
                 0.001f, corr_out.data());

  printf("==================output data=================\n");
  for (int i = 0; i < 1 * 28; i++) {
    std::cout << output[i] << " ,";
  }
  std::cout << std::endl;
  CompareOutputData(output.data(), corr_out.data(), 32, 0.00001);

  input0_tensor.SetData(nullptr);
  input1_tensor.SetData(nullptr);
  input2_tensor.SetData(nullptr);
  output0_tensor.SetData(nullptr);
}
}  // namespace mindspore
