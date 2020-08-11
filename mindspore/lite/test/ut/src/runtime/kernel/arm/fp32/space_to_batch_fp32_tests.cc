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
#include "mindspore/lite/src/runtime/kernel/arm/nnacl/fp32/space_to_batch.h"
#include "mindspore/lite/src/kernel_registry.h"
#include "mindspore/lite/src/lite_kernel.h"

namespace mindspore {

class SpaceToBatchTestFp32 : public mindspore::CommonTest {
 public:
  SpaceToBatchTestFp32() {}
};

void InitSpaceToBatchParameter(SpaceToBatchParameter *param) {
  param->n_dims_ = 4;
  param->n_space_dims_ = 2;

  param->block_sizes_[0] = 2;
  param->block_sizes_[1] = 2;

  param->paddings_[0] = 2;
  param->paddings_[1] = 0;
  param->paddings_[2] = 2;
  param->paddings_[3] = 2;

  param->in_shape_[0] = 1;
  param->in_shape_[1] = 4;
  param->in_shape_[2] = 4;
  param->in_shape_[3] = 1;

  param->padded_in_shape_[0] = 1;
  param->padded_in_shape_[1] = 6;
  param->padded_in_shape_[2] = 8;
  param->padded_in_shape_[3] = 1;

  param->num_elements_ = 16;
  param->num_elements_padded_ = 48;

  param->need_paddings_ = true;
}

void InitSpaceToBatchParameter2(SpaceToBatchParameter *param) {
  param->block_sizes_[0] = 2;
  param->block_sizes_[1] = 2;

  param->paddings_[0] = 2;
  param->paddings_[1] = 0;
  param->paddings_[2] = 2;
  param->paddings_[3] = 2;

  param->in_shape_[0] = 1;
  param->in_shape_[1] = 4;
  param->in_shape_[2] = 4;
  param->in_shape_[3] = 1;

  param->padded_in_shape_[0] = 1;
  param->padded_in_shape_[1] = 6;
  param->padded_in_shape_[2] = 8;
  param->padded_in_shape_[3] = 1;
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest1) {
  float input[16] = {1, 2, 5, 6, 10, 20, 3, 8, 18, 10, 3, 4, 11, 55, 15, 25};
  const int out_size = 16;
  float expect_out[16] = {1, 5, 18, 3, 2, 6, 10, 4, 10, 3, 11, 15, 20, 8, 55, 25};

  float output[16];
  int in_shape[4] = {1, 4, 4, 1};
  int out_shape[4] = {4, 2, 2, 1};
  int block_sizes[2] = {2, 2};
  SpaceToBatchForNHWC((const float *)input, output, in_shape, 4, block_sizes);
  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  CompareOutputData(output, expect_out, out_size, 0.000001);
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest2) {
  SpaceToBatchParameter param;
  InitSpaceToBatchParameter(&param);
  float input[16] = {1, 2, 5, 6, 10, 20, 3, 8, 18, 10, 3, 4, 11, 55, 15, 25};
  const int out_size = 48;
  float expect_out[48] = {0, 0, 0, 0, 0, 1,  5, 0, 0, 18, 3,  0, 0, 0, 0, 0, 0, 2,  6, 0, 0, 10, 4,  0,
                          0, 0, 0, 0, 0, 10, 3, 0, 0, 11, 15, 0, 0, 0, 0, 0, 0, 20, 8, 0, 0, 55, 25, 0};
  float output[48];
  int in_shape[4] = {1, 4, 4, 1};
  int out_shape[4] = {4, 3, 4, 1};
  int block_sizes[2] = {2, 2};

  float padded_input[48]{}, tmp[48]{}, tmp_zero[48]{};
  float *tmp_space[3] = {padded_input, tmp, tmp_zero};
  auto ret = SpaceToBatch((const float *)input, output, param, tmp_space);
  std::cout << "return " << ret << std::endl;
  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  CompareOutputData(output, expect_out, out_size, 0.000001);
}

TEST_F(SpaceToBatchTestFp32, SpaceToBatchTest3) {
  SpaceToBatchParameter param;
  InitSpaceToBatchParameter2(&param);
  param.op_parameter_.type_ = schema::PrimitiveType_SpaceToBatch;

  std::vector<float> input = {1, 2, 5, 6, 10, 20, 3, 8, 18, 10, 3, 4, 11, 55, 15, 25};
  std::vector<int> in_shape = {1, 4, 4, 1};
  lite::tensor::Tensor input_tensor;
  input_tensor.SetData(input.data());
  input_tensor.set_shape(in_shape);
  input_tensor.SetFormat(schema::Format_NHWC);
  input_tensor.set_data_type(kNumberTypeFloat32);
  std::vector<lite::tensor::Tensor *> inputs_tensor;
  inputs_tensor.emplace_back(&input_tensor);

  const int out_size = 48;
  float expect_out[48] = {0, 0, 0, 0, 0, 1,  5, 0, 0, 18, 3,  0, 0, 0, 0, 0, 0, 2,  6, 0, 0, 10, 4,  0,
                          0, 0, 0, 0, 0, 10, 3, 0, 0, 11, 15, 0, 0, 0, 0, 0, 0, 20, 8, 0, 0, 55, 25, 0};
  std::vector<float> output(48);
  std::vector<int> out_shape = {4, 3, 4, 1};
  lite::tensor::Tensor output_tensor;
  output_tensor.SetData(output.data());
  output_tensor.set_shape(out_shape);
  output_tensor.SetFormat(schema::Format_NHWC);
  output_tensor.set_data_type(kNumberTypeFloat32);
  std::vector<lite::tensor::Tensor *> outputs_tensor;
  outputs_tensor.emplace_back(&output_tensor);

  lite::Context ctx;
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, schema::PrimitiveType_SpaceToBatch};
  auto creator = lite::KernelRegistry::GetInstance()->GetCreator(desc);
  ASSERT_NE(creator, nullptr);
  kernel::LiteKernel *kernel =
    creator(inputs_tensor, outputs_tensor, reinterpret_cast<OpParameter *>(&param), &ctx, desc, nullptr);
  ASSERT_NE(kernel, nullptr);
  kernel->Run();

  for (int i = 0; i < out_size; ++i) {
    std::cout << output[i] << " ";
  }
  std::cout << "\n";
  CompareOutputData(output.data(), expect_out, out_size, 0.000001);
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
}

}  // namespace mindspore
