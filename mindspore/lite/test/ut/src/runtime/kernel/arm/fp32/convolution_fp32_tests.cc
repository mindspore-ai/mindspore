/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/src/litert/kernel/cpu/base/convolution_base.h"
#include "nnacl/nnacl_manager.h"

namespace mindspore {
class TestConvolutionFp32 : public mindspore::CommonTest {
 public:
  TestConvolutionFp32() {}
};

void InitConvParam(ConvParameter *conv_param) {
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 2;
  conv_param->input_w_ = 2;
  conv_param->input_channel_ = 2;

  conv_param->output_batch_ = 1;
  conv_param->output_h_ = 2;
  conv_param->output_w_ = 2;
  conv_param->output_channel_ = 2;

  conv_param->group_ = 1;

  conv_param->kernel_h_ = 1;
  conv_param->kernel_w_ = 1;

  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;
  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;

  conv_param->pad_u_ = 0;
  conv_param->pad_l_ = 0;
  conv_param->pad_r_ = 0;
  conv_param->pad_d_ = 0;
}

void InitConvTensor(std::vector<lite::Tensor *> *inputs, std::vector<lite::Tensor *> *outputs) {
  float input_data[] = {0, 1, 2, 3, 4, 5, -6, -7};
  auto *input = new lite::Tensor;
  input->set_data_type(kNumberTypeFloat32);
  input->set_format(mindspore::NHWC);
  input->set_shape({1, 2, 2, 2});
  input->MallocData();
  memcpy(input->MutableData(), input_data, input->Size());

  float weight_data[] = {1, 1, 1, 1, 1, 1, 1, 1};
  auto *weight = new lite::Tensor;
  weight->set_data_type(kNumberTypeFloat32);
  weight->set_format(mindspore::NHWC);
  weight->set_shape({2, 1, 1, 2});
  weight->MallocData();
  memcpy(weight->MutableData(), weight_data, weight->Size());

  auto *bias = new lite::Tensor;
  bias->set_data_type(kNumberTypeFloat32);
  bias->set_shape({2});
  bias->MallocData();
  memset(bias->MutableData(), 0, bias->ElementsNum() * sizeof(float));

  auto *output = new lite::Tensor;
  output->set_data_type(kNumberTypeFloat32);
  output->set_shape({1, 2, 2, 2});
  output->MallocData();
  memset(output->MutableData(), 0, output->Size());

  inputs->push_back(input);
  inputs->push_back(weight);
  inputs->push_back(bias);
  outputs->push_back(output);
}

TEST_F(TestConvolutionFp32, conv1) {
  // prepare stage
  int thread_num = 1;
  auto conv_param = new ConvParameter();
  conv_param->op_parameter_.type_ = PrimType_Conv2DFusion;
  InitConvParam(conv_param);

  // init tensor
  std::vector<lite::Tensor *> inputs;
  std::vector<lite::Tensor *> outputs;
  InitConvTensor(&inputs, &outputs);

  // init ctx
  auto ctx = new InnerContext();
  ctx->thread_num_ = thread_num;
  conv_param->op_parameter_.thread_num_ = thread_num;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_Conv2DFusion};

  // register op
  auto kernel = nnacl::NNACLKernelRegistry(&conv_param->op_parameter_, inputs, outputs, ctx, desc);
  ASSERT_NE(kernel, nullptr);
  // op run
  auto ret = kernel->Prepare();
  EXPECT_EQ(0, ret);
  ret = kernel->Run();
  EXPECT_EQ(0, ret);

  std::cout << "==================output data=================" << std::endl;
  auto output_ptr = reinterpret_cast<float *>(outputs[0]->MutableData());
  for (int i = 0; i < outputs[0]->ElementsNum(); i++) {
    std::cout << output_ptr[i] << ", ";
  }
  std::cout << std::endl;

  // compare
  float correct_data[] = {1, 1, 5, 5, 9, 9, -13, -13};
  ASSERT_EQ(0, CompareOutputData(output_ptr, correct_data, outputs[0]->ElementsNum(), 0.0001));

  for (auto &in_t : inputs) {
    delete in_t;
  }
  for (auto &out_t : outputs) {
    delete out_t;
  }
  delete kernel;
}
}  // namespace mindspore
