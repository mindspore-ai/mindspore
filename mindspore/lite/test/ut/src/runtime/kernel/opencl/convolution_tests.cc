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
#include <memory>
#include "utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/convolution.h"

using mindspore::kernel::ConvolutionOpenCLKernel;
using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;

namespace mindspore {

class TestConvolutionOpenCL : public mindspore::CommonTest {};

void LoadData(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &dst_size));
    memcpy(dst, src_data, dst_size);
  }
}

void MyCompareOutput(lite::tensor::Tensor *output_tensor, const std::string &file_path) {
  auto *output_data = reinterpret_cast<float *>(output_tensor->Data());
  printf("output[0:10]:");
  for (int i = 0; i < 10; i++) {
    printf("%d:%.3f ", i, output_data[i]);
  }
  printf("\n");

  size_t output_size = output_tensor->Size();
  auto expect_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(file_path.c_str(), &output_size));
  constexpr float atol = 0.5;
  for (int i = 0; i < output_tensor->ElementsNum(); ++i) {
    if (std::fabs(output_data[i] - expect_data[i]) > atol) {
      printf("error at idx[%d] expect=%.3f output=%.3f\n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f\n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f\n\n\n", i, expect_data[i], output_data[i]);
      return;
    }
  }
  printf("compare success!\n");
  printf("compare success!\n");
  printf("compare success!\n\n\n");
}

void TEST_MAIN(ConvParameter *param, schema::Format data_format, const std::string &input_file,
               const std::string &weight_file, const std::string &bias_file, const std::string &expect_file) {
  assert(data_format == schema::Format_NHWC || data_format == schema::Format_NHWC4);

  std::cout << "initialize OpenCLRuntime";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  std::cout << "create inputs/weights/outputs Tensors(framework do)";
  std::vector<int> input_shape = {param->input_batch_, param->input_h_, param->input_w_, param->input_channel_};
  std::vector<int> weight_shape = {param->output_channel_, param->kernel_h_, param->kernel_w_, param->input_channel_};
  std::vector<int> bias_shape = {param->output_channel_};
  std::vector<int> output_shape = {param->output_batch_, param->output_h_, param->output_w_, param->output_channel_};
  auto data_type = kNumberTypeFloat32;
  auto tensorType = schema::NodeType_ValueNode;
  auto input_tensor = new lite::tensor::Tensor(data_type, input_shape, data_format, tensorType);
  auto weight_tensor = new lite::tensor::Tensor(data_type, weight_shape, schema::Format_KHWC, tensorType);
  auto bias_tensor = new lite::tensor::Tensor(data_type, bias_shape, schema::Format_KHWC, tensorType);
  auto output_tensor = new lite::tensor::Tensor(data_type, output_shape, data_format, tensorType);
  std::vector<lite::tensor::Tensor *> inputs{input_tensor, weight_tensor, bias_tensor};
  std::vector<lite::tensor::Tensor *> outputs{output_tensor};

  std::cout << "initialize weight Tensors data(framework do)";
  std::vector<float> weight_vec(weight_tensor->ElementsNum());
  std::vector<float> bias_vec(weight_tensor->ElementsNum());
  weight_tensor->SetData(weight_vec.data());
  bias_tensor->SetData(bias_vec.data());
  LoadData(weight_tensor->Data(), weight_tensor->Size(), weight_file);
  LoadData(bias_tensor->Data(), bias_tensor->Size(), bias_file);

  std::cout << "create OpenCL Kernel";  // weight has been allcated by framework
  auto *conv_kernel = new ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  conv_kernel->Init();
  std::vector<LiteKernel *> kernels{conv_kernel};

  // freamework to do!!! allocate memory by hand
  inputs[0]->MallocData(allocator);

  std::cout << "create SubGraphOpenCLKernel";
  auto *sub_graph = new SubGraphOpenCLKernel({input_tensor}, outputs, kernels, kernels, kernels);
  sub_graph->Init();

  std::cout << "initialize input Tensors data";  // inputs has been allcated by sub_graph->Init()
  LoadData(input_tensor->Data(), input_tensor->Size(), input_file);
  printf("input[0] =%.3f\n", reinterpret_cast<float *>(input_tensor->Data())[0]);
  printf("weight[0]=%.3f\n", reinterpret_cast<float *>(weight_tensor->Data())[0]);
  printf("bias[0]  =%.3f\n", reinterpret_cast<float *>(bias_tensor->Data())[0]);

  std::cout << "sub_graph->Run()";
  sub_graph->Run();
  printf("output_tensor->Size() =%zu\n", output_tensor->Size());

  std::cout << "compare result";
  MyCompareOutput(output_tensor, expect_file);
  //  lite::CompareOutput(reinterpret_cast<float *>(output_tensor->Data()), expect_file);

  mindspore::lite::opencl::OpenCLRuntime::DeleteInstance();
}

std::array<std::string, 4> GenFilenames(ConvParameter *param, schema::Format data_format, const std::string &path) {
  auto full_path = path + "inputNHWC_" + std::to_string(param->input_batch_) + "x" + std::to_string(param->input_h_) +
                   "x" + std::to_string(param->input_w_) + "x" + std::to_string(param->input_channel_) +
                   "_outputNHWC_" + std::to_string(param->output_batch_) + "x" + std::to_string(param->output_h_) +
                   "x" + std::to_string(param->output_w_) + "x" + std::to_string(param->output_channel_) +
                   "_kernelHW_" + std::to_string(param->kernel_h_) + "x" + std::to_string(param->kernel_w_) +
                   "_strideHW_" + std::to_string(param->stride_h_) + "x" + std::to_string(param->stride_w_) +
                   "_padTopBottomLeftRight_" + std::to_string(param->pad_u_) + "x" + std::to_string(param->pad_d_) +
                   "x" + std::to_string(param->pad_l_) + "x" + std::to_string(param->pad_r_) + "_dilationHW_1x1/";

  if (data_format == schema::Format_NHWC4) {
    return std::array<std::string, 4>{full_path + "input_NHWC4.bin", full_path + "weight_OHWI.bin",
                                      full_path + "bias_C4.bin", full_path + "expect_NHWC4.bin"};
  } else {
    return std::array<std::string, 4>{full_path + "input_NHWC.bin", full_path + "weight_OHWI.bin",
                                      full_path + "bias_C.bin", full_path + "expect_NHWC.bin"};
  }
}

TEST_F(TestConvolutionOpenCL, in1x224x224x3_out1x112x112x32_k33_s22_p0101) {
  auto param = new ConvParameter;
  param->input_batch_ = 1, param->input_h_ = 224, param->input_w_ = 224, param->input_channel_ = 3;
  param->output_batch_ = 1, param->output_h_ = 112, param->output_w_ = 112, param->output_channel_ = 32;
  param->kernel_h_ = 3, param->kernel_w_ = 3;
  param->stride_h_ = 2, param->stride_w_ = 2;
  param->pad_u_ = 0, param->pad_d_ = 1, param->pad_l_ = 0, param->pad_r_ = 1;

  auto filenames = GenFilenames(param, schema::Format_NHWC4, "testcases/mobilenetv2_fp32/");
  //  std::cout << filenames[0] << std::endl;
  //  std::cout << filenames[1] << std::endl;
  //  std::cout << filenames[2] << std::endl;
  //  std::cout << filenames[3] << std::endl;
  TEST_MAIN(param, schema::Format_NHWC4, filenames[0], filenames[1], filenames[2], filenames[3]);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestConvolutionOpenCL, in1x1x64x512_out1x1x64x7358_k11_s11_p0000) {
  auto param = new ConvParameter;
  param->input_batch_ = 1, param->input_h_ = 1, param->input_w_ = 64, param->input_channel_ = 512;
  param->output_batch_ = 1, param->output_h_ = 1, param->output_w_ = 64, param->output_channel_ = 7358;
  param->kernel_h_ = 1, param->kernel_w_ = 1;
  param->stride_h_ = 1, param->stride_w_ = 1;
  param->pad_u_ = 0, param->pad_d_ = 0, param->pad_l_ = 0, param->pad_r_ = 0;

  auto filenames = GenFilenames(param, schema::Format_NHWC4, "testcases/02_fp32/");
  //  std::cout << filenames[0] << std::endl;
  //  std::cout << filenames[1] << std::endl;
  //  std::cout << filenames[2] << std::endl;
  //  std::cout << filenames[3] << std::endl;
  TEST_MAIN(param, schema::Format_NHWC4, filenames[0], filenames[1], filenames[2], filenames[3]);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

}  // namespace mindspore
