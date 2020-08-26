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
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/convolution.h"
#include "nnacl/pack.h"
#include "nnacl/fp32/common_func.h"

using mindspore::kernel::ConvolutionOpenCLKernel;
using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;

namespace mindspore {

class TestConvolutionOpenCL : public mindspore::CommonTest {};

void LoadData(void *dst, size_t dst_size, const std::string &file_path) {
  if (file_path.empty()) {
    memset(dst, 0x00, dst_size);
  } else {
    auto src_data = mindspore::lite::ReadFile(file_path.c_str(), &dst_size);
    memcpy(dst, src_data, dst_size);
  }
}

void MyCompareOutput(lite::tensor::Tensor *output_tensor, const std::string &file_path, const TypeId data_type,
                     const float atol) {
  size_t output_size = output_tensor->Size();
  auto output_data_ori = output_tensor->Data();
  auto expect_data_ori = mindspore::lite::ReadFile(file_path.c_str(), &output_size);
  std::vector<float> output_data_vec(output_tensor->ElementsC4Num());
  std::vector<float> expect_data_vec(output_tensor->ElementsC4Num());
  float *output_data, *expect_data;
  if (data_type == kNumberTypeFloat16) {
    for (int i = 0; i < output_data_vec.size(); ++i) {
      output_data_vec[i] = ShortToFloat32(reinterpret_cast<uint16_t *>(output_data_ori)[i]);
      expect_data_vec[i] = ShortToFloat32(reinterpret_cast<uint16_t *>(expect_data_ori)[i]);
    }
    output_data = output_data_vec.data();
    expect_data = expect_data_vec.data();
  } else {
    output_data = reinterpret_cast<float *>(output_data_ori);
    expect_data = reinterpret_cast<float *>(expect_data_ori);
  }

  printf("\noutput[0:10]:");
  for (int i = 0; i < 10; i++) {
    printf("%d:%.3f ", i, output_data[i]);
  }
  printf("\n");

  for (int i = 0; i < output_tensor->ElementsNum(); ++i) {
    if (std::fabs(output_data[i] - expect_data[i]) > atol) {
      printf("error at idx[%d] expect=%.3f output=%.3f\n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f\n", i, expect_data[i], output_data[i]);
      printf("error at idx[%d] expect=%.3f output=%.3f\n\n\n", i, expect_data[i], output_data[i]);
      return;
    }
  }
  printf("COMPARE SUCCESS!\n\n\n");
}

void TEST_MAIN(schema::Format input_format, schema::Format output_format, const TypeId data_type,
               const std::string &data_path, std::string attr_str) {
  auto param = new (std::nothrow) ConvParameter;
  if (param == nullptr) {
    return;
  }
  sscanf(attr_str.c_str(),
         "inputNHWC_%dx%dx%dx%d_outputNHWC_%dx%dx%dx%d_kernelHW_%dx%d_strideHW_%dx%d_padTopBottomLeftRight_%dx%dx%dx%d_"
         "dilationHW_%dx%d",
         &param->input_batch_, &param->input_h_, &param->input_w_, &param->input_channel_, &param->output_batch_,
         &param->output_h_, &param->output_w_, &param->output_channel_, &param->kernel_h_, &param->kernel_w_,
         &param->stride_h_, &param->stride_w_, &param->pad_u_, &param->pad_d_, &param->pad_l_, &param->pad_r_,
         &param->dilation_h_, &param->dilation_w_);
  auto testcase_path = data_path + "/" + attr_str + "/";
  auto input_file = testcase_path + (input_format == schema::Format_NHWC4 ? "input_NHWC4.bin" : "input_NHWC.bin");
  auto weight_file = testcase_path + "weight_OHWI.bin";
  auto bias_file = testcase_path + "bias_C.bin";
  auto expect_file = testcase_path + (output_format == schema::Format_NHWC4 ? "expect_NHWC4.bin" : "expect_NHWC.bin");
  std::cout << "input_file  :" << input_file << std::endl;
  std::cout << "weight_file :" << weight_file << std::endl;
  std::cout << "bias_file   :" << bias_file << std::endl;
  std::cout << "expect_file :" << expect_file << std::endl;

  std::cout << "initialize OpenCLRuntime and OpenCLAllocator";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  ocl_runtime->SetFp16Enable(data_type == kNumberTypeFloat16);
  auto allocator = ocl_runtime->GetAllocator();

  std::cout << "create Tensors";
  std::vector<int> input_shape = {param->input_batch_, param->input_h_, param->input_w_, param->input_channel_};
  std::vector<int> weight_shape = {param->output_channel_, param->kernel_h_, param->kernel_w_, param->input_channel_};
  std::vector<int> bias_shape = {param->output_channel_};
  std::vector<int> output_shape = {param->output_batch_, param->output_h_, param->output_w_, param->output_channel_};
  auto tensor_type = schema::NodeType_ValueNode;
  auto input_tensor = lite::tensor::Tensor(data_type, input_shape, input_format, tensor_type);
  auto weight_tensor = lite::tensor::Tensor(data_type, weight_shape, schema::Format_KHWC, tensor_type);
  auto bias_tensor = lite::tensor::Tensor(data_type, bias_shape, schema::Format_KHWC, tensor_type);
  auto output_tensor = lite::tensor::Tensor(data_type, output_shape, output_format, tensor_type);
  std::vector<lite::tensor::Tensor *> inputs{&input_tensor, &weight_tensor, &bias_tensor};
  std::vector<lite::tensor::Tensor *> outputs{&output_tensor};

  std::cout << "allocate memory and initialize weight/bias";
  weight_tensor.MallocData();
  bias_tensor.MallocData();
  LoadData(weight_tensor.Data(), weight_tensor.Size(), weight_file);
  LoadData(bias_tensor.Data(), bias_tensor.Size(), bias_file);

  std::cout << "create OpenCL Kernel";
  auto kernel = ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  kernel.Init();

  std::cout << "create SubGraph";
  auto sub_graph = new (std::nothrow) SubGraphOpenCLKernel({&input_tensor}, outputs, {&kernel}, {&kernel}, {&kernel});
  if (sub_graph == nullptr) {
    return;
  }
  input_tensor.MallocData(allocator);  // before MapBuffer()
  sub_graph->Init();
  LoadData(input_tensor.Data(), input_tensor.Size(), input_file);  // after MapBuffer()
  if (data_type == kNumberTypeFloat16) {
    printf("input[0] =%.3f\n", ShortToFloat32(reinterpret_cast<uint16_t *>(input_tensor.Data())[0]));
    printf("weight[0]=%.3f\n", ShortToFloat32(reinterpret_cast<uint16_t *>(weight_tensor.Data())[0]));
    printf("bias[0]  =%.3f\n", ShortToFloat32(reinterpret_cast<uint16_t *>(bias_tensor.Data())[0]));
  } else {
    printf("input[0] =%.3f\n", reinterpret_cast<float *>(input_tensor.Data())[0]);
    printf("weight[0]=%.3f\n", reinterpret_cast<float *>(weight_tensor.Data())[0]);
    printf("bias[0]  =%.3f\n", reinterpret_cast<float *>(bias_tensor.Data())[0]);
  }
  sub_graph->Run();
  MyCompareOutput(&output_tensor, expect_file, data_type, (data_type == kNumberTypeFloat16 ? 0.7f : 0.1f));

  std::cout << "release resources";
  weight_tensor.FreeData();
  bias_tensor.FreeData();
  input_tensor.SetData(nullptr);
  output_tensor.SetData(nullptr);
  weight_tensor.SetData(nullptr);
  bias_tensor.SetData(nullptr);
  delete param;
  delete sub_graph;
}

TEST_F(TestConvolutionOpenCL, in1x224x224x3_out1x112x112x32_k33_s22_p0101_fp32) {
  TEST_MAIN(
    schema::Format_NHWC, schema::Format_NHWC4, kNumberTypeFloat32, "testcases/mobilenetv2_fp32/",
    "inputNHWC_1x224x224x3_outputNHWC_1x112x112x32_kernelHW_3x3_strideHW_2x2_padTopBottomLeftRight_0x1x0x1_dilationHW_"
    "1x1");
}

TEST_F(TestConvolutionOpenCL, in1x224x224x3_out1x112x112x32_k33_s22_p0101_fp16) {
  TEST_MAIN(
    schema::Format_NHWC, schema::Format_NHWC4, kNumberTypeFloat16, "testcases/mobilenetv2_fp16/",
    "inputNHWC_1x224x224x3_outputNHWC_1x112x112x32_kernelHW_3x3_strideHW_2x2_padTopBottomLeftRight_0x1x0x1_dilationHW_"
    "1x1");
}

TEST_F(TestConvolutionOpenCL, winograd_02_origin_inputNHWC_1x16x256x96_outputNHWC_1x16x256x80_fp32) {
  TEST_MAIN(schema::Format_NHWC, schema::Format_NHWC4, kNumberTypeFloat32, "testcases/test_fp32/",
            "inputNHWC_1x16x256x96_outputNHWC_1x16x256x80_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_1x1x1x1_"
            "dilationHW_1x1");
}

TEST_F(TestConvolutionOpenCL, winograd_02_origin_inputNHWC_1x16x256x96_outputNHWC_1x16x256x80_fp16) {
  TEST_MAIN(schema::Format_NHWC, schema::Format_NHWC4, kNumberTypeFloat16, "testcases/test_fp16/",
            "inputNHWC_1x16x256x96_outputNHWC_1x16x256x80_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_1x1x1x1_"
            "dilationHW_1x1");
}

}  // namespace mindspore
