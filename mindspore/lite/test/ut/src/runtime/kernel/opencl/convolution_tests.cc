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

using mindspore::kernel::ConvolutionOpenCLKernel;
using mindspore::kernel::LiteKernel;
using mindspore::kernel::SubGraphOpenCLKernel;
using mindspore::lite::Tensor;
using mindspore::schema::Format;
using mindspore::schema::NodeType_ValueNode;
using mindspore::schema::Format::Format_KHWC;
using mindspore::schema::Format::Format_NC4HW4;
using mindspore::schema::Format::Format_NCHW;
using mindspore::schema::Format::Format_NHWC;
using mindspore::schema::Format::Format_NHWC4;

namespace mindspore {

class TestConvolutionOpenCL : public mindspore::CommonTest {};

void LoadData(Tensor *tensor, const float *src) {
  if (tensor->data_type() == kNumberTypeFloat16) {
    auto num = tensor->Size() / sizeof(float16_t);
    auto tensor_data = reinterpret_cast<float16_t *>(tensor->data_c());
    for (int i = 0; i < num; ++i) {
      tensor_data[i] = static_cast<float16_t>(src[i]);
    }
  } else {
    memcpy(tensor->data_c(), src, tensor->Size());
  }
}

void CompareOutput(Tensor *output, const float *expect_data, const float atol) {
  auto num = output->Size() / (output->data_type() == kNumberTypeFloat16 ? 2 : 4);
  std::vector<float> output_data(num);
  if (output->data_type() == kNumberTypeFloat16) {
    for (int i = 0; i < output_data.size(); ++i) {
      output_data[i] = static_cast<float>(reinterpret_cast<float16_t *>(output->data_c())[i]);
    }
  } else {
    memcpy(output_data.data(), output->data_c(), output->Size());
  }

  printf("output:");
  for (int i = 0; i < std::min(10, output->ElementsNum()); i++) {
    printf("%7.3f  ", output_data[i]);
  }
  printf("\n");

  float max_err = -1.0f;
  std::array<int, 5> idx_5d{};
  int max_err_idx = -1, first_err_idx = -1;
  auto SLICES = UP_DIV(output->Channel(), 4);
  int I = 1, J = 1, K = 1, L = 1, M = 1;
  switch (output->GetFormat()) {
    case Format_NHWC:
      I = output->Batch(), J = output->Height(), K = output->Width(), L = output->Channel();
      break;
    case Format_NCHW:
      I = output->Batch(), J = output->Channel(), K = output->Height(), L = output->Width();
      break;
    case Format_NHWC4:
      I = output->Batch(), J = output->Height(), K = output->Width(), L = SLICES, M = 4;
      break;
    case Format_NC4HW4:
      I = output->Batch(), J = SLICES, K = output->Height(), L = output->Width(), M = 4;
      break;
    default:
      break;
  }

  int cn = 0;
  for (int i = 0; i < I; ++i) {
    for (int j = 0; j < J; ++j) {
      for (int k = 0; k < K; ++k) {
        for (int l = 0; l < L; ++l) {
          for (int m = 0; m < M; ++m) {
            auto err = std::fabs(output_data[cn] - expect_data[cn]);
            if (first_err_idx == -1 && max_err > atol) {
              first_err_idx = cn;
            }
            if (err > max_err) {
              max_err = err;
              idx_5d = {i, j, k, l, m};
              max_err_idx = cn;
            }
            cn++;
          }
        }
      }
    }
  }

  if (max_err > atol) {
    printf("first error at %d expect=%.3f output=%.3f\n", first_err_idx, expect_data[first_err_idx],
           output_data[first_err_idx]);
    FAIL();
  } else {
    float relative_err = max_err / std::fabs(std::max(expect_data[max_err_idx], output_data[max_err_idx]));
    if (output->GetFormat() == Format_NHWC || output->GetFormat() == Format_NCHW) {
      printf("max relative error at [%d,%d,%d,%d]", idx_5d[0], idx_5d[1], idx_5d[2], idx_5d[3]);
    } else {
      printf("max relative error at [%d,%d,%d,%d,%d]", idx_5d[0], idx_5d[1], idx_5d[2], idx_5d[3], idx_5d[4]);
    }
    printf(" expect=%.3f output=%.3f absolute_err=%.2e relative_err=%.2f%%\n", expect_data[max_err_idx],
           output_data[max_err_idx], max_err, relative_err * 100);
    printf("COMPARE SUCCESS!\n\n");
  }
}

Format get_op_format(Format input_format) {
  switch (input_format) {
    case Format_NHWC:
    case Format_NHWC4:
      return Format_NHWC4;
    case Format_NCHW:
      return Format_NHWC4;
    default:
      return Format_NC4HW4;
  }
}

void TEST_MAIN(const std::string &attr, Format input_format, Format output_format, const TypeId data_type,
               const float atol, const float *input_data, const float *weight_data, const float *bias_data,
               const float *expect_data) {
  auto param = std::make_unique<ConvParameter>();
  if (param == nullptr) {
    MS_LOG(ERROR) << "ConvParameter create error.";
    return;
  }
  sscanf(attr.c_str(),
         "inputNHWC_%dx%dx%dx%d_outputNHWC_%dx%dx%dx%d_kernelHW_%dx%d_strideHW_%dx%d_padTopBottomLeftRight_%dx%dx%dx%d_"
         "dilationHW_%dx%d",
         &param->input_batch_, &param->input_h_, &param->input_w_, &param->input_channel_, &param->output_batch_,
         &param->output_h_, &param->output_w_, &param->output_channel_, &param->kernel_h_, &param->kernel_w_,
         &param->stride_h_, &param->stride_w_, &param->pad_u_, &param->pad_d_, &param->pad_l_, &param->pad_r_,
         &param->dilation_h_, &param->dilation_w_);

  MS_LOG(DEBUG) << "initialize OpenCLRuntime and OpenCLAllocator";
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  ocl_runtime->SetFp16Enable(data_type == kNumberTypeFloat16);
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(DEBUG) << "create Tensors";
  std::vector<int> input_shape = {param->input_batch_, param->input_h_, param->input_w_, param->input_channel_};
  std::vector<int> weight_shape = {param->output_channel_, param->kernel_h_, param->kernel_w_, param->input_channel_};
  std::vector<int> bias_shape = {param->output_channel_};
  std::vector<int> output_shape = {param->output_batch_, param->output_h_, param->output_w_, param->output_channel_};
  auto input = Tensor(data_type, input_shape, input_format, lite::TensorCategory(NodeType_ValueNode));
  auto weight = Tensor(data_type, weight_shape, Format_KHWC, lite::TensorCategory(NodeType_ValueNode));
  auto bias = Tensor(data_type, bias_shape, Format_KHWC, lite::TensorCategory(NodeType_ValueNode));
  auto output = Tensor(data_type, output_shape, output_format, lite::TensorCategory(NodeType_ValueNode));

  MS_LOG(DEBUG) << "allocate memory and initialize weight/bias";
  weight.MallocData();
  bias.MallocData();
  LoadData(&weight, weight_data);
  LoadData(&bias, bias_data);

  MS_LOG(DEBUG) << "create OpenCL Kernel";
  auto kernel =
    ConvolutionOpenCLKernel(reinterpret_cast<OpParameter *>(param.release()), {&input, &weight, &bias}, {&output});
  kernel.SetFormatType(get_op_format(input_format));
  kernel.Init();

  MS_LOG(DEBUG) << "create SubGraph";
  auto sub_graph = new (std::nothrow) SubGraphOpenCLKernel({&input}, {&output}, {&kernel}, {&kernel}, {&kernel});
  if (sub_graph == nullptr) {
    return;
  }
  input.MallocData(allocator);
  sub_graph->Init();
  LoadData(&input, input_data);
  sub_graph->Run();
  CompareOutput(&output, expect_data, atol);

  MS_LOG(DEBUG) << "release resources";
  weight.FreeData();
  bias.FreeData();
  input.SetData(nullptr);
  output.SetData(nullptr);
  delete sub_graph;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

void TEST_MAIN(const std::string &attr, Format input_format, Format output_format, const TypeId data_type,
               const float atol, const std::string &data_path) {
  auto testcase_path = data_path + "/" + attr + "/";
  std::map<Format, std::string> format_str{
    {Format_NCHW, "NCHW"}, {Format_NHWC, "NHWC"}, {Format_NHWC4, "NHWC4"}, {Format_NC4HW4, "NC4HW4"}};
  auto input_file = testcase_path + "input_" + format_str[input_format] + ".bin";
  auto weight_file = testcase_path + "weight_OHWI.bin";
  auto bias_file = testcase_path + "bias_C.bin";
  auto expect_file = testcase_path + "expect_" + format_str[output_format] + ".bin";
  MS_LOG(DEBUG) << "input_file  :" << input_file;
  MS_LOG(DEBUG) << "weight_file :" << weight_file;
  MS_LOG(DEBUG) << "bias_file   :" << bias_file;
  MS_LOG(DEBUG) << "expect_file :" << expect_file;

  size_t dst_size;
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_file.c_str(), &dst_size));
  auto weight_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(weight_file.c_str(), &dst_size));
  auto bias_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(bias_file.c_str(), &dst_size));
  auto expect_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(expect_file.c_str(), &dst_size));
  printf("input [0-3]: %7.3f  %7.3f  %7.3f\n", input_data[0], input_data[1], input_data[2]);
  printf("weight[0-3]: %7.3f  %7.3f  %7.3f\n", weight_data[0], weight_data[1], weight_data[2]);
  printf("bias  [0-3]: %7.3f  %7.3f  %7.3f\n", bias_data[0], bias_data[1], bias_data[2]);
  printf("expect[0-3]: %7.3f  %7.3f  %7.3f\n", expect_data[0], expect_data[1], expect_data[2]);

  TEST_MAIN(attr, input_format, output_format, data_type, atol, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, in1x224x224x3_out1x112x112x32_k33_s22_p0101) {
  std::string attr =
    "inputNHWC_1x224x224x3_outputNHWC_1x112x112x32_kernelHW_3x3_strideHW_2x2_padTopBottomLeftRight_0x1x0x1_dilationHW_"
    "1x1";
  TEST_MAIN(attr, Format_NC4HW4, Format_NC4HW4, kNumberTypeFloat32, 2e-6f, "testcases/mobilenetv2_fp32/");
  TEST_MAIN(attr, Format_NC4HW4, Format_NC4HW4, kNumberTypeFloat16, 2e-2f, "testcases/mobilenetv2_fp32/");
  TEST_MAIN(attr, Format_NHWC4, Format_NHWC4, kNumberTypeFloat32, 2e-6f, "testcases/mobilenetv2_fp32/");
  TEST_MAIN(attr, Format_NHWC4, Format_NHWC4, kNumberTypeFloat16, 2e-2f, "testcases/mobilenetv2_fp32/");
}

TEST_F(TestConvolutionOpenCL, winograd_inputNHWC_1x16x256x96_outputNHWC_1x16x256x80) {
  std::string attr =
    "inputNHWC_1x16x256x96_outputNHWC_1x16x256x80_kernelHW_3x3_strideHW_1x1_padTopBottomLeftRight_1x1x1x1_dilationHW_"
    "1x1";
  TEST_MAIN(attr, Format_NC4HW4, Format_NC4HW4, kNumberTypeFloat32, 1e-4f, "testcases/test_fp32/");
  TEST_MAIN(attr, Format_NC4HW4, Format_NC4HW4, kNumberTypeFloat16, 0.6f, "testcases/test_fp32/");
  TEST_MAIN(attr, Format_NHWC4, Format_NHWC4, kNumberTypeFloat32, 1e-4f, "testcases/test_fp32/");
  TEST_MAIN(attr, Format_NHWC4, Format_NHWC4, kNumberTypeFloat16, 0.6f, "testcases/test_fp32/");
}

TEST_F(TestConvolutionOpenCL, simple_test0_NHWC) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float bias_data[] = {0.0f, 0.0f};
  float expect_data[] = {1.0f, 1.0f, 5.0f, 5.0f, 9.0f, 9.0f, 13.0f, 13.0f};
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}
TEST_F(TestConvolutionOpenCL, simple_test0_NCHW) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 2.0f, 4.0f, 6.0f, 1.0f, 3.0f, 5.0f, 7.0f};
  float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float bias_data[] = {0.0f, 0.0f};
  float expect_data[] = {1.0f, 5.0f, 9.0f, 13.0f, 1.0f, 5.0f, 9.0f, 13.0f};
  TEST_MAIN(attr, Format_NCHW, Format_NCHW, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, Format_NCHW, Format_NCHW, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, simple_test0_NHWC4_and_NC4HW4) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 0.0f, 0.0f, 2.0f, 3.0f, 0.0f, 0.0f, 4.0f, 5.0f, 0.0f, 0.0f, 6.0f, 7.0f, 0.0f, 0.0f};
  float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float bias_data[] = {0.0f, 0.0f};
  float expect_data[] = {1.0f, 1.0f, 0.0f, 0.0f, 5.0f,  5.0f,  0.0f, 0.0f,
                         9.0f, 9.0f, 0.0f, 0.0f, 13.0f, 13.0f, 0.0f, 0.0f};
  TEST_MAIN(attr, Format_NHWC4, Format_NHWC4, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data,
            expect_data);
  TEST_MAIN(attr, Format_NHWC4, Format_NHWC4, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data,
            expect_data);
  TEST_MAIN(attr, Format_NC4HW4, Format_NC4HW4, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data,
            expect_data);
  TEST_MAIN(attr, Format_NC4HW4, Format_NC4HW4, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data,
            expect_data);
}

TEST_F(TestConvolutionOpenCL, simple_test1) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float bias_data[] = {0.5f, -0.5f};
  float expect_data[] = {2.5f, 3.5f, 8.5f, 17.5f, 14.5f, 31.5f, 20.5f, 45.5f};
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, simple_test2) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x1_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float bias_data[] = {0.0f};
  float expect_data[] = {28.0f, 18.0f, 22.0f, 13.0f};
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, simple_test3) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  float bias_data[] = {0.5f, -0.5f};
  float expect_data[] = {168.5f, 391.5f, 80.5f, 223.5f, 60.5f, 235.5f, 20.5f, 123.5f};
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, Format_NHWC, Format_NHWC, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

}  // namespace mindspore
