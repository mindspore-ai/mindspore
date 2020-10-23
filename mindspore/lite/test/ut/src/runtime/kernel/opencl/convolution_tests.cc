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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
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
using mindspore::schema::Format::Format_NHWC;

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

  bool not_equal = false;
  int idx = 0;
  std::array<int, 4> idx_4d{};
  auto N = output->Batch(), H = output->Height(), W = output->Width(), C = output->Channel();
  for (int i = 0, cn = 0; i < N; ++i) {
    for (int j = 0; j < H; ++j) {
      for (int k = 0; k < W; ++k) {
        for (int l = 0; l < C; ++l) {
          auto err = std::fabs(output_data[cn] - expect_data[cn]);
          if (err > atol) {
            not_equal = true;
            idx_4d = {i, j, k, l};
            goto End;
          }
          cn++;
        }
      }
    }
  }

End:
  if (not_equal) {
    printf("first error at [%d %d %d %d] expect=%.3f output=%.3f\n", idx_4d[0], idx_4d[1], idx_4d[2], idx_4d[3],
           expect_data[idx], output_data[idx]);
    FAIL();
  } else {
    printf("COMPARE SUCCESS!\n\n");
  }
}

void TEST_MAIN(const std::string &attr, const TypeId data_type, const float atol, const float *input_data,
               const float *weight_data, const float *bias_data, const float *expect_data) {
  auto param = static_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
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
  auto runtime_wrapper = lite::opencl::OpenCLRuntimeWrapper();
  auto ocl_runtime = runtime_wrapper.GetInstance();
  ocl_runtime->Init();
  ocl_runtime->SetFp16Enable(data_type == kNumberTypeFloat16);
  auto allocator = ocl_runtime->GetAllocator();

  MS_LOG(DEBUG) << "create Tensors";
  std::vector<int> input_shape = {param->input_batch_, param->input_h_, param->input_w_, param->input_channel_};
  std::vector<int> weight_shape = {param->output_channel_, param->kernel_h_, param->kernel_w_, param->input_channel_};
  std::vector<int> bias_shape = {param->output_channel_};
  std::vector<int> output_shape = {param->output_batch_, param->output_h_, param->output_w_, param->output_channel_};
  auto input = Tensor(data_type, input_shape, Format_NHWC, lite::Tensor::CONST_TENSOR);
  auto weight = Tensor(data_type, weight_shape, Format_KHWC, lite::Tensor::CONST_TENSOR);
  auto bias = Tensor(data_type, bias_shape, Format_KHWC, lite::Tensor::CONST_TENSOR);
  auto output = Tensor(data_type, output_shape, Format_NHWC, lite::Tensor::CONST_TENSOR);

  MS_LOG(DEBUG) << "allocate memory and initialize weight/bias";
  weight.MallocData();
  LoadData(&weight, weight_data);
  if (bias_data) {
    bias.MallocData();
    LoadData(&bias, bias_data);
  }

  MS_LOG(DEBUG) << "create OpenCL Kernel";
  std::vector<lite::Tensor *> inputs{&input, &weight};
  if (bias_data) {
    inputs.push_back(&bias);
  }
  std::vector<lite::Tensor *> outputs{&output};
  auto kernel = std::make_unique<ConvolutionOpenCLKernel>(reinterpret_cast<OpParameter *>(param), inputs, outputs);
  kernel->Init();

  MS_LOG(DEBUG) << "create SubGraph";
  std::vector<kernel::LiteKernel *> kernels{kernel.release()};
  auto sub_graph = new (std::nothrow) SubGraphOpenCLKernel({&input}, {&output}, kernels, kernels, kernels);
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
  if (bias_data) {
    bias.FreeData();
  }
  delete sub_graph;
}

TEST_F(TestConvolutionOpenCL, test0) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float bias_data[] = {0.0f, 0.0f};
  float expect_data[] = {1.0f, 1.0f, 5.0f, 5.0f, 9.0f, 9.0f, 13.0f, 13.0f};
  TEST_MAIN(attr, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, test0_no_bias) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float expect_data[] = {1.0f, 1.0f, 5.0f, 5.0f, 9.0f, 9.0f, 13.0f, 13.0f};
  TEST_MAIN(attr, kNumberTypeFloat32, 1e-3f, input_data, weight_data, nullptr, expect_data);
  TEST_MAIN(attr, kNumberTypeFloat16, 1e-6f, input_data, weight_data, nullptr, expect_data);
}

TEST_F(TestConvolutionOpenCL, test1) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_1x1_strideHW_1x1_padTopBottomLeftRight_0x0x0x0_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float bias_data[] = {0.5f, -0.5f};
  float expect_data[] = {2.5f, 3.5f, 8.5f, 17.5f, 14.5f, 31.5f, 20.5f, 45.5f};
  TEST_MAIN(attr, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, test2) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x1_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x1x0x1_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  float bias_data[] = {0.0f};
  float expect_data[] = {28.0f, 18.0f, 22.0f, 13.0f};
  TEST_MAIN(attr, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, test3) {
  std::string attr =
    "inputNHWC_1x2x2x2_outputNHWC_1x2x2x2_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x1x0x1_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  float bias_data[] = {0.5f, -0.5f};
  float expect_data[] = {168.5f, 391.5f, 80.5f, 223.5f, 60.5f, 235.5f, 20.5f, 123.5f};
  TEST_MAIN(attr, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

TEST_F(TestConvolutionOpenCL, test3_batch2) {
  std::string attr =
    "inputNHWC_2x2x2x2_outputNHWC_2x2x2x2_kernelHW_2x2_strideHW_1x1_padTopBottomLeftRight_0x1x0x1_dilationHW_1x1";
  float input_data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
  float weight_data[] = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
  float bias_data[] = {0.5f, -0.5f};
  float expect_data[] = {168.5f, 391.5f, 80.5f, 223.5f, 60.5f, 235.5f, 20.5f, 123.5f,
                         168.5f, 391.5f, 80.5f, 223.5f, 60.5f, 235.5f, 20.5f, 123.5f};
  TEST_MAIN(attr, kNumberTypeFloat32, 1e-3f, input_data, weight_data, bias_data, expect_data);
  TEST_MAIN(attr, kNumberTypeFloat16, 1e-6f, input_data, weight_data, bias_data, expect_data);
}

}  // namespace mindspore
