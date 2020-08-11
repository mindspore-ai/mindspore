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
#include "mindspore/lite/src/common/file_utils.h"
#include "src/runtime/kernel/arm/nnacl/pack.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/depthwise_conv2d.h"


#define SAFE_DELETE_ARRAY(a) \
  if (a != nullptr) {        \
    delete[] a;              \
    a = nullptr;             \
  }
#define SAFE_DELETE_PTR(a) \
  if (a != nullptr) {      \
    delete a;              \
    a = nullptr;           \
  }

bool IMAGE2D_OPEN = true;

namespace mindspore {
class TestConvolutionDwOpenCL : public mindspore::CommonTest {
 public:
  TestConvolutionDwOpenCL(){}
};

void DepthWiseTestMain(ConvParameter *conv_param, float_t *input_data, float_t *weight_data, float_t *gnd_data,
                       schema::Format format, bool is_compare = true) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // pack input
  int IC4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int pack_input_size = C4NUM * IC4 * conv_param->input_h_ * conv_param->input_w_;
  float *packed_input = new float[pack_input_size];
  memset(packed_input, 0, pack_input_size * sizeof(float));
  int plane = conv_param->input_w_ * conv_param->input_h_;
  if (format == schema::Format_NHWC4) {
    PackNHWCToNHWC4Fp32(input_data, packed_input, 1, plane, conv_param->input_channel_);
  } else {
    PackNHWCToNC4HW4Fp32(input_data, packed_input, 1, plane, conv_param->input_channel_);
  }

  // pack weight
  int OC4 = UP_DIV(conv_param->output_channel_, C4NUM);
  int pack_weight_size = conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_;
  float *packed_weight = weight_data;

  // float bias_data[] = {0.31856894, 0.6674104, 0.13179787, 0.7163272, 0.2894061, 0.0, 0.0, 0.0};
  float bias_data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  size_t packed_output_size = conv_param->output_batch_ * C4NUM * UP_DIV(conv_param->output_channel_, C4NUM) *
                              conv_param->output_h_ * conv_param->output_w_;

  std::vector<int> shape_in = {conv_param->input_batch_, conv_param->input_h_, conv_param->input_w_,
                               conv_param->input_channel_};  // Note!!!actual is NHWC4
  std::vector<int> shape_filter = {1, conv_param->kernel_h_, conv_param->kernel_w_, conv_param->output_channel_};
  std::vector<int> shape_bias = {conv_param->output_channel_};
  std::vector<int> shape_out = {conv_param->output_batch_, conv_param->output_h_, conv_param->output_w_,
                                conv_param->output_channel_};
  lite::tensor::Tensor *tensor_a =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_in, format);  // Note!!!actual is NHWC4
  lite::tensor::Tensor *tensor_b =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_filter, schema::Format_NHWC);
  lite::tensor::Tensor *tensor_c =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_bias, schema::Format_NHWC);
  lite::tensor::Tensor *tensor_d = new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_out, format);
  std::vector<lite::tensor::Tensor *> inputs{tensor_a, tensor_b, tensor_c};
  std::vector<lite::tensor::Tensor *> outputs{tensor_d};

  // freamework to do!!!
  inputs[1]->SetData(packed_weight);
  inputs[2]->SetData(bias_data);

  OpParameter * parameter = reinterpret_cast<OpParameter *>(conv_param);
  auto *pKernel = new kernel::DepthwiseConv2dOpenCLKernel(parameter, inputs, outputs);
  pKernel->Init();

  std::vector<kernel::LiteKernel *> kernels{pKernel};
  std::vector<lite::tensor::Tensor *> inputs_{tensor_a};
  size_t C4 = UP_DIV(inputs[0]->Channel(), C4NUM);
  // if (IMAGE2D_OPEN && format == schema::Format_NHWC4) {
  //   std::vector<size_t> img_size{inputs[0]->Width() * C4, (size_t)inputs[0]->Height(), CL_FLOAT};
  //   auto in_data = allocator->Malloc(inputs[0]->Size(), img_size);
  //   inputs[0]->SetData(in_data);
  // } else if (IMAGE2D_OPEN && format == schema::Format_NC4HW4) {
  //   std::vector<size_t> img_size{(size_t)inputs[0]->Width(), inputs[0]->Height() * C4, CL_FLOAT};
  //   auto in_data = allocator->Malloc(inputs[0]->Size(), img_size);
  //   inputs[0]->SetData(in_data);
  // } else {
    inputs[0]->MallocData(allocator);
  // }
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs_, outputs, kernels, kernels, kernels);
  pGraph->Init();

  // freamework to do!!!
  memcpy(inputs[0]->Data(), packed_input, sizeof(float) * pack_input_size);

  pGraph->Run();
  if (is_compare) {
    float_t* packed_output = reinterpret_cast<float *>(outputs[0]->Data());
    float_t *packed_correct_data = new float_t[packed_output_size];
    memset(packed_correct_data, 0, packed_output_size * sizeof(float_t));
    if (format == schema::Format_NC4HW4) {
      PackNHWCToNC4HW4Fp32(gnd_data, packed_correct_data, conv_param->output_batch_,
                          conv_param->output_h_ * conv_param->output_w_, conv_param->output_channel_);
    } else {
      PackNHWCToNHWC4Fp32(gnd_data, packed_correct_data, conv_param->output_batch_,
                          conv_param->output_h_ * conv_param->output_w_, conv_param->output_channel_);
    }

    printf("==================input_data=================\n");
    std::cout << std::endl;
    for (int i = 0; i < pack_input_size; i++) {
      std::cout << packed_input[i] << ", ";
    }
    std::cout << std::endl;
    printf("==================weight data=================\n");
    std::cout << std::endl;
    for (int i = 0; i < pack_weight_size; i++) {
      std::cout << packed_weight[i] << ", ";
    }
    std::cout << std::endl;
    printf("==================output data=================\n");
    std::cout << std::endl;
    for (int i = 0; i < 80/*packed_output_size*/; i++) {
      std::cout << packed_output[i] << ", ";
    }
    std::cout << std::endl;
    printf("==================expected output data=================\n");
    for (int i = 0; i < packed_output_size; i++) {
      std::cout << packed_correct_data[i] << ", ";
    }
    std::cout << std::endl;
    // compare
    CommonTest::CompareOutputData(packed_output, packed_correct_data, packed_output_size, 0.00001);
    SAFE_DELETE_ARRAY(packed_correct_data)
  }

  inputs[1]->SetData(nullptr);
  inputs[2]->SetData(nullptr);
  SAFE_DELETE_ARRAY(packed_input);
  for (auto tensor : inputs) {
    SAFE_DELETE_PTR(tensor)
  }
  for (auto tensor : outputs) {
    SAFE_DELETE_PTR(tensor)
  }
  SAFE_DELETE_PTR(pKernel)
  SAFE_DELETE_PTR(pGraph)
  return;
}

TEST_F(TestConvolutionDwOpenCL, NoPadNC4HW4Fp32) {
  ConvParameter *conv_param = new ConvParameter();
  {
    conv_param->input_batch_ = 1;
    conv_param->input_h_ = 4;
    conv_param->input_w_ = 4;
    conv_param->input_channel_ = 4;
    conv_param->output_batch_ = 1;
    conv_param->output_h_ = 2;
    conv_param->output_w_ = 2;
    conv_param->output_channel_ = 4;
    conv_param->kernel_h_ = 3;
    conv_param->kernel_w_ = 3;
    conv_param->stride_h_ = 1;
    conv_param->stride_w_ = 1;
    conv_param->dilation_h_ = 1;
    conv_param->dilation_w_ = 1;
    conv_param->pad_h_ = 0;
    conv_param->pad_w_ = 0;
  }

  // nhwc
  float input_data[] = {0.5488135,  0.0202184,  0.45615032, 0.31542835, 0.71518934, 0.83261985, 0.56843394, 0.36371076,
                        0.60276335, 0.77815676, 0.0187898,  0.57019675, 0.5448832,  0.87001216, 0.6176355,  0.43860152,
                        0.4236548,  0.9786183,  0.6120957,  0.9883738,  0.6458941,  0.7991586,  0.616934,   0.10204481,
                        0.4375872,  0.46147937, 0.94374806, 0.20887676, 0.891773,   0.7805292,  0.6818203,  0.16130951,
                        0.96366274, 0.11827443, 0.3595079,  0.6531083,  0.3834415,  0.639921,   0.43703195, 0.2532916,
                        0.79172504, 0.14335328, 0.6976312,  0.46631077, 0.5288949,  0.9446689,  0.06022547, 0.2444256,
                        0.56804454, 0.5218483,  0.6667667,  0.15896958, 0.92559665, 0.41466194, 0.67063785, 0.11037514,
                        0.07103606, 0.2645556,  0.21038257, 0.6563296,  0.0871293,  0.7742337,  0.12892629, 0.13818295};

  // co h w ci
  float weight_data[] = {0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,
                         0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772,
                         0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051,
                         0.5759465,  0.9292962,  0.31856894, 0.6674104,  0.13179787, 0.7163272,  0.2894061,  0.18319136,
                         0.5865129,  0.02010755, 0.82894003, 0.00469548};

  // pack correct data, nhwc
  float gnd_data[] = {3.3848767, 1.4446403, 1.8428744, 1.3194335, 2.5873442, 2.1384869, 2.04022,  1.1872686,
                      2.2294958, 1.6570128, 2.465089,  1.4294086, 2.7941442, 1.7871612, 2.188921, 1.0601988};

  DepthWiseTestMain(conv_param, input_data, weight_data, gnd_data, schema::Format_NC4HW4);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestConvolutionDwOpenCL, PadNC4HW4Fp32) {
  ConvParameter *conv_param = new ConvParameter();
  {
    conv_param->input_batch_ = 1;
    conv_param->input_h_ = 3;
    conv_param->input_w_ = 3;
    conv_param->input_channel_ = 5;
    conv_param->output_batch_ = 1;
    conv_param->output_h_ = 3;
    conv_param->output_w_ = 3;
    conv_param->output_channel_ = 5;
    conv_param->kernel_h_ = 3;
    conv_param->kernel_w_ = 3;
    conv_param->stride_h_ = 1;
    conv_param->stride_w_ = 1;
    conv_param->dilation_h_ = 1;
    conv_param->dilation_w_ = 1;
    conv_param->pad_h_ = 1;
    conv_param->pad_w_ = 1;
  }

  // nhwc
  float input_data[] = {0.5488135,  0.3834415,  0.77815676, 0.9446689, 0.6120957,  0.71518934, 0.79172504, 0.87001216,
                        0.5218483,  0.616934,   0.60276335, 0.5288949, 0.9786183,  0.41466194, 0.94374806, 0.5448832,
                        0.56804454, 0.7991586,  0.2645556,  0.6818203, 0.4236548,  0.92559665, 0.46147937, 0.7742337,
                        0.3595079,  0.6458941,  0.07103606, 0.7805292, 0.45615032, 0.43703195, 0.4375872,  0.0871293,
                        0.11827443, 0.56843394, 0.6976312,  0.891773,  0.0202184,  0.639921,   0.0187898,  0.06022547,
                        0.96366274, 0.83261985, 0.14335328, 0.6176355, 0.6667667};
  // float input_data[]={
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  };
  // co h w ci
  float weight_data[] = {0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,
                         0.10204481, 0.20887676, 0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958,
                         0.11037514, 0.6563296,  0.13818295, 0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,
                         0.09609841, 0.97645944, 0.4686512,  0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696,
                         0.12019656, 0.2961402,  0.11872772, 0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146,
                         0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962};
  // float weight_data[]={
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 };
  // pack correct data, nhwc
  float gnd_data[] = {1.189188,   1.0425153,  1.8012011,  0.6074867,  1.2120346,  1.5005531,  0.8346756, 2.4365785,
                      0.54975945, 1.6815965,  1.2690231,  0.60214907, 1.6158017,  0.42115876, 0.8854959, 1.1709145,
                      1.0929465,  1.3534508,  1.1985044,  1.2932993,  2.4621446,  1.7086457,  2.6977584, 2.1960166,
                      2.3769147,  2.3185873,  0.6133741,  0.9687358,  0.9987654,  1.0254729,  0.8368954, 0.74171704,
                      0.8749627,  0.8953936,  0.5093431,  1.5496738,  0.54936385, 0.7683113,  1.165742,  1.3682933,
                      1.0517888,  0.59817517, 0.75649744, 1.2075498,  0.38804203};

  DepthWiseTestMain(conv_param, input_data, weight_data, gnd_data, schema::Format_NC4HW4);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestConvolutionDwOpenCL, NoPadNHWC4Fp32) {
  ConvParameter *conv_param = new ConvParameter();
  {
    conv_param->input_batch_ = 1;
    conv_param->input_h_ = 4;
    conv_param->input_w_ = 4;
    conv_param->input_channel_ = 4;
    conv_param->output_batch_ = 1;
    conv_param->output_h_ = 2;
    conv_param->output_w_ = 2;
    conv_param->output_channel_ = 4;
    conv_param->kernel_h_ = 3;
    conv_param->kernel_w_ = 3;
    conv_param->stride_h_ = 1;
    conv_param->stride_w_ = 1;
    conv_param->dilation_h_ = 1;
    conv_param->dilation_w_ = 1;
    conv_param->pad_h_ = 0;
    conv_param->pad_w_ = 0;
  }

  // nhwc
  float input_data[] = {0.5488135,  0.0202184,  0.45615032, 0.31542835, 0.71518934, 0.83261985, 0.56843394, 0.36371076,
                        0.60276335, 0.77815676, 0.0187898,  0.57019675, 0.5448832,  0.87001216, 0.6176355,  0.43860152,
                        0.4236548,  0.9786183,  0.6120957,  0.9883738,  0.6458941,  0.7991586,  0.616934,   0.10204481,
                        0.4375872,  0.46147937, 0.94374806, 0.20887676, 0.891773,   0.7805292,  0.6818203,  0.16130951,
                        0.96366274, 0.11827443, 0.3595079,  0.6531083,  0.3834415,  0.639921,   0.43703195, 0.2532916,
                        0.79172504, 0.14335328, 0.6976312,  0.46631077, 0.5288949,  0.9446689,  0.06022547, 0.2444256,
                        0.56804454, 0.5218483,  0.6667667,  0.15896958, 0.92559665, 0.41466194, 0.67063785, 0.11037514,
                        0.07103606, 0.2645556,  0.21038257, 0.6563296,  0.0871293,  0.7742337,  0.12892629, 0.13818295};

  // co h w ci
  float weight_data[] = {0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,
                         0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772,
                         0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051,
                         0.5759465,  0.9292962,  0.31856894, 0.6674104,  0.13179787, 0.7163272,  0.2894061,  0.18319136,
                         0.5865129,  0.02010755, 0.82894003, 0.00469548};

  // pack correct data, nhwc
  float gnd_data[] = {3.3848767, 1.4446403, 1.8428744, 1.3194335, 2.5873442, 2.1384869, 2.04022,  1.1872686,
                      2.2294958, 1.6570128, 2.465089,  1.4294086, 2.7941442, 1.7871612, 2.188921, 1.0601988};

  DepthWiseTestMain(conv_param, input_data, weight_data, gnd_data, schema::Format_NHWC4);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestConvolutionDwOpenCL, PadNHWC4Fp32) {
  ConvParameter *conv_param = new ConvParameter();
  {
    conv_param->input_batch_ = 1;
    conv_param->input_h_ = 3;
    conv_param->input_w_ = 3;
    conv_param->input_channel_ = 5;
    conv_param->output_batch_ = 1;
    conv_param->output_h_ = 3;
    conv_param->output_w_ = 3;
    conv_param->output_channel_ = 5;
    conv_param->kernel_h_ = 3;
    conv_param->kernel_w_ = 3;
    conv_param->stride_h_ = 1;
    conv_param->stride_w_ = 1;
    conv_param->dilation_h_ = 1;
    conv_param->dilation_w_ = 1;
    conv_param->pad_h_ = 1;
    conv_param->pad_w_ = 1;
  }

  // nhwc
  float input_data[] = {0.5488135,  0.3834415,  0.77815676, 0.9446689, 0.6120957,  0.71518934, 0.79172504, 0.87001216,
                        0.5218483,  0.616934,   0.60276335, 0.5288949, 0.9786183,  0.41466194, 0.94374806, 0.5448832,
                        0.56804454, 0.7991586,  0.2645556,  0.6818203, 0.4236548,  0.92559665, 0.46147937, 0.7742337,
                        0.3595079,  0.6458941,  0.07103606, 0.7805292, 0.45615032, 0.43703195, 0.4375872,  0.0871293,
                        0.11827443, 0.56843394, 0.6976312,  0.891773,  0.0202184,  0.639921,   0.0187898,  0.06022547,
                        0.96366274, 0.83261985, 0.14335328, 0.6176355, 0.6667667};
  // float input_data[]={
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  };
  // co h w ci
  float weight_data[] = {0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,
                         0.10204481, 0.20887676, 0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958,
                         0.11037514, 0.6563296,  0.13818295, 0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,
                         0.09609841, 0.97645944, 0.4686512,  0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696,
                         0.12019656, 0.2961402,  0.11872772, 0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146,
                         0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962};
  // float weight_data[]={
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 };
  // pack correct data, nhwc
  float gnd_data[] = {1.189188,   1.0425153,  1.8012011,  0.6074867,  1.2120346,  1.5005531,  0.8346756, 2.4365785,
                      0.54975945, 1.6815965,  1.2690231,  0.60214907, 1.6158017,  0.42115876, 0.8854959, 1.1709145,
                      1.0929465,  1.3534508,  1.1985044,  1.2932993,  2.4621446,  1.7086457,  2.6977584, 2.1960166,
                      2.3769147,  2.3185873,  0.6133741,  0.9687358,  0.9987654,  1.0254729,  0.8368954, 0.74171704,
                      0.8749627,  0.8953936,  0.5093431,  1.5496738,  0.54936385, 0.7683113,  1.165742,  1.3682933,
                      1.0517888,  0.59817517, 0.75649744, 1.2075498,  0.38804203};

  DepthWiseTestMain(conv_param, input_data, weight_data, gnd_data, schema::Format_NHWC4);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}


TEST_F(TestConvolutionDwOpenCL, ConvDwNoPadFp32) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  ConvParameter *conv_param = new ConvParameter();
  {
    conv_param->input_batch_ = 1;
    conv_param->input_h_ = 4;
    conv_param->input_w_ = 4;
    conv_param->input_channel_ = 4;
    conv_param->output_batch_ = 1;
    conv_param->output_h_ = 2;
    conv_param->output_w_ = 2;
    conv_param->output_channel_ = 4;
    conv_param->kernel_h_ = 3;
    conv_param->kernel_w_ = 3;
    conv_param->stride_h_ = 1;
    conv_param->stride_w_ = 1;
    conv_param->dilation_h_ = 1;
    conv_param->dilation_w_ = 1;
    conv_param->pad_h_ = 0;
    conv_param->pad_w_ = 0;
  }

  // nhwc
  float input_data[] = {0.5488135,  0.0202184,  0.45615032, 0.31542835, 0.71518934, 0.83261985, 0.56843394, 0.36371076,
                        0.60276335, 0.77815676, 0.0187898,  0.57019675, 0.5448832,  0.87001216, 0.6176355,  0.43860152,
                        0.4236548,  0.9786183,  0.6120957,  0.9883738,  0.6458941,  0.7991586,  0.616934,   0.10204481,
                        0.4375872,  0.46147937, 0.94374806, 0.20887676, 0.891773,   0.7805292,  0.6818203,  0.16130951,
                        0.96366274, 0.11827443, 0.3595079,  0.6531083,  0.3834415,  0.639921,   0.43703195, 0.2532916,
                        0.79172504, 0.14335328, 0.6976312,  0.46631077, 0.5288949,  0.9446689,  0.06022547, 0.2444256,
                        0.56804454, 0.5218483,  0.6667667,  0.15896958, 0.92559665, 0.41466194, 0.67063785, 0.11037514,
                        0.07103606, 0.2645556,  0.21038257, 0.6563296,  0.0871293,  0.7742337,  0.12892629, 0.13818295};

  // pack input
  int IC4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int pack_input_size = C4NUM * IC4 * conv_param->input_h_ * conv_param->input_w_;
  float *packed_input = input_data;

  // co h w ci
  float weight_data[] = {0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,
                         0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772,
                         0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051,
                         0.5759465,  0.9292962,  0.31856894, 0.6674104,  0.13179787, 0.7163272,  0.2894061,  0.18319136,
                         0.5865129,  0.02010755, 0.82894003, 0.00469548};

  // pack weight
  int OC4 = UP_DIV(conv_param->output_channel_, C4NUM);
  int pack_weight_size = C4NUM * OC4 * conv_param->kernel_h_ * conv_param->kernel_w_;
  float *packed_weight = weight_data;

  // float bias_data[] = {0.31856894, 0.6674104, 0.13179787, 0.7163272, 0.2894061, 0.0, 0.0, 0.0};
  float bias_data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  size_t packed_output_size = conv_param->output_batch_ * C4NUM * UP_DIV(conv_param->output_channel_, C4NUM) *
                              conv_param->output_h_ * conv_param->output_w_;

  std::vector<int> shape_in = {conv_param->input_batch_, conv_param->input_h_, conv_param->input_w_,
                               IC4 * C4NUM};  // Note!!!actual is NHWC4
  std::vector<int> shape_filter = {1, conv_param->kernel_h_, conv_param->kernel_w_, conv_param->output_channel_};
  std::vector<int> shape_bias = {conv_param->output_channel_};
  std::vector<int> shape_out = {conv_param->output_batch_, conv_param->output_h_, conv_param->output_w_,
                                conv_param->output_channel_};
  lite::tensor::Tensor *tensor_a =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_in, schema::Format_NC4HW4);  // Note!!!actual is NHWC4
  lite::tensor::Tensor *tensor_b =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_filter, schema::Format_NHWC);
  lite::tensor::Tensor *tensor_c =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_bias, schema::Format_NHWC);
  lite::tensor::Tensor *tensor_d =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_out, schema::Format_NC4HW4);
  std::vector<lite::tensor::Tensor *> inputs{tensor_a, tensor_b, tensor_c};
  std::vector<lite::tensor::Tensor *> outputs{tensor_d};

  // freamework to do!!!
  inputs[1]->SetData(packed_weight);
  inputs[2]->SetData(bias_data);

  OpParameter * parameter = reinterpret_cast<OpParameter *>(conv_param);
  auto *pKernel = new kernel::DepthwiseConv2dOpenCLKernel(parameter, inputs, outputs);
  pKernel->Init();

  std::vector<kernel::LiteKernel *> kernels{pKernel};
  std::vector<lite::tensor::Tensor *> inputs_{tensor_a};
  inputs[0]->MallocData();
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs_, outputs, kernels, kernels, kernels);
  pGraph->Init();

  // freamework to do!!!
  memcpy(inputs[0]->Data(), packed_input, sizeof(float) * pack_input_size);

  pGraph->Run();
  float *packed_output = reinterpret_cast<float *>(outputs[0]->Data());

  // pack correct data, nhwc
  float packed_correct_data[] = {3.3848767, 1.4446403, 1.8428744, 1.3194335, 2.5873442, 2.1384869, 2.04022,  1.1872686,
                                 2.2294958, 1.6570128, 2.465089,  1.4294086, 2.7941442, 1.7871612, 2.188921, 1.0601988};

  printf("==================input_data=================\n");
  std::cout << std::endl;
  for (int i = 0; i < pack_input_size; i++) {
    std::cout << packed_input[i] << ", ";
  }
  std::cout << std::endl;
  printf("==================packed_weight data=================\n");
  std::cout << std::endl;
  for (int i = 0; i < pack_weight_size; i++) {
    std::cout << packed_weight[i] << ", ";
  }
  std::cout << std::endl;
  printf("==================output data=================\n");
  std::cout << std::endl;
  for (int i = 0; i < packed_output_size; i++) {
    std::cout << packed_output[i] << ", ";
  }
  std::cout << std::endl;
  printf("==================expected output data=================\n");
  for (int i = 0; i < packed_output_size; i++) {
    std::cout << packed_correct_data[i] << ", ";
  }
  std::cout << std::endl;
  // compare
  CommonTest::CompareOutputData(packed_output, packed_correct_data, packed_output_size, 0.00001);

  inputs[1]->SetData(nullptr);
  inputs[2]->SetData(nullptr);
  for (auto tensor : inputs) {
    SAFE_DELETE_PTR(tensor)
  }
  for (auto tensor : outputs) {
    SAFE_DELETE_PTR(tensor)
  }
  SAFE_DELETE_PTR(pKernel)
  SAFE_DELETE_PTR(pGraph)
  MS_LOG(INFO) << "TestConvolutionDwNoPadFp32 passed";
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestConvolutionDwOpenCL, ConvDwPadFp32) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  ConvParameter *conv_param = new ConvParameter();
  {
    conv_param->input_batch_ = 1;
    conv_param->input_h_ = 3;
    conv_param->input_w_ = 3;
    conv_param->input_channel_ = 5;
    conv_param->output_batch_ = 1;
    conv_param->output_h_ = 3;
    conv_param->output_w_ = 3;
    conv_param->output_channel_ = 5;
    conv_param->kernel_h_ = 3;
    conv_param->kernel_w_ = 3;
    conv_param->stride_h_ = 1;
    conv_param->stride_w_ = 1;
    conv_param->dilation_h_ = 1;
    conv_param->dilation_w_ = 1;
    conv_param->pad_h_ = 1;
    conv_param->pad_w_ = 1;
  }

  // nhwc
  float input_data[] = {0.5488135,  0.3834415,  0.77815676, 0.9446689, 0.6120957,  0.71518934, 0.79172504, 0.87001216,
                        0.5218483,  0.616934,   0.60276335, 0.5288949, 0.9786183,  0.41466194, 0.94374806, 0.5448832,
                        0.56804454, 0.7991586,  0.2645556,  0.6818203, 0.4236548,  0.92559665, 0.46147937, 0.7742337,
                        0.3595079,  0.6458941,  0.07103606, 0.7805292, 0.45615032, 0.43703195, 0.4375872,  0.0871293,
                        0.11827443, 0.56843394, 0.6976312,  0.891773,  0.0202184,  0.639921,   0.0187898,  0.06022547,
                        0.96366274, 0.83261985, 0.14335328, 0.6176355, 0.6667667};
  // float input_data[]={
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  ,
  //   1  , 1  , 1 , 1  , 1  };

  // pack input
  int IC4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int pack_input_size = C4NUM * IC4 * conv_param->input_h_ * conv_param->input_w_;
  float *packed_input = new float[pack_input_size];
  memset(packed_input, 0, pack_input_size * sizeof(float));
  int plane = conv_param->input_w_ * conv_param->input_h_;
  PackNHWCToNC4HW4Fp32(input_data, packed_input, 1, plane, conv_param->input_channel_);

  // co h w ci
  float weight_data[] = {0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,
                         0.10204481, 0.20887676, 0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958,
                         0.11037514, 0.6563296,  0.13818295, 0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,
                         0.09609841, 0.97645944, 0.4686512,  0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696,
                         0.12019656, 0.2961402,  0.11872772, 0.31798318, 0.41426298, 0.06414749, 0.6924721,  0.56660146,
                         0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962};
  // float weight_data[]={
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 ,
  //   1  , 1  , 1 };

  // pack weight
  int OC4 = UP_DIV(conv_param->output_channel_, C4NUM);
  int pack_weight_size = conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_;
  float *packed_weight = weight_data;

  // float bias_data[] = {0.31856894, 0.6674104, 0.13179787, 0.7163272, 0.2894061, 0.0, 0.0, 0.0};
  float bias_data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  size_t packed_output_size = conv_param->output_batch_ * C4NUM * UP_DIV(conv_param->output_channel_, C4NUM) *
                              conv_param->output_h_ * conv_param->output_w_;

  std::vector<int> shape_in = {conv_param->input_batch_, conv_param->input_h_, conv_param->input_w_,
                               IC4 * C4NUM};  // Note!!!actual is NHWC4
  std::vector<int> shape_filter = {1, conv_param->kernel_h_, conv_param->kernel_w_, conv_param->output_channel_};
  std::vector<int> shape_bias = {conv_param->output_channel_};
  std::vector<int> shape_out = {conv_param->output_batch_, conv_param->output_h_, conv_param->output_w_,
                                conv_param->output_channel_};
  lite::tensor::Tensor *tensor_a =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_in, schema::Format_NC4HW4);  // Note!!!actual is NHWC4
  lite::tensor::Tensor *tensor_b =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_filter, schema::Format_NHWC);
  lite::tensor::Tensor *tensor_c =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_bias, schema::Format_NHWC);
  lite::tensor::Tensor *tensor_d =
    new lite::tensor::Tensor(TypeId(kNumberTypeFloat32), shape_out, schema::Format_NC4HW4);
  std::vector<lite::tensor::Tensor *> inputs{tensor_a, tensor_b, tensor_c};
  std::vector<lite::tensor::Tensor *> outputs{tensor_d};

  // freamework to do!!!
  inputs[1]->SetData(packed_weight);
  inputs[2]->SetData(bias_data);

  OpParameter * parameter = reinterpret_cast<OpParameter *>(conv_param);
  auto *pKernel = new kernel::DepthwiseConv2dOpenCLKernel(parameter, inputs, outputs);
  pKernel->Init();

  std::vector<kernel::LiteKernel *> kernels{pKernel};
  std::vector<lite::tensor::Tensor *> inputs_{tensor_a};
  inputs[0]->MallocData();
  auto *pGraph = new kernel::SubGraphOpenCLKernel(inputs_, outputs, kernels, kernels, kernels);
  pGraph->Init();

  // freamework to do!!!
  memcpy(inputs[0]->Data(), packed_input, sizeof(float) * pack_input_size);

  pGraph->Run();
  float *packed_output = reinterpret_cast<float *>(outputs[0]->Data());

  // pack correct data, nhwc
  float correct_data[] = {1.189188,   1.0425153,  1.8012011,  0.6074867,  1.2120346,  1.5005531,  0.8346756, 2.4365785,
                          0.54975945, 1.6815965,  1.2690231,  0.60214907, 1.6158017,  0.42115876, 0.8854959, 1.1709145,
                          1.0929465,  1.3534508,  1.1985044,  1.2932993,  2.4621446,  1.7086457,  2.6977584, 2.1960166,
                          2.3769147,  2.3185873,  0.6133741,  0.9687358,  0.9987654,  1.0254729,  0.8368954, 0.74171704,
                          0.8749627,  0.8953936,  0.5093431,  1.5496738,  0.54936385, 0.7683113,  1.165742,  1.3682933,
                          1.0517888,  0.59817517, 0.75649744, 1.2075498,  0.38804203};
  float *packed_correct_data = new float[packed_output_size];
  memset(packed_correct_data, 0, packed_output_size * sizeof(float));
  PackNHWCToNC4HW4Fp32(correct_data, packed_correct_data, conv_param->output_batch_,
                       conv_param->output_h_ * conv_param->output_w_, conv_param->output_channel_);

  printf("==================input_data=================\n");
  std::cout << std::endl;
  for (int i = 0; i < pack_input_size; i++) {
    std::cout << packed_input[i] << ", ";
  }
  std::cout << std::endl;
  printf("==================weight data=================\n");
  std::cout << std::endl;
  for (int i = 0; i < pack_weight_size; i++) {
    std::cout << packed_weight[i] << ", ";
  }
  std::cout << std::endl;
  printf("==================output data=================\n");
  std::cout << std::endl;
  for (int i = 0; i < packed_output_size; i++) {
    std::cout << packed_output[i] << ", ";
  }
  std::cout << std::endl;
  printf("==================expected output data=================\n");
  for (int i = 0; i < packed_output_size; i++) {
    std::cout << packed_correct_data[i] << ", ";
  }
  std::cout << std::endl;
  // compare
  CommonTest::CompareOutputData(packed_output, packed_correct_data, packed_output_size, 0.00001);

  inputs[1]->SetData(nullptr);
  inputs[2]->SetData(nullptr);
  SAFE_DELETE_ARRAY(packed_input);
  SAFE_DELETE_ARRAY(packed_correct_data)
  for (auto tensor : inputs) {
    SAFE_DELETE_PTR(tensor)
  }
  for (auto tensor : outputs) {
    SAFE_DELETE_PTR(tensor)
  }
  SAFE_DELETE_PTR(pKernel)
  SAFE_DELETE_PTR(pGraph)
  MS_LOG(INFO) << "TestConvolutionDwPadFp32 passed";
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestConvolutionDwOpenCL, ProfilingMobilenetv2) {
  std::vector<std::vector<int>> src_shape{
    {1, 32, 112, 112},
    {1, 96, 112, 112},
    {1, 144, 56, 56},
    {1, 144, 56, 56},
    {1, 192, 28, 28},
    {1, 192, 28, 28},
    {1, 384, 14, 14},
    {1, 576, 14, 14},
    {1, 576, 14, 14},
    {1, 960, 7, 7},
  };
  std::vector<std::vector<int>> dst_shape{
    {1, 32, 112, 112},
    {1, 96, 56, 56},
    {1, 144, 56, 56},
    {1, 144, 28, 28},
    {1, 192, 28, 28},
    {1, 192, 14, 14},
    {1, 384, 14, 14},
    {1, 576, 14, 14},
    {1, 576, 7, 7},
    {1, 960, 7, 7},
  };
  std::vector<std::vector<int>> filter_shape{
    {32, 1, 1, 1},
    {96, 3, 3, 1},
    {144, 1, 1, 1},
    {144, 3, 3, 1},
    {192, 1, 1, 1},
    {192, 3, 3, 1},
    {384, 1, 1, 1},
    {576, 1, 1, 1},
    {576, 3, 3, 1},
    {960, 1, 1, 1},
  };

  // nhwc
  size_t in_size = 96*112*112;
  float_t *input_data = new float_t[in_size];
  memset(input_data, 0, in_size);
  for (auto i = 0; i < in_size; ++i) {
    input_data[i] = 1;
  }
  // co h w ci
  size_t wt_size = 576*3*3;
  float_t *weight_data = new float_t[wt_size];
  memset(weight_data, 0, wt_size);
  for (auto i = 0; i < wt_size; ++i) {
    weight_data[i] = 1;
  }
  size_t out_size = 96*112*112;
  float_t *gnd_data = new float_t[out_size];
  memset(gnd_data, 0, out_size);
//  for (auto i = 0; i < in_size; ++i) {
//    gnd_data[i] = 1;
//  }
  for (size_t i = 0; i < src_shape.size(); ++i) {
    const int MAX_RUN_TIMES = 1;
    for (int j = 0; j < MAX_RUN_TIMES; ++j) {
      printf("========profiling depthwise, in shape(%d,%d,%d,%d), out shape(%d,%d,%d,%d), iter%d========\n",
        src_shape[i][0], src_shape[i][1], src_shape[i][2], src_shape[i][3],
        dst_shape[i][0], dst_shape[i][1], dst_shape[i][2], dst_shape[i][3], j);
      ConvParameter *conv_param = new ConvParameter();
      {
        conv_param->input_batch_    = 1;
        conv_param->input_h_        = src_shape[i][2];
        conv_param->input_w_        = src_shape[i][3];
        conv_param->input_channel_  = src_shape[i][1];
        conv_param->output_batch_   = 1;
        conv_param->output_h_       = dst_shape[i][2];
        conv_param->output_w_       = dst_shape[i][3];
        conv_param->output_channel_ = dst_shape[i][1];
        conv_param->kernel_h_       = filter_shape[i][1];
        conv_param->kernel_w_       = filter_shape[i][2];
        conv_param->stride_h_       = conv_param->output_h_/conv_param->input_h_;
        conv_param->stride_w_       = conv_param->output_w_/conv_param->input_w_;
        conv_param->pad_h_          = (conv_param->kernel_h_-1)/2;
        conv_param->pad_w_          = (conv_param->kernel_w_-1)/2;
        conv_param->dilation_h_     = 1;
        conv_param->dilation_w_     = 1;
      }
//      DepthWiseTestMain(conv_param, input_data, weight_data, gnd_data, schema::Format_NC4HW4, false);
       DepthWiseTestMain(conv_param, input_data, weight_data, nullptr, schema::Format_NHWC4, false);
    }
  }
  SAFE_DELETE_ARRAY(input_data);
  SAFE_DELETE_ARRAY(weight_data);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}

TEST_F(TestConvolutionDwOpenCL, Buffer2Image) {
  std::vector<int> src_shape{1, 96, 64, 64};
  std::vector<int> dst_shape{1, 96, 32, 32};
  std::vector<int> filter_shape{96, 3, 3, 1};

  // nhwc
  size_t in_size = 96*112*112;
  float_t *input_data = new float_t[in_size];
  memset(input_data, 0, in_size);
  for (auto i = 0; i < in_size; ++i) {
    input_data[i] = 1;
  }
  // co h w ci
  size_t wt_size = 576*3*3;
  float_t *weight_data = new float_t[wt_size];
  memset(weight_data, 0, wt_size);
  for (auto i = 0; i < wt_size; ++i) {
    weight_data[i] = 1;
  }
  size_t out_size = 96*112*112;
  float_t *gnd_data = new float_t[out_size];
  memset(gnd_data, 0, out_size);
//  for (auto i = 0; i < in_size; ++i) {
//    gnd_data[i] = 1;
//  }
    ConvParameter *conv_param = new ConvParameter();
    {
      conv_param->input_batch_    = 1;
      conv_param->input_h_        = src_shape[2];
      conv_param->input_w_        = src_shape[3];
      conv_param->input_channel_  = src_shape[1];
      conv_param->output_batch_   = 1;
      conv_param->output_h_       = dst_shape[2];
      conv_param->output_w_       = dst_shape[3];
      conv_param->output_channel_ = dst_shape[1];
      conv_param->kernel_h_       = filter_shape[1];
      conv_param->kernel_w_       = filter_shape[2];
      conv_param->stride_h_       = conv_param->output_h_/conv_param->input_h_;
      conv_param->stride_w_       = conv_param->output_w_/conv_param->input_w_;
      conv_param->pad_h_          = (conv_param->kernel_h_-1)/2;
      conv_param->pad_w_          = (conv_param->kernel_w_-1)/2;
      conv_param->dilation_h_     = 1;
      conv_param->dilation_w_     = 1;
    }
//      DepthWiseTestMain(conv_param, input_data, weight_data, gnd_data, schema::Format_NC4HW4, true);
      DepthWiseTestMain(conv_param, input_data, weight_data, gnd_data, schema::Format_NHWC4, true);
  SAFE_DELETE_ARRAY(input_data);
  SAFE_DELETE_ARRAY(weight_data);
  lite::opencl::OpenCLRuntime::DeleteInstance();
}
}  // namespace mindspore
