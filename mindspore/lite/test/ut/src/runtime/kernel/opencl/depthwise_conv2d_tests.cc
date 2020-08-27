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
#include "nnacl/pack.h"
#include "src/runtime/kernel/opencl/utils.h"
#include "mindspore/lite/src/runtime/opencl/opencl_runtime.h"
#include "mindspore/lite/src/runtime/kernel/opencl/subgraph_opencl_kernel.h"
#include "mindspore/lite/src/runtime/kernel/opencl/kernel/depthwise_conv2d.h"

namespace mindspore {
class TestConvolutionDwOpenCL : public mindspore::CommonTest {
 public:
  TestConvolutionDwOpenCL() {}
};

template <class T1, class T2>
void DepthWiseTestMain(ConvParameter *conv_param, T2 *input_data, T1 *weight_data, T2 *gnd_data, schema::Format format,
                       TypeId dtype = kNumberTypeFloat32, bool is_compare = true, T2 err_max = 1e-5) {
  auto ocl_runtime = lite::opencl::OpenCLRuntime::GetInstance();
  ocl_runtime->Init();
  auto allocator = ocl_runtime->GetAllocator();

  // pack input
  int IC4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int pack_input_size = C4NUM * IC4 * conv_param->input_h_ * conv_param->input_w_;
  auto packed_input = new (std::nothrow) T2[pack_input_size];
  if (packed_input == nullptr) {
    return;
  }
  memset(packed_input, 0, pack_input_size * sizeof(T2));
  int plane = conv_param->input_w_ * conv_param->input_h_;
  std::function<T2(T2)> to_dtype = [](T2 x) -> T2 { return x; };
  if (format == schema::Format_NHWC4) {
    kernel::PackNHWCToNHWC4<T2, T2>(input_data, packed_input, 1, plane, conv_param->input_channel_, to_dtype);
  } else {
    kernel::PackNHWCToNC4HW4<T2, T2>(input_data, packed_input, 1, plane, conv_param->input_channel_, to_dtype);
  }

  // pack weight
  int pack_weight_size = conv_param->output_channel_ * conv_param->kernel_h_ * conv_param->kernel_w_;
  T1 *packed_weight = weight_data;

  // T1 bias_data[] = {0.31856894, 0.6674104, 0.13179787, 0.7163272, 0.2894061, 0.0, 0.0, 0.0};
  T1 bias_data[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  size_t packed_output_size = conv_param->output_batch_ * C4NUM * UP_DIV(conv_param->output_channel_, C4NUM) *
                              conv_param->output_h_ * conv_param->output_w_;

  std::vector<int> shape_filter = {1, conv_param->kernel_h_, conv_param->kernel_w_, conv_param->output_channel_};
  std::vector<int> shape_bias = {conv_param->output_channel_};
  std::vector<int> shape_out;
  std::vector<int> shape_in;
  if (format == schema::Format_NHWC || format == schema::Format_NHWC4) {
    shape_in = std::vector<int>(
      {conv_param->input_batch_, conv_param->input_h_, conv_param->input_w_, conv_param->input_channel_});
    shape_out = std::vector<int>(
      {conv_param->output_batch_, conv_param->output_h_, conv_param->output_w_, conv_param->output_channel_});
  } else if (format == schema::Format_NCHW || format == schema::Format_NC4HW4) {
    shape_in = std::vector<int>(
      {conv_param->input_batch_, conv_param->input_channel_, conv_param->input_h_, conv_param->input_w_});
    shape_out = std::vector<int>(
      {conv_param->output_batch_, conv_param->output_channel_, conv_param->output_h_, conv_param->output_w_});
  } else {
    MS_LOG(ERROR) << "Unsupported format: " << format;
    delete[] packed_input;
    return;
  }
  auto tensor_a = lite::tensor::Tensor(TypeId(dtype), shape_in, format);
  auto tensor_b = lite::tensor::Tensor(TypeId(dtype), shape_filter, schema::Format_NHWC);
  auto tensor_c = lite::tensor::Tensor(TypeId(dtype), shape_bias, schema::Format_NHWC);
  auto tensor_d = lite::tensor::Tensor(TypeId(dtype), shape_out, format);
  std::vector<lite::tensor::Tensor *> inputs{&tensor_a, &tensor_b, &tensor_c};
  std::vector<lite::tensor::Tensor *> outputs{&tensor_d};

  // freamework to do!!!
  inputs[1]->SetData(packed_weight);
  inputs[2]->SetData(bias_data);

  OpParameter *parameter = reinterpret_cast<OpParameter *>(conv_param);
  auto pKernel = std::make_unique<kernel::DepthwiseConv2dOpenCLKernel>(parameter, inputs, outputs);
  if (pKernel.get() == nullptr) {
    delete[] packed_input;
    return;
  }
  pKernel->Init();

  std::vector<kernel::LiteKernel *> kernels{pKernel.get()};
  std::vector<lite::tensor::Tensor *> inputs_{&tensor_a};
  auto pGraph = std::make_unique<kernel::SubGraphOpenCLKernel>(inputs_, outputs, kernels, kernels, kernels);
  if (pGraph.get() == nullptr) {
    delete[] packed_input;
    return;
  }
  pGraph->Init();

  // freamework to do!!!
  inputs[0]->MallocData(allocator);
  memcpy(inputs[0]->Data(), packed_input, sizeof(T2) * pack_input_size);

  pGraph->Run();
  if (is_compare) {
    T2 *packed_output = reinterpret_cast<T2 *>(outputs[0]->Data());
    auto packed_correct_data = std::make_unique<T2>(packed_output_size);
    if (packed_correct_data.get() == nullptr) {
      delete[] packed_input;
      return;
    }
    memset(packed_correct_data.get(), 0, packed_output_size * sizeof(T2));
    if (format == schema::Format_NC4HW4) {
      kernel::PackNHWCToNC4HW4<T2, T2>(gnd_data, packed_correct_data.get(), conv_param->output_batch_,
                                       conv_param->output_h_ * conv_param->output_w_, conv_param->output_channel_,
                                       to_dtype);
    } else {
      kernel::PackNHWCToNHWC4<T2, T2>(gnd_data, packed_correct_data.get(), conv_param->output_batch_,
                                      conv_param->output_h_ * conv_param->output_w_, conv_param->output_channel_,
                                      to_dtype);
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
    for (int i = 0; i < packed_output_size; i++) {
      std::cout << packed_output[i] << ", ";
    }
    std::cout << std::endl;
    printf("==================expected output data=================\n");
    for (int i = 0; i < packed_output_size; i++) {
      std::cout << packed_correct_data.get()[i] << ", ";
    }
    std::cout << std::endl;
    // compare
    CommonTest::CompareOutputData<T2>(packed_output, packed_correct_data.get(), packed_output_size, err_max);
  }

  inputs[1]->SetData(nullptr);
  inputs[2]->SetData(nullptr);
  delete[] packed_input;
  lite::opencl::OpenCLRuntime::DeleteInstance();
  return;
}

TEST_F(TestConvolutionDwOpenCL, NoPadNC4HW4Fp32) {
  auto conv_param = std::make_unique<ConvParameter>();
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
    conv_param->pad_u_ = 0;
    conv_param->pad_l_ = 0;
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

  DepthWiseTestMain<float, float>(conv_param.get(), input_data, weight_data, gnd_data, schema::Format_NC4HW4);
}

TEST_F(TestConvolutionDwOpenCL, PadNC4HW4Fp32) {
  auto conv_param = std::make_unique<ConvParameter>();
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
    conv_param->pad_u_ = 1;
    conv_param->pad_l_ = 1;
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

  DepthWiseTestMain<float, float>(conv_param.get(), input_data, weight_data, gnd_data, schema::Format_NC4HW4);
}

TEST_F(TestConvolutionDwOpenCL, NoPadNHWC4Fp32) {
  auto conv_param = std::make_unique<ConvParameter>();
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
    conv_param->pad_u_ = 0;
    conv_param->pad_l_ = 0;
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

  DepthWiseTestMain<float, float>(conv_param.get(), input_data, weight_data, gnd_data, schema::Format_NHWC4);
}

TEST_F(TestConvolutionDwOpenCL, PadNHWC4Fp32) {
  auto conv_param = std::make_unique<ConvParameter>();
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
    conv_param->pad_u_ = 1;
    conv_param->pad_l_ = 1;
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

  DepthWiseTestMain<float, float>(conv_param.get(), input_data, weight_data, gnd_data, schema::Format_NHWC4);
}

TEST_F(TestConvolutionDwOpenCL, NoPadNHWC4Fp16) {
  auto conv_param = std::make_unique<ConvParameter>();
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
    conv_param->pad_u_ = 0;
    conv_param->pad_l_ = 0;
  }

  // nhwc
  float16_t input_data[] = {
    0.5488135,  0.0202184,  0.45615032, 0.31542835, 0.71518934, 0.83261985, 0.56843394, 0.36371076,
    0.60276335, 0.77815676, 0.0187898,  0.57019675, 0.5448832,  0.87001216, 0.6176355,  0.43860152,
    0.4236548,  0.9786183,  0.6120957,  0.9883738,  0.6458941,  0.7991586,  0.616934,   0.10204481,
    0.4375872,  0.46147937, 0.94374806, 0.20887676, 0.891773,   0.7805292,  0.6818203,  0.16130951,
    0.96366274, 0.11827443, 0.3595079,  0.6531083,  0.3834415,  0.639921,   0.43703195, 0.2532916,
    0.79172504, 0.14335328, 0.6976312,  0.46631077, 0.5288949,  0.9446689,  0.06022547, 0.2444256,
    0.56804454, 0.5218483,  0.6667667,  0.15896958, 0.92559665, 0.41466194, 0.67063785, 0.11037514,
    0.07103606, 0.2645556,  0.21038257, 0.6563296,  0.0871293,  0.7742337,  0.12892629, 0.13818295};

  // co h w ci
  float16_t weight_data[] = {
    0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,  0.9767611,
    0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772, 0.31798318, 0.41426298,
    0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962,  0.31856894,
    0.6674104,  0.13179787, 0.7163272,  0.2894061,  0.18319136, 0.5865129,  0.02010755, 0.82894003, 0.00469548};

  // pack correct data, nhwc
  float16_t gnd_data[] = {3.3848767, 1.4446403, 1.8428744, 1.3194335, 2.5873442, 2.1384869, 2.04022,  1.1872686,
                          2.2294958, 1.6570128, 2.465089,  1.4294086, 2.7941442, 1.7871612, 2.188921, 1.0601988};

  lite::opencl::OpenCLRuntime::GetInstance()->SetFp16Enable(true);
  DepthWiseTestMain<float16_t, float16_t>(conv_param.get(), input_data, weight_data, gnd_data, schema::Format_NHWC4,
                                          kNumberTypeFloat16, true, 1e-2);
}

TEST_F(TestConvolutionDwOpenCL, PadNHWC4Fp16) {
  auto conv_param = std::make_unique<ConvParameter>();
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
    conv_param->pad_u_ = 1;
    conv_param->pad_l_ = 1;
  }

  // nhwc
  float16_t input_data[] = {
    0.5488135, 0.3834415,  0.77815676, 0.9446689,  0.6120957,  0.71518934, 0.79172504, 0.87001216, 0.5218483,
    0.616934,  0.60276335, 0.5288949,  0.9786183,  0.41466194, 0.94374806, 0.5448832,  0.56804454, 0.7991586,
    0.2645556, 0.6818203,  0.4236548,  0.92559665, 0.46147937, 0.7742337,  0.3595079,  0.6458941,  0.07103606,
    0.7805292, 0.45615032, 0.43703195, 0.4375872,  0.0871293,  0.11827443, 0.56843394, 0.6976312,  0.891773,
    0.0202184, 0.639921,   0.0187898,  0.06022547, 0.96366274, 0.83261985, 0.14335328, 0.6176355,  0.6667667};
  // float16_t input_data[]={
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
  float16_t weight_data[] = {
    0.67063785, 0.21038257, 0.12892629, 0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738,  0.10204481,
    0.20887676, 0.16130951, 0.6531083,  0.2532916,  0.46631077, 0.2444256,  0.15896958, 0.11037514, 0.6563296,
    0.13818295, 0.19658236, 0.36872518, 0.82099324, 0.09710128, 0.8379449,  0.09609841, 0.97645944, 0.4686512,
    0.9767611,  0.6048455,  0.7392636,  0.03918779, 0.28280696, 0.12019656, 0.2961402,  0.11872772, 0.31798318,
    0.41426298, 0.06414749, 0.6924721,  0.56660146, 0.2653895,  0.5232481,  0.09394051, 0.5759465,  0.9292962};
  // float16_t weight_data[]={
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
  float16_t gnd_data[] = {1.189188,   1.0425153,  1.8012011,  0.6074867,  1.2120346,  1.5005531,  0.8346756, 2.4365785,
                          0.54975945, 1.6815965,  1.2690231,  0.60214907, 1.6158017,  0.42115876, 0.8854959, 1.1709145,
                          1.0929465,  1.3534508,  1.1985044,  1.2932993,  2.4621446,  1.7086457,  2.6977584, 2.1960166,
                          2.3769147,  2.3185873,  0.6133741,  0.9687358,  0.9987654,  1.0254729,  0.8368954, 0.74171704,
                          0.8749627,  0.8953936,  0.5093431,  1.5496738,  0.54936385, 0.7683113,  1.165742,  1.3682933,
                          1.0517888,  0.59817517, 0.75649744, 1.2075498,  0.38804203};

  lite::opencl::OpenCLRuntime::GetInstance()->SetFp16Enable(true);
  DepthWiseTestMain<float16_t, float16_t>(conv_param.get(), input_data, weight_data, gnd_data, schema::Format_NHWC4,
                                          kNumberTypeFloat16, true, 1e-2);
}

TEST_F(TestConvolutionDwOpenCL, ProfilingMobilenetv2Fp32) {
  std::vector<std::vector<int>> src_shape{
    {1, 32, 112, 112}, {1, 96, 112, 112}, {1, 144, 56, 56}, {1, 144, 56, 56}, {1, 192, 28, 28},
    {1, 192, 28, 28},  {1, 384, 14, 14},  {1, 576, 14, 14}, {1, 576, 14, 14}, {1, 960, 7, 7},
  };
  std::vector<std::vector<int>> dst_shape{
    {1, 32, 112, 112}, {1, 96, 56, 56},  {1, 144, 56, 56}, {1, 144, 28, 28}, {1, 192, 28, 28},
    {1, 192, 14, 14},  {1, 384, 14, 14}, {1, 576, 14, 14}, {1, 576, 7, 7},   {1, 960, 7, 7},
  };
  std::vector<std::vector<int>> filter_shape{
    {32, 1, 1, 1},  {96, 3, 3, 1},  {144, 1, 1, 1}, {144, 3, 3, 1}, {192, 1, 1, 1},
    {192, 3, 3, 1}, {384, 1, 1, 1}, {576, 1, 1, 1}, {576, 3, 3, 1}, {960, 1, 1, 1},
  };

  // nhwc
  const size_t in_size = 96 * 112 * 112;
  float *input_data = new (std::nothrow) float[in_size];
  if (input_data == nullptr) {
    return;
  }
  memset(input_data, 0, in_size * sizeof(float_t));
  for (auto i = 0; i < in_size; ++i) {
    input_data[i] = 1;
  }
  // co h w ci
  const size_t wt_size = 576 * 3 * 3;
  float *weight_data = new (std::nothrow) float[wt_size];
  if (weight_data == nullptr) {
    delete [] input_data;
    return;
  }
  memset(weight_data, 0, wt_size);
  for (auto i = 0; i < wt_size; ++i) {
    weight_data[i] = 1;
  }
  for (size_t i = 0; i < src_shape.size(); ++i) {
    const int MAX_RUN_TIMES = 1;
    for (int j = 0; j < MAX_RUN_TIMES; ++j) {
      printf("========profiling depthwise, in shape(%d,%d,%d,%d), out shape(%d,%d,%d,%d), iter%d========\n",
             src_shape[i][0], src_shape[i][1], src_shape[i][2], src_shape[i][3], dst_shape[i][0], dst_shape[i][1],
             dst_shape[i][2], dst_shape[i][3], j);
      auto conv_param = ConvParameter();
      {
        conv_param.input_batch_ = 1;
        conv_param.input_h_ = src_shape[i][2];
        conv_param.input_w_ = src_shape[i][3];
        conv_param.input_channel_ = src_shape[i][1];
        conv_param.output_batch_ = 1;
        conv_param.output_h_ = dst_shape[i][2];
        conv_param.output_w_ = dst_shape[i][3];
        conv_param.output_channel_ = dst_shape[i][1];
        conv_param.kernel_h_ = filter_shape[i][1];
        conv_param.kernel_w_ = filter_shape[i][2];
        conv_param.stride_h_ = conv_param.output_h_ / conv_param.input_h_;
        conv_param.stride_w_ = conv_param.output_w_ / conv_param.input_w_;
        conv_param.pad_u_ = (conv_param.kernel_h_ - 1) / 2;
        conv_param.pad_l_ = (conv_param.kernel_w_ - 1) / 2;
        conv_param.dilation_h_ = 1;
        conv_param.dilation_w_ = 1;
      }
      DepthWiseTestMain<float, float>(&conv_param, input_data, weight_data, nullptr, schema::Format_NHWC4,
                                      kNumberTypeFloat32, false);
    }
  }
  delete [] input_data;
  delete [] weight_data;
  lite::opencl::OpenCLRuntime::DeleteInstance();
}
}  // namespace mindspore
