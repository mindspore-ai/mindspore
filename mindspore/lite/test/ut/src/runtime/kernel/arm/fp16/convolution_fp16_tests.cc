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
#include "mindspore/core/utils/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/utils.h"
#include "src/common/file_utils.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp16/convolution_fp16.h"
#include "mindspore/lite/src/runtime/kernel/arm/fp16/convolution_3x3_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/conv_fp16.h"

namespace mindspore {
class TestConvolutionFp16 : public mindspore::CommonTest {
 public:
  TestConvolutionFp16() {}
};

void InitConvParamGroup1Fp16(ConvParameter *conv_param) {
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 28;
  conv_param->input_w_ = 28;
  conv_param->input_channel_ = 3;

  conv_param->output_batch_ = 1;
  conv_param->output_h_ = 28;
  conv_param->output_w_ = 28;
  conv_param->output_channel_ = 32;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;

  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;

  conv_param->pad_h_ = 1;
  conv_param->pad_w_ = 1;
  conv_param->thread_num_ = 1;
}

void InitConvParamGroup2Fp16(ConvParameter *conv_param) {
  conv_param->input_batch_ = 1;
  conv_param->input_h_ = 128;
  conv_param->input_w_ = 128;
  conv_param->input_channel_ = 32;

  conv_param->output_batch_ = 1;
  conv_param->output_h_ = 128;
  conv_param->output_w_ = 128;
  conv_param->output_channel_ = 32;

  conv_param->kernel_h_ = 3;
  conv_param->kernel_w_ = 3;

  conv_param->stride_h_ = 1;
  conv_param->stride_w_ = 1;

  conv_param->dilation_h_ = 1;
  conv_param->dilation_w_ = 1;

  conv_param->pad_h_ = 1;
  conv_param->pad_w_ = 1;
  conv_param->thread_num_ = 1;
}

TEST_F(TestConvolutionFp16, ConvTest1) {
  // prepare stage
  auto conv_param = new ConvParameter();
  InitConvParamGroup1Fp16(conv_param);

  int tile_num = 16;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int kernel_plane = k_h * k_w;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int i_h = conv_param->input_h_;
  int i_w = conv_param->input_w_;
  int out_channel = conv_param->output_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int oc8 = UP_DIV(out_channel, C8NUM);

  size_t weight_size;
  std::string weight_path = "./test_data/conv/convfp32_weight_32_3_3_3.bin";
  auto weight_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(weight_path.c_str(), &weight_size));
  std::cout << "==============fp32 weight data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << weight_data[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "weight data size: " << weight_size / sizeof(float) << std::endl;

  int weight_ele_size = weight_size / sizeof(float);
  auto fp16_weight_data = new float16_t[weight_ele_size];
  for (int i = 0; i < weight_ele_size; i++) {
    fp16_weight_data[i] = static_cast<float16_t>(weight_data[i]);
  }

  std::cout << "==============fp16 weight data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << fp16_weight_data[i] << ", ";
  }
  std::cout << std::endl;

  auto packed_weight = reinterpret_cast<float16_t *>(malloc(k_h * k_w * ic4 * C4NUM * oc8 * C8NUM * sizeof(float16_t)));
  PackWeightFp16(fp16_weight_data, conv_param, packed_weight);

  std::cout << "==============fp16 packed weight data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << packed_weight[i] << ", ";
  }
  std::cout << std::endl;

  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::cout << "==============fp32 input data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << input_data[i] << ", ";
  }
  std::cout << std::endl;

  int input_ele_size = input_size / sizeof(float);
  auto fp16_input_data = new float16_t[input_ele_size];
  for (int i = 0; i < input_ele_size; i++) {
    fp16_input_data[i] = static_cast<float16_t>(input_data[i]);
  }

  auto nhwc4_input_data = reinterpret_cast<float16_t *>(malloc(i_h * i_w * ic4 * C4NUM* sizeof(float16_t)));
  PackNHWCToNHWC4Fp32(fp16_input_data, nhwc4_input_data, 1, i_h * i_w, in_channel);

  std::cout << "==============fp16 input data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << fp16_input_data[i] << ", ";
  }
  std::cout << std::endl;

  int output_count = conv_param->output_h_ * conv_param->output_w_;
  int output_tile_count = UP_DIV(output_count, tile_num);
  int unit_size = kernel_plane * ic4 * C4NUM;
  int packed_input_size = output_tile_count * tile_num * unit_size;
  auto packed_input = reinterpret_cast<float16_t *>(malloc(in_batch * packed_input_size * sizeof(float16_t)));
  memset(packed_input, 0, in_batch * packed_input_size * sizeof(float16_t));

  auto bias_data = reinterpret_cast<float16_t *>(malloc(conv_param->output_channel_ * sizeof(float16_t)));
  memset(bias_data, 0, conv_param->output_channel_ * sizeof(float16_t));

  size_t output_data_size =
    conv_param->output_batch_ * conv_param->output_channel_ * conv_param->output_h_ * conv_param->output_w_;
  auto output_data = new float16_t[output_data_size];
  auto tmp_output_block = reinterpret_cast<float16_t *>(malloc(tile_num * out_channel * sizeof(float16_t)));

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  // warmup
  for (int i = 0; i < 3; i++) {
    ConvFp16(nhwc4_input_data, packed_input, packed_weight, bias_data, tmp_output_block, output_data, 0, conv_param);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    ConvFp16(nhwc4_input_data, packed_input, packed_weight, bias_data, tmp_output_block, output_data, 0, conv_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::cout << "==============fp16 output data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << ", ";
  }
  std::cout << std::endl;

  auto fp32_output_data = new float[output_data_size];
  for (int i = 0; i < output_data_size; i++) {
    fp32_output_data[i] = static_cast<float>(output_data[i]);
  }
  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << fp32_output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/conv/convfp32_out_1_28_28_32.bin";
  lite::CompareOutput(fp32_output_data, output_path);

  free(nhwc4_input_data);
  free(packed_input);
  free(bias_data);
  free(packed_weight);
  free(tmp_output_block);
  delete conv_param;
  delete input_data;
  delete weight_data;
  delete[] fp16_weight_data;
  delete[] fp16_input_data;
  delete[] fp32_output_data;
  delete[] output_data;
  MS_LOG(INFO) << "TestConvolutionFp16 passed";
}

TEST_F(TestConvolutionFp16, ConvTest2) {
  // prepare stage
  auto conv_param = new ConvParameter();
  InitConvParamGroup2Fp16(conv_param);

  // parameter
  int tile_num = 16;
  int k_h = conv_param->kernel_h_;
  int k_w = conv_param->kernel_w_;
  int kernel_plane = k_h * k_w;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int out_channel = conv_param->output_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);
  int oc8 = UP_DIV(out_channel, C8NUM);

  // weight
  size_t weight_size;
  std::string weight_path = "./test_data/conv/convfp32_weight_32_3_3_32.bin";
  auto weight_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(weight_path.c_str(), &weight_size));
  int weight_ele_size = weight_size / sizeof(float);
  auto fp16_weight_data = new float16_t[weight_ele_size];
  for (int i = 0; i < weight_ele_size; i++) {
    fp16_weight_data[i] = static_cast<float16_t>(weight_data[i]);
  }
  auto packed_weight = reinterpret_cast<float16_t *>(malloc(k_h * k_w * ic4 * C4NUM * oc8 * C8NUM * sizeof(float16_t)));
  PackWeightFp16(fp16_weight_data, conv_param, packed_weight);

  // input
  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input_1_128_128_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  int input_ele_size = input_size / sizeof(float);
  auto fp16_input_data = new float16_t[input_ele_size];
  for (int i = 0; i < input_ele_size; i++) {
    fp16_input_data[i] = static_cast<float16_t>(input_data[i]);
  }
  int output_count = conv_param->output_h_ * conv_param->output_w_;
  int output_tile_count = UP_DIV(output_count, tile_num);
  int unit_size = kernel_plane * ic4 * C4NUM;
  int packed_input_size = output_tile_count * tile_num * unit_size;
  auto packed_input = reinterpret_cast<float16_t *>(malloc(in_batch * packed_input_size * sizeof(float16_t)));
  memset(packed_input, 0, in_batch * packed_input_size * sizeof(float16_t));

  // bias
  auto bias_data = reinterpret_cast<float16_t *>(malloc(conv_param->output_channel_ * sizeof(float16_t)));
  memset(bias_data, 0, conv_param->output_channel_ * sizeof(float16_t));

  // output
  auto tmp_output_block = reinterpret_cast<float16_t *>(malloc(tile_num * out_channel * sizeof(float16_t)));
  size_t output_data_size =
    conv_param->output_batch_ * conv_param->output_channel_ * conv_param->output_h_ * conv_param->output_w_;
  auto output_data = new float16_t[output_data_size];

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  // warmup
  for (int i = 0; i < 3; i++) {
    ConvFp16(fp16_input_data, packed_input, packed_weight, bias_data, tmp_output_block, output_data, 0, conv_param);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    ConvFp16(fp16_input_data, packed_input, packed_weight, bias_data, tmp_output_block, output_data, 0, conv_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::cout << "==============fp16 output data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << ", ";
  }
  std::cout << std::endl;

  auto fp32_output_data = new float[output_data_size];
  for (int i = 0; i < output_data_size; i++) {
    fp32_output_data[i] = static_cast<float>(output_data[i]);
  }
  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << fp32_output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/conv/convfp32_out_1_128_128_32.bin";
  lite::CompareOutput(fp32_output_data, output_path);

  free(packed_input);
  free(bias_data);
  free(packed_weight);
  free(tmp_output_block);
  delete conv_param;
  delete input_data;
  delete weight_data;
  delete[] fp16_weight_data;
  delete[] fp16_input_data;
  delete[] fp32_output_data;
  delete[] output_data;
  MS_LOG(INFO) << "TestConvolutionFp16 passed";
}

TEST_F(TestConvolutionFp16, Conv3x3Test1) {
  auto conv_param = new ConvParameter();
  InitConvParamGroup1Fp16(conv_param);
  // todo
  int thread_count = 1;
  int tile_num = 16;
  int output_batch = conv_param->output_batch_;
  int output_h = conv_param->output_h_;
  int output_w = conv_param->output_w_;
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);

  // tmp buffer
  int k_plane = 36;
  size_t tile_buffer_size = thread_count * tile_num * k_plane * ic4 * C4NUM * sizeof(float16_t);
  float16_t *tile_buffer = reinterpret_cast<float16_t *>(malloc(tile_buffer_size));
  memset(tile_buffer, 0, tile_buffer_size);

  size_t block_unit_buffer_size = thread_count * k_plane * C4NUM * sizeof(float16_t);
  float16_t *block_unit_buffer = reinterpret_cast<float16_t *>(malloc(block_unit_buffer_size));
  memset(block_unit_buffer, 0, block_unit_buffer_size);

  size_t tmp_dst_buffer_size = thread_count * tile_num * k_plane * oc8 * C8NUM * sizeof(float16_t);
  float16_t *tmp_dst_buffer = reinterpret_cast<float16_t *>(malloc(tmp_dst_buffer_size));
  memset(tmp_dst_buffer, 0, tmp_dst_buffer_size);

  size_t tmp_out_size = oc8 * C8NUM * output_batch * output_h * output_w * tile_num * sizeof(float16_t);
  float16_t *tmp_out = reinterpret_cast<float16_t *>(malloc(tmp_out_size));
  memset(tmp_out, 0, tmp_out_size);

  // weight
  size_t weight_size;
  std::string weight_path = "./test_data/conv/convfp32_weight_32_3_3_3.bin";
  auto weight_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(weight_path.c_str(), &weight_size));
  std::cout << "==============fp32 weight data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << weight_data[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "weight data size: " << weight_size / sizeof(float) << std::endl;

  int weight_ele_size = weight_size / sizeof(float);
  auto fp16_weight_data = new float16_t[weight_ele_size];
  for (int i = 0; i < weight_ele_size; i++) {
    fp16_weight_data[i] = (float16_t)weight_data[i];
  }

  std::cout << "==============fp16 weight data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << fp16_weight_data[i] << ", ";
  }
  std::cout << std::endl;

  size_t transformed_size = ic4 * C4NUM * oc8 * C8NUM * 36;
  auto transformed_weight_data = new float16_t[transformed_size];
  memset(transformed_weight_data, 0, transformed_size * sizeof(float16_t));
  kernel::ProcessFilterFp16(fp16_weight_data, transformed_weight_data, conv_param);

  // bias
  auto bias_data =
    reinterpret_cast<float16_t *>(malloc(UP_DIV(conv_param->output_channel_, 8) * 8 * sizeof(float16_t)));
  memset(bias_data, 0, UP_DIV(conv_param->output_channel_, 8) * 8 * sizeof(float16_t));

  // input
  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  std::cout << "==============fp32 input data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << input_data[i] << ", ";
  }
  std::cout << std::endl;

  int input_ele_size = input_size / sizeof(float);
  auto fp16_input_data = new float16_t[input_ele_size];
  for (int i = 0; i < input_ele_size; i++) {
    fp16_input_data[i] = static_cast<float16_t>(input_data[i]);
  }

  std::cout << "==============fp16 input data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << fp16_input_data[i] << ", ";
  }
  std::cout << std::endl;

  // output
  size_t output_data_size =
    conv_param->output_batch_ * conv_param->output_channel_ * conv_param->output_h_ * conv_param->output_w_;
  auto output_data = new float16_t[output_data_size];

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  // warmup
  for (int i = 0; i < 3; i++) {
    Conv3x3Fp16(fp16_input_data, transformed_weight_data, bias_data, output_data, tile_buffer, block_unit_buffer,
                tmp_dst_buffer, tmp_out, 0, conv_param);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    Conv3x3Fp16(fp16_input_data, transformed_weight_data, bias_data, output_data, tile_buffer, block_unit_buffer,
                tmp_dst_buffer, tmp_out, 0, conv_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::cout << "==============fp16 output data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << ", ";
  }
  std::cout << std::endl;

  auto fp32_output_data = new float[output_data_size];
  for (int i = 0; i < output_data_size; i++) {
    fp32_output_data[i] = static_cast<float>(output_data[i]);
  }
  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << fp32_output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/conv/convfp32_out_1_28_28_32.bin";
  lite::CompareOutput(fp32_output_data, output_path);

  free(bias_data);
  free(tile_buffer);
  free(block_unit_buffer);
  free(tmp_dst_buffer);
  free(tmp_out);
  delete input_data;
  delete weight_data;
  delete conv_param;
  delete[] fp16_weight_data;
  delete[] fp16_input_data;
  delete[] fp32_output_data;
  delete[] output_data;
  delete[] transformed_weight_data;
  MS_LOG(INFO) << "TestConvolutionFp16 Conv3x3 passed";
}

TEST_F(TestConvolutionFp16, Conv3x3Test2) {
  auto conv_param = new ConvParameter();
  InitConvParamGroup2Fp16(conv_param);
  // todo
  int thread_count = 1;
  int tile_num = 16;
  int output_batch = conv_param->output_batch_;
  int output_h = conv_param->output_h_;
  int output_w = conv_param->output_w_;
  int ic4 = UP_DIV(conv_param->input_channel_, C4NUM);
  int oc8 = UP_DIV(conv_param->output_channel_, C8NUM);

  // tmp buffer
  int k_plane = 36;
  size_t tile_buffer_size = thread_count * tile_num * k_plane * ic4 * C4NUM * sizeof(float16_t);
  float16_t *tile_buffer = reinterpret_cast<float16_t *>(malloc(tile_buffer_size));
  memset(tile_buffer, 0, tile_buffer_size);

  size_t block_unit_buffer_size = thread_count * k_plane * C4NUM * sizeof(float16_t);
  float16_t *block_unit_buffer = reinterpret_cast<float16_t *>(malloc(block_unit_buffer_size));
  memset(block_unit_buffer, 0, block_unit_buffer_size);

  size_t tmp_dst_buffer_size = thread_count * tile_num * k_plane * oc8 * C8NUM * sizeof(float16_t);
  float16_t *tmp_dst_buffer = reinterpret_cast<float16_t *>(malloc(tmp_dst_buffer_size));
  memset(tmp_dst_buffer, 0, tmp_dst_buffer_size);

  size_t tmp_out_size = oc8 * C8NUM * output_batch * output_h * output_w * tile_num * sizeof(float16_t);
  float16_t *tmp_out = reinterpret_cast<float16_t *>(malloc(tmp_out_size));
  memset(tmp_out, 0, tmp_out_size);

  // weight
  size_t weight_size;
  std::string weight_path = "./test_data/conv/convfp32_weight_32_3_3_32.bin";
  auto weight_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(weight_path.c_str(), &weight_size));
  int weight_ele_size = weight_size / sizeof(float);
  auto fp16_weight_data = new float16_t[weight_ele_size];
  for (int i = 0; i < weight_ele_size; i++) {
    fp16_weight_data[i] = static_cast<float16_t>(weight_data[i]);
  }
  size_t transformed_size = ic4 * C4NUM * oc8 * C8NUM * 36;
  auto transformed_weight_data = new float16_t[transformed_size];
  memset(transformed_weight_data, 0, transformed_size * sizeof(float16_t));
  kernel::ProcessFilterFp16(fp16_weight_data, transformed_weight_data, conv_param);

  // bias
  auto bias_data =
    reinterpret_cast<float16_t *>(malloc(UP_DIV(conv_param->output_channel_, 8) * 8 * sizeof(float16_t)));
  memset(bias_data, 0, UP_DIV(conv_param->output_channel_, 8) * 8 * sizeof(float16_t));

  // input
  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input_1_128_128_32.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  int input_ele_size = input_size / sizeof(float);
  auto fp16_input_data = new float16_t[input_ele_size];
  for (int i = 0; i < input_ele_size; i++) {
    fp16_input_data[i] = static_cast<float16_t>(input_data[i]);
  }

  // output
  size_t output_data_size =
    conv_param->output_batch_ * conv_param->output_channel_ * conv_param->output_h_ * conv_param->output_w_;
  auto output_data = new float16_t[output_data_size];

  // runtime part
  printf("Calculating runtime cost...\n");
  uint64_t time_avg = 0;
  // warmup
  for (int i = 0; i < 3; i++) {
    Conv3x3Fp16(fp16_input_data, transformed_weight_data, bias_data, output_data, tile_buffer, block_unit_buffer,
                tmp_dst_buffer, tmp_out, 0, conv_param);
  }

  int loop_count = 100;
  auto time_start = mindspore::lite::GetTimeUs();
  for (int i = 0; i < loop_count; i++) {
    Conv3x3Fp16(fp16_input_data, transformed_weight_data, bias_data, output_data, tile_buffer, block_unit_buffer,
                tmp_dst_buffer, tmp_out, 0, conv_param);
  }
  auto time_end = mindspore::lite::GetTimeUs();
  auto cost = time_end - time_start;
  time_avg = cost / loop_count;
  printf("single thread running time : %f ms\n", time_avg / 1000.0f);

  std::cout << "==============fp16 output data===========" << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << output_data[i] << ", ";
  }
  std::cout << std::endl;

  auto fp32_output_data = new float[output_data_size];
  for (int i = 0; i < output_data_size; i++) {
    fp32_output_data[i] = static_cast<float>(output_data[i]);
  }
  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << fp32_output_data[i] << " ,";
  }
  std::cout << std::endl;

  std::string output_path = "./test_data/conv/convfp32_out_1_128_128_32.bin";
  lite::CompareOutput(fp32_output_data, output_path);

  free(bias_data);
  free(tile_buffer);
  free(block_unit_buffer);
  free(tmp_dst_buffer);
  free(tmp_out);
  delete input_data;
  delete weight_data;
  delete conv_param;
  delete[] fp16_weight_data;
  delete[] fp16_input_data;
  delete[] fp32_output_data;
  delete[] output_data;
  delete[] transformed_weight_data;
  MS_LOG(INFO) << "TestConvolutionFp16 Conv3x3 passed";
}

}  // namespace mindspore
