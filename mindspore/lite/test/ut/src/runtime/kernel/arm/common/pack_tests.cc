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
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "mindspore/lite/src/common/file_utils.h"
#include "mindspore/lite/nnacl/pack.h"
#ifdef ENABLE_FP16
#include "mindspore/lite/nnacl/fp16/pack_fp16.h"
#endif

namespace mindspore {
class TestPack : public mindspore::CommonTest {
 public:
  TestPack() {}
};

void InitConvParamPack(ConvParameter *conv_param) {
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

  conv_param->pad_u_ = 1;
  conv_param->pad_l_ = 1;
}

TEST_F(TestPack, PackInputFp32) {
  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  auto conv_param = new ConvParameter;
  InitConvParamPack(conv_param);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int thread_count = 1;
  int tile_n = 8;
  int output_count = out_h * out_w;
  int output_tile_count = UP_DIV(output_count, tile_n);

  int inchannel_block = 4;
  int channel_block = UP_DIV(in_channel, inchannel_block);
  int kernel_plane = kernel_h * kernel_w;
  int unit_size = kernel_plane * channel_block * inchannel_block;
  int packed_input_size = output_tile_count * tile_n * unit_size;

  auto packed_input = reinterpret_cast<float *>(malloc(in_batch * packed_input_size * sizeof(float)));
  memset(packed_input, 0, in_batch * packed_input_size * sizeof(float));

  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * in_channel * in_h * in_w;
    int gemm_in_batch_offset = b * packed_input_size;
    for (int thread_id = 0; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - tile_n) : tile_n;
      float *gemm_input =
        reinterpret_cast<float *>(packed_input) + thread_id * unit_size * tile_n + gemm_in_batch_offset;
      Im2ColPackUnitFp32(input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);
    }
  }

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << packed_input[i] << " ,";
  }
  std::cout << std::endl;

  std::string file_path = "./test_data/conv/convfp32_packinput.txt";
  // mindspore::lite::WriteToTxt<float>(file_path, packed_data, in_batch * packed_input_size);

  delete input_data;
  delete conv_param;
  free(packed_input);
  MS_LOG(INFO) << "TestPackInputFp32 passed";
}

#ifdef ENABLE_FP16
TEST_F(TestPack, PackInputFp16) {
  size_t input_size;
  std::string input_path = "./test_data/conv/convfp32_input_1_28_28_3.bin";
  auto input_data = reinterpret_cast<float *>(mindspore::lite::ReadFile(input_path.c_str(), &input_size));
  int input_ele_size = input_size / sizeof(float);
  auto fp16_input_data = new float16_t[input_ele_size];
  for (int i = 0; i < input_ele_size; i++) {
    fp16_input_data[i] = (float16_t)input_data[i];
  }

  auto conv_param = new ConvParameter;
  InitConvParamPack(conv_param);
  int kernel_h = conv_param->kernel_h_;
  int kernel_w = conv_param->kernel_w_;
  int in_batch = conv_param->input_batch_;
  int in_channel = conv_param->input_channel_;
  int in_h = conv_param->input_h_;
  int in_w = conv_param->input_w_;
  int out_h = conv_param->output_h_;
  int out_w = conv_param->output_w_;

  int thread_count = 1;
  int tile_n = 16;
  int output_count = out_h * out_w;
  int output_tile_count = UP_DIV(output_count, tile_n);

  int inchannel_block = 8;
  int channel_block = UP_DIV(in_channel, inchannel_block);
  int kernel_plane = kernel_h * kernel_w;
  int unit_size = kernel_plane * channel_block * inchannel_block;
  int packed_input_size = output_tile_count * tile_n * unit_size;

  auto packed_input = reinterpret_cast<float *>(malloc(in_batch * packed_input_size * sizeof(float16_t)));
  memset(packed_input, 0, in_batch * packed_input_size * sizeof(float16_t));

  for (int b = 0; b < in_batch; b++) {
    int in_batch_offset = b * in_channel * in_h * in_w;
    int gemm_in_batch_offset = b * packed_input_size;
    for (int thread_id = 0; thread_id < output_tile_count; thread_id += thread_count) {
      int start_index = thread_id * tile_n;
      int real_cal_num = (output_count - start_index) < tile_n ? (output_count - tile_n) : tile_n;
      float16_t *gemm_input =
        reinterpret_cast<float16_t *>(packed_input) + thread_id * unit_size * tile_n + gemm_in_batch_offset;
      Im2ColPackUnitFp16(fp16_input_data + in_batch_offset, conv_param, gemm_input, real_cal_num, start_index);
    }
  }

  printf("==================output data=================\n");
  for (int i = 0; i < 20; i++) {
    std::cout << packed_input[i] << " ,";
  }
  std::cout << std::endl;

  delete input_data;
  delete[] fp16_input_data;
  delete conv_param;
  delete packed_input;
  MS_LOG(INFO) << "TestPackInputFp16 passed";
}
#endif

}  // namespace mindspore
