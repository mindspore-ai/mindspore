/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/conv2d_3x3_int8_coder.h"
#include <vector>
#include "securec/include/securec.h"
#include "nnacl/int8/conv3x3_int8.h"
#include "coder/opcoders/file_collector.h"
#include "coder/log.h"
#include "coder/opcoders/parallel.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"

namespace mindspore::lite::micro::nnacl {
void ProcessFilterUint8(int8_t *origin_weight, int16_t *dst_weight, ConvParameter *conv_param) {
  int input_channel = conv_param->input_channel_;
  int output_channel = conv_param->output_channel_;
  int kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  int iC8 = UP_DIV(input_channel, C8NUM);

  size_t tmp_size = output_channel * iC8 * C8NUM * kernel_plane * sizeof(int16_t);
  auto tmp_addr = reinterpret_cast<int16_t *>(malloc(tmp_size));
  MS_CHECK_PTR_IF_NULL(tmp_addr);
  int ret = memset_s(tmp_addr, tmp_size, 0, tmp_size);
  if (ret != EOK) {
    free(tmp_addr);
    MS_LOG(ERROR) << "memset_s tmp_addr failed.";
    return;
  }
  PackWeightToC8Int8(origin_weight, tmp_addr, conv_param);
  Conv3x3Int8FilterTransform(tmp_addr, dst_weight, iC8, output_channel, kernel_plane);
  free(tmp_addr);
}

int Conv2D3x3Int8Coder::InitWeightBias() {
  int input_channel = conv_param_->input_channel_;
  int output_channel = conv_param_->output_channel_;
  MS_CHECK_TRUE(input_channel > 0, "invalid input_channel");
  MS_CHECK_TRUE(output_channel > 0, "invalid output_channel");
  int iC8 = UP_DIV(input_channel, C8NUM);
  int oC4 = UP_DIV(output_channel, C4NUM);
  // init weight
  int transformed_size = iC8 * C8NUM * oC4 * C4NUM * 16 * sizeof(int16_t);
  transformed_filter_addr_ =
    static_cast<int16_t *>(allocator_->Malloc(kNumberTypeInt16, transformed_size, kOfflinePackWeight));
  MS_CHECK_PTR(transformed_filter_addr_);
  MS_CHECK_RET_CODE(memset_s(transformed_filter_addr_, transformed_size, 0, transformed_size),
                    "memset_s transformed_filter_addr_ failed.");
  auto *original_weight_addr = reinterpret_cast<int8_t *>(filter_tensor_->data_c());
  ProcessFilterUint8(original_weight_addr, transformed_filter_addr_, conv_param_);

  // init bias
  int new_bias_size = oC4 * C4NUM * sizeof(int32_t);
  new_bias_addr_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, new_bias_size, kOfflinePackWeight));
  MS_CHECK_PTR(new_bias_addr_);
  MS_CHECK_RET_CODE(memset_s(new_bias_addr_, new_bias_size, 0, new_bias_size), "memset_s new_bias_addr_ failed.");
  if (input_tensors_.size() == kInputSize2) {
    auto *ori_bias_addr = reinterpret_cast<int32_t *>(bias_tensor_->data_c());
    MS_CHECK_RET_CODE(memcpy_s(new_bias_addr_, new_bias_size, ori_bias_addr, output_channel * sizeof(int32_t)),
                      "memset_s new_bias_addr_ failed.");
  } else {
    MS_ASSERT(input_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int Conv2D3x3Int8Coder::InitTmpBuffer(CoderContext *const context) {
  int ic8 = UP_DIV(conv_param_->input_channel_, C8NUM);
  int oc4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  int in_batch = conv_param_->input_batch_;
  int input_w = conv_param_->input_w_;
  int input_h = conv_param_->input_h_;
  int output_batch = conv_param_->output_batch_;
  int output_w = conv_param_->output_w_;
  int output_h = conv_param_->output_h_;

  /*=============================tile_buffer_============================*/
  tile_buffer_size_ = thread_num_ * TILE_NUM * 16 * ic8 * C8NUM * sizeof(int16_t);
  tile_buffer_ = static_cast<int16_t *>(allocator_->Malloc(kNumberTypeInt16, tile_buffer_size_, kWorkspace));

  /*=============================block_unit_buffer_============================*/
  block_unit_buffer_size_ = thread_num_ * 4 * 4 * C8NUM * sizeof(int16_t);
  block_unit_buffer_ =
    static_cast<int16_t *>(allocator_->Malloc(kNumberTypeInt16, block_unit_buffer_size_, kWorkspace));

  /*=============================tmp_dst_buffer_============================*/
  tmp_dst_buffer_size_ = thread_num_ * TILE_NUM * 16 * oc4 * C4NUM * sizeof(int32_t);
  tmp_dst_buffer_ = static_cast<int32_t *>(allocator_->Malloc(kNumberTypeInt32, tmp_dst_buffer_size_, kWorkspace));

  /*=============================tmp_out_============================*/
  tmp_out_size_ = oc4 * C4NUM * output_batch * output_w * output_h * sizeof(uint8_t);
  tmp_out_ = static_cast<int8_t *>(allocator_->Malloc(kNumberTypeInt8, tmp_out_size_, kWorkspace));

  /*=============================input_data_============================*/
  c8_input_size_ = in_batch * input_h * input_w * ic8 * C8NUM * sizeof(int16_t);
  c8_input_ = static_cast<int16_t *>(allocator_->Malloc(kNumberTypeInt16, c8_input_size_, kWorkspace));
  return RET_OK;
}

void Conv2D3x3Int8Coder::ConfigInputOutput() { output_tensor_->set_format(schema::Format_NHWC); }

int Conv2D3x3Int8Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(Conv2DBaseCoder::Init(), "ConvolutionBase init failed.");
  conv_param_->thread_num_ = thread_num_;
  // to 1, task id is set to 0
  conv_param_->op_parameter_.thread_num_ = thread_num_;
  MS_CHECK_RET_CODE(SetQuantParam(), "Set quant param failed.");
  MS_CHECK_RET_CODE(InitWeightBias(), "Init weight bias failed.");
  // init tmp input, output
  MS_CHECK_RET_CODE(InitTmpBuffer(context), "Init tmp buffer failed.");
  // config input output
  ConfigInputOutput();
  return RET_OK;
}

int Conv2D3x3Int8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/int8/conv_int8.h", "nnacl/int8/conv3x3_int8.h"},
          {"pack_int8.c", "conv_int8.c", "conv3x3_int8.c", "fixed_point.c"});
  if (thread_num_ > 1) {
    Collect(context, {"wrapper/int8/conv3x3_run_int8_wrapper.h"}, {"conv3x3_run_int8_wrapper.c"});
  }
  nnacl::NNaclInt8Serializer code;
  code.precision(kPrecision);
  // call the op function
  code.CodeFunction("memset", tile_buffer_, 0, tile_buffer_size_);
  code.CodeFunction("memset", block_unit_buffer_, 0, block_unit_buffer_size_);
  code.CodeFunction("memset", tmp_dst_buffer_, 0, tmp_dst_buffer_size_);
  code.CodeFunction("memset", tmp_out_, 0, tmp_out_size_);
  code.CodeFunction("memset", c8_input_, 0, c8_input_size_);

  // define conv params
  code.CodeStruct("conv_param_", *conv_param_);
  // pack to c8
  code.CodeFunction("PackInputToC8Int8", input_tensor_, c8_input_, "&conv_param_");
  // code operator func
  if (thread_num_ > 1) {
    code.CodeBaseStruct("Conv3x3Int8Args", kRunArgs, c8_input_, transformed_filter_addr_, new_bias_addr_,
                        output_tensor_, tile_buffer_, block_unit_buffer_, tmp_dst_buffer_, tmp_out_, "&conv_param_");
    code.CodeFunction(kParallelLaunch, gThreadPool, "Conv3x3Int8Run", kRunArgsAddr, gThreadNum);
  } else {
    code.CodeFunction("Conv3x3Int8", c8_input_, transformed_filter_addr_, new_bias_addr_, output_tensor_, tile_buffer_,
                      block_unit_buffer_, tmp_dst_buffer_, tmp_out_, kDefaultTaskId, "&conv_param_");
  }
  code.CodeFunction("PackNC4HW4ToNHWCInt8", tmp_out_, output_tensor_, conv_param_->output_batch_,
                    conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  context->AppendCode(code.str());
  return RET_OK;
}
}  // namespace mindspore::lite::micro::nnacl
