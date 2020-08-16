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

#include "src/runtime/kernel/arm/fp16/convolution_3x3_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/conv_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/winograd_transform_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/fp16/layout_transform_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Conv2D;

namespace mindspore::kernel {
void ProcessFilterFp16(float16_t *origin_weight, float16_t *dst_weight, ConvParameter *conv_param) {
  auto input_channel = conv_param->input_channel_;
  auto output_channel = conv_param->output_channel_;
  auto kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  int iC4 = UP_DIV(input_channel, C4NUM);
  int oC8 = UP_DIV(output_channel, C8NUM);

  size_t tmp_size = oC8 * C8NUM * iC4 * C4NUM * kernel_plane * sizeof(float16_t);
  auto tmp_addr = reinterpret_cast<float16_t *>(malloc(tmp_size));
  memset(tmp_addr, 0, tmp_size);

  PackWeightToC4Fp16(origin_weight, tmp_addr, conv_param);
  Conv3x3Fp16FilterTransform(tmp_addr, dst_weight, iC4, output_channel, kernel_plane);

  free(tmp_addr);
}

int Convolution3x3FP16CPUKernel::InitWeightBias() {
  auto input_channel = conv_param_->input_channel_;
  int output_channel = conv_param_->output_channel_;
  int iC4 = UP_DIV(input_channel, C4NUM);
  int oC8 = UP_DIV(output_channel, C8NUM);
  // init weight
  size_t transformed_size = iC4 * C4NUM * oC8 * C8NUM * 36 * sizeof(float16_t);
  transformed_filter_addr_ = reinterpret_cast<float16_t *>(malloc(transformed_size));
  if (transformed_filter_addr_ == nullptr) {
    MS_LOG(ERROR) << "malloc transformed_filter_addr_ failed.";
    return RET_ERROR;
  }
  memset(transformed_filter_addr_, 0, transformed_size);
  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute filter failed.";
    return ret;
  }
  ProcessFilterFp16(execute_weight_, transformed_filter_addr_, conv_param_);

  // init bias
  size_t new_bias_size = oC8 * C8NUM * sizeof(float16_t);
  bias_data_ = malloc(new_bias_size);
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, new_bias_size);
  auto fp16_bias_data = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias_addr = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    for (int i = 0; i < output_channel; ++i) {
      fp16_bias_data[i] = (float16_t)ori_bias_addr[i];
    }
  } else {
    MS_ASSERT(inputs_.size() == kInputSize1);
  }
  return RET_OK;
}

int Convolution3x3FP16CPUKernel::InitTmpBuffer() {
  const int tile_num = 16;
  const int k_plane = 36;
  int iC4 = UP_DIV(conv_param_->input_channel_, C4NUM);
  int oC8 = UP_DIV(conv_param_->output_channel_, C8NUM);

  /*=============================tile_buffer_============================*/
  size_t tile_buffer_size = thread_count_ * tile_num * k_plane * iC4 * C4NUM * sizeof(float16_t);
  tile_buffer_ = reinterpret_cast<float16_t *>(malloc(tile_buffer_size));
  if (tile_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tile_buffer_ failed.";
    return RET_ERROR;
  }
  memset(tile_buffer_, 0, tile_buffer_size);

  /*=============================block_unit_buffer_============================*/
  size_t block_unit_buffer_size = thread_count_ * k_plane * C4NUM * sizeof(float16_t);
  block_unit_buffer_ = reinterpret_cast<float16_t *>(malloc(block_unit_buffer_size));
  if (block_unit_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc block_unit_buffer_ failed.";
    return RET_ERROR;
  }
  memset(block_unit_buffer_, 0, block_unit_buffer_size);

  /*=============================tmp_dst_buffer_============================*/
  size_t tmp_dst_buffer_size = thread_count_ * tile_num * k_plane * oC8 * C8NUM * sizeof(float16_t);
  tmp_dst_buffer_ = reinterpret_cast<float16_t *>(malloc(tmp_dst_buffer_size));
  if (tmp_dst_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_dst_buffer_ failed.";
    return RET_ERROR;
  }
  memset(tmp_dst_buffer_, 0, tmp_dst_buffer_size);

  /*=============================tmp_out_============================*/
  int new_out_plane = UP_DIV(conv_param_->output_h_, C4NUM) * UP_DIV(conv_param_->output_w_, C4NUM) * C4NUM * C4NUM;
  size_t tmp_out_size = oC8 * C8NUM * conv_param_->output_batch_ * new_out_plane * sizeof(float16_t);
  tmp_out_ = reinterpret_cast<float16_t *>(malloc(tmp_out_size));
  if (tmp_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_out_ failed.";
    return RET_ERROR;
  }
  memset(tmp_out_, 0, tmp_out_size);

  /*=============================fp16_input_============================*/
  size_t fp16_input_size = conv_param_->input_channel_ * conv_param_->input_batch_ * conv_param_->input_h_ *
                           conv_param_->input_w_ * sizeof(float16_t);
  fp16_input_ = reinterpret_cast<float16_t *>(malloc(fp16_input_size));
  if (fp16_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_input_ failed.";
    return RET_ERROR;
  }
  memset(fp16_input_, 0, fp16_input_size);

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size =
    iC4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float16_t);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4_input_ failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);

  /*=============================fp16_out_============================*/
  size_t fp16_output_size = conv_param_->output_channel_ * conv_param_->output_batch_ * conv_param_->output_h_ *
                            conv_param_->output_w_ * sizeof(float16_t);
  fp16_out_ = reinterpret_cast<float16_t *>(malloc(fp16_output_size));
  if (fp16_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_out_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void Convolution3x3FP16CPUKernel::ConfigInputOutput() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_format = input_tensor->GetFormat();
  schema::Format execute_format = schema::Format_NHWC4;
  convert_func_ = LayoutTransformFp16(input_format, execute_format);
  if (convert_func_ == nullptr) {
    MS_LOG(ERROR) << "layout convert func is nullptr.";
    return;
  }
}

int Convolution3x3FP16CPUKernel::Init() {
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  ConfigInputOutput();
  return RET_OK;
}

int Convolution3x3FP16CPUKernel::ReSize() {
  if (tile_buffer_ != nullptr) {
    free(tile_buffer_);
  }
  if (block_unit_buffer_ != nullptr) {
    free(block_unit_buffer_);
  }
  if (tmp_dst_buffer_ != nullptr) {
    free(tmp_dst_buffer_);
  }
  if (tmp_out_ != nullptr) {
    free(tmp_out_);
  }
  if (fp16_out_ != nullptr) {
    free(fp16_out_);
  }
  if (fp16_input_ != nullptr) {
    free(fp16_input_);
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
  }

  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3FP16CPUKernel::RunImpl(int task_id) {
  Conv3x3Fp16(reinterpret_cast<float16_t *>(nhwc4_input_), transformed_filter_addr_,
              reinterpret_cast<float16_t *>(bias_data_), execute_output_, tile_buffer_, block_unit_buffer_,
              tmp_dst_buffer_, tmp_out_, task_id, conv_param_);
  return RET_OK;
}

int Convolution3x3Fp16Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<Convolution3x3FP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution3x3 Fp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3FP16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  ret = ConvolutionBaseFP16CPUKernel::GetExecuteTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get execute tensor failed.";
    return ret;
  }
  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  convert_func_(reinterpret_cast<void *>(execute_input_), nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(Convolution3x3Fp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv3x3 fp16 error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  // get real output
  // todo
  int out_w_block = UP_DIV(conv_param_->output_w_, C4NUM);
  int out_h_block = UP_DIV(conv_param_->output_h_, C4NUM);
  int oc8 = UP_DIV(conv_param_->output_channel_, C8NUM);
  bool relu = conv_param_->is_relu_;
  bool relu6 = conv_param_->is_relu6_;
  for (int batch = 0; batch < conv_param_->output_batch_; batch++) {
    int tmp_out_batch_offset =
      batch * oc8 * C8NUM * out_w_block * out_h_block * conv_param_->output_unit_ * conv_param_->output_unit_;
    int ro_batch_size = batch * conv_param_->output_channel_ * conv_param_->output_h_ * conv_param_->output_w_;
    const float16_t *batch_tmp_out = tmp_out_ + tmp_out_batch_offset;
    float16_t *batch_out = execute_output_ + ro_batch_size;
    for (int h = 0; h < conv_param_->output_h_; h++) {
      for (int w = 0; w < conv_param_->output_w_; w++) {
        for (int c = 0; c < conv_param_->output_channel_; c++) {
          int oc8_block = c / C8NUM;
          int oc8_res = c % C8NUM;
          int src_offset = oc8_block * C8NUM * out_w_block * out_h_block * C4NUM * C4NUM +
                           C8NUM * (h * out_w_block * conv_param_->output_unit_ + w) + oc8_res;
          int dst_offset = (h * conv_param_->output_w_ + w) * conv_param_->output_channel_ + c;
          (batch_out + dst_offset)[0] = (batch_tmp_out + src_offset)[0];
          if (relu) {
            (batch_out + dst_offset)[0] = (batch_out + dst_offset)[0] < 0 ? 0 : (batch_out + dst_offset)[0];
          } else if (relu6) {
            (batch_out + dst_offset)[0] = (batch_out + dst_offset)[0] < 0 ? 0 : (batch_out + dst_offset)[0];
            (batch_out + dst_offset)[0] = (batch_out + dst_offset)[0] > 6 ? 6 : (batch_out + dst_offset)[0];
          }
        }
      }
    }
  }

  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  return RET_OK;
}
}  // namespace mindspore::kernel
