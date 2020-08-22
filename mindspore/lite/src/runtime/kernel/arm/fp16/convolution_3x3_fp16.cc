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
#include "nnacl/fp16/conv_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/winograd_transform_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
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
  int iC8 = UP_DIV(input_channel, C8NUM);
  int oC8 = UP_DIV(output_channel, C8NUM);

  size_t tmp_size = oC8 * C8NUM * iC8 * C8NUM * kernel_plane * sizeof(float16_t);
  auto tmp_addr = reinterpret_cast<float16_t *>(malloc(tmp_size));
  memset(tmp_addr, 0, tmp_size);

  PackWeightToC4Fp16(origin_weight, tmp_addr, conv_param);
  Conv3x3Fp16FilterTransform(tmp_addr, dst_weight, iC8 * 2, output_channel, kernel_plane);

  free(tmp_addr);
}

int Convolution3x3FP16CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  auto input_channel = filter_tensor->Channel();
  auto output_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = input_channel;
  conv_param_->output_channel_ = output_channel;
  int iC8 = UP_DIV(input_channel, C8NUM);
  int oC8 = UP_DIV(output_channel, C8NUM);

  size_t transformed_size = iC8 * C8NUM * oC8 * C8NUM * 36 * sizeof(float16_t);
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
  int oC8 = UP_DIV(conv_param_->output_channel_, C8NUM);
  MS_ASSERT(ctx_->allocator != nullptr);

  size_t block_unit_buffer_size = thread_count_ * k_plane * C8NUM * sizeof(float16_t);
  block_unit_buffer_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(block_unit_buffer_size));
  if (block_unit_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc block_unit_buffer_ failed.";
    return RET_ERROR;
  }

  size_t tmp_dst_buffer_size = thread_count_ * tile_num * k_plane * oC8 * C8NUM * sizeof(float16_t);
  tmp_dst_buffer_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(tmp_dst_buffer_size));
  if (tmp_dst_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_dst_buffer_ failed.";
    return RET_ERROR;
  }

  int new_out_plane = UP_DIV(conv_param_->output_h_, C4NUM) * UP_DIV(conv_param_->output_w_, C4NUM) * C4NUM * C4NUM;
  size_t tmp_out_size = oC8 * C8NUM * conv_param_->output_batch_ * new_out_plane * sizeof(float16_t);
  tmp_out_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(tmp_out_size));
  if (tmp_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_out_ failed.";
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
  if (!InferShapeDone()) {
    return RET_OK;
  }
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return ReSize();
}

int Convolution3x3FP16CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  if (tile_buffer_ != nullptr) {
    free(tile_buffer_);
    tile_buffer_ = nullptr;
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
    nhwc4_input_ = nullptr;
  }

  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return ret;
  }
  const int tile_num = 16;
  const int k_plane = 36;
  int iC8 = UP_DIV(conv_param_->input_channel_, C8NUM);

  size_t nhwc8_input_size =
    iC8 * C8NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float16_t);
  nhwc4_input_ = malloc(nhwc8_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4_input_ failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc8_input_size);

  size_t tile_buffer_size = thread_count_ * tile_num * k_plane * iC8 * C8NUM * sizeof(float16_t);
  tile_buffer_ = reinterpret_cast<float16_t *>(malloc(tile_buffer_size));
  if (tile_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tile_buffer_ failed.";
    return RET_ERROR;
  }
  memset(tile_buffer_, 0, tile_buffer_size);

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
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  PackNHWCToNHWC8Fp16(reinterpret_cast<void *>(execute_input_), nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(Convolution3x3Fp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv3x3 fp16 error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  // get real output
  bool relu = conv_param_->is_relu_;
  bool relu6 = conv_param_->is_relu6_;
  if (relu) {
    UnPack3x3ReluOutputFp16(tmp_out_, execute_output_, conv_param_->output_batch_, conv_param_->output_h_,
                            conv_param_->output_w_, conv_param_->output_channel_);
  } else if (relu6) {
    UnPack3x3Relu6OutputFp16(tmp_out_, execute_output_, conv_param_->output_batch_, conv_param_->output_h_,
                             conv_param_->output_w_, conv_param_->output_channel_);
  } else {
    UnPack3x3OutputFp16(tmp_out_, execute_output_, conv_param_->output_batch_, conv_param_->output_h_,
                        conv_param_->output_w_, conv_param_->output_channel_);
  }

  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  ConvolutionBaseFP16CPUKernel::FreeTmpBuffer();
  FreeTmpBuffer();
  return RET_OK;
}
}  // namespace mindspore::kernel
