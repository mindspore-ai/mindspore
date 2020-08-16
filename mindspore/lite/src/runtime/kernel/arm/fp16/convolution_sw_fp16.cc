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
#include "src/runtime/kernel/arm/fp16/convolution_sw_fp16.h"
#include <vector>
#include "src/runtime/kernel/arm/nnacl/fp16/conv_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp16/pack_fp16.h"
#include "src/runtime/kernel/arm/nnacl/fp32/conv_depthwise.h"
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
int ConvolutionSWFP16CPUKernel::ProcessFilter() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_channel = conv_param_->input_channel_;
  int out_channel = conv_param_->output_channel_;
  int ic4 = UP_DIV(in_channel, C4NUM);

  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute filter failed.";
    return ret;
  }

  for (int oc = 0; oc < out_channel; ++oc) {
    int src_oc_offset = oc * kernel_h * kernel_w * in_channel;
    int dst_oc_offset = oc * kernel_h * kernel_w * ic4 * C4NUM;
    for (int i = 0; i < kernel_h * kernel_w; ++i) {
      const float16_t *src = execute_weight_ + src_oc_offset + i * in_channel;
      float16_t *dst = packed_weight_ + dst_oc_offset + i * ic4 * C4NUM;
      memcpy(dst, src, in_channel * sizeof(float16_t));
    }
  }

  return RET_OK;
}

int ConvolutionSWFP16CPUKernel::InitWeightBias() {
  int kernel_h = conv_param_->kernel_h_;
  int kernel_w = conv_param_->kernel_w_;
  int in_channel = conv_param_->input_channel_;
  int out_channel = conv_param_->output_channel_;
  int oc4 = UP_DIV(out_channel, C4NUM);
  int ic4 = UP_DIV(in_channel, C4NUM);
  int kernel_plane = kernel_h * kernel_w;
  int pack_weight_size = oc4 * ic4 * C4NUM * C4NUM * kernel_plane;

  // init weight
  packed_weight_ = reinterpret_cast<float16_t *>(malloc(pack_weight_size * sizeof(float16_t)));
  if (packed_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed_weight_ failed.";
    return RET_ERROR;
  }
  memset(packed_weight_, 0, pack_weight_size * sizeof(float16_t));
  auto ret = ProcessFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Process filter failed.";
    return ret;
  }

  // init bias
  bias_data_ = malloc(oc4 * C4NUM * sizeof(float16_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc4 * C4NUM * sizeof(float16_t));
  auto fp16_bias_data = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    for (int i = 0; i < out_channel; ++i) {
      fp16_bias_data[i] = (float16_t)ori_bias[i];
    }
  } else {
    MS_ASSERT(in_tensor_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionSWFP16CPUKernel::InitTmpBuffer() {
  int in_channel = conv_param_->input_channel_;
  int out_channel = conv_param_->output_channel_;
  int channel_block = UP_DIV(in_channel, C4NUM);
  int oc4 = UP_DIV(out_channel, C4NUM);

  /*=============================fp16_input_============================*/
  size_t fp16_input_size =
    in_channel * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float16_t);
  fp16_input_ = reinterpret_cast<float16_t *>(malloc(fp16_input_size));
  if (fp16_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_input_ failed.";
    return RET_ERROR;
  }

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size = channel_block * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ *
                            conv_param_->input_w_ * sizeof(float16_t);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4_input_ failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);

  /*=============================tmp_output_block_============================*/
  tmp_output_block_ = reinterpret_cast<float16_t *>(malloc(conv_param_->output_batch_ * conv_param_->output_h_ *
                                                           conv_param_->output_w_ * oc4 * C4NUM * sizeof(float16_t)));
  if (tmp_output_block_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_output_block_ failed.";
    return RET_ERROR;
  }

  /*=============================fp16_out_============================*/
  size_t fp16_output_size =
    out_channel * conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * sizeof(float16_t);
  fp16_out_ = reinterpret_cast<float16_t *>(malloc(fp16_output_size));
  if (fp16_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc fp16_out_ failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

void ConvolutionSWFP16CPUKernel::ConfigInputOutput() {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_format = input_tensor->GetFormat();
  schema::Format execute_format = schema::Format_NHWC4;
  convert_func_ = LayoutTransformFp16(input_format, execute_format);
  if (convert_func_ == nullptr) {
    MS_LOG(ERROR) << "layout convert func is nullptr.";
    return;
  }
}

int ConvolutionSWFP16CPUKernel::Init() {
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init fail!ret: " << ret;
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

  // init sliding window param
  slidingWindow_param_ = new SlidingWindowParam;
  InitSlidingParamConv(slidingWindow_param_, conv_param_, C4NUM);
  return RET_OK;
}

int ConvolutionSWFP16CPUKernel::ReSize() {
  if (tmp_output_block_ != nullptr) {
    free(tmp_output_block_);
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
  }
  if (fp16_input_ != nullptr) {
    free(fp16_input_);
  }
  if (fp16_out_ != nullptr) {
    free(fp16_out_);
  }
  delete slidingWindow_param_;

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
  // init sliding window param
  slidingWindow_param_ = new SlidingWindowParam;
  InitSlidingParamConv(slidingWindow_param_, conv_param_, C4NUM);
  return RET_OK;
}

int ConvolutionSWFP16CPUKernel::RunImpl(int task_id) {
  ConvSWFp16(reinterpret_cast<float16_t *>(nhwc4_input_), packed_weight_, reinterpret_cast<float16_t *>(bias_data_),
             tmp_output_block_, execute_output_, task_id, conv_param_, slidingWindow_param_);
  return RET_OK;
}

int ConvolutionSWFp16Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv = reinterpret_cast<ConvolutionSWFP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionFp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionSWFP16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  ret = ConvolutionBaseFP16CPUKernel::GetExecuteTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute tensor failed.";
    return ret;
  }

  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  convert_func_(reinterpret_cast<void *>(execute_input_), nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(ConvolutionSWFp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv fp16 error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  // output nhwc4
  int oc4_res = conv_param_->output_channel_ % C4NUM;
  if (oc4_res != 0) {
    PackNHWC4ToNHWCFp16(reinterpret_cast<const void *>(tmp_output_block_), reinterpret_cast<void *>(execute_output_),
                        conv_param_->output_batch_, conv_param_->output_h_ * conv_param_->output_w_,
                        conv_param_->output_channel_);
  }
  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  return RET_OK;
}
}  // namespace mindspore::kernel
