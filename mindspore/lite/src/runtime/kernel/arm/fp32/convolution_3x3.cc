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

#include "src/runtime/kernel/arm/fp32/convolution_3x3.h"
#include "src/runtime/kernel/arm/nnacl/fp32/conv.h"
#include "src/runtime/kernel/arm/base/layout_transform.h"
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
void ProcessFilter(float *origin_weight, float *dst_weight, ConvParameter *conv_param, int oc_block, int oc_block_num) {
  auto input_channel = conv_param->input_channel_;
  auto output_channel = conv_param->output_channel_;
  auto kernel_plane = conv_param->kernel_w_ * conv_param->kernel_h_;
  int iC4 = UP_DIV(input_channel, C4NUM);

  size_t tmp_size = oc_block_num * oc_block * iC4 * C4NUM * kernel_plane * sizeof(float);
  auto tmp_addr = reinterpret_cast<float *>(malloc(tmp_size));
  if (tmp_addr == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_addr failed.";
    return;
  }
  memset(tmp_addr, 0, tmp_size);

  PackNHWCToNC4HW4Fp32(origin_weight, tmp_addr, output_channel, kernel_plane, input_channel);
  Conv3x3Fp32FilterTransform(tmp_addr, dst_weight, iC4, output_channel, kernel_plane, oc_block);
  free(tmp_addr);
}

int Convolution3x3CPUKernel::InitWeightBias() {
  auto input_channel = conv_param_->input_channel_;
  auto output_channel = conv_param_->output_channel_;
  int iC4 = UP_DIV(input_channel, C4NUM);
  int oC4 = UP_DIV(output_channel, C4NUM);
  int oc_block, oc_block_num;
#ifdef ENABLE_ARM32
  oc_block = C4NUM;
  oc_block_num = UP_DIV(output_channel, C4NUM);
#else
  oc_block = C8NUM;
  oc_block_num = UP_DIV(output_channel, C8NUM);
#endif
  const int k_plane = 16;
  // init weight
  size_t transformed_size = iC4 * C4NUM * oc_block_num * oc_block * k_plane * sizeof(float);
  transformed_filter_addr_ = reinterpret_cast<float *>(malloc(transformed_size));
  if (transformed_filter_addr_ == nullptr) {
    MS_LOG(ERROR) << "malloc transformed filter addr failed.";
    return RET_ERROR;
  }
  memset(transformed_filter_addr_, 0, transformed_size);
  auto weight_data = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->Data());
  ProcessFilter(weight_data, transformed_filter_addr_, conv_param_, oc_block, oc_block_num);

  // init bias
  size_t new_bias_size = oC4 * C4NUM * sizeof(float);
  bias_data_ = reinterpret_cast<float *>(malloc(new_bias_size));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias data failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, new_bias_size);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias_addr = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->Data());
    memcpy(bias_data_, ori_bias_addr, output_channel * sizeof(float));
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }
  return RET_OK;
}

int Convolution3x3CPUKernel::InitTmpBuffer() {
  int iC4 = UP_DIV(conv_param_->input_channel_, C4NUM);
  int oC4 = UP_DIV(conv_param_->output_channel_, C4NUM);
  const int k_plane = 16;

  /*=============================tile_buffer_============================*/
  size_t tile_buffer_size = thread_count_ * TILE_NUM * k_plane * iC4 * C4NUM * sizeof(float);
  tile_buffer_ = reinterpret_cast<float *>(malloc(tile_buffer_size));
  if (tile_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tile buffer failed.";
    return RET_ERROR;
  }
  memset(tile_buffer_, 0, tile_buffer_size);

  /*=============================block_unit_buffer_============================*/
  size_t block_unit_buffer_size = thread_count_ * k_plane * C4NUM * sizeof(float);
  block_unit_buffer_ = reinterpret_cast<float *>(malloc(block_unit_buffer_size));
  if (block_unit_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc block_unit_buffer_ failed.";
    return RET_ERROR;
  }
  memset(block_unit_buffer_, 0, block_unit_buffer_size);

  /*=============================tmp_dst_buffer_============================*/
  size_t tmp_dst_buffer_size = thread_count_ * TILE_NUM * k_plane * oC4 * C4NUM * sizeof(float);
  tmp_dst_buffer_ = reinterpret_cast<float *>(malloc(tmp_dst_buffer_size));
  if (tmp_dst_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_dst_buffer_ failed.";
    return RET_ERROR;
  }
  memset(tmp_dst_buffer_, 0, tmp_dst_buffer_size);

  /*=============================nhwc4_input_============================*/
  size_t nhwc4_input_size =
    iC4 * C4NUM * conv_param_->input_batch_ * conv_param_->input_h_ * conv_param_->input_w_ * sizeof(float);
  nhwc4_input_ = malloc(nhwc4_input_size);
  if (nhwc4_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc nhwc4_input_ failed.";
    return RET_ERROR;
  }
  memset(nhwc4_input_, 0, nhwc4_input_size);

  /*=============================nc4hw4_out_============================*/
  size_t nc4hw4_out_size =
    oC4 * C4NUM * conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * sizeof(float);
  nc4hw4_out_ = reinterpret_cast<float *>(malloc(nc4hw4_out_size));
  if (nc4hw4_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc nc4hw4_out_ failed.";
    return RET_ERROR;
  }
  memset(nc4hw4_out_, 0, nc4hw4_out_size);
  tmp_buffer_address_list_[0] = tile_buffer_;
  tmp_buffer_address_list_[1] = block_unit_buffer_;
  tmp_buffer_address_list_[2] = tmp_dst_buffer_;
  tmp_buffer_address_list_[3] = nc4hw4_out_;
  return RET_OK;
}

void Convolution3x3CPUKernel::ConfigInputOutput() {
  auto output_tensor = out_tensors_.at(kOutputIndex);
  output_tensor->SetFormat(schema::Format_NHWC);

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ret = CheckLayout(input_tensor);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Check layout failed.";
    return;
  }
#ifdef ENABLE_ARM32
  gemm_func_ = IndirectGemmFp32_8x4;
#else
  gemm_func_ = IndirectGemmFp32_8x8;
#endif
}

int Convolution3x3CPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }
  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
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

int Convolution3x3CPUKernel::ReSize() {
  if (tile_buffer_ != nullptr) {
    free(tile_buffer_);
  }
  if (block_unit_buffer_ != nullptr) {
    free(block_unit_buffer_);
  }
  if (tmp_dst_buffer_ != nullptr) {
    free(tmp_dst_buffer_);
  }
  if (nhwc4_input_ != nullptr) {
    free(nhwc4_input_);
  }
  if (nc4hw4_out_ != nullptr) {
    free(nc4hw4_out_);
  }

  auto ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3CPUKernel::RunImpl(int task_id) {
  if (gemm_func_ == nullptr) {
    MS_LOG(ERROR) << "gemm_func is nullptr.";
    return RET_ERROR;
  }
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  Conv3x3Fp32(reinterpret_cast<float *>(nhwc4_input_), transformed_filter_addr_, reinterpret_cast<float *>(bias_data_),
              output_addr, tmp_buffer_address_list_, task_id, conv_param_, gemm_func_);
  return RET_OK;
}

int Convolution3x3Impl(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto conv3x3 = reinterpret_cast<Convolution3x3CPUKernel *>(cdata);
  auto error_code = conv3x3->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution3x3 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int Convolution3x3CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto ori_input_data = input_tensor->Data();
  int in_batch = conv_param_->input_batch_;
  int in_h = conv_param_->input_h_;
  int in_w = conv_param_->input_w_;
  int in_channel = conv_param_->input_channel_;
  convert_func_(ori_input_data, nhwc4_input_, in_batch, in_h * in_w, in_channel);

  int error_code = LiteBackendParallelLaunch(Convolution3x3Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv3x3 error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  auto is_relu = conv_param_->is_relu_;
  auto is_relu6 = conv_param_->is_relu6_;
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->Data());
  PackNC4HW4ToNHWCFp32(nc4hw4_out_, output_addr, conv_param_->output_batch_,
                       conv_param_->output_h_ * conv_param_->output_w_, conv_param_->output_channel_);
  int output_num =
    conv_param_->output_channel_ * conv_param_->output_h_ * conv_param_->output_w_ * conv_param_->output_batch_;
  if (is_relu) {
    ReluFp32(output_addr, output_addr, output_num);
  } else if (is_relu6) {
    Relu6Fp32(output_addr, output_addr, output_num);
  } else {
    // do nothing
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
