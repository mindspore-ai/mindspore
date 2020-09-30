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

#include "src/runtime/kernel/arm/fp16/convolution_winograd_fp16.h"
#include "nnacl/fp16/matrix_fp16.h"
#include "nnacl/fp16/conv_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/fp16/winograd_transform_fp16.h"
#include "nnacl/fp16/winograd_utils_fp16.h"
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
int ConvolutionWinogradFP16CPUKernel::WinogradFilterTransformFp16(const float16_t *weight_data, float *matrix_g,
                                                                  float *matrix_gt, int oc_block) {
  // original weight format : ohwi
  auto channel_in = conv_param_->input_channel_;
  auto channel_out = conv_param_->output_channel_;
  int input_unit_square = input_unit_ * input_unit_;
  int oc_block_num = UP_DIV(channel_out, oc_block);

  auto matrix_g_data_fp16 = reinterpret_cast<float16_t *>(malloc(input_unit_ * kernel_unit_ * sizeof(float16_t)));
  if (matrix_g_data_fp16 == nullptr) {
    MS_LOG(ERROR) << "malloc matrix_g_data_fp16 failed.";
    return RET_ERROR;
  }
  auto matrix_gt_data_fp16 = reinterpret_cast<float16_t *>(malloc(input_unit_ * kernel_unit_ * sizeof(float16_t)));
  if (matrix_gt_data_fp16 == nullptr) {
    free(matrix_g_data_fp16);
    MS_LOG(ERROR) << "malloc matrix_gt_data_fp16 failed.";
    return RET_ERROR;
  }
  Float32ToFloat16(matrix_g, matrix_g_data_fp16, input_unit_ * kernel_unit_);
  Float32ToFloat16(matrix_gt, matrix_gt_data_fp16, input_unit_ * kernel_unit_);

  // trans_filter = G*g*GT (g represents weight_data)
  // separate into two steps ===> tmp = G*g ===> out = tmp * GT
  auto tmp_weight_data = reinterpret_cast<float16_t *>(malloc(kernel_unit_ * kernel_unit_ * sizeof(float16_t)));
  if (tmp_weight_data == nullptr) {
    free(matrix_g_data_fp16);
    free(matrix_gt_data_fp16);
    MS_LOG(ERROR) << "malloc tmp_weight_data failed.";
    return RET_ERROR;
  }
  auto tmp_data = reinterpret_cast<float16_t *>(malloc(input_unit_ * kernel_unit_ * sizeof(float16_t)));
  if (tmp_data == nullptr) {
    free(tmp_weight_data);
    free(matrix_g_data_fp16);
    free(matrix_gt_data_fp16);
    MS_LOG(ERROR) << "malloc tmp_data failed.";
    return RET_ERROR;
  }
  auto trans_out_data = reinterpret_cast<float16_t *>(malloc(input_unit_ * input_unit_ * sizeof(float16_t)));
  if (trans_out_data == nullptr) {
    free(tmp_data);
    free(tmp_weight_data);
    free(matrix_g_data_fp16);
    free(matrix_gt_data_fp16);
    MS_LOG(ERROR) << "malloc trans_out_data failed.";
    return RET_ERROR;
  }

  if (oc_block == 0) {
    MS_LOG(ERROR) << "Divide by zero";
    free(tmp_weight_data);
    free(tmp_data);
    free(trans_out_data);
    free(matrix_g_data_fp16);
    free(matrix_gt_data_fp16);
    return RET_ERROR;
  }
  int stride1 = channel_in * oc_block;
  for (int i = 0; i < channel_out; i++) {
    int out_c_block = i / oc_block;
    int out_c_res = i % oc_block;
    int input_oz_offset = i * kernel_unit_ * kernel_unit_ * channel_in;
    int output_oz_offset = out_c_block * stride1 + out_c_res;
    for (int j = 0; j < channel_in; j++) {
      int input_iz_offset = input_oz_offset + j;
      int output_iz_offset = output_oz_offset + j * oc_block;
      for (int k = 0; k < kernel_unit_ * kernel_unit_; k++) {
        int input_xy_offset = input_iz_offset + k * channel_in;
        tmp_weight_data[k] = *(weight_data + input_xy_offset);
      }
      // now we only support row-major matrix-multiply
      // tmp = G * g
      MatrixMultiplyFp16(matrix_g_data_fp16, tmp_weight_data, tmp_data, input_unit_, kernel_unit_, kernel_unit_);
      // out = tmp * GT
      MatrixMultiplyFp16(tmp_data, matrix_gt_data_fp16, trans_out_data, input_unit_, kernel_unit_, input_unit_);

      for (int z = 0; z < input_unit_square; z++) {
        int output_xy_offset = output_iz_offset + z * oc_block_num * stride1;
        trans_weight_[output_xy_offset] = trans_out_data[z];
      }
    }
  }
  free(tmp_weight_data);
  free(tmp_data);
  free(trans_out_data);
  free(matrix_g_data_fp16);
  free(matrix_gt_data_fp16);
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::InitWeightBias() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = filter_tensor->Channel();
  int out_channel = filter_tensor->Batch();
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;

  const int oc_block = C8NUM;
  int oc_block_num = UP_DIV(out_channel, C8NUM);

  // init weight
  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteFilter();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute filter failed.";
    return ret;
  }

  // set data
  auto trans_matrix_data_size = input_unit_ * input_unit_ * in_channel * oc_block_num * oc_block * sizeof(float16_t);
  trans_weight_ = reinterpret_cast<float16_t *>(malloc(trans_matrix_data_size));
  if (trans_weight_ == nullptr) {
    MS_LOG(ERROR) << "malloc trans_weight_ failed.";
    return RET_ERROR;
  }
  memset(trans_weight_, 0, trans_matrix_data_size);

  float matrix_g[64];
  float matrix_gt[64];
  float matrix_a[64];
  float matrix_at[64];
  float matrix_b[64];
  float matrix_bt[64];
  float coef = 1.0f;
  if (input_unit_ == 8) {
    coef = 0.5f;
  }
  ret = CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g, matrix_gt, coef, output_unit_, kernel_unit_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "get matrix g from CookToomFilter failed.";
    return ret;
  }
  ret = WinogradFilterTransformFp16(execute_weight_, matrix_g, matrix_gt, oc_block);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "winograd filter transfrom failed.";
    return ret;
  }

  // init bias
  bias_data_ = malloc(oc_block_num * oc_block * sizeof(float16_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, oc_block_num * oc_block * sizeof(float16_t));
  auto fp16_bias_data = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == kInputSize2) {
    auto ori_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->MutableData());
    for (int i = 0; i < out_channel; ++i) {
      fp16_bias_data[i] = (float16_t)ori_bias[i];
    }
  } else {
    MS_ASSERT(inputs_.size() == kInputSize1);
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::InitTmpBuffer() {
  const int cal_num = 16;
  int channel_out = conv_param_->output_channel_;
  int oc8 = UP_DIV(channel_out, C8NUM);

  size_t tile_buffer_size =
    thread_count_ * cal_num * input_unit_ * input_unit_ * conv_param_->input_channel_ * sizeof(float16_t);
  trans_input_ = reinterpret_cast<float16_t *>(ctx_->allocator->Malloc(tile_buffer_size));
  if (trans_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc trans_input_ failed.";
    return RET_ERROR;
  }

  gemm_out_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(thread_count_ * cal_num * input_unit_ * input_unit_ * oc8 * C8NUM * sizeof(float16_t)));
  if (gemm_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc gemm_out_ failed.";
    return RET_ERROR;
  }

  tmp_data_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(thread_count_ * C8NUM * input_unit_ * input_unit_ * sizeof(float16_t)));
  if (tmp_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_data_ failed.";
    return RET_ERROR;
  }

  col_buffer_ = reinterpret_cast<float16_t *>(
    ctx_->allocator->Malloc(thread_count_ * cal_num * conv_param_->input_channel_ * sizeof(float16_t)));
  if (col_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_buffer_ failed.";
    return RET_ERROR;
  }

  tmp_buffer_address_list_[0] = trans_input_;
  tmp_buffer_address_list_[1] = gemm_out_;
  tmp_buffer_address_list_[2] = tmp_data_;
  tmp_buffer_address_list_[3] = col_buffer_;
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::ConfigInputOutput() {
  in_func_ = GetInputTransFp16Func(input_unit_);
  if (in_func_ == nullptr) {
    MS_LOG(ERROR) << "in_func_ is null.";
    return RET_ERROR;
  }
  out_func_ = GetOutputTransFp16Func(input_unit_, output_unit_, conv_param_->act_type_);
  if (out_func_ == nullptr) {
    MS_LOG(ERROR) << "out_func_ is null.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::Init() {
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;
  auto ret = InitWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ConvolutionWinogradFP16CPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  ret = ConvolutionBaseCPUKernel::Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionBase init failed.";
    return RET_ERROR;
  }
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;

  ret = ConfigInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConfigInputOutput failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::RunImpl(int task_id) {
  ConvWinogardFp16(execute_input_, trans_weight_, reinterpret_cast<const float16_t *>(bias_data_), execute_output_,
                   tmp_buffer_address_list_, task_id, conv_param_, in_func_, out_func_);
  return RET_OK;
}

static int ConvolutionWinogradFp16Impl(void *cdata, int task_id) {
  auto conv = reinterpret_cast<ConvolutionWinogradFP16CPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionWinograd Fp16 Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradFP16CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }

  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteTensor();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute tensor failed.";
    return ret;
  }

  ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    return RET_ERROR;
  }

  int error_code = ParallelLaunch(this->context_->thread_pool_, ConvolutionWinogradFp16Impl, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "conv winograd error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  ConvolutionBaseFP16CPUKernel::IfCastOutput();
  ConvolutionBaseFP16CPUKernel::FreeTmpBuffer();
  FreeTmpBuffer();
  return RET_OK;
}
}  // namespace mindspore::kernel
