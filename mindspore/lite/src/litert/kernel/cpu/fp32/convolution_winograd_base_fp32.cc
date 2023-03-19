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

#include "src/litert/kernel/cpu/fp32/convolution_winograd_base_fp32.h"
#include "nnacl/fp32/conv_winograd_fp32.h"
#include "nnacl/pack.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
#define CONV_MIN_CALC_BLOCK C1NUM
void ConvolutionWinogradBaseCPUKernel::InitGlobalVariable() {
  oc_block_ = C8NUM;
  tmp_data_tile_ = C4NUM;
  tile_num_ = C12NUM;
}

int ConvolutionWinogradBaseCPUKernel::WinogradFilterTransform(const float *weight_data, float *matrix_g,
                                                              const float *matrix_gt, int oc_block) {
  if (oc_block == 0) {
    MS_LOG(ERROR) << "Divide by zero";
    return RET_ERROR;
  }

  return WinogradWeightTransform(weight_data, reinterpret_cast<float *>(packed_weight_), matrix_g, matrix_gt, oc_block,
                                 input_unit_, kernel_unit_, conv_param_->input_channel_, conv_param_->output_channel_,
                                 true);
}

int ConvolutionWinogradBaseCPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  int input_plane = input_unit_ * input_unit_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(thread_count_, input_plane, RET_ERROR);
  int thread_input_plane = thread_count_ * input_plane;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(tile_num_, thread_input_plane, RET_ERROR);
  int total_thread_input_plane = tile_num_ * thread_input_plane;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thread_input_plane, conv_param_->input_channel_, RET_ERROR);
  size_t tile_buffer_size = static_cast<size_t>(total_thread_input_plane * conv_param_->input_channel_) * sizeof(float);
  trans_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(tile_buffer_size));
  if (trans_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc trans_input_ failed.";
    return RET_MEMORY_FAILED;
  }

  int oc8 = UP_ROUND(conv_param_->output_channel_, C8NUM);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thread_input_plane, oc8, RET_ERROR);
  gemm_out_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(static_cast<size_t>(total_thread_input_plane * oc8) * sizeof(float)));
  if (gemm_out_ == nullptr) {
    MS_LOG(ERROR) << "malloc gemm_out_ failed.";
    return RET_ERROR;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(tmp_data_tile_, thread_input_plane, RET_ERROR);
  tmp_data_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(static_cast<size_t>(tmp_data_tile_ * thread_input_plane) * sizeof(float)));
  if (tmp_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc tmp_data_ failed.";
    return RET_MEMORY_FAILED;
  }

  col_buffer_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(thread_count_ * tile_num_ * conv_param_->input_channel_ * sizeof(float)));
  if (col_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_buffer_ failed.";
    return RET_ERROR;
  }

  auto tile = UP_ROUND(conv_param_->input_channel_, tmp_data_tile_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_thread_input_plane, tile, RET_ERROR);
  opt_input_trans_ = reinterpret_cast<float *>(
    ctx_->allocator->Malloc(static_cast<size_t>(total_thread_input_plane * tile) * sizeof(float)));
  if (opt_input_trans_ == nullptr) {
    MS_LOG(ERROR) << "malloc opt_input_trans_ failed.";
    return RET_ERROR;
  }

  tmp_buffer_address_list_[C0NUM] = trans_input_;
  tmp_buffer_address_list_[C1NUM] = gemm_out_;
  tmp_buffer_address_list_[C2NUM] = tmp_data_;
  tmp_buffer_address_list_[C3NUM] = col_buffer_;
  tmp_buffer_address_list_[C4NUM] = opt_input_trans_;
  return RET_OK;
}

int ConvolutionWinogradBaseCPUKernel::ConfigInputOutput() {
  trans_func_.in_func_ = GetInputTransFunc(input_unit_);
  if (trans_func_.in_func_ == nullptr) {
    MS_LOG(ERROR) << "in_func_ is null.";
    return RET_ERROR;
  }

  trans_func_.out_func_ = GetOutputTransFunc(input_unit_, output_unit_, conv_param_->act_type_);
  if (trans_func_.out_func_ == nullptr) {
    MS_LOG(ERROR) << "out_func_ is null.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradBaseCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);

  InitGlobalVariable();
  kernel_unit_ = conv_param_->kernel_h_;
  input_unit_ = output_unit_ + kernel_unit_ - 1;
  conv_param_->input_unit_ = input_unit_;
  conv_param_->output_unit_ = output_unit_;
  if (op_parameter_->is_train_session_) {
    auto filter_tensor = in_tensors_.at(kWeightIndex);
    MS_CHECK_TRUE_MSG(filter_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    CHECK_NULL_RETURN(filter_tensor);
    int in_channel = filter_tensor->Channel();
    int out_channel = filter_tensor->Batch();
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_unit_, input_unit_, RET_ERROR);
    int input_plane = input_unit_ * input_unit_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(input_plane, in_channel, RET_ERROR);
    int in_chw = input_plane * in_channel;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(in_chw, UP_ROUND(out_channel, oc_block_), RET_ERROR);
    auto trans_matrix_data_size = static_cast<size_t>(in_chw * UP_ROUND(out_channel, oc_block_)) * sizeof(float);
    set_workspace_size(trans_matrix_data_size);
  }
  auto ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradBaseCPUKernel::UpdateThreadNumProcess(int32_t kernel_type, int64_t per_unit_load_num,
                                                             int64_t per_unit_store_num, int64_t unit_num) {
  if (conv_param_->input_batch_ % conv_param_->thread_num_ == 0) {
    use_batch_cut_flag_ = true;
    return RET_OK;
  } else {
    use_batch_cut_flag_ = false;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
  auto output_hw = conv_param_->output_h_ * conv_param_->output_w_;
  const int tile_num = C12NUM;

  conv_param_->thread_num_ =
    MSMIN(UP_DIV(UP_DIV(output_hw, tile_num), CONV_MIN_CALC_BLOCK), op_parameter_->thread_num_);
  thread_count_ = conv_param_->thread_num_;
  return RET_OK;
}

int ConvolutionWinogradBaseCPUKernel::ReSize() {
  auto ret = ConvolutionBaseCPUKernel::CheckResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }
  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv base init failed.";
    return ret;
  }
  if (UpdateThreadNumPass(TC_PTYPE(type_), 0, 0, 0) != RET_OK) {
    return RET_ERROR;
  }
  ret = ConfigInputOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ConfigInputOutput failed.";
    return RET_ERROR;
  }
  conv_param_->out_format_ = out_tensors_[0]->format();
  return RET_OK;
}

int ConvolutionWinogradBaseCPUKernel::RunImpl(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  CHECK_NULL_RETURN(input_tensor);
  auto ori_input_data = reinterpret_cast<float *>(input_tensor->data());
  CHECK_NULL_RETURN(ori_input_data);
  CHECK_NULL_RETURN(out_tensors_.front());
  auto output_data = reinterpret_cast<float *>(out_tensors_.front()->data());
  CHECK_NULL_RETURN(output_data);

  if (use_batch_cut_flag_) {
    ConvWinogardFp32CutByBatch(ori_input_data, reinterpret_cast<float *>(packed_weight_),
                               reinterpret_cast<const float *>(bias_data_), output_data, tmp_buffer_address_list_,
                               task_id, conv_param_, trans_func_);
  } else {
    ConvWinogardFp32(ori_input_data, reinterpret_cast<float *>(packed_weight_),
                     reinterpret_cast<const float *>(bias_data_), output_data, tmp_buffer_address_list_, task_id,
                     conv_param_, trans_func_);
  }

  return RET_OK;
}

int ConvolutionWinogradImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<ConvolutionWinogradBaseCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ConvolutionWinograd Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionWinogradBaseCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  if (RepackWeight() != RET_OK) {
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }

  ret = ParallelLaunch(this->ms_context_, ConvolutionWinogradImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv winograd error error_code[" << ret << "]";
  }

  FreeTmpBuffer();
  return ret;
}

int ConvolutionWinogradBaseCPUKernel::MallocWeightBiasData() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int in_channel = filter_tensor->Channel();
  if (in_channel < 0) {
    MS_LOG(ERROR) << "get channel from filter tensor failed.";
    return RET_ERROR;
  }
  int out_channel = filter_tensor->Batch();
  if (out_channel < 0) {
    MS_LOG(ERROR) << "get batch from filter tensor failed.";
    return RET_ERROR;
  }
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;

  // set data
  auto trans_matrix_data_size =
    static_cast<size_t>(input_unit_ * input_unit_ * in_channel * UP_ROUND(out_channel, oc_block_)) * sizeof(float);
  if (!op_parameter_->is_train_session_) {
    if (packed_weight_ == nullptr) {
      CHECK_LESS_RETURN(MAX_MALLOC_SIZE, trans_matrix_data_size);
      packed_weight_ = GetConvPackWeightData(trans_matrix_data_size);
      if (packed_weight_ == nullptr) {
        MS_LOG(ERROR) << "malloc matrix_buffer failed.";
        return RET_MEMORY_FAILED;
      }
    }
  }

  float matrix_a[64];
  float matrix_at[64];
  float matrix_b[64];
  float matrix_bt[64];
  float coef = 1.0f;
  if (input_unit_ == CONV_INPUT_UNIT_SIZE) {
    coef = 0.5f;
  }
  auto ret =
    CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g_, matrix_gt_, coef, output_unit_, kernel_unit_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "get matrix g from CookToomFilter failed.";
    return ret;
  }

  // init bias
  size_t new_bias_size = static_cast<size_t>(UP_ROUND(out_channel, C4NUM)) * sizeof(float);
  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, new_bias_size);
    bias_data_ = malloc(new_bias_size);
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias_data_ failed.";
      return RET_MEMORY_FAILED;
    }
  }
  memset(bias_data_, 0, new_bias_size);
  return RET_OK;
}

void ConvolutionWinogradBaseCPUKernel::PackWeight() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  void *origin_weight = (op_parameter_->is_train_session_) ? weight_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
  WinogradFilterTransform(reinterpret_cast<float *>(origin_weight), matrix_g_, matrix_gt_, oc_block_);
}
}  // namespace mindspore::kernel
