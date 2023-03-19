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

#include "src/litert/kernel/cpu/fp32/convolution_fp32.h"
#include "src/litert/pack_weight_manager.h"
#include "include/errorcode.h"
#include "nnacl/common_func.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/conv_common_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
#define CONV_MIN_CALC_BLOCK C1NUM
#ifdef ENABLE_AVX
#define OC_BLOCK C16NUM
#elif defined(ENABLE_ARM32)
#define OC_BLOCK C4NUM
#else
#define OC_BLOCK C8NUM
#endif
int ConvolutionCPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]->MutableData());
  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->kernel_h_, conv_param_->kernel_w_, RET_ERROR);
  int kernel_hw = conv_param_->kernel_h_ * conv_param_->kernel_w_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(kernel_hw, conv_param_->input_channel_, RET_ERROR);
  int kernel_chw = kernel_hw * conv_param_->input_channel_;
  MS_CHECK_INT_MUL_NOT_OVERFLOW(kernel_chw, thread_count_, RET_ERROR);
  int total_kernel_chw = kernel_chw * thread_count_;
#ifdef ENABLE_AVX
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_kernel_chw, C6NUM, RET_ERROR);
  int unit_size = total_kernel_chw * C6NUM;
#elif defined(ENABLE_SSE)
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_kernel_chw, C4NUM, RET_ERROR);
  int unit_size = total_kernel_chw * C4NUM;
#else
  MS_CHECK_INT_MUL_NOT_OVERFLOW(total_kernel_chw, C12NUM, RET_ERROR);
  int unit_size = total_kernel_chw * C12NUM;
#endif
  packed_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed input failed.";
    return RET_ERROR;
  }

  col_major_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (col_major_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_major_input_ failed.";
    return RET_ERROR;
  }

#ifdef ENABLE_AVX
  if (conv_param_->output_channel_ % OC_BLOCK != 0 && out_tensors_[0]->format() == NC4HW4) {
    output_need_align_ = true;
    int oc_algin = UP_DIV(conv_param_->output_channel_, OC_BLOCK);
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
    int output_hw = conv_param_->output_h_ * conv_param_->output_w_;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_batch_, output_hw, RET_ERROR);
    int output_bhw = conv_param_->output_batch_ * output_hw;
    MS_CHECK_INT_MUL_NOT_OVERFLOW(output_bhw, OC_BLOCK * oc_algin, RET_ERROR);
    int pack_output_size = output_bhw * OC_BLOCK * oc_algin;
    tmp_output_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_output_size * sizeof(float)));
    if (tmp_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc tmp_output_ buffer is failed.";
      return RET_ERROR;
    }
  }
#endif
  return RET_OK;
}

int ConvolutionCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (op_parameter_->is_train_session_) {
    auto filter_tensor = in_tensors_.at(kWeightIndex);
    MS_CHECK_TRUE_MSG(filter_tensor->shape().size() == C4NUM, RET_ERROR, "Conv-like: weight-shape only support 4D.");
    CHECK_NULL_RETURN(filter_tensor);
    size_t in_channel = filter_tensor->Channel();
    size_t out_channel = filter_tensor->Batch();
    size_t oc_block_num = UP_ROUND(out_channel, OC_BLOCK);
    size_t kernel_plane = filter_tensor->Height() * filter_tensor->Width();
    size_t pack_weight_size = oc_block_num * in_channel * kernel_plane;
    set_workspace_size(pack_weight_size * sizeof(float));
  }
  auto ret = InitConvWeightBias();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init weight bias failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::UpdateThreadNumProcess(int32_t kernel_type, int64_t per_unit_load_num,
                                                 int64_t per_unit_store_num, int64_t unit_num) {
  if (conv_param_->input_batch_ % conv_param_->thread_num_ == 0) {
    use_batch_cut_flag_ = true;
    return RET_OK;
  } else {
    use_batch_cut_flag_ = false;
  }

  MS_CHECK_INT_MUL_NOT_OVERFLOW(conv_param_->output_h_, conv_param_->output_w_, RET_ERROR);
  auto output_hw = conv_param_->output_h_ * conv_param_->output_w_;
#ifdef ENABLE_AVX
  const int cal_num = C6NUM;
#elif defined(ENABLE_SSE)
  const int cal_num = C4NUM;
#elif defined(ENABLE_ARM64)
  int cal_num = 0;
  if (output_hw <= C4NUM) {
    cal_num = C4NUM;
  } else if (output_hw <= C8NUM) {
    cal_num = C8NUM;
  } else {
    cal_num = C12NUM;
  }
#elif defined(ENABLE_ARM32)
  const int cal_num = C12NUM;
#else
  const int cal_num = C12NUM;
#endif

  conv_param_->thread_num_ = MSMIN(UP_DIV(UP_DIV(output_hw, cal_num), CONV_MIN_CALC_BLOCK), op_parameter_->thread_num_);
  thread_count_ = conv_param_->thread_num_;
  return RET_OK;
}

int ConvolutionCPUKernel::ReSize() {
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
  return RET_OK;
}

int ConvolutionCPUKernel::RunImpl(int task_id) {
  auto ori_input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data());
  CHECK_NULL_RETURN(ori_input_data);
  if (out_tensors_[0]->format() != NC4HW4) {
    if (use_batch_cut_flag_) {
      ConvFp32CutByBatch(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                         reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
    } else {
      ConvFp32(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
               reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
    }
  } else {
#if defined(ENABLE_ARM64) || defined(ENABLE_AVX)
    ConvFp32OutNC4HW4(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                      reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
#else
    if (use_batch_cut_flag_) {
      ConvFp32CutByBatch(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                         reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
    } else {
      ConvFp32(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
               reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
    }
#endif
  }
  return RET_OK;
}

int ConvolutionImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto conv = reinterpret_cast<ConvolutionCPUKernel *>(cdata);
  auto error_code = conv->RunImpl(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Convolution Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionCPUKernel::Run() {
  auto ret = InitTmpBuffer();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init tmp buffer failed.";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData());
  CHECK_NULL_RETURN(output_addr);
  if (!output_need_align_) {
    tmp_output_ = output_addr;
  }
  if (RepackWeight() != RET_OK) {
    FreeTmpBuffer();
    MS_LOG(ERROR) << "Repack weight failed.";
    return RET_ERROR;
  }
  ret = ParallelLaunch(this->ms_context_, ConvolutionImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << ret << "]";
  }
#ifdef ENABLE_AVX
  if (output_need_align_) {
    PackNC8HW8AlignedToNC8HW8NotAlignedFp32(tmp_output_, output_addr, conv_param_->output_batch_,
                                            conv_param_->output_h_ * conv_param_->output_w_,
                                            conv_param_->output_channel_);
  }
#endif
  FreeTmpBuffer();
  return ret;
}

void ConvolutionCPUKernel::PackWeight() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int32_t in_channel = filter_tensor->Channel();
  if (in_channel < 0) {
    MS_LOG(ERROR) << "get channel from filter_tensor failed.";
    return;
  }
  int32_t out_channel = filter_tensor->Batch();
  if (out_channel < 0) {
    MS_LOG(ERROR) << "get batch from filter_tensor failed.";
    return;
  }
  int32_t kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  if (kernel_plane < 0) {
    MS_LOG(ERROR) << "get height and width from filter_tensor failed.";
    return;
  }
  void *origin_weight = (op_parameter_->is_train_session_) ? filter_tensor->data() : origin_weight_;
  MS_ASSERT(origin_weight != nullptr);
#ifdef ENABLE_AVX
  RowMajor2Col16Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), out_channel,
                      in_channel * kernel_plane);
#elif defined(ENABLE_ARM32)
  RowMajor2Col4Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), out_channel,
                     in_channel * kernel_plane);
#else
  RowMajor2Col8Major(reinterpret_cast<float *>(origin_weight), reinterpret_cast<float *>(packed_weight_), out_channel,
                     in_channel * kernel_plane);
#endif
}

int ConvolutionCPUKernel::MallocWeightBiasData() {
  auto filter_tensor = in_tensors_.at(kWeightIndex);
  int32_t in_channel = filter_tensor->Channel();
  int32_t out_channel = filter_tensor->Batch();
  MS_CHECK_TRUE_RET(in_channel > 0 && out_channel > 0, RET_ERROR);
  conv_param_->input_channel_ = in_channel;
  conv_param_->output_channel_ = out_channel;
  size_t oc_block_num = UP_ROUND(out_channel, OC_BLOCK);
  size_t kernel_plane = filter_tensor->Height() * filter_tensor->Width();
  size_t pack_weight_size = oc_block_num * in_channel * kernel_plane;
  if (!op_parameter_->is_train_session_) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, pack_weight_size * sizeof(float));
    packed_weight_ = GetConvPackWeightData(static_cast<size_t>(pack_weight_size) * sizeof(float));
    if (packed_weight_ == nullptr) {
      MS_LOG(ERROR) << "malloc packed weight failed.";
      return RET_ERROR;
    }
  }

  if (bias_data_ == nullptr) {
    CHECK_LESS_RETURN(MAX_MALLOC_SIZE, oc_block_num * sizeof(float));
    bias_data_ = malloc(oc_block_num * sizeof(float));
    if (bias_data_ == nullptr) {
      MS_LOG(ERROR) << "malloc bias failed.";
      return RET_ERROR;
    }
  }
  memset(bias_data_, 0, oc_block_num * sizeof(float));
  return RET_OK;
}
}  // namespace mindspore::kernel
