/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/convolution_im2col_avx_fp32.h"
#include "nnacl/fp32/conv_common_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_INFER_INVALID;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionIm2ColAVXCPUKernel::InitGlobalVariable() {
  oc_tile_ = C16NUM;
  row_tile_ = C6NUM;

  rowMajor2ColNMajorFunc = RowMajor2Col16Major;
}

int ConvolutionIm2ColAVXCPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]->MutableData());

  int unit_size =
    conv_param_->kernel_h_ * conv_param_->kernel_w_ * conv_param_->input_channel_ * row_tile_ * thread_count_;

  if (packed_input_ != nullptr) {
    ctx_->allocator->Free(packed_input_);
    packed_input_ = nullptr;
  }
  packed_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (packed_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc packed input failed.";
    return RET_ERROR;
  }

  if (col_major_input_ != nullptr) {
    ctx_->allocator->Free(col_major_input_);
    col_major_input_ = nullptr;
  }
  col_major_input_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(unit_size * sizeof(float)));
  if (col_major_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc col_major_input_ failed.";
    return RET_ERROR;
  }

  if (conv_param_->output_channel_ % oc_tile_ != 0 && out_tensors_[0]->format() == NC4HW4) {
    output_need_align_ = true;
    int oc_algin = UP_DIV(conv_param_->output_channel_, oc_tile_);
    int pack_output_size =
      conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * oc_tile_ * oc_algin;
    if (tmp_output_ != nullptr) {
      ctx_->allocator->Free(tmp_output_);
      tmp_output_ = nullptr;
    }
    tmp_output_ = reinterpret_cast<float *>(ms_context_->allocator->Malloc(pack_output_size * sizeof(float)));
    if (tmp_output_ == nullptr) {
      MS_LOG(ERROR) << "Malloc tmp_output_ buffer is failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ConvolutionIm2ColAVXCPUKernel::RunImpl(int task_id) {
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
    ConvFp32OutNC4HW4(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                      reinterpret_cast<float *>(bias_data_), col_major_input_, tmp_output_, task_id, conv_param_);
  }
  return RET_OK;
}

int ConvolutionIm2ColAVXCPUKernel::Run() {
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
  ret = ParallelLaunch(this->ms_context_, ConvolutionIm2ColImpl, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "conv error error_code[" << ret << "]";
  }

  if (output_need_align_) {
    PackNC8HW8AlignedToNC8HW8NotAlignedFp32(tmp_output_, output_addr, conv_param_->output_batch_,
                                            conv_param_->output_h_ * conv_param_->output_w_,
                                            conv_param_->output_channel_);
  }

  FreeTmpBuffer();
  return ret;
}
}  // namespace mindspore::kernel
