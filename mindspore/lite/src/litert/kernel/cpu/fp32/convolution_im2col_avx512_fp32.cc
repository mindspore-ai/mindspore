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

#include "src/litert/kernel/cpu/fp32/convolution_im2col_avx512_fp32.h"
#include "nnacl/fp32/conv_im2col_avx512_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
void ConvolutionIm2ColAVX512CPUKernel::InitGlobalVariable() {
  oc_tile_ = C16NUM;
  row_tile_ = MSMIN(UP_DIV(conv_param_->output_h_ * conv_param_->output_w_, op_parameter_->thread_num_), C150NUM);

  rowMajor2ColNMajorFunc = RowMajor2Col64Major;
}

int ConvolutionIm2ColAVX512CPUKernel::InitTmpBuffer() {
  MS_ASSERT(ctx_->allocator != nullptr);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]->MutableData());

  size_t unit_size =
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

  if (conv_param_->output_channel_ % oc_tile_ != 0) {
    output_need_align_ = true;
    if (tmp_output_ != nullptr) {
      ctx_->allocator->Free(tmp_output_);
    }

    // avx512 need to malloc dst aligned to C16NUM
    size_t oc_algin = UP_ROUND(conv_param_->output_channel_, oc_tile_);
    size_t pack_output_size = conv_param_->output_batch_ * conv_param_->output_h_ * conv_param_->output_w_ * oc_algin;
    tmp_output_ =
      reinterpret_cast<float *>(ctx_->allocator->Malloc(pack_output_size * static_cast<size_t>(sizeof(float))));
    if (tmp_output_ == nullptr) {
      MS_LOG(ERROR) << "malloc tmp output data failed.";
      return RET_NULL_PTR;
    }
  }

  return RET_OK;
}

int ConvolutionIm2ColAVX512CPUKernel::RunImpl(int task_id) {
  auto ori_input_data = reinterpret_cast<float *>(in_tensors_.at(kInputIndex)->data());
  CHECK_NULL_RETURN(ori_input_data);
  if (out_tensors_[0]->format() != NC4HW4) {
    if (use_batch_cut_flag_) {
      ConvIm2ColAVX512Fp32CutByBatch(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                                     reinterpret_cast<float *>(bias_data_), tmp_output_, task_id, conv_param_,
                                     row_tile_);
    } else {
      ConvIm2ColAVX512Fp32(ori_input_data, packed_input_, reinterpret_cast<float *>(packed_weight_),
                           reinterpret_cast<float *>(bias_data_), tmp_output_, task_id, conv_param_, row_tile_);
    }
  } else {
    MS_LOG(ERROR) << "ConvolutionIm2ColAVX512CPUKernel do not support NC4HW4 output-format's avx512 version";
    return RET_ERROR;
  }
  return RET_OK;
}

int ConvolutionIm2ColAVX512CPUKernel::Run() {
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
    PackNHWCXToNHWCFp32(tmp_output_, output_addr, conv_param_->output_batch_,
                        conv_param_->output_w_ * conv_param_->output_h_, conv_param_->output_channel_, oc_tile_);
  } else {
    tmp_output_ = nullptr;
  }

  FreeTmpBuffer();
  return ret;
}
}  // namespace mindspore::kernel
