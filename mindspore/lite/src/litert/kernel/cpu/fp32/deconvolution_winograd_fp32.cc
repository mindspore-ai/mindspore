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

#include <algorithm>
#include "src/litert/kernel/cpu/fp32/deconvolution_winograd_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
const int kDeconvWinogradMaxPixel = 3145728;
DeConvolutionWinogradCPUKernel::~DeConvolutionWinogradCPUKernel() {
  FreeResizeBuf();
  FreeDeconvParam();
}

void DeConvolutionWinogradCPUKernel::FreeResizeBuf() {
  if (deconv_param_ == nullptr) {
    return;
  }
  for (int i = 0; i < deconv_param_->compute_size_; i++) {
    DeConvComputeUnit &unit = deconv_param_->compute_units_[i];
    if (unit.tmp_buffer_ != nullptr) {
      free(unit.tmp_buffer_);
      unit.tmp_buffer_ = nullptr;
    }

    if (unit.use_winograd_) {
      if (unit.winograd_.b_buffer_ != nullptr) {
        free(unit.winograd_.b_buffer_);
        unit.winograd_.b_buffer_ = nullptr;
      }
    }
  }

  for (auto &wg : deconv_param_->a_buffer_) {
    if (wg.buf_init_) {
      if (wg.dest_buffer_ != nullptr) {
        free(wg.dest_buffer_);
        wg.dest_buffer_ = nullptr;
      }
      if (wg.middle_buffer_ != nullptr) {
        free(wg.middle_buffer_);
        wg.middle_buffer_ = nullptr;
      }
    }
    wg.buf_init_ = false;
  }

  if (tile_input_ != nullptr) {
    free(tile_input_);
    tile_input_ = nullptr;
  }
}

void DeConvolutionWinogradCPUKernel::FreeDeconvParam() {
  if (deconv_param_ != nullptr) {
    for (int i = 0; i < deconv_param_->compute_size_; i++) {
      DeConvComputeUnit &unit = deconv_param_->compute_units_[i];

      if (unit.weight_ != nullptr) {
        free(unit.weight_);
        unit.weight_ = nullptr;
      }

      if (unit.use_winograd_) {
        if (unit.winograd_.AT_ != nullptr) {
          free(unit.winograd_.AT_);
          unit.winograd_.AT_ = nullptr;
        }
        if (unit.winograd_.BT_ != nullptr) {
          free(unit.winograd_.BT_);
          unit.winograd_.BT_ = nullptr;
        }
      }
    }

    if (deconv_param_->compute_units_ != nullptr) {
      free(deconv_param_->compute_units_);
      deconv_param_->compute_units_ = nullptr;
    }

    delete (deconv_param_);
    deconv_param_ = nullptr;
  }
}

int DeConvolutionWinogradCPUKernel::InitParameter() {
  deconv_param_->input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  deconv_param_->output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  deconv_param_->in_tile_w_count_ = UP_DIV(conv_param_->input_w_, DECONV_WINOGRAD_DEFAULT_UNIT);
  deconv_param_->in_tile_h_count_ = UP_DIV(conv_param_->input_h_, DECONV_WINOGRAD_DEFAULT_UNIT);

  deconv_param_->in_tile_count_ =
    UP_DIV(deconv_param_->in_tile_w_count_ * deconv_param_->in_tile_h_count_, DECONV_WINOGRAD_DEFAULT_TILE);
  deconv_param_->thread_num_ = MSMAX(1, op_parameter_->thread_num_);
  deconv_param_->thread_num_ = MSMIN(deconv_param_->thread_num_, deconv_param_->in_tile_count_);

  thread_num_hw_ = MSMIN(op_parameter_->thread_num_, deconv_param_->output_plane_);
  MS_CHECK_TRUE_RET(thread_num_hw_ != 0, RET_ERROR);
  thread_stride_hw_ = UP_DIV(deconv_param_->output_plane_, thread_num_hw_);

  int size = deconv_param_->thread_num_ * DECONV_WINOGRAD_DEFAULT_UNIT * DECONV_WINOGRAD_DEFAULT_UNIT *
             DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->ic_up_;
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, size * sizeof(float));
  tile_input_ = reinterpret_cast<float *>(malloc(size * sizeof(float)));
  if (tile_input_ == nullptr) {
    MS_LOG(ERROR) << "tile_input_ error!";
    return RET_NULL_PTR;
  }
  (void)memset(tile_input_, 0, size * sizeof(float));

  deconv_param_->out_tile_w_ = (DECONV_WINOGRAD_DEFAULT_UNIT - 1) * conv_param_->stride_w_ + conv_param_->kernel_w_;
  deconv_param_->out_tile_h_ = (DECONV_WINOGRAD_DEFAULT_UNIT - 1) * conv_param_->stride_h_ + conv_param_->kernel_h_;

  for (int i = 0; i < deconv_param_->compute_size_; i++) {
    DeConvComputeUnit &unit = deconv_param_->compute_units_[i];
    if (unit.use_winograd_) {
      if (!deconv_param_->a_buffer_[unit.winograd_.kh_].buf_init_) {
        deconv_param_->a_buffer_[unit.winograd_.kh_].buf_init_ = true;

        size = unit.winograd_.kh_ * unit.winograd_.kw_ * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->ic_up_;
        deconv_param_->a_buffer_[unit.winograd_.kh_].middle_buffer_ =
          malloc(deconv_param_->thread_num_ * size * sizeof(float));
        if (deconv_param_->a_buffer_[unit.winograd_.kh_].middle_buffer_ == nullptr) {
          MS_LOG(ERROR) << "middle_buffer_ error!";
          return RET_NULL_PTR;
        }
        deconv_param_->a_buffer_[unit.winograd_.kh_].dest_buffer_ =
          malloc(deconv_param_->thread_num_ * size * sizeof(float));
        if (deconv_param_->a_buffer_[unit.winograd_.kh_].dest_buffer_ == nullptr) {
          MS_LOG(ERROR) << "dest_buffer_ error!";
          return RET_NULL_PTR;
        }
      }

      unit.winograd_.b_buffer_ = malloc(deconv_param_->thread_num_ * unit.winograd_.kh_ * unit.winograd_.kw_ *
                                        deconv_param_->oc_up_ * DECONV_WINOGRAD_DEFAULT_TILE * sizeof(float));
      if (unit.winograd_.b_buffer_ == nullptr) {
        MS_LOG(ERROR) << "b_buffer_ error!";
        return RET_NULL_PTR;
      }
      unit.tmp_buffer_ = malloc(deconv_param_->thread_num_ * unit.winograd_.kh_ * unit.winograd_.kw_ *
                                deconv_param_->oc_div_ * DECONV_WINOGRAD_DEFAULT_TILE * tile_num_ * sizeof(float));
      if (unit.tmp_buffer_ == nullptr) {
        MS_LOG(ERROR) << "tmp_buffer_ error!";
        return RET_NULL_PTR;
      }
    } else {
      unit.tmp_buffer_ = malloc(deconv_param_->thread_num_ * deconv_param_->oc_div_ * unit.w_size_ * unit.h_size_ *
                                DECONV_WINOGRAD_DEFAULT_TILE * tile_num_ * sizeof(float));
      if (unit.tmp_buffer_ == nullptr) {
        MS_LOG(ERROR) << "tmp_buffer_ error!";
        return RET_NULL_PTR;
      }
    }
  }
  return RET_OK;
}

int DeConvWgFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto deconvWg = reinterpret_cast<DeConvolutionWinogradCPUKernel *>(cdata);
  auto ret = deconvWg->DoDeconv(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoDeconv error!";
    return ret;
  }
  return RET_OK;
}

int DeConvWgPostFp32Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto deconvWg = reinterpret_cast<DeConvolutionWinogradCPUKernel *>(cdata);
  auto ret = deconvWg->DeDeconvPost(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DeDeconv post error!";
    return ret;
  }
  return RET_OK;
}

int DeConvolutionWinogradCPUKernel::InitComputeParam() {
  MS_CHECK_TRUE_RET(conv_param_->stride_h_ != 0, RET_ERROR);
  MS_CHECK_TRUE_RET(conv_param_->stride_w_ != 0, RET_ERROR);
  CHECK_NULL_RETURN(in_tensors_[1]);
#ifdef ENABLE_AVX
  tile_num_ = C8NUM;
#else
  tile_num_ = C4NUM;
#endif
  auto weight_tensor = in_tensors_[1];
  auto shape = weight_tensor->shape();
  if (std::find(shape.begin(), shape.end(), -1) != shape.end()) {
    MS_LOG(WARNING) << "The shape of weight tensor is invalid.";
    valid_weight_shape_ = false;
    return RET_OK;
  }
  valid_weight_shape_ = true;
  conv_param_->input_channel_ = weight_tensor->Batch();
  conv_param_->output_channel_ = weight_tensor->Channel();
  conv_param_->kernel_w_ = weight_tensor->Width();
  conv_param_->kernel_h_ = weight_tensor->Height();

  deconv_param_->kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  deconv_param_->ic_div_ = UP_DIV(conv_param_->input_channel_, tile_num_);
  deconv_param_->oc_div_ = UP_DIV(conv_param_->output_channel_, tile_num_);
  deconv_param_->ic_up_ = deconv_param_->ic_div_ * tile_num_;
  deconv_param_->oc_up_ = deconv_param_->oc_div_ * tile_num_;

  deconv_param_->compute_size_ = 0;
  for (int si_h = 0; si_h < conv_param_->stride_h_; si_h++) {
    for (int si_w = 0; si_w < conv_param_->stride_w_; si_w++) {
      if (si_h < conv_param_->kernel_h_ && si_w < conv_param_->kernel_w_) {
        deconv_param_->compute_size_++;
      }
    }
  }

  size_t size = (size_t)deconv_param_->compute_size_ * sizeof(DeConvComputeUnit);
  deconv_param_->compute_units_ = reinterpret_cast<DeConvComputeUnit *>(malloc(size));
  if (deconv_param_->compute_units_ == nullptr) {
    return RET_NULL_PTR;
  }
  int cur_count = 0;
  if (conv_param_->stride_h_ == 0 || conv_param_->stride_w_ == 0) {
    MS_LOG(ERROR) << "conv_param_->stride_w_ or conv_param_->stride_h_ is 0";
    return RET_ERROR;
  }
  for (int si_h = 0; si_h < conv_param_->stride_h_; si_h++) {
    if (si_h >= conv_param_->kernel_h_) {
      continue;
    }
    for (int si_w = 0; si_w < conv_param_->stride_w_; si_w++) {
      if (si_w >= conv_param_->kernel_w_) {
        continue;
      }

      int h_size = 1 + (conv_param_->kernel_h_ - si_h - 1) / conv_param_->stride_h_;
      int w_size = 1 + (conv_param_->kernel_w_ - si_w - 1) / conv_param_->stride_w_;

      DeConvComputeUnit unit;
      unit.winograd_.AT_ = nullptr;
      unit.winograd_.BT_ = nullptr;

      unit.h_start_ = si_h;
      unit.w_start_ = si_w;
      unit.h_size_ = h_size;
      unit.w_size_ = w_size;

      unit.use_winograd_ = false;
      if (h_size == w_size) {
        unit.winograd_.k_ = unit.h_size_;
        unit.winograd_.i_ = DECONV_WINOGRAD_DEFAULT_UNIT;
        unit.winograd_.o_ = DECONV_WINOGRAD_DEFAULT_UNIT + unit.h_size_ - 1;
        unit.winograd_.kh_ = unit.h_size_ + DECONV_WINOGRAD_DEFAULT_UNIT - 1;
        unit.winograd_.kw_ = unit.w_size_ + DECONV_WINOGRAD_DEFAULT_UNIT - 1;
        unit.use_winograd_ =
          unit.winograd_.kh_ < DECONV_WINOGRAD_BUFFER_COUNT && unit.winograd_.kw_ < DECONV_WINOGRAD_BUFFER_COUNT;
      }
      if (unit.use_winograd_) {
        unit.winograd_.b_buffer_ = nullptr;
        unit.weight_ = malloc(unit.winograd_.kh_ * unit.winograd_.kw_ * deconv_param_->oc_up_ * deconv_param_->ic_up_ *
                              sizeof(float));
        if (unit.weight_ == nullptr) {
          MS_LOG(ERROR) << "weight_ error!";
          return RET_NULL_PTR;
        }
      } else {
        unit.weight_ = malloc(h_size * w_size * deconv_param_->ic_up_ * deconv_param_->oc_up_ * sizeof(float));
        if (unit.weight_ == nullptr) {
          MS_LOG(ERROR) << "weight_ error!";
          return RET_NULL_PTR;
        }
      }
      unit.tmp_buffer_ = nullptr;
      deconv_param_->compute_units_[cur_count] = unit;
      cur_count++;
    }
  }
  return RET_OK;
}

int DeConvolutionWinogradCPUKernel::InitDataParam() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  auto nhwc_weight = reinterpret_cast<float *>(weight_tensor->data());
  if (nhwc_weight == nullptr) {
    MS_LOG(WARNING) << "The weight data is nullptr, will init data parameter in runtime.";
    is_repack_ = true;
    return RET_OK;
  }

  /* unit data : weight & winograd data */
  for (int i = 0; i < deconv_param_->compute_size_; i++) {
    DeConvComputeUnit *unit = &deconv_param_->compute_units_[i];
    int ret = PackDeConvWgDataFp32(nhwc_weight, unit, conv_param_, deconv_param_);
    if (ret != RET_OK) {
      return ret;
    }
  }

  /* bias */
  if (bias_data_ != nullptr) {
    free(bias_data_);
  }
  bias_data_ = malloc(deconv_param_->oc_up_ * sizeof(float));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "bias_data_ error!";
    return RET_NULL_PTR;
  }
  (void)memset(bias_data_, 0, deconv_param_->oc_up_ * sizeof(float));

  if (in_tensors_.size() == kInputSize2) {
    auto bias_tensor = in_tensors_.at(kBiasIndex);
    CHECK_NULL_RETURN(bias_tensor);
    CHECK_NULL_RETURN(bias_tensor->data());
    if (bias_tensor->shape().size() == 1 && bias_tensor->DimensionSize(0) == conv_param_->output_channel_) {
      (void)memcpy(bias_data_, bias_tensor->data(), conv_param_->output_channel_ * sizeof(float));
    }
  }
  return RET_OK;
}

int DeConvolutionWinogradCPUKernel::ReSize() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(out_tensors_.at(kOutputIndex));
  CHECK_NULL_RETURN(conv_param_);
  CHECK_NULL_RETURN(deconv_param_);

  auto ret = ConvolutionBaseCPUKernel::CheckDeconvResizeValid();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Resize is invalid.";
    return ret;
  }

  auto weight_tensor = in_tensors_.at(kWeightIndex);
  CHECK_NULL_RETURN(weight_tensor);
  CHECK_NOT_EQUAL_RETURN(conv_param_->kernel_h_, weight_tensor->Height());
  CHECK_NOT_EQUAL_RETURN(conv_param_->kernel_w_, weight_tensor->Width());

  FreeResizeBuf();
  ret = ConvolutionBaseCPUKernel::Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "prepare is failed!";
    return ret;
  }
  if (!valid_weight_shape_) {
    if (InitComputeParam() != RET_OK) {
      MS_LOG(ERROR) << "InitComputeParam error!";
      return RET_ERROR;
    } else if (!valid_weight_shape_) {
      return RET_OK;
    }
    if (InitDataParam() != RET_OK) {
      MS_LOG(ERROR) << "InitDataParam error!";
      return RET_ERROR;
    }
  }

  int error_code = InitParameter();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "InitParameter error! ret: " << error_code;
    return error_code;
  }
  if (conv_param_->output_channel_ * conv_param_->output_h_ * conv_param_->output_w_ <= kDeconvWinogradMaxPixel) {
    deconv_param_->thread_num_ = MSMIN(deconv_param_->thread_num_, C3NUM);
  }
  return RET_OK;
}

int DeConvolutionWinogradCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(kInputIndex));
  CHECK_NULL_RETURN(in_tensors_.at(kWeightIndex));
  CHECK_NULL_RETURN(out_tensors_.at(kOutputIndex));
  CHECK_NULL_RETURN(conv_param_);
  UpdateOriginWeightAndBias();

  deconv_param_ = new (std::nothrow) DeConvParam();
  if (deconv_param_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  for (auto &wg : deconv_param_->a_buffer_) {
    wg.buf_init_ = false;
  }

  if (InitComputeParam() != RET_OK) {
    MS_LOG(ERROR) << "InitDataParam error!";
    return RET_ERROR;
  }
  if (valid_weight_shape_ && InitDataParam() != RET_OK) {
    MS_LOG(ERROR) << "InitDataParam error!";
    return RET_ERROR;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int DeConvolutionWinogradCPUKernel::DoDeconv(int task_id) {
  for (int tile_index = task_id; tile_index < deconv_param_->in_tile_count_; tile_index += deconv_param_->thread_num_) {
    float *tile_in = tile_input_ + task_id * DECONV_WINOGRAD_DEFAULT_UNIT * DECONV_WINOGRAD_DEFAULT_UNIT *
                                     DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->ic_up_;
    int size = deconv_param_->out_tile_w_ * deconv_param_->out_tile_h_ * DECONV_WINOGRAD_DEFAULT_TILE *
               deconv_param_->oc_div_ * tile_num_;
    float *tile_out = tile_output_ + task_id * size;
    (void)memset(tile_out, 0, size * sizeof(float));

    int start_index = tile_index * DECONV_WINOGRAD_DEFAULT_TILE;
    int calculate_count = MSMIN(DECONV_WINOGRAD_DEFAULT_TILE,
                                deconv_param_->in_tile_w_count_ * deconv_param_->in_tile_h_count_ - start_index);

    auto ret =
      DeconvWg(nhwc_input_, tile_in, tile_out, start_index, calculate_count, conv_param_, deconv_param_, task_id);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "DeconvWg is error";
      return ret;
    }
    std::unique_lock<std::mutex> merge_lock(lock_);
    ret = DeconvWgPost(tile_out, nc4hw4_output_, conv_param_, deconv_param_, calculate_count, tile_index);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "DeconvWgPost is error";
      return ret;
    }
  }
  return RET_OK;
}

int DeConvolutionWinogradCPUKernel::DeDeconvPost(int task_id) {
  int rest_plane = deconv_param_->output_plane_ - task_id * thread_stride_hw_;
  int current_plane = MSMIN(rest_plane, thread_stride_hw_);
  if (current_plane <= 0) {
    return RET_OK;
  }

  WinogradPostConvFuncFp32CX(nc4hw4_output_ + task_id * thread_stride_hw_ * tile_num_,
                             nhwc_output_ + task_id * thread_stride_hw_ * conv_param_->output_channel_,
                             reinterpret_cast<float *>(bias_data_), conv_param_->output_channel_, current_plane,
                             deconv_param_->output_plane_, conv_param_->act_type_);
  return RET_OK;
}

int DeConvolutionWinogradCPUKernel::InitRunBuf() {
  int size = deconv_param_->oc_up_ * deconv_param_->output_plane_;
  nc4hw4_output_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(size * sizeof(float)));
  if (nc4hw4_output_ == nullptr) {
    MS_LOG(ERROR) << "de conv wg Malloc nc4hw4_output_ error!";
    return RET_MEMORY_FAILED;
  }

  size = deconv_param_->thread_num_ * deconv_param_->out_tile_w_ * deconv_param_->out_tile_h_ *
         DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->oc_up_;
  tile_output_ = reinterpret_cast<float *>(ctx_->allocator->Malloc(size * sizeof(float)));
  if (tile_output_ == nullptr) {
    MS_LOG(ERROR) << "de conv wg Malloc tile_output_ error!";
    return RET_MEMORY_FAILED;
  }
  return RET_OK;
}

void DeConvolutionWinogradCPUKernel::FreeRunBuf() {
  if (nc4hw4_output_ != nullptr) {
    ctx_->allocator->Free(nc4hw4_output_);
    nc4hw4_output_ = nullptr;
  }

  if (tile_output_ != nullptr) {
    ctx_->allocator->Free(tile_output_);
    tile_output_ = nullptr;
  }
}

int DeConvolutionWinogradCPUKernel::Run() {
  auto ret = InitRunBuf();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InitRunBuf fail!ret: " << ret;
    FreeRunBuf();
    return ret;
  }

  if (!valid_weight_shape_) {
    if (InitComputeParam() != RET_OK) {
      MS_LOG(ERROR) << "InitDataParam error!";
      FreeRunBuf();
      return RET_ERROR;
    }
    if (!valid_weight_shape_ || InitParameter() != RET_OK) {
      MS_LOG(ERROR) << "InitDataParam error!";
      FreeRunBuf();
      return RET_ERROR;
    }
  }
  if (IsRepack() && InitDataParam() != RET_OK) {
    MS_LOG(ERROR) << "InitDataParam error!";
    FreeRunBuf();
    return RET_ERROR;
  }

  auto input_tensor = in_tensors_.at(kInputIndex);
  auto output_tensor = out_tensors_.at(kOutputIndex);
  auto src_in = reinterpret_cast<float *>(input_tensor->data());
  auto src_out = reinterpret_cast<float *>(output_tensor->data());
  CHECK_NULL_RETURN(src_in);
  CHECK_NULL_RETURN(src_out);

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    nhwc_input_ = src_in + batch_index * deconv_param_->input_plane_ * conv_param_->input_channel_;
    nhwc_output_ = src_out + batch_index * deconv_param_->output_plane_ * conv_param_->output_channel_;

    (void)memset(nc4hw4_output_, 0, deconv_param_->output_plane_ * deconv_param_->oc_div_ * tile_num_ * sizeof(float));
    ret = ParallelLaunch(this->ms_context_, DeConvWgFp32Run, this, deconv_param_->thread_num_);
    if (ret != RET_OK) {
      FreeRunBuf();
      MS_LOG(ERROR) << "DeConvWgFp32Run failed!";
      return ret;
    }

    /* post bias activate and nhwc */
    ret = ParallelLaunch(this->ms_context_, DeConvWgPostFp32Run, this, thread_num_hw_);
    if (ret != RET_OK) {
      FreeRunBuf();
      MS_LOG(ERROR) << "DeConvWgPostFp32Run failed!";
      return ret;
    }
  }

  FreeRunBuf();
  return RET_OK;
}
}  // namespace mindspore::kernel
