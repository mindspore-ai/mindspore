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

#include "src/runtime/kernel/arm/fp16/deconvolution_winograd_fp16.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
DeConvWinogradFp16CPUKernel::~DeConvWinogradFp16CPUKernel() {
  FreeResizeBuf();
  FreeDeconvParam();
  return;
}

void DeConvWinogradFp16CPUKernel::FreeResizeBuf() {
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

  for (int i = 0; i < DECONV_WINOGRAD_BUFFER_COUNT; i++) {
    DeConvWgABuffer &wg = deconv_param_->a_buffer_[i];
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

  if (tile_output_ != nullptr) {
    free(tile_output_);
    tile_output_ = nullptr;
  }

  if (nc4hw4_output_ != nullptr) {
    free(nc4hw4_output_);
    nc4hw4_output_ = nullptr;
  }
  return;
}

void DeConvWinogradFp16CPUKernel::FreeDeconvParam() {
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
  return;
}

int DeConvWinogradFp16CPUKernel::InitParameter() {
  deconv_param_->input_plane_ = conv_param_->input_h_ * conv_param_->input_w_;
  deconv_param_->output_plane_ = conv_param_->output_h_ * conv_param_->output_w_;

  nc4hw4_output_ =
    reinterpret_cast<float16_t *>(malloc(deconv_param_->oc_up4_ * deconv_param_->output_plane_ * sizeof(float16_t)));
  if (nc4hw4_output_ == nullptr) {
    return RET_NULL_PTR;
  }

  deconv_param_->in_tile_w_count_ = UP_DIV(conv_param_->input_w_, DECONV_WINOGRAD_DEFAULT_UNIT);
  deconv_param_->in_tile_h_count_ = UP_DIV(conv_param_->input_h_, DECONV_WINOGRAD_DEFAULT_UNIT);

  deconv_param_->in_tile_count_ =
    UP_DIV(deconv_param_->in_tile_w_count_ * deconv_param_->in_tile_h_count_, DECONV_WINOGRAD_DEFAULT_TILE);
  deconv_param_->thread_num_ = MSMAX(1, op_parameter_->thread_num_);
  deconv_param_->thread_num_ = MSMIN(deconv_param_->thread_num_, deconv_param_->in_tile_count_);

  thread_num_hw_ = MSMIN(op_parameter_->thread_num_, deconv_param_->output_plane_);
  thread_stride_hw_ = UP_DIV(deconv_param_->output_plane_, thread_num_hw_);

  int size = deconv_param_->thread_num_ * DECONV_WINOGRAD_DEFAULT_UNIT * DECONV_WINOGRAD_DEFAULT_UNIT *
             DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->ic_up4_;
  tile_input_ = reinterpret_cast<float16_t *>(malloc(size * sizeof(float16_t)));
  if (tile_input_ == nullptr) {
    return RET_NULL_PTR;
  }
  memset(tile_input_, 0, size * sizeof(float16_t));

  deconv_param_->out_tile_w_ = (DECONV_WINOGRAD_DEFAULT_UNIT - 1) * conv_param_->stride_w_ + conv_param_->kernel_w_;
  deconv_param_->out_tile_h_ = (DECONV_WINOGRAD_DEFAULT_UNIT - 1) * conv_param_->stride_h_ + conv_param_->kernel_h_;
  size = deconv_param_->thread_num_ * deconv_param_->out_tile_w_ * deconv_param_->out_tile_h_ *
         DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->oc_up4_;
  tile_output_ = reinterpret_cast<float16_t *>(malloc(size * sizeof(float16_t)));
  if (tile_output_ == nullptr) {
    return RET_NULL_PTR;
  }

  for (int i = 0; i < deconv_param_->compute_size_; i++) {
    DeConvComputeUnit &unit = deconv_param_->compute_units_[i];
    if (unit.use_winograd_) {
      if (deconv_param_->a_buffer_[unit.winograd_.kh_].buf_init_ == false) {
        deconv_param_->a_buffer_[unit.winograd_.kh_].buf_init_ = true;

        size = unit.winograd_.kh_ * unit.winograd_.kw_ * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->ic_up4_;
        deconv_param_->a_buffer_[unit.winograd_.kh_].middle_buffer_ =
          malloc(deconv_param_->thread_num_ * size * sizeof(float16_t));
        if (deconv_param_->a_buffer_[unit.winograd_.kh_].middle_buffer_ == nullptr) {
          return RET_NULL_PTR;
        }
        deconv_param_->a_buffer_[unit.winograd_.kh_].dest_buffer_ =
          malloc(deconv_param_->thread_num_ * size * sizeof(float16_t));
        if (deconv_param_->a_buffer_[unit.winograd_.kh_].dest_buffer_ == nullptr) {
          return RET_NULL_PTR;
        }
      }

      unit.winograd_.b_buffer_ = malloc(deconv_param_->thread_num_ * unit.winograd_.kh_ * unit.winograd_.kw_ *
                                        deconv_param_->oc_up4_ * DECONV_WINOGRAD_DEFAULT_TILE * sizeof(float16_t));
      if (unit.winograd_.b_buffer_ == nullptr) {
        return RET_NULL_PTR;
      }
      unit.tmp_buffer_ = malloc(deconv_param_->thread_num_ * unit.winograd_.kh_ * unit.winograd_.kw_ *
                                deconv_param_->oc_div4_ * DECONV_WINOGRAD_DEFAULT_TILE * C4NUM * sizeof(float16_t));
      if (unit.tmp_buffer_ == nullptr) {
        return RET_NULL_PTR;
      }

    } else {
      unit.tmp_buffer_ = malloc(deconv_param_->thread_num_ * deconv_param_->oc_div4_ * unit.w_size_ * unit.h_size_ *
                                DECONV_WINOGRAD_DEFAULT_TILE * C4NUM * sizeof(float16_t));
      if (unit.tmp_buffer_ == nullptr) {
        return RET_NULL_PTR;
      }
    }
  }

  return RET_OK;
}

int DeConvWinogradFp16CPUKernel::DoDeconv(int task_id) {
  for (int tile_index = task_id; tile_index < deconv_param_->in_tile_count_; tile_index += deconv_param_->thread_num_) {
    float16_t *tile_in = tile_input_ + task_id * DECONV_WINOGRAD_DEFAULT_UNIT * DECONV_WINOGRAD_DEFAULT_UNIT *
                                         DECONV_WINOGRAD_DEFAULT_TILE * deconv_param_->ic_up4_;
    int size = deconv_param_->out_tile_w_ * deconv_param_->out_tile_h_ * DECONV_WINOGRAD_DEFAULT_TILE *
               deconv_param_->oc_div4_ * C4NUM;
    float16_t *tile_out = tile_output_ + task_id * size;
    memset(tile_out, 0, size * sizeof(float16_t));

    int start_index = tile_index * DECONV_WINOGRAD_DEFAULT_TILE;
    int calculate_count = MSMIN(DECONV_WINOGRAD_DEFAULT_TILE,
                                deconv_param_->in_tile_w_count_ * deconv_param_->in_tile_h_count_ - start_index);

    DeconvWgFp16(nhwc_input_, tile_in, tile_out, start_index, calculate_count, conv_param_, deconv_param_, task_id);

    std::unique_lock<std::mutex> merge_lock(lock_);
    DeconvWgPostFp16(tile_out, nc4hw4_output_, conv_param_, deconv_param_, calculate_count, tile_index);
  }
  return RET_OK;
}

int DeConvWinogradFp16CPUKernel::DeDeconvPost(int task_id) {
  int rest_plane = deconv_param_->output_plane_ - task_id * thread_stride_hw_;
  int current_plane = MSMIN(rest_plane, thread_stride_hw_);
  if (current_plane <= 0) {
    return RET_OK;
  }

  PostConvFuncFp16C4(nc4hw4_output_ + task_id * thread_stride_hw_ * C4NUM,
                     nhwc_output_ + task_id * thread_stride_hw_ * conv_param_->output_channel_,
                     reinterpret_cast<float16_t *>(bias_data_), conv_param_->output_channel_, current_plane,
                     deconv_param_->output_plane_, conv_param_->act_type_);
  return RET_OK;
}

int DeConvWgFp16Run(void *cdata, int task_id) {
  auto deconvWg = reinterpret_cast<DeConvWinogradFp16CPUKernel *>(cdata);
  deconvWg->DoDeconv(task_id);
  return RET_OK;
}

int DeConvWgPostFp16Run(void *cdata, int task_id) {
  auto deconvWg = reinterpret_cast<DeConvWinogradFp16CPUKernel *>(cdata);
  deconvWg->DeDeconvPost(task_id);
  return RET_OK;
}

int DeConvWinogradFp16CPUKernel::InitComputeParam() {
  auto weight_tensor = in_tensors_.at(1);

  conv_param_->input_channel_ = weight_tensor->Batch();
  conv_param_->output_channel_ = weight_tensor->Channel();
  conv_param_->kernel_w_ = weight_tensor->Width();
  conv_param_->kernel_h_ = weight_tensor->Height();

  deconv_param_->kernel_plane_ = conv_param_->kernel_w_ * conv_param_->kernel_h_;
  deconv_param_->ic_div4_ = UP_DIV(conv_param_->input_channel_, C4NUM);
  deconv_param_->oc_div4_ = UP_DIV(conv_param_->output_channel_, C4NUM);
  deconv_param_->ic_up4_ = deconv_param_->ic_div4_ * C4NUM;
  deconv_param_->oc_up4_ = deconv_param_->oc_div4_ * C4NUM;

  deconv_param_->compute_size_ = 0;
  for (int si_h = 0; si_h < conv_param_->stride_h_; si_h++) {
    for (int si_w = 0; si_w < conv_param_->stride_w_; si_w++) {
      if (si_h < conv_param_->kernel_h_ && si_w < conv_param_->kernel_w_) {
        deconv_param_->compute_size_++;
      }
    }
  }

  int size = deconv_param_->compute_size_ * sizeof(DeConvComputeUnit);
  deconv_param_->compute_units_ = reinterpret_cast<DeConvComputeUnit *>(malloc(size));
  if (deconv_param_->compute_units_ == nullptr) {
    return RET_NULL_PTR;
  }
  int cur_count = 0;
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

      unit.h_start_ = si_h;
      unit.w_start_ = si_w;
      unit.h_size_ = h_size;
      unit.w_size_ = w_size;

      if (h_size == w_size) {
        unit.use_winograd_ = true;

        unit.winograd_.k_ = unit.h_size_;
        unit.winograd_.i_ = DECONV_WINOGRAD_DEFAULT_UNIT;
        unit.winograd_.o_ = DECONV_WINOGRAD_DEFAULT_UNIT + unit.h_size_ - 1;
        unit.winograd_.kh_ = unit.h_size_ + DECONV_WINOGRAD_DEFAULT_UNIT - 1;
        unit.winograd_.kw_ = unit.w_size_ + DECONV_WINOGRAD_DEFAULT_UNIT - 1;

        unit.winograd_.b_buffer_ = nullptr;
        unit.weight_ = malloc(unit.winograd_.kh_ * unit.winograd_.kw_ * deconv_param_->oc_up4_ *
                              deconv_param_->ic_up4_ * sizeof(float16_t));
        if (unit.weight_ == nullptr) {
          return RET_NULL_PTR;
        }
      } else {
        unit.use_winograd_ = false;
        unit.weight_ = malloc(h_size * w_size * deconv_param_->ic_up4_ * deconv_param_->oc_up4_ * sizeof(float16_t));
        if (unit.weight_ == nullptr) {
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

int DeConvWinogradFp16CPUKernel::InitDataParam() {
  /* unit data : weight & winograd data*/
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto ret = ConvolutionBaseFP16CPUKernel::GetExecuteFilter(weight_tensor, weight_tensor->data_c());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Get Execute filter failed.";
    return ret;
  }

  for (int i = 0; i < deconv_param_->compute_size_; i++) {
    DeConvComputeUnit *unit = &deconv_param_->compute_units_[i];
    ret = PackDeConvWgDataFp16(execute_weight_, unit, conv_param_, deconv_param_);
    if (ret != RET_OK) {
      return ret;
    }
  }

  /* bias */
  bias_data_ = malloc(deconv_param_->oc_up4_ * sizeof(float16_t));
  if (bias_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc bias_data_ failed.";
    return RET_ERROR;
  }
  memset(bias_data_, 0, deconv_param_->oc_up4_ * sizeof(float16_t));
  auto fp16_bias_data = reinterpret_cast<float16_t *>(bias_data_);
  if (in_tensors_.size() == 3 && in_tensors_.at(kBiasIndex)->shape().size() == 1 &&
      in_tensors_.at(kBiasIndex)->DimensionSize(0) == conv_param_->output_channel_) {
    auto src_bias = reinterpret_cast<float *>(in_tensors_.at(kBiasIndex)->MutableData());
    MS_ASSERT(src_bias);
    for (int i = 0; i < conv_param_->output_channel_; ++i) {
      fp16_bias_data[i] = (float16_t)src_bias[i];
    }
  } else {
    MS_ASSERT(in_tensors_.size() == kInputSize1);
  }

  return RET_OK;
}

int DeConvWinogradFp16CPUKernel::ReSize() {
  FreeResizeBuf();
  ConvolutionBaseCPUKernel::Init();
  InitParameter();
  return RET_OK;
}

int DeConvWinogradFp16CPUKernel::Init() {
  deconv_param_ = new (std::nothrow) DeConvParam();
  if (deconv_param_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  for (auto &wg : deconv_param_->a_buffer_) {
    wg.buf_init_ = false;
    wg.dest_buffer_ = nullptr;
    wg.middle_buffer_ = nullptr;
  }
  int error_code = InitComputeParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "InitComputeParam error! ret: " << error_code;
    return error_code;
  }

  error_code = InitDataParam();
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "InitWeightBias error! ret: " << error_code;
    return error_code;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int DeConvWinogradFp16CPUKernel::Run() {
  ConvolutionBaseFP16CPUKernel::GetExecuteTensor();

  for (int batch_index = 0; batch_index < conv_param_->input_batch_; batch_index++) {
    nhwc_input_ = execute_input_ + batch_index * deconv_param_->input_plane_ * conv_param_->input_channel_;
    nhwc_output_ = execute_output_ + batch_index * deconv_param_->output_plane_ * conv_param_->output_channel_;

    ::memset(nc4hw4_output_, 0, deconv_param_->output_plane_ * deconv_param_->oc_div4_ * C4NUM * sizeof(float16_t));
    ParallelLaunch(this->context_->thread_pool_, DeConvWgFp16Run, this, deconv_param_->thread_num_);

    /*post bias activate and nhwc */
    ParallelLaunch(this->context_->thread_pool_, DeConvWgPostFp16Run, this, thread_num_hw_);
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
