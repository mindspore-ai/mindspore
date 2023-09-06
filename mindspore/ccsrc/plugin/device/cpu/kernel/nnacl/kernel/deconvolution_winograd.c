/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef _WIN32
#ifndef ENABLE_MCU
#include "nnacl/kernel/deconvolution_winograd.h"
#include "nnacl/infer/common_infer.h"
#include "nnacl/fp32/deconv_winograd_fp32.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/kernel/deconvolution.h"

void DeConvWinogradFreeResizeBuf(DeConvWinogradStruct *deconv) {
  DeConvParam *param = &deconv->param_;

  for (int i = 0; i < param->compute_size_; i++) {
    DeConvComputeUnit *unit = &param->compute_units_[i];
    if (unit->tmp_buffer_ != NULL) {
      free(unit->tmp_buffer_);
      unit->tmp_buffer_ = NULL;
    }

    if (unit->use_winograd_) {
      if (unit->winograd_.b_buffer_ != NULL) {
        free(unit->winograd_.b_buffer_);
        unit->winograd_.b_buffer_ = NULL;
      }
    }
  }

  for (int i = 0; i < DECONV_WINOGRAD_BUFFER_COUNT; i++) {
    DeConvWgABuffer *wg = &param->a_buffer_[i];
    if (wg->buf_init_) {
      if (wg->dest_buffer_ != NULL) {
        free(wg->dest_buffer_);
        wg->dest_buffer_ = NULL;
      }
      if (wg->middle_buffer_ != NULL) {
        free(wg->middle_buffer_);
        wg->middle_buffer_ = NULL;
      }
    }
    wg->buf_init_ = false;
  }

  if (deconv->tile_input_ != NULL) {
    free(deconv->tile_input_);
    deconv->tile_input_ = NULL;
  }
}

void DeConvWinogradFreeDeconvParam(DeConvWinogradStruct *deconv) {
  DeConvParam *param = &deconv->param_;

  for (int i = 0; i < param->compute_size_; i++) {
    DeConvComputeUnit *unit = &param->compute_units_[i];

    if (unit->weight_ != NULL) {
      free(unit->weight_);
      unit->weight_ = NULL;
    }

    if (unit->use_winograd_) {
      if (unit->winograd_.AT_ != NULL) {
        free(unit->winograd_.AT_);
        unit->winograd_.AT_ = NULL;
      }
      if (unit->winograd_.BT_ != NULL) {
        free(unit->winograd_.BT_);
        unit->winograd_.BT_ = NULL;
      }
    }
  }

  if (param->compute_units_ != NULL) {
    free(param->compute_units_);
    param->compute_units_ = NULL;
  }
}

int DeConvWinogradInitParameter(DeConvWinogradStruct *deconv) {
  DeConvParam *param = &deconv->param_;
  ConvComputeParam *compute = &deconv->conv_.compute_;

  int thread_num = deconv->conv_.base_.thread_nr_;
  NNACL_CHECK_ZERO_RETURN_ERR(thread_num);

  param->input_plane_ = compute->in_hw_;
  param->output_plane_ = compute->out_hw_;

  param->in_tile_w_count_ = UP_DIV(compute->in_w_, WINOGRAD_DEFAULT_UNIT);
  NNACL_CHECK_ZERO_RETURN_ERR(param->in_tile_w_count_);
  param->in_tile_h_count_ = UP_DIV(compute->in_h_, WINOGRAD_DEFAULT_UNIT);
  NNACL_CHECK_ZERO_RETURN_ERR(param->in_tile_h_count_);
  param->in_tile_count_ = UP_DIV(param->in_tile_w_count_ * param->in_tile_h_count_, WINOGRAD_DEFAULT_TILE);

  deconv->conv_.base_.thread_nr_ = NNACL_MAX(1, deconv->conv_.base_.thread_nr_);
  deconv->conv_.base_.thread_nr_ = NNACL_MIN(deconv->conv_.base_.thread_nr_, param->in_tile_count_);

  deconv->thread_num_hw_ = NNACL_MIN(deconv->conv_.base_.thread_nr_, compute->out_hw_);
  NNACL_CHECK_ZERO_RETURN_ERR(deconv->thread_num_hw_);
  deconv->thread_stride_hw_ = UP_DIV(compute->out_hw_, deconv->thread_num_hw_);

  int total_ic_up = WINOGRAD_DEFAULT_UNIT * WINOGRAD_DEFAULT_UNIT * WINOGRAD_DEFAULT_TILE * param->ic_up_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->conv_.base_.thread_nr_, total_ic_up, NNACL_ERR);
  int size = deconv->conv_.base_.thread_nr_ * total_ic_up;
  NNACL_CHECK_MALLOC_SIZE(size * sizeof(float));
  deconv->tile_input_ = (float *)malloc(size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->tile_input_);
  (void)memset(deconv->tile_input_, 0, size * sizeof(float));

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW((WINOGRAD_DEFAULT_UNIT - 1), compute->stride_w_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW((WINOGRAD_DEFAULT_UNIT - 1), compute->stride_h_, NNACL_ERR);
  param->out_tile_w_ = (WINOGRAD_DEFAULT_UNIT - 1) * compute->stride_w_ + compute->kernel_w_;
  param->out_tile_h_ = (WINOGRAD_DEFAULT_UNIT - 1) * compute->stride_h_ + compute->kernel_h_;

  for (int i = 0; i < param->compute_size_; i++) {
    DeConvComputeUnit *unit = &param->compute_units_[i];
    if (unit->use_winograd_) {
      if (!param->a_buffer_[unit->winograd_.kh_].buf_init_) {
        param->a_buffer_[unit->winograd_.kh_].buf_init_ = true;
        size = unit->winograd_.kh_ * unit->winograd_.kw_ * WINOGRAD_DEFAULT_TILE * param->ic_up_;

        param->a_buffer_[unit->winograd_.kh_].middle_buffer_ = malloc(thread_num * size * sizeof(float));
        NNACL_MALLOC_CHECK_NULL_RETURN_ERR(param->a_buffer_[unit->winograd_.kh_].middle_buffer_);

        param->a_buffer_[unit->winograd_.kh_].dest_buffer_ = malloc(thread_num * size * sizeof(float));
        NNACL_MALLOC_CHECK_NULL_RETURN_ERR(param->a_buffer_[unit->winograd_.kh_].dest_buffer_);
      }

      size = unit->winograd_.kh_ * unit->winograd_.kw_ * param->oc_up_ * WINOGRAD_DEFAULT_TILE;
      unit->winograd_.b_buffer_ = malloc(thread_num * size * sizeof(float));
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(unit->winograd_.b_buffer_);

      size = unit->winograd_.kh_ * unit->winograd_.kw_ * param->oc_div_ * WINOGRAD_DEFAULT_TILE * compute->tile_num_;
      unit->tmp_buffer_ = malloc(thread_num * size * sizeof(float));
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(unit->tmp_buffer_);
    } else {
      size = param->oc_div_ * unit->w_size_ * unit->h_size_ * WINOGRAD_DEFAULT_TILE * compute->tile_num_;
      unit->tmp_buffer_ = malloc(thread_num * size * sizeof(float));
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(unit->tmp_buffer_);
    }
  }

  return NNACL_OK;
}

int DeConvWgFp32Run(void *cdata, int task_id, float l, float r) {
  DeConvWinogradStruct *deconv = (DeConvWinogradStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);
  ConvParameter *conv_param = (ConvParameter *)deconv->conv_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);
  DeConvParam *param = &deconv->param_;
  ConvComputeParam *compute = &deconv->conv_.compute_;

  for (int tile_index = task_id; tile_index < param->in_tile_count_; tile_index += deconv->conv_.base_.thread_nr_) {
    int size = WINOGRAD_DEFAULT_UNIT * WINOGRAD_DEFAULT_UNIT * WINOGRAD_DEFAULT_TILE * param->ic_up_;
    float *tile_in = deconv->tile_input_ + task_id * size;
    size = param->out_tile_w_ * param->out_tile_h_ * WINOGRAD_DEFAULT_TILE * param->oc_div_ * compute->tile_num_;
    float *tile_out = deconv->tile_output_ + task_id * size;
    (void)memset(tile_out, 0, size * sizeof(float));

    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(tile_index, WINOGRAD_DEFAULT_TILE, NNACL_ERR);
    int start_index = tile_index * WINOGRAD_DEFAULT_TILE;
    int cal_count = NNACL_MIN(WINOGRAD_DEFAULT_TILE, param->in_tile_w_count_ * param->in_tile_h_count_ - start_index);

    int ret = DeconvWg(deconv->nhwc_input_, tile_in, tile_out, start_index, cal_count, conv_param, param, task_id);
    if (ret != NNACL_OK) {
      return ret;
    }

    (void)pthread_mutex_lock(&deconv->lock_);
    (void)DeconvWgPost(tile_out, deconv->nc4hw4_output_, conv_param, param, cal_count, tile_index);
    (void)pthread_mutex_unlock(&deconv->lock_);
  }
  return NNACL_OK;
}

int DeConvWgPostFp32Run(void *cdata, int task_id, float l, float r) {
  DeConvWinogradStruct *deconv = (DeConvWinogradStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);
  ConvComputeParam *compute = &deconv->conv_.compute_;

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, deconv->thread_stride_hw_, NNACL_ERR);
  int output_stride_plane = task_id * deconv->thread_stride_hw_;
  int rest_plane = compute->out_hw_ - output_stride_plane;
  int current_plane = MSMIN(rest_plane, deconv->thread_stride_hw_);
  if (current_plane <= 0) {
    return NNACL_OK;
  }

  ActType act = ((ConvParameter *)deconv->conv_.base_.param_)->act_type_;
  float *bias = (float *)deconv->conv_.bias_data_;

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(output_stride_plane, deconv->conv_.compute_.tile_num_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(output_stride_plane, deconv->conv_.compute_.out_c_, NNACL_ERR);
  WinogradPostConvFuncFp32CX(deconv->nc4hw4_output_ + output_stride_plane * compute->tile_num_,
                             deconv->nhwc_output_ + output_stride_plane * compute->out_c_, bias, compute->out_c_,
                             current_plane, compute->out_hw_, act);
  return NNACL_OK;
}

int DeConvWinogradInitComputeParam(DeConvWinogradStruct *deconv) {
  deconv->valid_weight_shape_ = CheckShaleValid(&deconv->conv_.base_.in_[SECOND_INPUT], Num1);
  if (deconv->valid_weight_shape_ == false) {
    return NNACL_OK;
  }

  ConvComputeParam *compute = &deconv->conv_.compute_;
  DeConvParam *param = &deconv->param_;

  param->kernel_plane_ = compute->kernel_hw_;
  param->ic_div_ = UP_DIV(compute->in_c_, compute->tile_num_);
  param->oc_div_ = UP_DIV(compute->out_c_, compute->tile_num_);
  param->ic_up_ = param->ic_div_ * compute->tile_num_;
  param->oc_up_ = param->oc_div_ * compute->tile_num_;

  param->compute_size_ = 0;
  for (int si_h = 0; si_h < compute->stride_h_; si_h++) {
    for (int si_w = 0; si_w < compute->stride_w_; si_w++) {
      if (si_h < compute->kernel_h_ && si_w < compute->kernel_w_) {
        param->compute_size_++;
      }
    }
  }

  size_t size = (size_t)param->compute_size_ * sizeof(DeConvComputeUnit);
  param->compute_units_ = (DeConvComputeUnit *)(malloc(size));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(param->compute_units_);

  int cur_count = 0;
  for (int si_h = 0; si_h < compute->stride_h_; si_h++) {
    if (si_h >= compute->kernel_h_) {
      continue;
    }
    for (int si_w = 0; si_w < compute->stride_w_; si_w++) {
      if (si_w >= compute->kernel_w_) {
        continue;
      }

      int h_size = 1 + (compute->kernel_h_ - si_h - 1) / compute->stride_h_;
      int w_size = 1 + (compute->kernel_w_ - si_w - 1) / compute->stride_w_;

      DeConvComputeUnit unit;
      unit.winograd_.AT_ = NULL;
      unit.winograd_.BT_ = NULL;

      unit.h_start_ = si_h;
      unit.w_start_ = si_w;
      unit.h_size_ = h_size;
      unit.w_size_ = w_size;

      unit.use_winograd_ = false;
      if (h_size == w_size) {
        unit.winograd_.k_ = unit.h_size_;
        unit.winograd_.i_ = WINOGRAD_DEFAULT_UNIT;
        unit.winograd_.o_ = WINOGRAD_DEFAULT_UNIT + unit.h_size_ - 1;
        unit.winograd_.kh_ = unit.h_size_ + WINOGRAD_DEFAULT_UNIT - 1;
        unit.winograd_.kw_ = unit.w_size_ + WINOGRAD_DEFAULT_UNIT - 1;
        unit.use_winograd_ = unit.winograd_.kh_ < WINOGRAD_MAX_COUNT && unit.winograd_.kw_ < WINOGRAD_MAX_COUNT;
      }
      if (unit.use_winograd_) {
        unit.winograd_.b_buffer_ = NULL;
        unit.weight_ = malloc(unit.winograd_.kh_ * unit.winograd_.kw_ * param->oc_up_ * param->ic_up_ * sizeof(float));
        NNACL_MALLOC_CHECK_NULL_RETURN_ERR(unit.weight_);
      } else {
        unit.weight_ = malloc(h_size * w_size * param->ic_up_ * param->oc_up_ * sizeof(float));
        NNACL_MALLOC_CHECK_NULL_RETURN_ERR(unit.weight_);
      }
      unit.tmp_buffer_ = NULL;
      param->compute_units_[cur_count] = unit;
      cur_count++;
    }
  }
  return NNACL_OK;
}

int DeConvWinogradInitDataParam(DeConvWinogradStruct *deconv) {
  TensorC *weight_tensor = deconv->conv_.base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(weight_tensor);
  float *nhwc_weight = weight_tensor->data_;
  if (nhwc_weight == NULL) {
    deconv->conv_.is_repack_ = true;
    return NNACL_OK;
  }

  DeConvParam *param = &deconv->param_;

  /* unit data : weight & winograd data */
  for (int i = 0; i < param->compute_size_; i++) {
    DeConvComputeUnit *unit = &param->compute_units_[i];
    int ret = PackDeConvWgDataFp32(nhwc_weight, unit, (ConvParameter *)deconv->conv_.base_.param_, param);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  /* bias */
  ExecEnv *env = deconv->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);
  if (deconv->conv_.bias_data_ != NULL) {
    env->Free(env->allocator_, deconv->conv_.bias_data_);
    deconv->conv_.bias_data_ = NULL;
  }
  deconv->conv_.bias_data_ = env->Alloc(env->allocator_, param->oc_up_ * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->conv_.bias_data_);
  (void)memset(deconv->conv_.bias_data_, 0, param->oc_up_ * sizeof(float));

  if (deconv->conv_.base_.in_size_ == THREE_TENSOR) {
    TensorC *bias_tensor = deconv->conv_.base_.in_[THIRD_INPUT];
    if (bias_tensor->shape_size_ == Num1 && GetElementNum(bias_tensor) == deconv->conv_.compute_.out_c_) {
      (void)memcpy(deconv->conv_.bias_data_, bias_tensor->data_, deconv->conv_.compute_.out_c_ * sizeof(float));
    }
  }
  return NNACL_OK;
}

int DeConvWinogradInitRunBuf(DeConvWinogradStruct *deconv) {
  ExecEnv *env = deconv->conv_.base_.env_;

  int size = deconv->param_.oc_up_ * deconv->conv_.compute_.out_hw_;
  deconv->nc4hw4_output_ = (float *)env->Alloc(env->allocator_, size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->nc4hw4_output_);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->param_.out_tile_w_, deconv->param_.out_tile_h_, NNACL_ERR);
  int out_tile_hw = deconv->param_.out_tile_w_ * deconv->param_.out_tile_h_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->conv_.base_.thread_nr_, out_tile_hw, NNACL_ERR);
  int total_out_tile_hw = deconv->conv_.base_.thread_nr_ * out_tile_hw;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(WINOGRAD_DEFAULT_TILE, deconv->param_.oc_up_, NNACL_ERR);
  int tile_oc_up = WINOGRAD_DEFAULT_TILE * deconv->param_.oc_up_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_out_tile_hw, tile_oc_up, NNACL_ERR);
  size = total_out_tile_hw * tile_oc_up;
  deconv->tile_output_ = (float *)env->Alloc(env->allocator_, size * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->tile_output_);

  return NNACL_OK;
}

void DeConvWinogradFreeRunBuf(DeConvWinogradStruct *deconv) {
  ExecEnv *env = deconv->conv_.base_.env_;

  if (deconv->nc4hw4_output_ != NULL) {
    env->Free(env->allocator_, deconv->nc4hw4_output_);
    deconv->nc4hw4_output_ = NULL;
  }

  if (deconv->tile_output_ != NULL) {
    env->Free(env->allocator_, deconv->tile_output_);
    deconv->tile_output_ = NULL;
  }
}

int InitTrainComputeInit(DeConvWinogradStruct *deconv) {
  if (!deconv->valid_weight_shape_) {
    int ret = DeConvWinogradInitComputeParam(deconv);
    if (ret != NNACL_OK) {
      DeConvWinogradFreeRunBuf(deconv);
      return ret;
    }
    if (!deconv->valid_weight_shape_ || DeConvWinogradInitParameter(deconv) != NNACL_OK) {
      DeConvWinogradFreeRunBuf(deconv);
      return NNACL_DECONVOLUTION_DEPTHWISE_INVALID_WEIGHT_SHAPE;
    }
  }

  if (deconv->conv_.is_repack_ && DeConvWinogradInitDataParam(deconv) != NNACL_OK) {
    DeConvWinogradFreeRunBuf(deconv);
    return NNACL_DECONVOLUTION_DEPTHWISE_INVALID_WEIGHT_REPACK;
  }

  return NNACL_OK;
}

int DeConvWinogradPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_ERR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_ERR);

  DeConvWinogradStruct *deconv = (DeConvWinogradStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);
  ConvComputeParam *compute = &deconv->conv_.compute_;
  NNACL_CHECK_FALSE(compute->dilation_h_ != Num1, NNACL_DECONVOLUTION_DEPTHWISE_DILATION_INVALID);
  NNACL_CHECK_FALSE(compute->dilation_w_ != Num1, NNACL_DECONVOLUTION_DEPTHWISE_DILATION_INVALID);
  NNACL_CHECK_FALSE(compute->stride_h_ == Num0, NNACL_DECONVOLUTION_DEPTHWISE_STRIDE_INVALID);
  NNACL_CHECK_FALSE(compute->stride_w_ == Num0, NNACL_DECONVOLUTION_DEPTHWISE_STRIDE_INVALID);

#ifdef ENABLE_AVX
  compute->tile_num_ = C8NUM;
#else
  compute->tile_num_ = C4NUM;
#endif

  ConvBaseUpdateOriginWeightAndBias(&deconv->conv_);

  int ret = DeConvWinogradInitComputeParam(deconv);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (deconv->valid_weight_shape_) {
    ret = DeConvWinogradInitDataParam(deconv);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  // when input data is const tensor, save data in kernel
  TensorC *input_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  if (IsConst(input_tensor)) {
    deconv->origin_input_ = (float *)malloc(GetSize(input_tensor));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(deconv->origin_input_);
    (void)memcpy(deconv->origin_input_, input_tensor->data_, GetSize(input_tensor));
  }
  return NNACL_OK;
}

int DeConvWinogradResize(KernelBase *self) {
  DeConvWinogradStruct *deconv = (DeConvWinogradStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);
  (void)ConvBaseUpdateComputeInfo(&deconv->conv_);

  int ret = DeConvCheckvResizeValid(&deconv->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  DeConvWinogradFreeResizeBuf(deconv);

  ret = ConvBasePrepare(&deconv->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (!deconv->valid_weight_shape_) {
    ret = DeConvWinogradInitComputeParam(deconv);
    if (ret != NNACL_OK) {
      return ret;
    }
    if (!deconv->valid_weight_shape_) {
      return NNACL_OK;
    }
    ret = DeConvWinogradInitDataParam(deconv);
    if (ret != NNACL_OK) {
      return ret;
    }
  }

  ret = DeConvWinogradInitParameter(deconv);
  if (ret != NNACL_OK) {
    return ret;
  }

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(deconv->conv_.compute_.out_hw_, deconv->conv_.compute_.out_c_, NNACL_ERR);
  int output_chw = deconv->conv_.compute_.out_hw_ * deconv->conv_.compute_.out_c_;
  if (output_chw <= kDeconvWinogradMaxPixel) {
    self->thread_nr_ = NNACL_MIN(self->thread_nr_, Num3);
  }
  return NNACL_OK;
}

int DeConvWinogradRelease(KernelBase *self) {
  DeConvWinogradStruct *deconv = (DeConvWinogradStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);

  DeConvWinogradFreeResizeBuf(deconv);
  DeConvWinogradFreeDeconvParam(deconv);

  if (deconv->origin_input_ != NULL) {
    free(deconv->origin_input_);
    deconv->origin_input_ = NULL;
  }
  return NNACL_OK;
}

int DeConvWinogradCompute(KernelBase *self) {
  DeConvWinogradStruct *deconv = (DeConvWinogradStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(deconv);
  DeConvParam *param = &deconv->param_;
  ConvComputeParam *compute_ = &deconv->conv_.compute_;

  int ret = DeConvWinogradInitRunBuf(deconv);
  if (ret != NNACL_OK) {
    DeConvWinogradFreeRunBuf(deconv);
    return ret;
  }

  ret = InitTrainComputeInit(deconv);
  if (ret != NNACL_OK) {
    DeConvWinogradFreeRunBuf(deconv);
    return ret;
  }

  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  float *src_in = deconv->origin_input_ != NULL ? deconv->origin_input_ : (float *)in_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(src_in);
  TensorC *output_tensor = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  float *src_out = (float *)output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(src_out);

  int input_chw = compute_->in_hw_ * compute_->in_c_;
  int output_chw = compute_->out_hw_ * compute_->out_c_;
  for (int batch_index = 0; batch_index < compute_->in_n_; batch_index++) {
    deconv->nhwc_input_ = src_in + batch_index * input_chw;
    deconv->nhwc_output_ = src_out + batch_index * output_chw;

    (void)memset(deconv->nc4hw4_output_, 0, compute_->out_hw_ * param->oc_div_ * compute_->tile_num_ * sizeof(float));
    ret = self->env_->ParallelLaunch(self->env_->thread_pool_, DeConvWgFp32Run, self, self->thread_nr_);
    if (ret != NNACL_OK) {
      DeConvWinogradFreeRunBuf(deconv);
      return ret;
    }

    /* post bias activate and nhwc */
    ret = self->env_->ParallelLaunch(self->env_->thread_pool_, DeConvWgPostFp32Run, self, self->thread_nr_);
    if (ret != NNACL_OK) {
      DeConvWinogradFreeRunBuf(deconv);
      return ret;
    }
  }

  DeConvWinogradFreeRunBuf(deconv);
  return NNACL_OK;
}

ConvolutionBaseStruct *CreateDeConvWinograd(ConvParameter *param) {
  DeConvWinogradStruct *deconv_winograd = (DeConvWinogradStruct *)malloc(sizeof(DeConvWinogradStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(deconv_winograd);
  memset(deconv_winograd, 0, sizeof(DeConvWinogradStruct));

  deconv_winograd->conv_.base_.Prepare = DeConvWinogradPrepare;
  deconv_winograd->conv_.base_.Resize = DeConvWinogradResize;
  deconv_winograd->conv_.base_.Release = DeConvWinogradRelease;
  deconv_winograd->conv_.base_.Compute = DeConvWinogradCompute;
  return &deconv_winograd->conv_;
}
#endif
#endif
