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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either convolutionress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nnacl/kernel/convolution_winograd_base.h"
#include "nnacl/base/minimal_filtering_generator.h"
#include "nnacl/fp32/winograd_transform.h"
#include "nnacl/fp32/conv_winograd_fp32.h"

int ConvWinoBaseMallocWeightBiasData(ConvolutionBaseStruct *conv) {
  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)conv;
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(winograd);

  // set data
  size_t trans_matrix_data_size = winograd->input_unit_ * winograd->input_unit_ * conv->compute_.in_c_ *
                                  UP_ROUND(conv->compute_.out_c_, winograd->oc_block_) * sizeof(float);
  if (!conv->base_.train_session_) {
    if (conv->packed_weight_ == NULL) {
      NNACL_CHECK_MALLOC_SIZE(trans_matrix_data_size);
      conv->packed_weight_ = ConvBaseGetConvPackWeightData(conv, trans_matrix_data_size);
      NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->packed_weight_);
    }
  }

  float matrix_a[CONVOLUTION_WINOGRAD_MATRIX_SIZE];
  float matrix_at[CONVOLUTION_WINOGRAD_MATRIX_SIZE];
  float matrix_b[CONVOLUTION_WINOGRAD_MATRIX_SIZE];
  float matrix_bt[CONVOLUTION_WINOGRAD_MATRIX_SIZE];
  float coef = 1.0f;
  if (winograd->input_unit_ == CONVOLUTION_WINOGRAD_INPUT_UNIT_SIZE) {
    coef = 0.5f;
  }
  int ret = CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, winograd->matrix_g_, winograd->matrix_gt_, coef,
                           winograd->output_unit_, winograd->kernel_unit_);
  if (ret != NNACL_OK) {
    return ret;
  }

  // init bias
  size_t new_bias_size = UP_ROUND(conv->compute_.out_c_, C4NUM) * sizeof(float);
  if (conv->bias_data_ == NULL) {
    NNACL_CHECK_MALLOC_SIZE(new_bias_size);
    conv->bias_data_ = conv->base_.env_->Alloc(conv->base_.env_->allocator_, new_bias_size);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(conv->bias_data_);
  }
  memset(conv->bias_data_, 0, new_bias_size);
  return NNACL_OK;
}

void ConvWinoBaseFreeTmpBuffer(ConvolutionWinogradBaseStruct *winograd) {
  ExecEnv *env = winograd->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_VOID(env);

  if (winograd->trans_input_ != NULL) {
    env->Free(env->allocator_, winograd->trans_input_);
    winograd->trans_input_ = NULL;
  }
  if (winograd->tmp_data_ != NULL) {
    env->Free(env->allocator_, winograd->tmp_data_);
    winograd->tmp_data_ = NULL;
  }
  if (winograd->gemm_out_ != NULL) {
    env->Free(env->allocator_, winograd->gemm_out_);
    winograd->gemm_out_ = NULL;
  }
  if (winograd->col_buffer_ != NULL) {
    env->Free(env->allocator_, winograd->col_buffer_);
    winograd->col_buffer_ = NULL;
  }
  if (winograd->opt_input_trans_ != NULL) {
    env->Free(env->allocator_, winograd->opt_input_trans_);
    winograd->opt_input_trans_ = NULL;
  }
}

void ConvWinoBaseInitGlobalVariable(ConvolutionBaseStruct *conv) {
  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)conv;
  winograd->oc_block_ = C8NUM;
  winograd->tmp_data_tile_ = C4NUM;
  winograd->tile_num_ = C12NUM;
}

int ConvWinoBaseWinogradFilterTransform(ConvolutionWinogradBaseStruct *winograd, const float *weight_data) {
  NNACL_CHECK_ZERO_RETURN_ERR(winograd->oc_block_);
  return WinogradWeightTransform(weight_data, (float *)winograd->conv_.packed_weight_, winograd->matrix_g_,
                                 winograd->matrix_gt_, winograd->oc_block_, winograd->input_unit_,
                                 winograd->kernel_unit_, winograd->conv_.compute_.in_c_,
                                 winograd->conv_.compute_.out_c_, true);
}

void ConvWinoBasePackWeight(ConvolutionBaseStruct *conv) {
  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)conv;
  NNACL_CHECK_NULL_RETURN_VOID(winograd);
  TensorC *weight_tensor = conv->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_VOID(weight_tensor);
  void *origin_weight = (conv->base_.train_session_) ? weight_tensor->data_ : conv->origin_weight_;
  NNACL_CHECK_NULL_RETURN_VOID(origin_weight);
  ConvWinoBaseWinogradFilterTransform(winograd, (float *)origin_weight);
}

int ConvolutionWinogradBasePrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(winograd);

  winograd->conv_.init_global_variable_(&winograd->conv_);

  winograd->kernel_unit_ = winograd->conv_.compute_.kernel_h_;
  winograd->input_unit_ = winograd->output_unit_ + winograd->kernel_unit_ - 1;

  if (self->train_session_) {
    TensorC *filter_tensor = self->in_[SECOND_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(filter_tensor);
    NNACL_CHECK_FALSE(filter_tensor->shape_size_ != DIMENSION_4D, NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);

    int input_plane = winograd->input_unit_ * winograd->input_unit_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(input_plane, winograd->conv_.compute_.in_c_, NNACL_ERR);
    int in_chw = input_plane * winograd->conv_.compute_.in_c_;
    NNACL_CHECK_INT_MUL_NOT_OVERFLOW(in_chw, UP_ROUND(winograd->conv_.compute_.out_c_, winograd->oc_block_), NNACL_ERR);
    int trans_matrix_data_size =
      in_chw * UP_ROUND(winograd->conv_.compute_.out_c_, winograd->oc_block_) * sizeof(float);
    self->work_size_ = trans_matrix_data_size;
  }

  return ConvBaseInitConvWeightBias(&winograd->conv_);
}

int ConvoWinoBaseUpdateThreadNumProcess(ConvolutionWinogradBaseStruct *winograd) {
  if (winograd->conv_.compute_.in_n_ % winograd->conv_.base_.thread_nr_ == 0) {
    winograd->conv_.use_batch_cut_flag_ = true;
    return NNACL_OK;
  } else {
    winograd->conv_.use_batch_cut_flag_ = false;
  }

  int update_thread = UP_DIV(UP_DIV(winograd->conv_.compute_.out_hw_, C12NUM), ConvMinBlock);
  winograd->conv_.base_.thread_nr_ = NNACL_MIN(update_thread, winograd->conv_.base_.thread_nr_);
  return NNACL_OK;
}

int ConvoWinoBaseUpdateThread(ConvolutionWinogradBaseStruct *winograd) {
#ifdef DYNAMIC_THREAD_DISTRIBUTE
  ConvoWinoBaseUpdateThreadNumProcess(winograd);
#else
  KernelBase *base = &winograd->conv_.base_;
  base->thread_nr_ = base->UpdateThread(TC_PTYPE(PrimType_Conv2DFusion), 0, 0, 0, base->thread_nr_);
#endif
  return NNACL_OK;
}

int ConvWinoBaseConfigInputOutput(ConvolutionWinogradBaseStruct *winograd) {
  winograd->transfer_functions_.in_func_ = GetInputTransFunc(winograd->input_unit_);
  NNACL_CHECK_NULL_RETURN_ERR(winograd->transfer_functions_.in_func_);

  ConvParameter *conv_param = (ConvParameter *)winograd->conv_.base_.param_;
  winograd->transfer_functions_.out_func_ =
    GetOutputTransFunc(winograd->input_unit_, winograd->output_unit_, conv_param->act_type_);
  NNACL_CHECK_NULL_RETURN_ERR(winograd->transfer_functions_.out_func_);

  return NNACL_OK;
}

int ConvoWinoBaseInitTmpBuffer(ConvolutionWinogradBaseStruct *winograd) {
  ExecEnv *env = winograd->conv_.base_.env_;
  NNACL_CHECK_NULL_RETURN_ERR(env);

  int thread_input_plane = winograd->conv_.base_.thread_nr_ * winograd->input_unit_ * winograd->input_unit_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(winograd->tile_num_, thread_input_plane, NNACL_ERR);
  int total_thread_input_plane = winograd->tile_num_ * thread_input_plane;
  size_t tile_buffer_size = total_thread_input_plane * winograd->conv_.compute_.in_c_ * sizeof(float);
  winograd->trans_input_ = (float *)env->Alloc(env->allocator_, tile_buffer_size);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(winograd->trans_input_);

  int oc8 = UP_ROUND(winograd->conv_.compute_.out_c_, C8NUM);
  winograd->gemm_out_ = env->Alloc(env->allocator_, total_thread_input_plane * oc8 * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(winograd->gemm_out_);

  winograd->tmp_data_ = env->Alloc(env->allocator_, winograd->tmp_data_tile_ * thread_input_plane * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(winograd->tmp_data_);

  winograd->col_buffer_ = env->Alloc(env->allocator_, winograd->conv_.base_.thread_nr_ * winograd->tile_num_ *
                                                        winograd->conv_.compute_.in_c_ * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(winograd->col_buffer_);

  int tile = UP_ROUND(winograd->conv_.compute_.in_c_, winograd->tmp_data_tile_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(total_thread_input_plane, tile, NNACL_ERR);
  winograd->opt_input_trans_ = env->Alloc(env->allocator_, total_thread_input_plane * tile * sizeof(float));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(winograd->opt_input_trans_);

  winograd->tmp_buffer_address_list_[Index0] = winograd->trans_input_;
  winograd->tmp_buffer_address_list_[Index1] = winograd->gemm_out_;
  winograd->tmp_buffer_address_list_[Index2] = winograd->tmp_data_;
  winograd->tmp_buffer_address_list_[Index3] = winograd->col_buffer_;
  winograd->tmp_buffer_address_list_[Index4] = winograd->opt_input_trans_;
  return NNACL_OK;
}

int ConvWinoBaseRunImpl(ConvolutionBaseStruct *conv, int task_id) {
  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)conv;
  NNACL_CHECK_NULL_RETURN_ERR(winograd);
  ConvParameter *conv_param = (ConvParameter *)conv->base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  TensorC *input_tensor = conv->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  float *input_data = (float *)input_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);

  TensorC *output_tensor = conv->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output_tensor);
  float *output_data = (float *)output_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);

  if (conv->use_batch_cut_flag_) {
    ConvWinogardFp32CutByBatch(input_data, (float *)conv->packed_weight_, (float *)conv->bias_data_, output_data,
                               winograd->tmp_buffer_address_list_, task_id, conv_param, winograd->transfer_functions_);
  } else {
    ConvWinogardFp32(input_data, (float *)conv->packed_weight_, (float *)conv->bias_data_, output_data,
                     winograd->tmp_buffer_address_list_, task_id, conv_param, winograd->transfer_functions_);
  }

  return NNACL_OK;
}

int ConvWinoImpl(void *cdata, int task_id, float l, float r) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  return conv->run_impl_(conv, task_id);
}

void ConvWinoBaseUpdateParam(ConvParameter *param, ConvolutionWinogradBaseStruct *winograd) {
  param->input_unit_ = winograd->input_unit_;
  param->output_unit_ = winograd->output_unit_;
}

int ConvolutionWinogradBaseResize(KernelBase *self) {
  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(winograd);

  int ret = ConvBaseCheckResizeValid(&winograd->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ConvBasePrepare(&winograd->conv_);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = ConvoWinoBaseUpdateThread(winograd);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = winograd->config_input_output_(winograd);
  if (ret != NNACL_OK) {
    return ret;
  }

  ConvWinoBaseUpdateParam((ConvParameter *)self->param_, winograd);
  return NNACL_OK;
}

int ConvolutionWinogradBaseCompute(KernelBase *self) {
  ConvolutionWinogradBaseStruct *winograd = (ConvolutionWinogradBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(winograd);

  int ret = ConvoWinoBaseInitTmpBuffer(winograd);
  if (ret != NNACL_OK) {
    ConvWinoBaseFreeTmpBuffer(winograd);
    return ret;
  }

  ret = ConvBaseRepackWeight(&winograd->conv_);
  if (ret != NNACL_OK) {
    ConvWinoBaseFreeTmpBuffer(winograd);
    return ret;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, ConvWinoImpl, self, self->thread_nr_);
  ConvWinoBaseFreeTmpBuffer(winograd);
  return ret;
}

int ConvolutionWinogradBaseRelease(KernelBase *self) {
  ConvolutionBaseStruct *conv = (ConvolutionBaseStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(conv);
  ConvBaseRelease(conv);
  return NNACL_OK;
}

ConvolutionWinogradBaseStruct *CreateConvWinogradBase(ConvParameter *conv_param) {
  ConvolutionWinogradBaseStruct *winograd =
    (ConvolutionWinogradBaseStruct *)malloc(sizeof(ConvolutionWinogradBaseStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(winograd);
  memset(winograd, 0, sizeof(ConvolutionWinogradBaseStruct));

  winograd->config_input_output_ = ConvWinoBaseConfigInputOutput;
  winograd->conv_.init_global_variable_ = ConvWinoBaseInitGlobalVariable;

  winograd->conv_.base_.Prepare = ConvolutionWinogradBasePrepare;
  winograd->conv_.base_.Resize = ConvolutionWinogradBaseResize;
  winograd->conv_.base_.Release = ConvolutionWinogradBaseRelease;
  winograd->conv_.base_.Compute = ConvolutionWinogradBaseCompute;
  return (ConvolutionWinogradBaseStruct *)winograd;
}
