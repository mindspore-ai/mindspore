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

#include "nnacl/kernel/group_convolution.h"
#include "nnacl/kernel/convolution_delegate.h"
#include "nnacl/base/conv_common_base.h"

int GroupConvBasePrepare(GroupConvolutionStruct *group_conv) {
  ConvParameter *conv_param = (ConvParameter *)group_conv->conv_base_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  for (int i = 0; i < conv_param->group_; ++i) {
    KernelBase *sub_conv = group_conv->group_convs_[i];
    NNACL_CHECK_NULL_RETURN_ERR(sub_conv);

    int ret = sub_conv->prepare(sub_conv);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  conv_param->input_channel_ *= conv_param->group_;
  conv_param->output_channel_ *= conv_param->group_;
  // if infer shape is done, resize func will be invoked in sub kernels
  return NNACL_OK;
}

ConvParameter *CreateNewConvParameter(const ConvParameter *parameter) {
  ConvParameter *new_conv = (ConvParameter *)malloc(sizeof(ConvParameter));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(new_conv);
  memcpy(new_conv, parameter, sizeof(ConvParameter));
  return new_conv;
}

int group_convolution_release(KernelBase *self) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  if (group_conv->group_convs_ != NULL) {
    for (int i = 0; i < conv_param->group_; i++) {
      KernelBase *current_conv = group_conv->group_convs_[i];
      if (current_conv->param_ != NULL) {
        free(current_conv->param_);
        current_conv->param_ = NULL;
      }

      if (current_conv->in_ != NULL) {
        for (int j = 0; j < current_conv->in_size_; j++) {
          if (IsConst(current_conv->in_[j])) {
            free(current_conv->in_[j]->data_);
            current_conv->in_[j]->data_ = NULL;
          }
          if (current_conv->in_[j] != NULL) {
            free(current_conv->in_[j]);
            current_conv->in_[j] = NULL;
          }
        }
        free(current_conv->in_);
        current_conv->in_ = NULL;
      }

      if (current_conv->out_ != NULL) {
        for (int j = 0; j < current_conv->out_size_; j++) {
          if (current_conv->out_[j] != NULL) {
            free(current_conv->out_[j]);
            current_conv->out_[j] = NULL;
          }
        }
        free(current_conv->out_);
        current_conv->out_ = NULL;
      }

      if (group_conv->group_convs_[i] != NULL) {
        free(group_conv->group_convs_[i]);
        group_conv->group_convs_[i] = NULL;
      }
    }
    free(group_conv->group_convs_);
    group_conv->group_convs_ = NULL;
  }
  return NNACL_OK;
}

int GroupConvCreatorNewInputTensor(GroupConvolutionStruct *group_conv, KernelBase *new_conv) {
  TensorC *in_tensor = (TensorC *)malloc(sizeof(TensorC));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(in_tensor);
  in_tensor->format_ = NHWC;
  in_tensor->category_ = VarTensor;
  in_tensor->data_type_ = group_conv->data_type_;
  memcpy(in_tensor->shape_, group_conv->input_shape_, DIMENSION_4D * sizeof(int));
  new_conv->in_[FIRST_INPUT] = in_tensor;
  return NNACL_OK;
}

int GroupConvCreatorNewOutputTensor(GroupConvolutionStruct *group_conv, KernelBase *new_conv) {
  if (group_conv->conv_base_.base_.param_->quant_type_ != Quant_None) {
    return NNACL_ERR;
  }

  TensorC *out_tensor = (TensorC *)malloc(sizeof(TensorC));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(out_tensor);
  out_tensor->format_ = NHWC;
  out_tensor->category_ = VarTensor;
  out_tensor->data_type_ = group_conv->data_type_;
  memcpy(out_tensor->shape_, group_conv->output_shape_, DIMENSION_4D * sizeof(int));
  new_conv->out_[OUTPUT_INDEX] = out_tensor;
  return NNACL_OK;
}

TensorC *CreateConstTensor(const TensorC *tensor, const int *shape, const int shape_size, const int index) {
  NNACL_CHECK_NULL_RETURN_NULL(tensor->data_);

  TensorC *new_tensor = (TensorC *)malloc(sizeof(TensorC));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(new_tensor);
  new_tensor->data_type_ = tensor->data_type_;
  new_tensor->format_ = NHWC;
  new_tensor->category_ = ConstTensor;
  memcpy(new_tensor->shape_, shape, shape_size * sizeof(int));

  int size = GetSize(new_tensor);
  if (size <= 0) {
    free(new_tensor);
    return NULL;
  }

  void *data = malloc(size);
  if (data == NULL) {
    free(new_tensor);
    return NULL;
  }
  new_tensor->data_ = data;

  uint8_t *new_tensor_data = (uint8_t *)tensor->data_ + index * size;
  memcpy(new_tensor->data_, new_tensor_data, size);
  return new_tensor;
}

int GroupConvCreatorNewConstTensor(GroupConvolutionStruct *group_conv, KernelBase *new_conv, int group_id) {
  TensorC *weight_tensor = CreateConstTensor(group_conv->conv_base_.base_.in_[SECOND_INPUT], group_conv->filter_shape_,
                                             DIMENSION_4D, group_id);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(weight_tensor);
  new_conv->in_[SECOND_INPUT] = weight_tensor;

  if (group_conv->conv_base_.base_.in_size_ == THREE_TENSOR) {
    TensorC *bias_tensor =
      CreateConstTensor(group_conv->conv_base_.base_.in_[THIRD_INPUT], group_conv->bias_shape_, DIMENSION_1D, group_id);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(bias_tensor);
    new_conv->in_[THIRD_INPUT] = bias_tensor;
  }
  return NNACL_OK;
}

int GroupConvCreatorSetShapeOfTensors(GroupConvolutionStruct *group_conv) {
  TensorC *weight_tensor = group_conv->conv_base_.base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(weight_tensor);
  ConvParameter *conv_param = (ConvParameter *)group_conv->conv_base_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  NNACL_CHECK_FALSE(conv_param->group_ == 0, NNACL_GROUP_CONVOLUTION_GROUP_INVALID);
  NNACL_CHECK_FALSE(weight_tensor->shape_size_ != DIMENSION_4D, NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);
  NNACL_CHECK_FALSE(conv_param->kernel_h_ != GetHeight(weight_tensor), NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);
  NNACL_CHECK_FALSE(conv_param->kernel_w_ != GetWidth(weight_tensor), NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);

  int new_in_channel = GetChannel(weight_tensor);
  int new_out_channel = GetBatch(weight_tensor) / conv_param->group_;

  /* set shape */
  int filter_shape[] = {new_out_channel, conv_param->kernel_h_, conv_param->kernel_w_, new_in_channel};
  memcpy(group_conv->filter_shape_, filter_shape, DIMENSION_4D * sizeof(int));
  group_conv->bias_shape_[Index0] = new_out_channel;
  conv_param->input_channel_ = new_in_channel;
  conv_param->output_channel_ = new_out_channel;

  if (CheckInferShapeDone(group_conv->conv_base_.base_.in_, group_conv->conv_base_.base_.in_size_,
                          group_conv->conv_base_.base_.out_, group_conv->conv_base_.base_.out_size_)) {
    TensorC *input_tensor = group_conv->conv_base_.base_.in_[FIRST_INPUT];
    int input_shape[] = {GetBatch(input_tensor), GetHeight(input_tensor), GetWidth(input_tensor), new_in_channel};
    memcpy(group_conv->input_shape_, input_shape, DIMENSION_4D * sizeof(int));
    int output_shape[] = {GetBatch(input_tensor), GetHeight(input_tensor), GetWidth(input_tensor), new_out_channel};
    memcpy(group_conv->output_shape_, output_shape, DIMENSION_4D * sizeof(int));
  } else {
    int infershape_not_done[] = {-1, 0, 0, 0};
    memcpy(group_conv->input_shape_, infershape_not_done, DIMENSION_4D * sizeof(int));
    memcpy(group_conv->output_shape_, infershape_not_done, DIMENSION_4D * sizeof(int));
  }
  return NNACL_OK;
}

int GroupConvCreatorGetSingleConvParam(GroupConvolutionStruct *group_conv, KernelBase *new_conv, int group_id) {
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  NNACL_CHECK_NULL_RETURN_ERR(new_conv);

  new_conv->in_ = (TensorC **)malloc(new_conv->in_size_ * sizeof(TensorC *));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(new_conv->in_);
  new_conv->out_ = (TensorC **)malloc(new_conv->out_size_ * sizeof(TensorC *));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(new_conv->out_);

  // create new input for each group
  int ret = GroupConvCreatorNewInputTensor(group_conv, new_conv);
  if (ret != NNACL_OK) {
    group_conv->conv_base_.base_.release((KernelBase *)group_conv);
    return ret;
  }

  // const tensor
  ret = GroupConvCreatorNewConstTensor(group_conv, new_conv, group_id);
  if (ret != NNACL_OK) {
    group_conv->conv_base_.base_.release((KernelBase *)group_conv);
    return ret;
  }

  // create new output tensor
  ret = GroupConvCreatorNewOutputTensor(group_conv, new_conv);
  if (ret != NNACL_OK) {
    group_conv->conv_base_.base_.release((KernelBase *)group_conv);
    return ret;
  }
  return NNACL_OK;
}

int GroupConvConcatOutputRun(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)cdata;

  int plane_step = UP_DIV(group_conv->out_plane_, group_conv->conv_base_.base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(plane_step, task_id, NNACL_ERR);
  int begin_plane = plane_step * task_id;
  int end_plane = MSMIN(group_conv->out_plane_, plane_step * (task_id + 1));
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(begin_plane, group_conv->sub_out_channel_, NNACL_ERR);
  float *src_ptr = group_conv->sub_out_src_ + begin_plane * group_conv->sub_out_channel_;
  float *dst_ptr = group_conv->sub_out_dst_ + begin_plane * group_conv->ori_out_channel_;
  for (int i = begin_plane; i < end_plane; ++i) {
    (void)memcpy(dst_ptr, src_ptr, group_conv->sub_out_channel_ * sizeof(float));
    src_ptr += group_conv->sub_out_channel_;
    dst_ptr += group_conv->ori_out_channel_;
  }
  return NNACL_OK;
}

int GroupConvPostConcat(GroupConvolutionStruct *group_conv, int group_id) {
  group_conv->sub_out_src_ = (float *)group_conv->group_convs_[group_id]->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_out_src_);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(group_id, group_conv->sub_out_channel_, NNACL_ERR);
  group_conv->sub_out_dst_ = (float *)(group_conv->origin_output_data_) + group_id * group_conv->sub_out_channel_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_out_dst_);

  return group_conv->conv_base_.base_.env_->parallel_launch(group_conv->conv_base_.base_.env_->thread_pool_,
                                                            GroupConvConcatOutputRun, group_conv,
                                                            group_conv->conv_base_.base_.thread_nr_);
}

int GroupConvSeparateInputRun(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)cdata;

  int plane_step = UP_DIV(group_conv->in_plane_, group_conv->conv_base_.base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(plane_step, task_id, NNACL_ERR);
  int begin_plane = plane_step * task_id;
  int end_plane = MSMIN(group_conv->in_plane_, plane_step * (task_id + 1));
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(begin_plane, group_conv->ori_in_channel_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(begin_plane, group_conv->sub_in_channel_, NNACL_ERR);
  float *src_ptr = group_conv->sub_in_src_ + begin_plane * group_conv->ori_in_channel_;
  float *dst_ptr = group_conv->sub_in_dst_ + begin_plane * group_conv->sub_in_channel_;
  for (int i = begin_plane; i < end_plane; ++i) {
    (void)memcpy(dst_ptr, src_ptr, group_conv->sub_in_channel_ * sizeof(float));
    src_ptr += group_conv->ori_in_channel_;
    dst_ptr += group_conv->sub_in_channel_;
  }

  return NNACL_OK;
}

int GroupConvSeparateInput(GroupConvolutionStruct *group_conv, int group_id) {
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(group_id, group_conv->sub_in_channel_, NNACL_ERR);

  group_conv->sub_in_src_ = (float *)(group_conv->origin_input_data_) + group_id * group_conv->sub_in_channel_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_in_src_);
  group_conv->sub_in_dst_ = (float *)(group_conv->group_convs_[group_id]->in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_in_dst_);

  return group_conv->conv_base_.base_.env_->parallel_launch(group_conv->conv_base_.base_.env_->thread_pool_,
                                                            GroupConvSeparateInputRun, group_conv,
                                                            group_conv->conv_base_.base_.thread_nr_);
}

int GroupConvBaseInitGroupParam(GroupConvolutionStruct *group_conv) {
  ConvParameter *conv_param = (ConvParameter *)group_conv->conv_base_.base_.param_;

  TensorC *in_tensor = group_conv->conv_base_.base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  group_conv->in_plane_ = GetHeight(in_tensor) * GetWidth(in_tensor) * GetBatch(in_tensor);
  NNACL_CHECK_FALSE(group_conv->in_plane_ < 0, NNACL_GROUP_CONVOLUTION_SHAPE_INVALID);

  group_conv->sub_in_channel_ = conv_param->input_channel_ / conv_param->group_;
  group_conv->ori_in_channel_ = conv_param->input_channel_;
  group_conv->conv_base_.base_.thread_nr_ =
    MSMIN(MSMAX(1, group_conv->conv_base_.base_.thread_nr_), group_conv->in_plane_);

  TensorC *out_tensor = group_conv->conv_base_.base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out_tensor);
  group_conv->out_plane_ = GetHeight(out_tensor) * GetWidth(out_tensor) * GetBatch(out_tensor);
  NNACL_CHECK_FALSE(group_conv->out_plane_ < 0, NNACL_GROUP_CONVOLUTION_SHAPE_INVALID);

  group_conv->sub_out_channel_ = conv_param->output_channel_ / conv_param->group_;
  group_conv->ori_out_channel_ = conv_param->output_channel_;
  group_conv->conv_base_.base_.thread_nr_ =
    MSMIN(MSMAX(1, group_conv->conv_base_.base_.thread_nr_), group_conv->out_plane_);
  return NNACL_OK;
}

int group_convolution_resize(KernelBase *self) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);
  NNACL_CHECK_FALSE(conv_param->group_ == 0, NNACL_GROUP_CONVOLUTION_GROUP_INVALID);

  for (int i = 0; i < conv_param->group_; ++i) {
    int ret = group_conv->group_convs_[i]->resize(group_conv->group_convs_[i]);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

int group_convolution_compute(KernelBase *self) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  int ret = GroupConvBaseInitGroupParam(group_conv);
  if (ret != NNACL_OK) {
    return ret;
  }

  group_conv->origin_input_data_ = self->in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->origin_input_data_);
  group_conv->origin_output_data_ = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->origin_output_data_);

  for (int i = 0; i < conv_param->group_; ++i) {
    // first, malloc data for sub_kernel's tensors.
    TensorC *sub_kernel_in_tensor = group_conv->group_convs_[i]->in_[FIRST_INPUT];
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(sub_kernel_in_tensor);
    void *sub_in_data = malloc(GetSize(sub_kernel_in_tensor));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(sub_in_data);
    sub_kernel_in_tensor->data_ = sub_in_data;

    TensorC *sub_kernel_out_tensor = group_conv->group_convs_[i]->out_[OUTPUT_INDEX];
    void *sub_out_data = malloc(GetSize(sub_kernel_out_tensor));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(sub_out_data);
    sub_kernel_out_tensor->data_ = sub_out_data;

    // second, separate group conv input into several parts. This step must be in runtime stage.
    ret = GroupConvSeparateInput(group_conv, i);
    if (ret != NNACL_OK) {
      return ret;
    }

    // sun kernels run
    ret = group_conv->group_convs_[i]->compute(group_conv->group_convs_[i]);
    if (ret != NNACL_OK) {
      return ret;
    }

    // post process, concat all outputs of sub-kernels into one output
    ret = GroupConvPostConcat(group_conv, i);
    if (ret != NNACL_OK) {
      return ret;
    }

    // free data
    free(sub_in_data);
    sub_in_data = NULL;
    sub_kernel_in_tensor->data_ = NULL;
    free(sub_out_data);
    sub_out_data = NULL;
    sub_kernel_out_tensor->data_ = NULL;
  }
  return NNACL_OK;
}

int group_convolution_prepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  GroupConvCreatorSetShapeOfTensors(group_conv);

  group_conv->group_convs_ = (KernelBase **)malloc(conv_param->group_ * sizeof(KernelBase *));
  for (int i = 0; i < conv_param->group_; ++i) {
    ConvParameter *new_conv_param = CreateNewConvParameter(conv_param);

    KernelBase *new_conv = CreateConvlutionDelegate(new_conv_param);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(new_conv);
    group_conv->group_convs_[i] = new_conv;

    int ret = GroupConvCreatorGetSingleConvParam(group_conv, new_conv, i);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return GroupConvBasePrepare(group_conv);
}

KernelBase *CreateGroupConvolution(ConvParameter *conv_param) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)malloc(sizeof(GroupConvolutionStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(group_conv);
  memset(group_conv, 0, sizeof(GroupConvolutionStruct));

  group_conv->conv_base_.base_.compute = group_convolution_compute;
  group_conv->conv_base_.base_.resize = group_convolution_resize;
  group_conv->conv_base_.base_.prepare = group_convolution_prepare;
  group_conv->conv_base_.base_.release = group_convolution_release;
  return (KernelBase *)group_conv;
}
