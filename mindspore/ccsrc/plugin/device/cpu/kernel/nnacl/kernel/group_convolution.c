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
#include "nnacl/tensor_c_utils.h"

int GroupConvBasePrepare(GroupConvolutionStruct *group_conv) {
  for (int i = 0; i < group_conv->group_; ++i) {
    KernelBase *sub_conv = group_conv->group_convs_[i];
    NNACL_CHECK_NULL_RETURN_ERR(sub_conv);
    int ret = sub_conv->Prepare(sub_conv);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

int GroupConvCreatorNewInputTensor(GroupConvolutionStruct *group_conv, KernelBase *new_conv) {
  TensorC *in_tensor = (TensorC *)malloc(sizeof(TensorC));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(in_tensor);
  in_tensor->format_ = Format_NHWC;
  in_tensor->category_ = VarTensor;
  in_tensor->data_type_ = group_conv->data_type_;
  in_tensor->shape_size_ = DIMENSION_4D;
  in_tensor->shape_[Index0] = INVALID_SHAPE;
  new_conv->in_[FIRST_INPUT] = in_tensor;
  return NNACL_OK;
}

int GroupConvCreatorNewOutputTensor(GroupConvolutionStruct *group_conv, KernelBase *new_conv) {
  TensorC *out_tensor = (TensorC *)malloc(sizeof(TensorC));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(out_tensor);
  out_tensor->format_ = Format_NHWC;
  out_tensor->category_ = VarTensor;
  out_tensor->data_type_ = group_conv->data_type_;
  out_tensor->shape_size_ = DIMENSION_4D;
  out_tensor->shape_[Index0] = INVALID_SHAPE;
  new_conv->out_[OUTPUT_INDEX] = out_tensor;
  return NNACL_OK;
}

TensorC *CreateConstTensor(const TensorC *tensor, const int *shape, const int shape_size, const int index) {
  NNACL_CHECK_NULL_RETURN_NULL(tensor->data_);

  TensorC *new_tensor = (TensorC *)malloc(sizeof(TensorC));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(new_tensor);
  new_tensor->data_type_ = tensor->data_type_;
  new_tensor->format_ = Format_NHWC;
  new_tensor->category_ = ConstTensor;
  new_tensor->shape_size_ = shape_size;
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
  TensorC *origin_weight = group_conv->conv_base_.base_.in_[SECOND_INPUT];
  int shape[] = {group_conv->sub_out_c_, GetHeight(origin_weight), GetWidth(origin_weight), group_conv->sub_in_c_};
  TensorC *weight_tensor = CreateConstTensor(origin_weight, shape, DIMENSION_4D, group_id);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(weight_tensor);
  new_conv->in_[SECOND_INPUT] = weight_tensor;

  if (group_conv->conv_base_.base_.in_size_ == THREE_TENSOR) {
    TensorC *bias_weight = group_conv->conv_base_.base_.in_[THIRD_INPUT];
    TensorC *bias_tensor = CreateConstTensor(bias_weight, &group_conv->sub_out_c_, DIMENSION_1D, group_id);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(bias_tensor);
    new_conv->in_[THIRD_INPUT] = bias_tensor;
  }
  return NNACL_OK;
}

int GroupConvCreatorSetShapeOfTensors(GroupConvolutionStruct *group_conv) {
  ConvParameter *origin_conv_param = (ConvParameter *)group_conv->conv_base_.base_.param_;
  NNACL_CHECK_NULL_RETURN_ERR(origin_conv_param);
  ConvParameter *new_conv_param = &group_conv->new_conv_param_;
  NNACL_CHECK_NULL_RETURN_ERR(new_conv_param);
  memcpy(new_conv_param, origin_conv_param, sizeof(ConvParameter));

  TensorC *weight_tensor = group_conv->conv_base_.base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(weight_tensor);
  NNACL_CHECK_FALSE(origin_conv_param->group_ == 0, NNACL_GROUP_CONVOLUTION_GROUP_INVALID);
  NNACL_CHECK_FALSE(weight_tensor->shape_size_ != DIMENSION_4D, NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);
  NNACL_CHECK_FALSE(origin_conv_param->kernel_h_ != GetHeight(weight_tensor), NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);
  NNACL_CHECK_FALSE(origin_conv_param->kernel_w_ != GetWidth(weight_tensor), NNACL_CONVOLUTION_WEIGHT_SHAPE_INVALID);

  ConvComputeParam *compute = &group_conv->conv_base_.compute_;
  group_conv->ori_in_c_ = compute->in_c_;
  group_conv->ori_out_c_ = compute->out_c_;
  group_conv->sub_in_c_ = compute->in_c_ / group_conv->group_;
  group_conv->sub_out_c_ = compute->out_c_ / group_conv->group_;

  new_conv_param->input_channel_ = group_conv->sub_in_c_;
  new_conv_param->output_channel_ = group_conv->sub_out_c_;
  new_conv_param->group_ = origin_conv_param->group_;

  return NNACL_OK;
}

int GroupConvSetSubConvInfo(GroupConvolutionStruct *group_conv, KernelBase *new_conv, int group_id) {
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  NNACL_CHECK_NULL_RETURN_ERR(new_conv);

  ConvolutionBaseStruct *sub_conv = (ConvolutionBaseStruct *)new_conv;
  (void)ConvBaseUpdateParamInfo(&sub_conv->compute_, &group_conv->new_conv_param_);

  sub_conv->infershape_done_ = group_conv->conv_base_.infershape_done_;
  sub_conv->shaing_manager_ = group_conv->conv_base_.shaing_manager_;
  sub_conv->get_sharing_weight_ = group_conv->conv_base_.get_sharing_weight_;
  sub_conv->free_sharing_weight_ = group_conv->conv_base_.free_sharing_weight_;
  sub_conv->is_sharing_pack_ = group_conv->conv_base_.is_sharing_pack_;

  new_conv->env_ = group_conv->conv_base_.base_.env_;
  new_conv->param_ = &group_conv->new_conv_param_.op_parameter_;
  new_conv->thread_nr_ = group_conv->conv_base_.base_.thread_nr_;
  new_conv->train_session_ = group_conv->conv_base_.base_.train_session_;
  new_conv->UpdateThread = group_conv->conv_base_.base_.UpdateThread;
  new_conv->in_size_ = group_conv->conv_base_.base_.in_size_;
  new_conv->out_size_ = group_conv->conv_base_.base_.out_size_;

  new_conv->in_ = (TensorC **)malloc(new_conv->in_size_ * sizeof(TensorC *));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(new_conv->in_);
  memset(new_conv->in_, 0, new_conv->in_size_ * sizeof(TensorC *));
  new_conv->out_ = (TensorC **)malloc(new_conv->out_size_ * sizeof(TensorC *));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(new_conv->out_);
  memset(new_conv->out_, 0, new_conv->out_size_ * sizeof(TensorC *));

  // create new input for each group
  int ret = GroupConvCreatorNewInputTensor(group_conv, new_conv);
  if (ret != NNACL_OK) {
    group_conv->conv_base_.base_.Release((KernelBase *)group_conv);
    return ret;
  }

  // const tensor
  ret = GroupConvCreatorNewConstTensor(group_conv, new_conv, group_id);
  if (ret != NNACL_OK) {
    group_conv->conv_base_.base_.Release((KernelBase *)group_conv);
    return ret;
  }

  // create new output tensor
  ret = GroupConvCreatorNewOutputTensor(group_conv, new_conv);
  if (ret != NNACL_OK) {
    group_conv->conv_base_.base_.Release((KernelBase *)group_conv);
    return ret;
  }
  return NNACL_OK;
}

int GroupConvConcatOutputRun(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)cdata;

  int plane_step = UP_DIV(group_conv->conv_base_.compute_.out_hw_, group_conv->conv_base_.base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(plane_step, task_id, NNACL_ERR);
  int begin_plane = plane_step * task_id;
  int end_plane = NNACL_MIN(group_conv->conv_base_.compute_.out_hw_, plane_step * (task_id + 1));
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(begin_plane, group_conv->sub_out_c_, NNACL_ERR);
  float *src_ptr = group_conv->sub_out_src_ + begin_plane * group_conv->sub_out_c_;
  float *dst_ptr = group_conv->sub_out_dst_ + begin_plane * group_conv->ori_out_c_;
  for (int i = begin_plane; i < end_plane; ++i) {
    (void)memcpy(dst_ptr, src_ptr, group_conv->sub_out_c_ * sizeof(float));
    src_ptr += group_conv->sub_out_c_;
    dst_ptr += group_conv->ori_out_c_;
  }
  return NNACL_OK;
}

int GroupConvPostConcat(GroupConvolutionStruct *group_conv, int group_id) {
  group_conv->sub_out_src_ = (float *)group_conv->group_convs_[group_id]->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_out_src_);

  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(group_id, group_conv->sub_out_c_, NNACL_ERR);
  group_conv->sub_out_dst_ = (float *)(group_conv->origin_output_data_) + group_id * group_conv->sub_out_c_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_out_dst_);

  return group_conv->conv_base_.base_.env_->ParallelLaunch(group_conv->conv_base_.base_.env_->thread_pool_,
                                                           GroupConvConcatOutputRun, group_conv,
                                                           group_conv->conv_base_.base_.thread_nr_);
}

int GroupConvSeparateInputRun(void *cdata, int task_id, float l, float r) {
  NNACL_CHECK_NULL_RETURN_ERR(cdata);
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)cdata;

  int plane_step = UP_DIV(group_conv->conv_base_.compute_.in_hw_, group_conv->conv_base_.base_.thread_nr_);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(plane_step, task_id, NNACL_ERR);
  int begin_plane = plane_step * task_id;
  int end_plane = NNACL_MIN(group_conv->conv_base_.compute_.in_hw_, plane_step * (task_id + 1));
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(begin_plane, group_conv->ori_in_c_, NNACL_ERR);
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(begin_plane, group_conv->sub_in_c_, NNACL_ERR);
  float *src_ptr = group_conv->sub_in_src_ + begin_plane * group_conv->ori_in_c_;
  float *dst_ptr = group_conv->sub_in_dst_ + begin_plane * group_conv->sub_in_c_;
  for (int i = begin_plane; i < end_plane; ++i) {
    (void)memcpy(dst_ptr, src_ptr, group_conv->sub_in_c_ * sizeof(float));
    src_ptr += group_conv->ori_in_c_;
    dst_ptr += group_conv->sub_in_c_;
  }

  return NNACL_OK;
}

int GroupConvSeparateInput(GroupConvolutionStruct *group_conv, int group_id) {
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(group_id, group_conv->sub_in_c_, NNACL_ERR);

  group_conv->sub_in_src_ = (float *)(group_conv->origin_input_data_) + group_id * group_conv->sub_in_c_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_in_src_);
  group_conv->sub_in_dst_ = (float *)(group_conv->group_convs_[group_id]->in_[FIRST_INPUT]->data_);
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->sub_in_dst_);

  return group_conv->conv_base_.base_.env_->ParallelLaunch(group_conv->conv_base_.base_.env_->thread_pool_,
                                                           GroupConvSeparateInputRun, group_conv,
                                                           group_conv->conv_base_.base_.thread_nr_);
}

void GroupConvUpdateShape(GroupConvolutionStruct *group_conv) {
  for (int i = 0; i < group_conv->group_; i++) {
    TensorC *in_tensor = group_conv->conv_base_.base_.in_[FIRST_INPUT];
    int in_shape[] = {GetBatch(in_tensor), GetHeight(in_tensor), GetWidth(in_tensor), group_conv->sub_in_c_};
    memcpy(group_conv->group_convs_[i]->in_[FIRST_INPUT]->shape_, in_shape, DIMENSION_4D * sizeof(float));

    TensorC *out_tensor = group_conv->conv_base_.base_.out_[OUTPUT_INDEX];
    int out_shape[] = {GetBatch(out_tensor), GetHeight(out_tensor), GetWidth(out_tensor), group_conv->sub_out_c_};
    memcpy(group_conv->group_convs_[i]->out_[OUTPUT_INDEX]->shape_, out_shape, DIMENSION_4D * sizeof(float));
  }
  return;
}

int GroupConvolutionResize(KernelBase *self) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);

  (void)ConvBaseUpdateComputeInfo(&group_conv->conv_base_);
  self->thread_nr_ = NNACL_MIN(NNACL_MAX(1, self->thread_nr_), group_conv->conv_base_.compute_.in_hw_);
  self->thread_nr_ = NNACL_MIN(NNACL_MAX(1, self->thread_nr_), group_conv->conv_base_.compute_.in_hw_);

  GroupConvUpdateShape(group_conv);

  for (int i = 0; i < group_conv->group_; ++i) {
    group_conv->group_convs_[i]->thread_nr_ = self->thread_nr_;
    int ret = group_conv->group_convs_[i]->Resize(group_conv->group_convs_[i]);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return NNACL_OK;
}

int GroupConvolutionCompute(KernelBase *self) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);

  group_conv->origin_input_data_ = self->in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->origin_input_data_);
  group_conv->origin_output_data_ = self->out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv->origin_output_data_);

  for (int i = 0; i < group_conv->group_; ++i) {
    // first, malloc data for sub_kernel's tensors.
    TensorC *sub_kernel_in_tensor = group_conv->group_convs_[i]->in_[FIRST_INPUT];
    sub_kernel_in_tensor->data_ = self->env_->Alloc(self->env_->allocator_, GetSize(sub_kernel_in_tensor));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(sub_kernel_in_tensor->data_);

    TensorC *sub_kernel_out_tensor = group_conv->group_convs_[i]->out_[OUTPUT_INDEX];
    sub_kernel_out_tensor->data_ = self->env_->Alloc(self->env_->allocator_, GetSize(sub_kernel_out_tensor));
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(sub_kernel_out_tensor->data_);

    // second, separate group conv input into several parts. This step must be in runtime stage.
    int ret = GroupConvSeparateInput(group_conv, i);
    if (ret != NNACL_OK) {
      return ret;
    }

    // sun kernels run
    ret = group_conv->group_convs_[i]->Compute(group_conv->group_convs_[i]);
    if (ret != NNACL_OK) {
      return ret;
    }

    // post process, concat all outputs of sub-kernels into one output
    ret = GroupConvPostConcat(group_conv, i);
    if (ret != NNACL_OK) {
      return ret;
    }

    // Free data
    self->env_->Free(self->env_->allocator_, sub_kernel_in_tensor->data_);
    sub_kernel_in_tensor->data_ = NULL;
    self->env_->Free(self->env_->allocator_, sub_kernel_out_tensor->data_);
    sub_kernel_out_tensor->data_ = NULL;
  }
  return NNACL_OK;
}

int GroupConvolutionPrepare(KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < TWO_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ != ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  NNACL_CHECK_FALSE(group_conv->group_ == 0, NNACL_GROUP_CONVOLUTION_GROUP_INVALID);

  GroupConvCreatorSetShapeOfTensors(group_conv);

  group_conv->group_convs_ = (KernelBase **)malloc(group_conv->group_ * sizeof(KernelBase *));
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(group_conv->group_convs_);
  memset(group_conv->group_convs_, 0, group_conv->group_ * sizeof(KernelBase *));

  for (int i = 0; i < group_conv->group_; ++i) {
    KernelBase *new_conv = CreateConvlutionDelegate(&group_conv->new_conv_param_);
    NNACL_MALLOC_CHECK_NULL_RETURN_ERR(new_conv);
    group_conv->group_convs_[i] = new_conv;

    int ret = GroupConvSetSubConvInfo(group_conv, new_conv, i);
    if (ret != NNACL_OK) {
      return ret;
    }
  }
  return GroupConvBasePrepare(group_conv);
}

void GroupConvReleaseSubConv(KernelBase *current_conv) {
  (void)current_conv->Release(current_conv);

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
}

int GroupConvolutionRelease(KernelBase *self) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(group_conv);
  ConvParameter *conv_param = (ConvParameter *)self->param_;
  NNACL_CHECK_NULL_RETURN_ERR(conv_param);

  if (group_conv->group_convs_ != NULL) {
    for (int i = 0; i < conv_param->group_; i++) {
      if (group_conv->group_convs_[i] != NULL) {
        GroupConvReleaseSubConv(group_conv->group_convs_[i]);
        free(group_conv->group_convs_[i]);
        group_conv->group_convs_[i] = NULL;
      }
    }
    free(group_conv->group_convs_);
    group_conv->group_convs_ = NULL;
  }
  return NNACL_OK;
}

KernelBase *CreateGroupConvolution(ConvParameter *conv_param, TypeIdC data_type) {
  GroupConvolutionStruct *group_conv = (GroupConvolutionStruct *)malloc(sizeof(GroupConvolutionStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(group_conv);
  memset(group_conv, 0, sizeof(GroupConvolutionStruct));

  group_conv->data_type_ = data_type;
  group_conv->group_ = conv_param->group_;
  group_conv->conv_base_.base_.Compute = GroupConvolutionCompute;
  group_conv->conv_base_.base_.Resize = GroupConvolutionResize;
  group_conv->conv_base_.base_.Prepare = GroupConvolutionPrepare;
  group_conv->conv_base_.base_.Release = GroupConvolutionRelease;
  return (KernelBase *)group_conv;
}
