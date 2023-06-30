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

#include "nnacl/kernel/pad.h"
#include "nnacl/kernel/default_kernel_base.h"
#include "nnacl/tensor_c_utils.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/common_func.h"
#ifdef ENABLE_FP16
#include "nnacl/fp16/pad_fp16.h"
#endif
#include "nnacl/fp32/pad_fp32.h"

int PadInitMirrorPadBlock(PadStruct *pad) {
  int left_pads[DEFAULT_PAD_NDIMS] = {0};
  for (size_t i = 0; i < DEFAULT_PAD_NDIMS; ++i) {
    left_pads[i] = pad->paddings_[Num2 * i];
  }

  int input_separate_dims[DEFAULT_PAD_NDIMS] = {0};
  int output_separate_dims[DEFAULT_PAD_NDIMS] = {0};
  int separate_offset[DEFAULT_PAD_NDIMS] = {0};
  int separate_size = 0;

  /* init separate dims */
  for (size_t i = 0; i < DEFAULT_PAD_NDIMS; ++i) {
    input_separate_dims[separate_size] = pad->in_[i];
    output_separate_dims[separate_size] = pad->out_[i];
    separate_offset[separate_size] = left_pads[i];
    separate_size++;
  }

  /* init separate stride */
  int output_separate_stride[DEFAULT_PAD_NDIMS] = {0};
  (void)GetStride(output_separate_stride, output_separate_dims, separate_size);
  int remain_stride_size = 0;
  int remain_size = 1;
  int right_pads[DEFAULT_PAD_NDIMS] = {0};
  for (size_t i = 0; i < DEFAULT_PAD_NDIMS; i++) {
    right_pads[i] = output_separate_dims[i] - input_separate_dims[i] - separate_offset[i];
  }

  /* init pad region */
  int pad_region[DEFAULT_PAD_NDIMS] = {0};
  int pad_region_size = 0;
  for (int i = remain_stride_size; i < separate_size; ++i) {
    int r = 1;
    r = (separate_offset[i] > 0) ? (r + 1) : r;
    r = (right_pads[i] > 0) ? (r + 1) : r;
    pad_region[pad_region_size++] = r;
  }
  int pad_region_stride[DEFAULT_PAD_NDIMS] = {0};
  int region_size = GetStride(pad_region_stride, pad_region, pad_region_size);

  /* init mirror block info */
  int max_block_size = remain_size * region_size * sizeof(MirrorPadBlock);
  pad->mirror_pad_block_ = (MirrorPadBlock *)pad->base_.env_->Alloc(pad->base_.env_->allocator_, max_block_size);
  NNACL_MALLOC_CHECK_NULL_RETURN_ERR(pad->mirror_pad_block_);

  // 0: center, 1: left, 2: right
  int pad_cord[DEFAULT_PAD_NDIMS] = {0};

  for (int pos = 0; pos < remain_size; ++pos) {
    const int dst_basic_offset = 0;
    for (int index = 1; index < region_size; ++index) {
      int dst_offset = dst_basic_offset;
      int value = index;
      for (size_t i = 0; i < pad_region_size && pad_region_stride[i] != 0; ++i) {
        NNACL_CHECK_ZERO_RETURN_ERR(pad_region_stride[i]);
        pad_cord[i] = value / pad_region_stride[i];
        value = value % pad_region_stride[i];
      }
      MirrorPadBlock block;
      const int size_offset = DEFAULT_PAD_NDIMS - pad_region_size;
      for (size_t i = 0; i < pad_region_size; ++i) {
        int di = size_offset + i;
        int si = remain_stride_size + i;
        if (di >= DEFAULT_PAD_NDIMS) {
          continue;
        }
        switch (pad_cord[i]) {
          case Num0:
            dst_offset += separate_offset[si] * output_separate_stride[si];
            block.size_[di] = input_separate_dims[si];
            block.out_stride_[di] = output_separate_stride[si];
            break;
          case Num2:
            dst_offset += (separate_offset[si] + input_separate_dims[si]) * output_separate_stride[si];
            block.size_[di] = right_pads[si];
            block.out_stride_[di] = output_separate_stride[si];
            break;
          case Num1:
            if (separate_offset[si] > 0) {
              block.size_[di] = separate_offset[si];
              block.out_stride_[di] = output_separate_stride[si];
            } else {
              dst_offset += (separate_offset[si] + input_separate_dims[si]) * output_separate_stride[si];
              block.size_[di] = right_pads[si];
              block.out_stride_[di] = output_separate_stride[si];
            }
            break;
          default:
            break;
        }
      }
      block.out_offset_ = dst_offset;
      pad->mirror_pad_block_[pad->mirror_pad_block_size_++] = block;
    }
  }
  return NNACL_OK;
}

int PadExtendDims(int *dims, const int *origin_dims, int max_dim, int origin_dim, int init_value) {
  NNACL_CHECK_NULL_RETURN_ERR(dims);
  NNACL_CHECK_NULL_RETURN_ERR(origin_dims);
  for (int i = 0; i < max_dim - origin_dim; ++i) {
    dims[i] = init_value;
  }
  for (int i = max_dim - origin_dim; i < max_dim; ++i) {
    dims[i] = origin_dims[i - (max_dim - origin_dim)];
  }
  return NNACL_OK;
}

int PadImpl(void *cdata, int task_id, float l, float r) {
  PadStruct *pad = (PadStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(pad);
  void *input = pad->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input);
  void *output = pad->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output);

  if (pad->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    PadFp16(input, output, pad->in_, pad->out_, pad->paddings_, task_id, pad->base_.thread_nr_);
#endif
  } else {
    Pad((float *)input, (float *)output, pad->in_, pad->out_, pad->paddings_, task_id, pad->base_.thread_nr_);
  }
  return NNACL_OK;
}

int PadFastMirrorRunImpl(PadStruct *pad, int task_id) {
  void *in = pad->base_.in_[FIRST_INPUT]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(in);
  void *out = pad->base_.out_[OUTPUT_INDEX]->data_;
  NNACL_CHECK_NULL_RETURN_ERR(out);

  /* copy center part */
  if (pad->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    PadFp16((float16_t *)in, (float16_t *)out, pad->in_, pad->out_, pad->paddings_, task_id, pad->base_.thread_nr_);
#endif
  } else {
    Pad((float *)in, (float *)out, pad->in_, pad->out_, pad->paddings_, task_id, pad->base_.thread_nr_);
  }

  /* calculate region part */
  for (int i = task_id; i < pad->mirror_pad_block_size_; i += pad->base_.thread_nr_) {
    MirrorPadBlock *block = &pad->mirror_pad_block_[i];
    for (int a = 0; a < block->size_[FIRST_INPUT]; a++) {
      int out_a_index = block->out_offset_ + a * block->out_stride_[FIRST_INPUT];
      for (int b = 0; b < block->size_[SECOND_INPUT]; b++) {
        int out_b_index = out_a_index + b * block->out_stride_[SECOND_INPUT];
        for (int c = 0; c < block->size_[THIRD_INPUT]; ++c) {
          int out_c_index = out_b_index + c * block->out_stride_[THIRD_INPUT];
          for (int d = 0; d < block->size_[FOURTH_INPUT]; ++d) {
            int out_d_index = out_c_index + d * block->out_stride_[FOURTH_INPUT];
            for (int e = 0; e < block->size_[FIFTH_INPUT]; ++e) {
              int start_index = out_d_index + e * block->out_stride_[FIFTH_INPUT];
              int end_index = start_index + block->size_[SIXTH_INPUT];
              if (pad->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
                MirrorPadFp16(in, out, pad->in_, pad->in_strides_, pad->out_strides_, pad->paddings_,
                              pad->mirror_offset_, start_index, end_index);
#endif
              } else {
                MirrorPad(in, out, pad->in_, pad->in_strides_, pad->out_strides_, pad->paddings_, pad->mirror_offset_,
                          start_index, end_index);
              }
            }
          }
        }
      }
    }
  }
  return NNACL_OK;
}

int MirrorPadImpl(void *cdata, int task_id, float l, float r) {
  PadStruct *pad = (PadStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(pad);

  /* Fast Mirror pad */
  if (pad->mirror_pad_block_size_ != 0) {
    return PadFastMirrorRunImpl(pad, task_id);
  }

  TensorC *input = pad->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  void *input_data = input->data_;
  NNACL_CHECK_NULL_RETURN_ERR(input_data);
  TensorC *output = pad->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);
  void *output_data = output->data_;
  NNACL_CHECK_NULL_RETURN_ERR(output_data);

  /* Common Mirror pad */
  int unit = UP_DIV(GetElementNum(output), pad->base_.thread_nr_);
  int begin = unit * task_id;
  int end = NNACL_MIN(begin + unit, GetElementNum(output));
  if (pad->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
    MirrorPadFp16((float16_t *)input_data, (float16_t *)output_data, pad->in_, pad->in_strides_, pad->out_strides_,
                  pad->paddings_, pad->mirror_offset_, begin, end);
#endif
  } else {
    MirrorPad((float *)input_data, (float *)output_data, pad->in_, pad->in_strides_, pad->out_strides_, pad->paddings_,
              pad->mirror_offset_, begin, end);
  }
  return NNACL_OK;
}

int PadCheckPaddings(const int *paddings, int length, const int *input_shape, int mode) {
  NNACL_CHECK_NULL_RETURN_ERR(paddings);
  NNACL_CHECK_NULL_RETURN_ERR(input_shape);
  int offset = mode == PaddingMode_Symmetric ? 0 : 1;
  for (int i = 0; i < length; ++i) {
    int max_valid = input_shape[i] - offset;
    if (paddings[i * Num2] > max_valid) {
      return NNACL_PAD_MIRROR_PAD_SIZE_INVALID;
    }
    if (paddings[i * Num2 + 1] > max_valid) {
      return NNACL_PAD_MIRROR_PAD_SIZE_INVALID;
    }
  }
  return NNACL_OK;
}

int PadCopyPaddingFromInput(PadStruct *pad) {
  TensorC *input_tensor = pad->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input_tensor);
  TensorC *padding_tensor = pad->base_.in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(padding_tensor);
  int *padding_data = padding_tensor->data_;
  NNACL_CHECK_NULL_RETURN_ERR(padding_data);

  (void)PadExtendDims(pad->in_, input_tensor->shape_, DEFAULT_PAD_NDIMS, input_tensor->shape_size_, 1);
  (void)PadExtendDims(pad->paddings_, padding_data, MAX_PAD_SIZE, GetElementNum(padding_tensor), 0);
  pad->paddings_size_ = MAX_PAD_SIZE;

  return NNACL_OK;
}

void PadCalculateStrides(PadStruct *pad) {
  pad->in_strides_[DEFAULT_PAD_NDIMS - 1] = 1;
  for (int i = DEFAULT_PAD_NDIMS - Num2; i >= 0; --i) {
    pad->in_strides_[i] = pad->in_[i + 1] * pad->in_strides_[i + 1];
  }
  for (int i = 0; i < DEFAULT_PAD_NDIMS; ++i) {
    pad->out_[i] = pad->in_[i] + pad->paddings_[i * Num2] + pad->paddings_[i * Num2 + 1];
  }
  pad->out_strides_[DEFAULT_PAD_NDIMS - 1] = 1;
  for (int i = DEFAULT_PAD_NDIMS - Num2; i >= 0; --i) {
    pad->out_strides_[i] = pad->out_[i + 1] * pad->out_strides_[i + 1];
  }
}

int PadHandleMirrorPad(PadStruct *pad) {
  pad->mirror_offset_ = pad->pad_mode_ == PaddingMode_Reflect ? 1 : 0;
  (void)PadCheckPaddings(pad->paddings_, DEFAULT_PAD_NDIMS, pad->in_, pad->pad_mode_);
  PadCalculateStrides(pad);
  return PadInitMirrorPadBlock(pad);
}

int PadCompute(KernelBase *self) {
  PadStruct *pad = (PadStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(pad);

  if (self->in_size_ == THREE_TENSOR) {
    TensorC *pad_value_tensor = self->in_[THIRD_INPUT];
    NNACL_CHECK_NULL_RETURN_ERR(pad_value_tensor);
    NNACL_CHECK_FALSE(GetElementNum(pad_value_tensor) != 1, NNACL_PAD_PADDING_VALID_INVALID);
    void *pad_valud = pad_value_tensor->data_;
    if (pad->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
      pad->constant_value_ = ((float16_t *)pad_valud)[Index0];
#endif
    } else {
      pad->constant_value_ = ((float *)pad_valud)[Index0];
    }
  }

  int ret = PadCopyPaddingFromInput(pad);
  if (ret != NNACL_OK) {
    return ret;
  }

  if (pad->pad_mode_ == PaddingMode_Constant) {
    TensorC *output = self->out_[OUTPUT_INDEX];
    NNACL_CHECK_NULL_RETURN_ERR(output);
    size_t output_size = GetElementNum(output);
    void *output_data = output->data_;
    if (fabsf(pad->constant_value_ - 0.0f) < 1e-5) {
      memset(output_data, 0, output_size * (int)DataTypeCSize(pad->data_type_));
    } else {
      for (size_t i = 0; i < output_size; ++i) {
        if (pad->data_type_ == kNumberTypeFloat16) {
#ifdef ENABLE_FP16
          ((float16_t *)output_data)[i] = pad->constant_value_;
#endif
        } else {
          ((float *)output_data)[i] = pad->constant_value_;
        }
      }
    }
    ret = self->env_->ParallelLaunch(self->env_->thread_pool_, PadImpl, self, self->thread_nr_);
    return ret;
  }

  /* not constant pad mod using mirror pad algorithm */
  ret = PadHandleMirrorPad(pad);
  if (ret != NNACL_OK) {
    return ret;
  }

  ret = self->env_->ParallelLaunch(self->env_->thread_pool_, MirrorPadImpl, self, self->thread_nr_);

  self->env_->Free(self->env_->allocator_, pad->mirror_pad_block_);
  pad->mirror_pad_block_ = NULL;
  pad->mirror_pad_block_size_ = 0;
  return ret;
}

int PadResize(KernelBase *self) {
  PadStruct *pad = (PadStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(pad);
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  TensorC *padding = self->in_[SECOND_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(padding);
  TensorC *output = self->out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(output);

  int rank = input->shape_size_;
  NNACL_CHECK_FALSE(input->shape_size_ > DEFAULT_PAD_NDIMS, NNACL_PAD_SHAPE_INVALID);
  NNACL_CHECK_FALSE(GetElementNum(padding) != rank + rank, NNACL_PAD_SHAPE_INVALID);

  if (pad->pad_mode_ == PaddingMode_Constant) {
    (void)PadExtendDims(pad->in_, input->shape_, DEFAULT_PAD_NDIMS, rank, 1);
    (void)PadExtendDims(pad->out_, output->shape_, DEFAULT_PAD_NDIMS, rank, 1);

    if (pad->paddings_size_ < MAX_PAD_SIZE) {
      int ori_paddings[MAX_PAD_SIZE];
      memcpy(ori_paddings, pad->paddings_, MAX_PAD_SIZE * sizeof(int));
      (void)PadExtendDims(pad->paddings_, ori_paddings, MAX_PAD_SIZE, pad->paddings_size_, 0);
      pad->paddings_size_ = MAX_PAD_SIZE;
    }
  }
  return NNACL_OK;
}

int PadPrepare(KernelBase *self) {
  NNACL_CHECK_TRUE_RET(self->in_size_ == TWO_TENSOR || self->in_size_ == THREE_TENSOR, NNACL_ERR);
  NNACL_CHECK_TRUE_RET(self->out_size_ == ONE_TENSOR, NNACL_ERR);
  TensorC *input = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(input);
  NNACL_CHECK_FALSE(input->data_type_ != kNumberTypeFloat32 && input->data_type_ != kNumberTypeFloat16, NNACL_ERR);
  return NNACL_OK;
}

KernelBase *CreatePad(OpParameter *param, int data_type) {
  PadStruct *pad = (PadStruct *)malloc(sizeof(PadStruct));
  NNACL_CHECK_NULL_RETURN_NULL(pad);
  memset(pad, 0, sizeof(PadStruct));

  pad->data_type_ = data_type;

  PadParameter *pad_param = (PadParameter *)param;
  pad->pad_mode_ = pad_param->pad_mode_;
  pad->constant_value_ = pad_param->constant_value_;
  pad->paddings_size_ = pad_param->padding_length;
  memcpy(pad->paddings_, pad_param->paddings_, MAX_PAD_SIZE * sizeof(int));

  pad->base_.Release = DefaultRelease;
  pad->base_.Prepare = PadPrepare;
  pad->base_.Resize = PadResize;
  pad->base_.Compute = PadCompute;
  return (KernelBase *)pad;
}

REG_KERNEL_CREATOR(PrimType_PadFusion, kNumberTypeFloat32, CreatePad)
REG_KERNEL_CREATOR(PrimType_PadFusion, kNumberTypeFloat16, CreatePad)
