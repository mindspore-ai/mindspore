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

#include "nnacl/kernel/cast.h"
#include "nnacl/op_base.h"
#include "nnacl/base/cast_base.h"
#include "nnacl/nnacl_common.h"
#include "nnacl/tensor_c_utils.h"

#ifdef ENABLE_FP16
#include "nnacl/fp16/cast_fp16.h"
#endif

int CastToFp32(const TensorC *input, TensorC *output, int offset, int data_num) {
  int input_data_type = input->data_type_;
  float *output_data = (float *)output->data_;
  switch (input_data_type) {
    case kNumberTypeBool:
      BoolToFloat32((const bool *)(input->data_) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeUInt8:
      Uint8ToFloat32((const uint8_t *)(input->data_) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeInt32:
      Int32ToFloat32((const int32_t *)(input->data_) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeFloat16:
#ifdef ENABLE_FP16
      Fp16ToFloat32((const float16_t *)(input->data_) + offset, output_data + offset, data_num);
#else
      Fp16ToFloat32((const uint16_t *)(input->data_) + offset, output_data + offset, data_num);
#endif
      break;
    case kNumberTypeInt64:
      Int64ToFloat32((const int64_t *)(input->data_) + offset, output_data + offset, data_num);
      break;
    default:
      return NNACL_ERR;
  }
  return NNACL_OK;
}

int CastToFp16(const TensorC *input, TensorC *output, int offset, int data_num) {
  int input_data_type = input->data_type_;
#ifdef ENABLE_FP16
  float16_t *output_data = (float16_t *)output->data_;
  switch (input_data_type) {
    case kNumberTypeFloat32:
      Float32ToFp16((const float *)(input->data_) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeInt64:
      Int64ToFp16((const int64_t *)(input->data_) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeInt32:
      Int32ToFp16((const int32_t *)(input->data_) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeBool:
      BoolToFp16((const bool *)(input->data_) + offset, output_data + offset, data_num);
      break;
    case kNumberTypeUInt8:
      Uint8ToFp16((const uint8_t *)(input->data_) + offset, output_data + offset, data_num);
      break;
    default:
      return NNACL_ERR;
  }
#else
  if (input_data_type == kNumberTypeFloat32) {
    Float32ToFp16((const float *)(input->data_) + offset, (uint16_t *)(output->data_) + offset, data_num);
  } else {
    return NNACL_ERR;
  }
#endif
  return NNACL_OK;
}

int CastToOthers(const TensorC *input, TensorC *output, int offset, int data_num) {
  int input_data_type = input->data_type_;
  int output_data_type = output->data_type_;
  void *output_data = output->data_;
  if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt64) {
    Float32ToInt64((const float *)(input->data_) + offset, (int64_t *)(output_data) + offset, data_num);
  } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt32) {
    Float32ToInt32((const float *)(input->data_) + offset, (int32_t *)(output_data) + offset, data_num);
  } else if (input_data_type == kNumberTypeInt32 && output_data_type == kNumberTypeInt64) {
    Int32ToInt64((const int32_t *)(input->data_) + offset, (int64_t *)(output_data) + offset, data_num);
  } else if (input_data_type == kNumberTypeInt64 && output_data_type == kNumberTypeInt32) {
    Int64ToInt32((const int64_t *)(input->data_) + offset, (int32_t *)(output_data) + offset, data_num);
  } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeInt16) {
    Float32ToInt16((const float *)(input->data_) + offset, (int16_t *)(output_data) + offset, data_num);
  } else if (input_data_type == kNumberTypeBool && output_data_type == kNumberTypeInt32) {
    BoolToInt32((const bool *)(input->data_) + offset, (int32_t *)(output_data) + offset, data_num);
  } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeBool) {
    Float32ToBool((const float *)(input->data_) + offset, (bool *)(output_data) + offset, data_num);
  } else if (input_data_type == kNumberTypeFloat32 && output_data_type == kNumberTypeUInt8) {
    Float32ToUint8((const float *)(input->data_) + offset, (uint8_t *)(output_data) + offset, data_num);
  } else {
    return NNACL_ERR;
  }
  return NNACL_OK;
}

int CastLaunch(void *cdata, int task_id, float l, float r) {
  CastStruct *cast = (CastStruct *)cdata;
  NNACL_CHECK_NULL_RETURN_ERR(cast);

  NNACL_CHECK_FALSE(cast->base_.in_size_ < ONE_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(cast->base_.out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);

  TensorC *in = cast->base_.in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in);
  NNACL_CHECK_NULL_RETURN_ERR(in->data_);
  TensorC *out = cast->base_.out_[OUTPUT_INDEX];
  NNACL_CHECK_NULL_RETURN_ERR(out);
  NNACL_CHECK_NULL_RETURN_ERR(out->data_);

  int stride = cast->stride_;
  NNACL_CHECK_INT_MUL_NOT_OVERFLOW(task_id, stride, NNACL_ERR);

  int data_num = MSMIN(stride, cast->data_num_ - task_id * stride);
  if (data_num <= 0) {
    return NNACL_OK;
  }

  int offset = task_id * stride;
  int input_data_type = in->data_type_;
  int output_data_type = out->data_type_;
  if (input_data_type == output_data_type) {
    size_t datalen = DataTypeCSize((TypeIdC)input_data_type);
    memcpy((int8_t *)(out->data_) + offset * datalen, (int8_t *)(in->data_) + offset * datalen, data_num * datalen);
    return NNACL_OK;
  }

  if (output_data_type == kNumberTypeFloat32) {
    return CastToFp32(in, out, offset, data_num);
  } else if (output_data_type == kNumberTypeFloat16) {
    return CastToFp16(in, out, offset, data_num);
  } else {
    return CastToOthers(in, out, offset, data_num);
  }
  return NNACL_OK;
}

int cast_prepare(struct KernelBase *self) {
  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  return NNACL_OK;
}

// Kernel resize input shape
int cast_resize(struct KernelBase *self) {
  CastStruct *cast = (CastStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(cast);
  NNACL_CHECK_FALSE(self->in_size_ < ONE_TENSOR, NNACL_INPUT_TENSOR_ERROR);
  TensorC *in_tensor = self->in_[FIRST_INPUT];
  NNACL_CHECK_NULL_RETURN_ERR(in_tensor);
  int data_num = GetElementNum(in_tensor);
  if (data_num == 0) {
    return NNACL_OK;
  }

  cast->data_num_ = data_num;
  NNACL_CHECK_FALSE(self->out_size_ < ONE_TENSOR, NNACL_OUTPUT_TENSOR_ERROR);
  // update thread num
  cast->base_.thread_nr_ = cast->base_.UpdateThread(
    TC_PTYPE(PrimType_Cast), 1, 1, GetElementNum(cast->base_.out_[FIRST_INPUT]), cast->base_.thread_nr_);
  cast->stride_ = UP_DIV(data_num, cast->base_.thread_nr_);
  return NNACL_OK;
}

int cast_release(struct KernelBase *self) { return NNACL_OK; }

// Cast Op Compute
int cast_compute(struct KernelBase *self) {
  CastStruct *cast = (CastStruct *)self;
  NNACL_CHECK_NULL_RETURN_ERR(cast);
  if (cast->data_num_ == 0) {
    return NNACL_OK;
  }

  return self->env_->ParallelLaunch(self->env_->thread_pool_, CastLaunch, self, self->thread_nr_);
}

KernelBase *CreateCast(OpParameter *param, int data_type) {
  CastStruct *cast = (CastStruct *)malloc(sizeof(CastStruct));
  NNACL_MALLOC_CHECK_NULL_RETURN_NULL(cast);
  memset(cast, 0, sizeof(CastStruct));
  cast->base_.Prepare = cast_prepare;
  cast->base_.Resize = cast_resize;
  cast->base_.Release = cast_release;
  cast->base_.Compute = cast_compute;
  cast->stride_ = 0;
  cast->data_num_ = 0;
  return (KernelBase *)cast;
}

// todo register kernel
