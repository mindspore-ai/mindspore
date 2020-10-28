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
#include "internal/src/kernel/fp32/arithmetic.h"
#include "internal/src/lite_log.h"
#include "internal/include/errorcode.h"
#include "internal/include/model.h"
#include "internal/include/ms_tensor.h"
#include "internal/include/lite_utils.h"
#include "nnacl/arithmetic_common.h"
#include "nnacl/fp32/arithmetic.h"

typedef int (*ArithmeticRun)(const float *input0, const float *input1, float *output, const int element_size);
typedef int (*ArithmeticOptRun)(const float *input0, const float *input1, float *output, const int element_size,
                                const ArithmeticParameter *param);

int BroadcastRun(float *input0, float *input1, float *output, int dim, int out_count, int break_pos,
                 ArithmeticRun arithmetic_run, ArithmeticParameter *params) {
  if (dim > break_pos) {
    return arithmetic_run(input0, input1, output, out_count);
  }
  for (int i = 0; i < params->out_shape_[dim]; ++i) {
    int pos0_ = params->in_shape0_[dim] == 1 ? 0 : i;
    int pos1_ = params->in_shape1_[dim] == 1 ? 0 : i;
    int error_code =
      BroadcastRun(input0 + pos0_ * params->in_strides0_[dim], input1 + pos1_ * params->in_strides1_[dim],
                   output + i * params->out_strides_[dim], dim + 1, out_count, break_pos, arithmetic_run, params);
    if (error_code != RET_OK) {
      return error_code;
    }
  }
  return RET_OK;
}

int CalBroadCasting(const TensorPtrVector &in_tensors, int *outside, int *break_pos, ArithmeticParameter *params) {
  params->broadcasting_ = false;
  for (size_t i = 0; i < params->ndim_; ++i) {
    if (params->in_shape0_[i] != params->in_shape1_[i]) {
      if (params->in_shape0_[i] == 1) {
        params->out_shape_[i] = params->in_shape1_[i];
      } else if (params->in_shape1_[i] == 1) {
        params->out_shape_[i] = params->in_shape0_[i];
      } else {
        LITE_LOG_ERROR("shapes of input tensors can not be broadCasted");
        return RET_INPUT_TENSOR_ERROR;
      }
      params->broadcasting_ = true;
    } else {
      params->out_shape_[i] = params->in_shape0_[i];
    }
  }
  if (params->broadcasting_) {
    *outside = 1;
    for (auto i = params->ndim_ - 1; i >= 0; --i) {
      if (params->in_shape0_[i] != params->in_shape1_[i]) {
        *break_pos = i;
        break;
      }
      (*outside) *= params->out_shape_[i];
    }
    ComputeStrides(params->in_shape0_, params->in_strides0_, params->ndim_);
    ComputeStrides(params->in_shape1_, params->in_strides1_, params->ndim_);
    ComputeStrides(params->out_shape_, params->out_strides_, params->ndim_);
  }
  return RET_OK;
}

int RunArithmetic(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, ArithmeticRun arithmetic_run,
                  ArithmeticOptRun arithmetic_opt_run, int outside, int break_pos, ArithmeticParameter *params) {
  int error_code = RET_OK;
  int count = out_tensors[0]->ElementsNum();
  float *input0_data = reinterpret_cast<float *>(in_tensors[0]->data_);
  float *input1_data1 = reinterpret_cast<float *>(in_tensors[1]->data_);
  float *output_data = reinterpret_cast<float *>(out_tensors[0]->data_);
  if (params->broadcasting_) {
    error_code = BroadcastRun(input0_data, input1_data1, output_data, 0, outside, break_pos, arithmetic_run, params);
  } else if (arithmetic_opt_run != NULL) {
    error_code = arithmetic_opt_run(input0_data, input1_data1, output_data, count, params);
  } else {
    error_code = arithmetic_run(input0_data, input1_data1, output_data, count);
  }
  if (error_code != RET_OK) {
    return error_code;
  }
  return RET_OK;
}

int DoArithmeticInferShape(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param) {
  if (in_tensors.size() != 2 || in_tensors[0]->data_ == NULL || in_tensors[1]->data_ == NULL) {
    LITE_LOG_ERROR("input tensors num not correct or input data is NULL!");
    return RET_INPUT_TENSOR_ERROR;
  }
  if (out_tensors.size() != 1) {
    LITE_LOG_ERROR("output tensors num not correct!");
    return RET_ERROR;
  }

  int in_datatype[2] = {in_tensors[0]->data_type_, in_tensors[1]->data_type_};
  int in_format[2] = {static_cast<int>(in_tensors[0]->format_), static_cast<int>(in_tensors[1]->format_)};
  size_t dim_size[2] = {in_tensors[0]->shape_.size(), in_tensors[1]->shape_.size()};
  int *in_shape[2] = {in_tensors[0]->shape_.data(), in_tensors[1]->shape_.data()};
  int out_format;
  int out_datatype;
  int ret = ArithmeticInferShape(in_shape, dim_size, out_tensors[0]->shape_.data(), in_format, &out_format, in_datatype,
                                 &out_datatype, param);
  if (ret != NNACL_OK) {
    LITE_ERROR_LOG("arithmetic infershape failed! ret: %d", ret);
    return RET_ERROR;
  }
  out_tensors[0]->format_ = static_cast<Format>(out_format);
  out_tensors[0]->data_type_ = static_cast<TypeId>(out_datatype);
  return RET_OK;
}

int ChooseKernel(const int kernel_type, ArithmeticRun *arithmetic_run, ArithmeticParameter *params) {
  if (kernel_type == KernelType::KernelType_Mul) {
    if (params->activation_type_ == ActivationType::RELU) {
      *arithmetic_run = ElementMulRelu;
    } else if (params->activation_type_ == ActivationType::RELU6) {
      *arithmetic_run = ElementMulRelu6;
    } else {
      *arithmetic_run = ElementMul;
    }
  } else {
    LITE_LOG_INFO("unsupported operator type");
    return RET_ERROR;
  }
  return RET_OK;
}

int ChooseOptKernel(const int kernel_type, ArithmeticOptRun *arithmetic_opt_run, ArithmeticParameter *params) {
  if (kernel_type == KernelType::KernelType_Mul) {
    if (params->activation_type_ == ActivationType::RELU) {
      *arithmetic_opt_run = ElementOptMulRelu;
    } else if (params->activation_type_ == ActivationType::RELU6) {
      *arithmetic_opt_run = ElementOptMulRelu6;
    } else {
      *arithmetic_opt_run = ElementOptMul;
    }
  } else {
    LITE_LOG_INFO("kernel not have opt version");
  }
  return RET_OK;
}

int DoArithmetic(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
                 mindspore::lite::Allocator *allocator) {
  if (in_tensors.size() != 2 || in_tensors[0]->data_ == NULL || in_tensors[1]->data_ == NULL) {
    LITE_LOG_ERROR("input tensors num not correct or input data is NULL!");
    return RET_INPUT_TENSOR_ERROR;
  }
  if (out_tensors.size() != 1 || out_tensors[0]->data_ == NULL) {
    LITE_LOG_ERROR("output tensors num not correct or output data is NULL!");
    return RET_ERROR;
  }
  if (allocator == NULL) {
    LITE_LOG_ERROR("allocator is NULL!");
    return RET_ERROR;
  }
  ArithmeticParameter *params = reinterpret_cast<ArithmeticParameter *>(node->primitive_);

  ArithmeticRun arithmetic_run = NULL;
  int kernel_type = params->op_parameter_.type_;
  int status = ChooseKernel(kernel_type, &arithmetic_run, params);
  if (status != RET_OK) {
    return status;
  }
  int outside = 0;
  int break_pos = 0;
  // when one of input only has one element
  params->in_elements_num0_ = in_tensors[0]->ElementsNum();
  params->in_elements_num1_ = in_tensors[1]->ElementsNum();
  params->out_elements_num_ = out_tensors[0]->ElementsNum();
  ArithmeticOptRun arithmetic_opt_run = NULL;
  if (params->in_elements_num0_ == 1 || params->in_elements_num1_ == 1) {
    params->broadcasting_ = false;
    ChooseOptKernel(kernel_type, &arithmetic_opt_run, params);
  } else {
    int ret = CalBroadCasting(in_tensors, &outside, &break_pos, params);
    if (ret != RET_OK) {
      return ret;
    }
  }
  return RunArithmetic(in_tensors, out_tensors, arithmetic_run, arithmetic_opt_run, outside, break_pos, params);
}
