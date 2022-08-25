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

#include "coder/opcoders/nnacl/int8/gather_int8_coder.h"
#include <cfloat>
#include "include/errorcode.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::lite::micro::nnacl {
int GatherInt8Coder::Prepare(CoderContext *context) {
  CHECK_LESS_RETURN(input_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(output_tensors_.size(), 1);
  if (input_tensors_.size() == kInputSize2) {
    auto axis_data = reinterpret_cast<int *>(input_tensors_.at(C2NUM)->data());
    if (axis_data == nullptr) {
      MS_LOG(ERROR) << "GatherInt8CPUkernel input[2] data nullptr.";
      return RET_ERROR;
    }
    axis_ = *axis_data;
  } else {
    axis_ = (reinterpret_cast<GatherParameter *>(parameter_))->axis_;
  }
  auto in_quant_args = input_tensors_.at(0)->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  auto out_quant_args = output_tensors_.at(0)->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  param_.alpha_ = in_quant_args.front().scale / out_quant_args.front().scale;
  param_.zp_in_ = in_quant_args.front().zeroPoint;
  param_.zp_out_ = out_quant_args.front().zeroPoint;

  return RET_OK;
}

int GatherInt8Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/int8/gather_int8.h",
          },
          {
            "gather_int8.c",
          });
  auto input_tensor = input_tensors_.at(0);
  auto indices_tensor = input_tensors_.at(1);

  auto in_shape = input_tensor->shape();
  int in_rank = in_shape.size();
  const int limit = in_shape.at(axis_);
  MS_CHECK_LT(axis_, in_rank, RET_ERROR);
  int indices_element_size = indices_tensor->ElementsNum();
  MS_CHECK_GT(indices_element_size, 0, RET_ERROR);

  if (indices_tensor->data_type() == kNumberTypeInt32) {
    auto indices_ptr = reinterpret_cast<int32_t *>(indices_tensor->data());
    CHECK_NULL_RETURN(indices_ptr);
    for (int i = 0; i < indices_element_size; ++i) {
      if (indices_ptr[i] >= limit) {
        MS_LOG(ERROR) << " indice data: " << indices_ptr[i] << " is not in [ 0, " << (limit - 1) << " ]";
        return RET_ERROR;
      }
    }
  } else if (indices_tensor->data_type() == kNumberTypeInt64) {
    auto indices_ptr = reinterpret_cast<int64_t *>(indices_tensor->data());
    CHECK_NULL_RETURN(indices_ptr);
    for (int i = 0; i < indices_element_size; ++i) {
      if (indices_ptr[i] >= limit) {
        MS_LOG(ERROR) << " indice data: " << indices_ptr[i] << " is not in [ 0, " << (limit - 1) << " ]";
        return RET_ERROR;
      }
    }
  } else {
    MS_LOG(ERROR) << "Unsupported data type:" << indices_tensor->data_type();
    return RET_ERROR;
  }

  int outer_size = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size *= in_shape.at(i);
  }

  int inner_size = 1;
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size *= in_shape.at(i);
  }

  NNaclInt8Serializer code;
  code.CodeStruct("param", param_);
  if (indices_tensor->data_type() == kNumberTypeInt32) {
    code.CodeFunction("GatherInt8Int32Index", input_tensor_, output_tensor_, outer_size, inner_size, limit,
                      input_tensors_.at(1), indices_element_size, "param");
  } else {
    code.CodeFunction("GatherInt8Int64Index", input_tensor_, output_tensor_, outer_size, inner_size, limit,
                      input_tensors_.at(1), indices_element_size, "param");
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Gather, CPUOpCoderCreator<GatherInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
