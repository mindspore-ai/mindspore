/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/nnacl/int8/transpose_int8_coder.h"
#include "include/errorcode.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::lite::micro::nnacl {
int TransposeInt8Coder::Prepare(CoderContext *const context) {
  auto in_tensor = input_tensors_.front();
  auto out_tensor = output_tensors_.front();
  auto in_shape = in_tensor->shape();
  auto out_shape = out_tensor->shape();
  param_ = reinterpret_cast<TransposeParameter *>(parameter_);
  param_->data_num_ = in_tensor->ElementsNum();

  auto perm_tensor = input_tensors_.at(1);
  int *perm_data = reinterpret_cast<int *>(perm_tensor->data());
  MS_ASSERT(perm_data != nullptr);
  param_->num_axes_ = perm_tensor->ElementsNum();
  for (int i = 0; i < param_->num_axes_; ++i) {
    param_->perm_[i] = perm_data[i];
  }
  param_->strides_[param_->num_axes_ - 1] = 1;
  param_->out_strides_[param_->num_axes_ - 1] = 1;
  for (int i = param_->num_axes_ - kTwo; i >= 0; i--) {
    param_->strides_[i] = in_shape.at(i + 1) * param_->strides_[i + 1];
    param_->out_strides_[i] = out_shape.at(i + 1) * param_->out_strides_[i + 1];
  }
  return RET_OK;
}

int TransposeInt8Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/int8/pack_int8.h", "nnacl/int8/transpose_int8.h"}, {"pack_int8.c", "transpose_int8.c"});

  NNaclInt8Serializer code;
  auto out_shape = output_tensors_[0]->shape();
  if (param_->num_axes_ == DIMENSION_4D && param_->perm_[0] == 0 && param_->perm_[1] == kTwo &&
      param_->perm_[kTwo] == kThree && param_->perm_[kThree] == 1) {
    code.CodeFunction("PackNCHWToNHWCInt8", input_tensors_[0], output_tensors_[0], out_shape[0],
                      out_shape[1] * out_shape[kTwo], out_shape[kThree]);
  } else if (param_->num_axes_ == DIMENSION_4D && param_->perm_[0] == 0 && param_->perm_[1] == kThree &&
             param_->perm_[kTwo] == 1 && param_->perm_[kThree] == kTwo) {
    code.CodeFunction("PackNHWCToNCHWInt8", input_tensors_[0], output_tensors_[0], out_shape[0],
                      out_shape[kTwo] * out_shape[kThree], out_shape[1]);
  } else {
    code << "int out_shape[" << param_->num_axes_ << "] = {";
    for (int i = 0; i < param_->num_axes_; i++) {
      code << out_shape[i] << ",";
    }
    code << "};\n";

    code.CodeStruct("transpore_param", *param_);
    code.CodeFunction("DoTransposeInt8", input_tensors_[0], output_tensors_[0], "out_shape", "&transpore_param");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Transpose, CPUOpCoderCreator<TransposeInt8Coder>)
}  // namespace mindspore::lite::micro::nnacl
