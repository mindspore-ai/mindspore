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

#include "coder/opcoders/nnacl/fp32/pad_fp32_coder.h"
#include <string>
#include <vector>
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_PadFusion;

namespace mindspore::lite::micro::nnacl {

int PadFP32Coder::Prepare(CoderContext *const context) {
  pad_param_ = reinterpret_cast<PadParameter *>(parameter_);
  return ReSize();
}

int PadFP32Coder::ReSize() {
  size_t rank = input_tensor_->shape().size();
  if (rank > DEFAULT_PAD_NDIMS) {
    MS_LOG(ERROR) << "Pad input rank should <= " << DEFAULT_PAD_NDIMS << ", got " << rank;
    return RET_ERROR;
  }
  if (pad_param_->pad_mode_ == static_cast<int>(schema::PaddingMode_CONSTANT)) {
    MS_CHECK_RET_CODE(ExtendShape(in_, DEFAULT_PAD_NDIMS, input_tensor_->shape().data(), rank),
                      "ExtendShape input error");
    MS_CHECK_RET_CODE(ExtendShape(out_, DEFAULT_PAD_NDIMS, output_tensor_->shape().data(), rank),
                      "ExtendShape output error");
    if (pad_param_->padding_length < MAX_PAD_SIZE) {
      int ori_paddings[MAX_PAD_SIZE];
      for (int i = 0; i < pad_param_->padding_length; ++i) {
        ori_paddings[i] = pad_param_->paddings_[i];
      }
      MS_CHECK_RET_CODE(ExtendPaddings(pad_param_->paddings_, MAX_PAD_SIZE, ori_paddings, pad_param_->padding_length),
                        "Extendpadding error");
      pad_param_->padding_length = MAX_PAD_SIZE;
    }
  }
  return RET_OK;
}

int PadFP32Coder::ExtendShape(int *shape, int length, const int *ori_shape, int rank) {
  MS_CHECK_PTR(shape);
  MS_CHECK_PTR(ori_shape);
  for (int i = 0; i < length - rank; ++i) {
    shape[i] = 1;
  }
  for (int i = length - rank; i < length; ++i) {
    shape[i] = ori_shape[i - (length - rank)];
  }
  return RET_OK;
}

int PadFP32Coder::ExtendPaddings(int *paddings, int length, const int *ori_paddings, int ori_length) {
  MS_CHECK_PTR(paddings);
  MS_CHECK_PTR(ori_paddings);
  for (int i = 0; i < length - ori_length; ++i) {
    paddings[i] = 0;
  }
  for (int i = length - ori_length; i < length; ++i) {
    paddings[i] = ori_paddings[i - (length - ori_length)];
  }
  return RET_OK;
}

int PadFP32Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/fp32/pad.h", "nnacl/pad_parameter.h"}, {"nnacl/fp32/pad.c"});

  NNaclFp32Serializer code;
  code.CodeArray("in_", in_, DEFAULT_PAD_NDIMS);
  code.CodeArray("out_", out_, DEFAULT_PAD_NDIMS);
  code.CodeArray("padding_", pad_param_->paddings_, MAX_PAD_SIZE);

  int output_size = output_tensor_->ElementsNum();
  if (pad_param_->constant_value_ - 0.0f < 1e-5) {
    code.CodeFunction("memset", output_tensor_, "0", output_size * sizeof(float));
  } else {
    std::vector<float> constant_values(output_size, pad_param_->constant_value_);
    code.CodeArray("output_tensor_", constant_values.data(), output_size);
  }
  code.CodeFunction("Pad", input_tensor_, output_tensor_, "in_", "out_", "padding_", kDefaultTaskId, thread_num_);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_PadFusion, CPUOpCoderCreator<PadFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
