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

#include "coder/opcoders/nnacl/fp16/transpose_fp16_coder.h"
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_Transpose;
namespace mindspore::lite::micro::nnacl {
int TransposeFp16Coder::Prepare(CoderContext *const context) {
  if (input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(INFO) << "Input tensor data type is invalid";
    return lite::RET_INPUT_PARAM_INVALID;
  }
  MS_CHECK_RET_CODE(Init(), "init failed");
  return RET_OK;
}

void TransposeFp16Coder::GetNHNCTransposeFunc() {
  auto out_shape = output_tensor_->shape();
  if (input_tensor_->shape().size() == DIMENSION_4D && param_->perm_[0] == 0 && param_->perm_[1] == kTwo &&
      param_->perm_[kTwo] == kThree && param_->perm_[kThree] == 1 && out_shape.size() >= kThree) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[1] * out_shape[kTwo];
    nhnc_param_[kTwo] = out_shape[kThree];
    if (input_tensor_->data_type() == kNumberTypeFloat16) {
      NHNCTransposeFunc_ = "PackNCHWToNHWCFp16";
    }
  }
  if (input_tensor_->shape().size() == DIMENSION_4D && param_->perm_[0] == 0 && param_->perm_[1] == kThree &&
      param_->perm_[kTwo] == 1 && param_->perm_[kThree] == kTwo && out_shape.size() >= kThree) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[kTwo] * out_shape[kThree];
    nhnc_param_[kTwo] = out_shape[1];
    if (input_tensor_->data_type() == kNumberTypeFloat16) {
      NHNCTransposeFunc_ = "PackNHWCToNCHWFp16";
    }
  }
}

int TransposeFp16Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "nnacl/transpose.h",
            "nnacl/errorcode.h",
            "nnacl/fp16/transpose_fp16.h",
          },
          {
            "transpose_fp16.c",
          });

  NNaclFp32Serializer code;
  if (input_tensor_->shape().size() != static_cast<size_t>(param_->num_axes_)) {
    code.CodeFunction("memcpy", output_tensor_, input_tensor_, input_tensor_->Size());
    context->AppendCode(code.str());
    return RET_OK;
  }
  if (input_tensors_.size() == DIMENSION_2D) {
    auto input_perm = input_tensors_.at(1);
    MS_CHECK_TRUE_RET(input_perm != nullptr, RET_ERROR);
    MS_CHECK_TRUE_RET(input_perm->data() != nullptr, RET_ERROR);
    int *perm_data = reinterpret_cast<int *>(input_perm->data());
    for (int i = 0; i < input_perm->ElementsNum(); ++i) {
      param_->perm_[i] = perm_data[i];
    }
    for (int i = input_perm->ElementsNum(); i < MAX_SHAPE_SIZE; ++i) {
      param_->perm_[i] = 0;
    }
  }
  GetNHNCTransposeFunc();
  if (!NHNCTransposeFunc_.empty()) {
    Collect(context,
            {
              "nnacl/fp16/pack_fp16.h",
            },
            {
              "pack_fp16.c",
            });
    code.CodeFunction(NHNCTransposeFunc_, input_tensor_, output_tensor_, nhnc_param_[0], nhnc_param_[1],
                      nhnc_param_[kTwo], kDefaultTaskId, 1);
    context->AppendCode(code.str());
    return RET_OK;
  }

  code.CodeStruct("trans_param", *param_);
  dims_ = output_tensor_->shape().size();
  if (dims_ > MAX_TRANSPOSE_DIM_SIZE) {
    int *dim_size = reinterpret_cast<int *>(malloc(dims_ * sizeof(int)));
    if (dim_size == nullptr) {
      return RET_NULL_PTR;
    }
    *(dim_size + dims_ - 1) = 1;
    for (int i = dims_ - 1; i > 0; --i) {
      *(dim_size + i - 1) = *(dim_size + i) * out_shape_[i];
    }
    code.CodeArray("dim_size", dim_size, dims_);
    int *position = reinterpret_cast<int *>(malloc(dims_ * thread_num_ * sizeof(int)));
    if (position == nullptr) {
      free(dim_size);
      return RET_NULL_PTR;
    }
    code.CodeArray("position", position, dims_ * thread_num_);
    code.CodeFunction("TransposeDimsFp16", input_tensor_, output_tensor_, out_shape_, "dim_size", "position",
                      "&trans_param", kDefaultTaskId, thread_num_);
    free(dim_size);
    free(position);
  } else {
    code.CodeFunction("DoTransposeFp16", input_tensor_, output_tensor_, out_shape_, "&trans_param");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Transpose, CPUOpCoderCreator<TransposeFp16Coder>)
}  // namespace mindspore::lite::micro::nnacl
