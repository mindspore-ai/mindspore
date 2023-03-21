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

#include "coder/opcoders/nnacl/fp32/transpose_fp32_coder.h"
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/common.h"

using mindspore::schema::PrimitiveType_Transpose;
namespace mindspore::lite::micro::nnacl {
int TransposeFp32Coder::Resize() {
  if (input_tensors_.size() == DIMENSION_2D) {
    param_->num_axes_ = input_tensors_.at(1)->ElementsNum();
  }
  if (input_tensors_.at(kInputIndex)->shape().size() != static_cast<size_t>(param_->num_axes_)) {
    return RET_OK;
  }
  // get perm data
  MS_CHECK_TRUE_RET(input_tensors_.size() == DIMENSION_2D, RET_ERROR);
  auto perm_tensor = input_tensors_.at(1);
  int *perm_data = reinterpret_cast<int *>(perm_tensor->data());
  MS_CHECK_TRUE_RET(perm_data != nullptr, RET_ERROR);
  for (int i = 0; i < param_->num_axes_; ++i) {
    param_->perm_[i] = perm_data[i];
  }
  auto in_shape = input_tensor_->shape();
  auto out_shape = output_tensor_->shape();
  param_->strides_[param_->num_axes_ - 1] = 1;
  param_->out_strides_[param_->num_axes_ - 1] = 1;
  param_->data_num_ = input_tensor_->ElementsNum();
  for (int i = param_->num_axes_ - 2; i >= 0; i--) {
    param_->strides_[i] = in_shape.at(i + 1) * param_->strides_[i + 1];
    param_->out_strides_[i] = out_shape.at(i + 1) * param_->out_strides_[i + 1];
  }

  return RET_OK;
}

int TransposeFp32Coder::Init() {
  param_ = reinterpret_cast<TransposeParameter *>(parameter_);
  MS_CHECK_PTR(param_);
  return Resize();
}

int TransposeFp32Coder::Prepare(CoderContext *const context) {
  MS_CHECK_RET_CODE(Init(), "init failed");
  return RET_OK;
}

void TransposeFp32Coder::GetNHNCTransposeFunc() {
  auto out_shape = output_tensor_->shape();
  if (input_tensor_->shape().size() == DIMENSION_4D && param_->perm_[0] == 0 && param_->perm_[1] == kTwo &&
      param_->perm_[kTwo] == kThree && param_->perm_[kThree] == 1) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[1] * out_shape[kTwo];
    nhnc_param_[kTwo] = out_shape[kThree];
    if (input_tensor_->data_type() == kNumberTypeFloat32) {
      NHNCTransposeFunc_ = "PackNCHWToNHWCFp32";
    }
  }
  if (input_tensor_->shape().size() == DIMENSION_4D && param_->perm_[0] == 0 && param_->perm_[1] == kThree &&
      param_->perm_[kTwo] == 1 && param_->perm_[kThree] == kTwo) {
    nhnc_param_[0] = out_shape[0];
    nhnc_param_[1] = out_shape[kTwo] * out_shape[kThree];
    nhnc_param_[kTwo] = out_shape[1];
    if (input_tensor_->data_type() == kNumberTypeFloat32) {
      NHNCTransposeFunc_ = "PackNHWCToNCHWFp32";
    }
  }
}

int TransposeFp32Coder::DoCode(CoderContext *const context) {
  Collect(context,
          {
            "wrapper/fp32/transpose_fp32_wrapper.h",
            "nnacl/transpose.h",
            "nnacl/errorcode.h",
            "nnacl/fp32/transpose_fp32.h",
          },
          {
            "transpose_fp32_wrapper.c",
            "transpose_fp32.c",
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
    if (!support_parallel_) {
      code.CodeFunction(NHNCTransposeFunc_, input_tensor_, output_tensor_, nhnc_param_[0], nhnc_param_[1],
                        nhnc_param_[kTwo], kDefaultTaskId, 1);
    } else {
      code.CodeStruct("transpose_param", *param_);
      code.CodeBaseStruct("TransposeFp32Args", kRunArgs, input_tensor_, output_tensor_, nhnc_param_[0], nhnc_param_[1],
                          nhnc_param_[kTwo], "&transpose_param");
      if (NHNCTransposeFunc_ == "PackNCHWToNHWCFp32") {
        code.CodeFunction(kParallelLaunch, "DoTransposeNCHWToNHWC", kRunArgsAddr,
                          "transpose_param.op_parameter_.thread_num_");
      } else if (NHNCTransposeFunc_ == "PackNHWCToNCHWFp32") {
        code.CodeFunction(kParallelLaunch, "DoTransposeNHWCToNCHW", kRunArgsAddr,
                          "transpose_param.op_parameter_.thread_num_");
      }
    }
    context->AppendCode(code.str());
    return RET_OK;
  }

  code.CodeStruct("trans_param", *param_);
  auto out_shape = output_tensor_->shape();
  dims_ = static_cast<int>(out_shape.size());
  code.CodeArray("output_shape", out_shape.data(), dims_, true);
  if (dims_ > MAX_TRANSPOSE_DIM_SIZE) {
    int *dim_size = reinterpret_cast<int *>(malloc(dims_ * sizeof(int)));
    if (dim_size == nullptr) {
      return RET_NULL_PTR;
    }
    *(dim_size + dims_ - 1) = 1;
    for (int i = dims_ - 1; i > 0; --i) {
      *(dim_size + i - 1) = *(dim_size + i) * out_shape[i];
    }
    code.CodeArray("dim_size", dim_size, dims_);
    int *position = reinterpret_cast<int *>(malloc(dims_ * thread_num_ * sizeof(int)));
    if (position == nullptr) {
      free(dim_size);
      return RET_NULL_PTR;
    }
    code.CodeArray("position", position, dims_ * thread_num_);
    code.CodeFunction("TransposeDimsFp32", input_tensor_, output_tensor_, "output_shape", "dim_size", "position",
                      "&trans_param", kDefaultTaskId, thread_num_);
    free(dim_size);
    free(position);
  } else {
    code.CodeFunction("DoTransposeFp32", input_tensor_, output_tensor_, "output_shape", "&trans_param");
  }
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Transpose, CPUOpCoderCreator<TransposeFp32Coder>)
}  // namespace mindspore::lite::micro::nnacl
