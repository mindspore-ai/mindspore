/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "coder/opcoders/base/strided_slice_dynamic_base_coder.h"
#include <cmath>
#include "mindspore/lite/src/common/log_util.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/parallel.h"
#include "coder/utils/coder_utils.h"
#include "tools/common/string_util.h"
#include "base/float16.h"

using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::lite::micro {
namespace {
size_t GetInnerSize(TypeId type_id, size_t inner_elements) {
  switch (type_id) {
    case kNumberTypeInt8:
      return inner_elements * sizeof(int8_t);
    case kNumberTypeFloat32:
      return inner_elements * sizeof(float);
    case kNumberTypeInt32:
      return inner_elements * sizeof(int32_t);
    case kNumberTypeFloat16:
      return inner_elements * sizeof(float16);
    default:
      MS_LOG(ERROR) << "Not supported data type: " << type_id;
      return 0;
  }
}
}  // namespace

int StridedSliceDynamicBaseCoder::Prepare(CoderContext *context) {
  CHECK_LESS_RETURN(input_tensors_.size(), C2NUM);
  for (size_t i = 1; i < input_tensors_.size(); ++i) {
    MS_CHECK_TRUE_MSG(input_tensors_[i]->IsConst(), RET_PARAM_INVALID,
                      "The " << i << " input of strided slice should be const.");
    MS_CHECK_TRUE_MSG(input_tensors_[i]->data_type() == kNumberTypeInt32, RET_PARAM_INVALID,
                      "The " << i << " input tensor data type should be int32.");
  }
  CHECK_LESS_RETURN(output_tensors_.size(), C1NUM);
  strided_slice_param_ = reinterpret_cast<StridedSliceParameter *>(parameter_);
  CHECK_NULL_RETURN(strided_slice_param_);
  MS_CHECK_TRUE_MSG(memset_s(&struct_, sizeof(struct_), 0, sizeof(struct_)) == EOK, RET_ERROR,
                    "memset_s struct_ failed.");
  if (input_tensor_->data_type() != kNumberTypeInt8 && input_tensor_->data_type() != kNumberTypeFloat32 &&
      input_tensor_->data_type() != kNumberTypeInt32 && input_tensor_->data_type() != kNumberTypeFloat16) {
    MS_LOG(ERROR) << "Not supported data type: " << input_tensor_->data_type();
    return RET_ERROR;
  }
  struct_.data_type_ = static_cast<TypeIdC>(input_tensor_->data_type());
  struct_.in_shape_size_ = strided_slice_param_->in_shape_length_;
  struct_.outer_ = 1;
  struct_.inner_ = 1;
  struct_.inner_size_ = GetInnerSize(input_tensor_->data_type(), inner_);
  auto begin_tensor = input_tensors_.at(1);
  input_shape_ = shape_info_container_->GetTemplateShape(input_tensor_);
  if (input_shape_.size() > DIMENSION_8D || begin_tensor->shape().size() > DIMENSION_8D) {
    MS_LOG(ERROR) << "StridedSlice not support input rank or begin num exceeds " << DIMENSION_8D;
    return RET_ERROR;
  }
  PadStridedSliceParamTo8D();
  return RET_OK;
}

int StridedSliceDynamicBaseCoder::DoCode(CoderContext *ctx) {
  Collect(ctx,
          {
            "nnacl/fp32/strided_slice_fp32.h",
          },
          {
            "strided_slice_fp32.c",
          });
  nnacl::NNaclFp32Serializer code;
  code.CodeStruct("strided_slice_parameter", struct_, dynamic_param_);
  std::string input_data = GetTensorAddr(input_tensor_, input_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  std::string output_data = GetTensorAddr(output_tensor_, output_tensor_->IsConst(), dynamic_mem_manager_, allocator_);
  code.CodeFunction("DoStridedSliceIn8D", input_data, output_data, "&strided_slice_parameter");
  ctx->AppendCode(code.str());
  return RET_OK;
}

void StridedSliceDynamicBaseCoder::PadStridedSliceParamTo8D() {
  std::vector<int32_t> begins(DIMENSION_8D, 0);
  std::vector<std::string> ends(DIMENSION_8D, "");
  std::vector<int32_t> strides(DIMENSION_8D, 0);
  std::vector<std::string> input_shape(DIMENSION_8D, "");
  for (int i = 0; i < struct_.in_shape_size_; ++i) {
    begins[i] = strided_slice_param_->begins_[i];
    ends[i] = "MSMIN(" + std::to_string(strided_slice_param_->ends_[i]) + ", " + input_shape_[i] + ")";
    strides[i] = strided_slice_param_->strides_[i];
    input_shape[i] = input_shape_[i];
  }
  end_.resize(DIMENSION_8D);
  input_shape_.resize(DIMENSION_8D);
  int real_index = struct_.in_shape_size_ - 1;
  for (int i = DIMENSION_8D - 1; i >= 0; --i) {
    if (real_index >= 0) {
      struct_.begins_[i] = begins[real_index];
      end_[i] = ends[real_index];
      struct_.strides_[i] = strides[real_index];
      input_shape_[i] = input_shape[real_index--];
    } else {
      struct_.begins_[i] = 0;
      end_[i] = "1";
      struct_.strides_[i] = 1;
      input_shape_[i] = "1";
    }
  }
  struct_.in_shape_size_ = DIMENSION_8D;
  dynamic_param_.in_shape_.clear();
  dynamic_param_.end_.clear();
  dynamic_param_.in_shape_ = "{";
  dynamic_param_.end_ = "{";
  for (int i = 0; i < DIMENSION_8D; ++i) {
    dynamic_param_.end_ += end_[i] + ", ";
    dynamic_param_.in_shape_ += input_shape_[i] + ", ";
  }
  dynamic_param_.in_shape_ += "}";
  dynamic_param_.end_ += "}";
}

REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat32, PrimitiveType_StridedSlice,
                           CPUOpCoderCreator<StridedSliceDynamicBaseCoder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeFloat16, PrimitiveType_StridedSlice,
                           CPUOpCoderCreator<StridedSliceDynamicBaseCoder>)
REG_DYNAMIC_OPERATOR_CODER(kARM64, kNumberTypeInt32, PrimitiveType_StridedSlice,
                           CPUOpCoderCreator<StridedSliceDynamicBaseCoder>)
}  // namespace mindspore::lite::micro
