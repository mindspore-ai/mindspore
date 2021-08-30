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

#include "coder/opcoders/base/strided_slice_base_coder.h"
#include <math.h>
#include <string>
#include "mindspore/lite/src/common/log_util.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/parallel.h"
using mindspore::schema::PrimitiveType_StridedSlice;

namespace mindspore::lite::micro {

namespace {
size_t GetInnerSize(TypeId type_id, int inner_elements) {
  switch (type_id) {
    case kNumberTypeInt8:
      return inner_elements * sizeof(int8_t);
    case kNumberTypeFloat32:
      return inner_elements * sizeof(float);
    case kNumberTypeInt32:
      return inner_elements * sizeof(int32_t);
    default:
      MS_LOG(ERROR) << "Not supported data type: " << type_id;
      return 0;
  }
}
}  // namespace

bool StridedSliceBaseCoder::MatchFastPattern() {
  std::vector<int> in_shape = input_tensor_->shape();
  std::vector<int> out_shape = output_tensor_->shape();
  if (in_shape.size() != out_shape.size()) {
    return false;
  }
  std::vector<int> axis_list;
  for (int i = 0; i < static_cast<int>(in_shape.size()); ++i) {
    if (in_shape[i] != out_shape[i]) {
      axis_list.emplace_back(i);
    }
  }
  if (axis_list.size() == 1) {
    split_axis_ = axis_list.front();
    return true;
  }
  return false;
}

int StridedSliceBaseCoder::InitFastRunParam() {
  std::vector<int> in_shape = input_tensor_->shape();
  std::vector<int> out_shape = output_tensor_->shape();
  MS_CHECK_LT(static_cast<unsigned int>(split_axis_), in_shape.size(), RET_ERROR);
  // reset && cal inner, outer
  for (int i = 0; i < split_axis_; ++i) {
    outer_ *= in_shape[i];
  }
  for (size_t i = split_axis_ + 1; i < in_shape.size(); i++) {
    inner_ *= in_shape[i];
  }
  int thread_num = strided_slice_parameter_->op_parameter_.thread_num_;
  // decide multi-thread launch strategy
  if (outer_ == 1) {
    parallel_on_split_axis_ = true;
    cal_num_per_thread_ = UP_DIV(out_shape[split_axis_], thread_num);
  } else {
    parallel_on_outer_ = true;
    cal_num_per_thread_ = UP_DIV(outer_, thread_num);
  }
  return RET_OK;
}

int StridedSliceBaseCoder::ReSize() {
  fast_run_ = MatchFastPattern();
  if (fast_run_) {
    return InitFastRunParam();
  }
  return RET_OK;
}

int StridedSliceBaseCoder::Prepare(CoderContext *context) {
  strided_slice_parameter_ = reinterpret_cast<StridedSliceParameter *>(parameter_);
  return ReSize();
}

int StridedSliceBaseCoder::DoFastCode(CoderContext *ctx) {
  std::vector<int> in_shape = input_tensor_->shape();
  std::vector<int> out_shape = output_tensor_->shape();
  int begin_index = strided_slice_parameter_->begins_[split_axis_];
  int caled_num = kDefaultTaskId * cal_num_per_thread_;
  nnacl::NNaclFp32Serializer code;
  std::string input_ptr_str = allocator_->GetRuntimeAddr(input_tensor_);
  std::string output_ptr_str = allocator_->GetRuntimeAddr(output_tensor_);
  if (parallel_on_outer_) {
    int cur_outer = outer_ - caled_num;
    if (cur_outer <= 0) {
      return RET_OK;
    }
    if (cur_outer > cal_num_per_thread_) {
      cur_outer = cal_num_per_thread_;
    }
    code << "uint8_t *cur_in_ptr = "
         << "(uint8_t *)(" << input_ptr_str << ")"
         << " + " << (caled_num * in_shape[split_axis_] + begin_index) * inner_size_ << ";\n";
    code << " uint8_t *cur_out_ptr = "
         << "(uint8_t *)(" << output_ptr_str << ")"
         << " + " << caled_num * out_shape[split_axis_] * inner_size_ << ";\n";
    code.CodeFunction("FastStride", "cur_in_ptr", "cur_out_ptr", out_shape.at(split_axis_),
                      strided_slice_parameter_->strides_[split_axis_], cur_outer, inner_size_,
                      in_shape.at(split_axis_) * inner_size_);
  } else {
    int cal_axis_num = out_shape.at(split_axis_) - caled_num;
    if (cal_axis_num <= 0) {
      return RET_OK;
    }
    if (cal_axis_num > cal_num_per_thread_) {
      cal_axis_num = cal_num_per_thread_;
    }
    code << "uint8_t *cur_in_ptr = "
         << "(uint8_t *)(" << input_ptr_str << ")"
         << " + " << (caled_num * strided_slice_parameter_->strides_[split_axis_] + begin_index) * inner_size_ << ";\n";
    code << "uint8_t *cur_out_ptr = "
         << "(uint8_t *)(" << output_ptr_str << ")"
         << " + " << caled_num * inner_size_ << ";\n";
    code.CodeFunction("FastStride", "cur_in_ptr", "cur_out_ptr", cal_axis_num,
                      strided_slice_parameter_->strides_[split_axis_], 1, inner_size_, 0);
  }
  ctx->AppendCode(code.str());
  return RET_OK;
}

int StridedSliceBaseCoder::DoNormalCode(CoderContext *ctx) {
  nnacl::NNaclFp32Serializer code;
  code.CodeStruct("strided_slice_parameter", *strided_slice_parameter_);
  code.CodeFunction("DoStridedSlice", input_tensor_, output_tensor_,
                    "(StridedSliceParameter *)&strided_slice_parameter");
  ctx->AppendCode(code.str());
  return RET_OK;
}

int StridedSliceBaseCoder::DoCode(CoderContext *ctx) {
  inner_size_ = GetInnerSize(input_tensor_->data_type(), inner_);
  Collect(ctx,
          {
            "nnacl/fp32/strided_slice_fp32.h",
          },
          {
            "strided_slice_fp32.c",
          });
  if (fast_run_) {
    return DoFastCode(ctx);
  }
  return DoNormalCode(ctx);
}
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_StridedSlice,
                   CPUOpCoderCreator<StridedSliceBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_StridedSlice, CPUOpCoderCreator<StridedSliceBaseCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_StridedSlice, CPUOpCoderCreator<StridedSliceBaseCoder>)
}  // namespace mindspore::lite::micro
