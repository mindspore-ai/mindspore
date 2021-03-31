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

#include "coder/opcoders/nnacl/fp32/power_fp32_coder.h"
#include <string>
#include <vector>
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"

using mindspore::schema::PrimitiveType_PowFusion;

namespace mindspore::lite::micro::nnacl {

int PowerFP32Coder::DoCode(CoderContext *const context) {
  scale_ = reinterpret_cast<PowerParameter *>(parameter_)->scale_;
  shift_ = reinterpret_cast<PowerParameter *>(parameter_)->shift_;

  Tensor *filter_tensor = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor);
  int size = input_tensor_->ElementsNum();
  MS_CHECK_TRUE(thread_num_ > 0, "thread_num_ <= 0");
  int stride = UP_DIV(size, thread_num_);
  int len = MSMIN(stride, size - stride * kDefaultTaskId);
  std::string exp_addr;
  bool broadcast = true;
  if (input_tensors_.size() == 2) {
    exp_addr = allocator_->GetRuntimeAddr(filter_tensor);
    broadcast = !(input_tensor_->shape() == filter_tensor->shape());
  }
  std::string cur_exp_str;
  if (broadcast) {
    cur_exp_str = input_tensors_.size() == 2 ? exp_addr : "&power";
  } else {
    cur_exp_str = exp_addr;
  }
  // generate code .h .c
  Collect(context, {"nnacl/power.h"}, {"power.c"});
  NNaclFp32Serializer code;
  code.CodeFunction("Power", input_tensor_, cur_exp_str, output_tensor_, len, scale_, shift_, broadcast);
  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_PowFusion, CPUOpCoderCreator<PowerFP32Coder>)

}  // namespace mindspore::lite::micro::nnacl
