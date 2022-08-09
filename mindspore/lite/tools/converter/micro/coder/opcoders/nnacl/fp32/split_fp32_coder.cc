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
#include "coder/opcoders/nnacl/fp32/split_fp32_coder.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_fp32_serializer.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/parallel.h"
#include "src/litert/kernel/cpu/base/split_base.h"

using mindspore::schema::PrimitiveType_Split;

namespace mindspore::lite::micro::nnacl {
int SplitFP32Coder::Prepare(CoderContext *const context) {
  auto status = mindspore::kernel::SplitBaseCPUKernel::CheckAndInitSplitParam(
    *input_tensor_, reinterpret_cast<SplitParameter *>(parameter_));
  if (RET_OK != status) {
    MS_LOG(ERROR) << "CheckAndInitSplitParam failed";
    return status;
  }
  return RET_OK;
}

int SplitFP32Coder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/base/split_base.h"}, {"split_base.c"});
  if (support_parallel_) {
    Collect(context, {"wrapper/fp32/split_fp32_wrapper.h"}, {"split_fp32_wrapper.c"});
  }
  auto param = reinterpret_cast<SplitParameter *>(parameter_);
  int num_unit = param->split_count_ * param->num_split_;

  NNaclFp32Serializer code;
  code << "    void *output_ptrs[" << output_tensors_.size() << "] = {";
  for (int i = 0; i < param->num_split_; i++) {
    code << allocator_->GetRuntimeAddr(output_tensors_.at(i)) << ",";
  }
  code << "};\n";
  code << "    int input_dim[" << input_tensor_->shape().size() << "] = {";
  for (auto &dim : input_tensor_->shape()) {
    code << dim << ",";
  }
  code << "};\n";
  code << "    int split_sizes[" << param->num_split_ << "] = {";
  for (int i = 0; i < param->num_split_; i++) {
    code << param->split_sizes_[i] << ",";
  }
  code << "};\n";

  code.CodeStruct("split_param", *param);
  if (!support_parallel_) {
    code.CodeFunction("DoSplit", input_tensor_, "(void *)output_ptrs", "input_dim", "0", num_unit, "&split_param",
                      lite::DataTypeSize(input_tensor_->data_type()));
  } else {
    code.CodeBaseStruct("SplitFp32Args", kRunArgs, input_tensor_, "(void *)output_ptrs", "input_dim", num_unit,
                        lite::DataTypeSize(input_tensor_->data_type()), "&split_param");
    code.CodeFunction(kParallelLaunch, "DoSplitRun", kRunArgsAddr, "split_param.op_parameter_.thread_num_");
  }

  context->AppendCode(code.str());
  return RET_OK;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Split, CPUOpCoderCreator<SplitFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt32, PrimitiveType_Split, CPUOpCoderCreator<SplitFP32Coder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat16, PrimitiveType_Split, CPUOpCoderCreator<SplitFP32Coder>)
}  // namespace mindspore::lite::micro::nnacl
