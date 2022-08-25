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

#include "coder/opcoders/nnacl/int8/prelu_int8_coder.h"
#include <algorithm>
#include "include/errorcode.h"
#include "coder/log.h"
#include "coder/opcoders/serializers/nnacl_serializer/nnacl_int8_serializer.h"
#include "coder/opcoders/file_collector.h"

using mindspore::schema::PrimitiveType_PReLUFusion;
constexpr int kSlopeIndex = 1;

namespace mindspore::lite::micro::nnacl {
int PReluInt8Coder::Prepare(CoderContext *context) {
  if (LeakyReluInt8Coder::Prepare(context) != RET_OK) {
    MS_LOG(ERROR) << "LeakyReluInt8Coder::Prepare failed.";
    return RET_ERROR;
  }
  CHECK_LESS_RETURN(input_tensors_.size(), C2NUM);
  CHECK_NULL_RETURN(input_tensors_[kSlopeIndex]);
  if ((input_tensors_[kSlopeIndex]->ElementsNum() == 1) && (input_tensors_[kSlopeIndex]->IsConst())) {
    auto quant = input_tensors_[kSlopeIndex]->quant_params();
    auto scale = static_cast<float>(quant.front().scale);
    auto zp = quant.front().zeroPoint;
    auto dat = static_cast<int8_t *>(input_tensors_[kSlopeIndex]->data())[0];
    quant_prelu_parm_.slope_ = (dat - zp) / scale;
  } else {
    MS_LOG(ERROR) << "prelu int8 operator is not fully supported at present.";
    return RET_ERROR;
  }
  return RET_OK;
}
std::unique_ptr<OperatorCoder> CPUPreluINT8CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                        const std::vector<Tensor *> &out_tensors,
                                                        const LiteGraph::Node *node, size_t node_index, Target target,
                                                        int schema_version) {
  if (node == nullptr) {
    MS_LOG(ERROR) << "node is null";
    return nullptr;
  }
  if (in_tensors.size() < C2NUM) {
    MS_LOG(ERROR) << "prelu in_tensors is less than 2.";
    return nullptr;
  }
  if (in_tensors[kSlopeIndex] == nullptr) {
    MS_LOG(ERROR) << "tensor is nullptr.";
    return nullptr;
  }
  std::unique_ptr<OperatorCoder> coder;
  if ((in_tensors[kSlopeIndex]->ElementsNum() == 1) && (in_tensors[kSlopeIndex]->IsConst())) {
    coder = CPUOpCoderCreator<PReluInt8Coder>(in_tensors, out_tensors, node, node_index, target, schema_version);
    if (coder == nullptr) {
      MS_LOG(ERROR) << "create PReluInt8Coder failed.";
      return nullptr;
    }
    return coder;
  } else {
    MS_LOG(ERROR) << "prelu int8 operator is not supported in nnacl.";
    return nullptr;
  }
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_PReLUFusion, CPUPreluINT8CoderCreator)
}  // namespace mindspore::lite::micro::nnacl
