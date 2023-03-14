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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/fp32_grad/softmax_grad.h"
#include "ops/sparse_softmax_cross_entropy_with_logits.h"
using mindspore::ops::kNameSparseSoftmaxCrossEntropyWithLogits;
using mindspore::schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits;

namespace mindspore {
namespace lite {
OpParameter *PopulateSparseSoftmaxCrossEntropyWithLogitsOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SoftmaxCrossEntropyParameter *>(PopulateOpParameter<SoftmaxCrossEntropyParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new TopkParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::SparseSoftmaxCrossEntropyWithLogits *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }

  param->is_grad_ = op->get_is_grad();
  return reinterpret_cast<OpParameter *>(param);
}
REG_OPERATOR_POPULATE(kNameSparseSoftmaxCrossEntropyWithLogits, PrimitiveType_SparseSoftmaxCrossEntropyWithLogits,
                      PopulateSparseSoftmaxCrossEntropyWithLogitsOpParameter)
}  // namespace lite
}  // namespace mindspore
