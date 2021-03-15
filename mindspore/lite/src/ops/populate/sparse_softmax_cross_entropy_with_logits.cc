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
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32_grad/softmax_grad.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateSparseSoftmaxCrossEntropyWithLogitsParameter(const void *prim) {
  SoftmaxCrossEntropyParameter *softmax_cross_entropy_param_ =
    reinterpret_cast<SoftmaxCrossEntropyParameter *>(malloc(sizeof(SoftmaxCrossEntropyParameter)));
  if (softmax_cross_entropy_param_ == nullptr) {
    MS_LOG(ERROR) << "malloc SoftmaxCrossEntropyParameter failed.";
    return nullptr;
  }
  memset(softmax_cross_entropy_param_, 0, sizeof(SoftmaxCrossEntropyParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  softmax_cross_entropy_param_->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(softmax_cross_entropy_param_);
}
Registry SparseSoftmaxCrossEntropyWithLogitsParameterRegistry(schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits,
                                                              PopulateSparseSoftmaxCrossEntropyWithLogitsParameter,
                                                              SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
