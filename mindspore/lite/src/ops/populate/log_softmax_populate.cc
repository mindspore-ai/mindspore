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
#include "nnacl/softmax_parameter.h"
using mindspore::schema::PrimitiveType_LogSoftmax;

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateLogSoftmaxParameter(const void *prim) {
  auto *log_softmax_param = reinterpret_cast<SoftmaxParameter *>(malloc(sizeof(SoftmaxParameter)));
  if (log_softmax_param == nullptr) {
    MS_LOG(ERROR) << "malloc LogSoftmaxParameter failed.";
    return nullptr;
  }
  memset(log_softmax_param, 0, sizeof(SoftmaxParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  log_softmax_param->op_parameter_.type_ = primitive->value_type();
  auto prim_log_softmax = primitive->value_as_LogSoftmax();
  if (prim_log_softmax == nullptr) {
    MS_LOG(ERROR) << "prim_log_softmax is nullptr";
    return nullptr;
  }
  log_softmax_param->axis_ = prim_log_softmax->axis();
  return reinterpret_cast<OpParameter *>(log_softmax_param);
}
}  // namespace

REG_POPULATE(PrimitiveType_LogSoftmax, PopulateLogSoftmaxParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
