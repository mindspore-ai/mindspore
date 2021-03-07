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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/reduce_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateReduceParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto reduce_prim = primitive->value_as_Reduce();
  ReduceParameter *reduce_param = reinterpret_cast<ReduceParameter *>(malloc(sizeof(ReduceParameter)));
  if (reduce_param == nullptr) {
    MS_LOG(ERROR) << "malloc ReduceParameter failed.";
    return nullptr;
  }
  memset(reduce_param, 0, sizeof(ReduceParameter));
  reduce_param->op_parameter_.type_ = schema::PrimitiveType_ReduceFusion;

  reduce_param->keep_dims_ = reduce_prim->keepDims();
  reduce_param->reduce_to_end_ = reduce_prim->reduceToEnd();
  reduce_param->coeff = reduce_prim->coeff();
  auto axisVector = reduce_prim->axes();
  if (axisVector->size() > MAX_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Reduce axes size " << axisVector->size() << " exceed limit " << MAX_SHAPE_SIZE;
    free(reduce_param);
    return nullptr;
  }
  reduce_param->num_axes_ = static_cast<int>(axisVector->size());
  int i = 0;
  for (auto iter = axisVector->begin(); iter != axisVector->end(); iter++) {
    reduce_param->axes_[i++] = *iter;
  }
  reduce_param->mode_ = static_cast<int>(reduce_prim->mode());
  return reinterpret_cast<OpParameter *>(reduce_param);
}
}  // namespace

Registry g_reduceV0ParameterRegistry(schema::v0::PrimitiveType_Reduce, PopulateReduceParameter, SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
