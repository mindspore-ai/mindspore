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
#include "src/ops/populate/v0/strided_slice_populate_v0.h"
#include <limits>
#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/strided_slice_parameter.h"

namespace mindspore {
namespace lite {
OpParameter *PopulateStridedSliceParameterV0(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto strided_slice_prim = primitive->value_as_StridedSlice();
  StridedSliceParameter *strided_slice_param =
    reinterpret_cast<StridedSliceParameter *>(malloc(sizeof(StridedSliceParameter)));
  if (strided_slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc StridedSliceParameter failed.";
    return nullptr;
  }
  memset(strided_slice_param, 0, sizeof(StridedSliceParameter));
  strided_slice_param->op_parameter_.type_ = schema::PrimitiveType_StridedSlice;

  auto begin = strided_slice_prim->begin();
  if (begin != nullptr) {
    if (((size_t)begin->size()) > std::numeric_limits<size_t>::max() / sizeof(int)) {
      MS_LOG(ERROR) << "The value of begin.size() is too big";
      free(strided_slice_param);
      return nullptr;
    }
    memcpy(strided_slice_param->begins_, (begin->data()), begin->size() * sizeof(int));
  }
  auto end = strided_slice_prim->end();
  if (end != nullptr) {
    if (((size_t)end->size()) > std::numeric_limits<size_t>::max() / sizeof(int)) {
      MS_LOG(ERROR) << "The value of end.size() is too big";
      free(strided_slice_param);
      return nullptr;
    }
    memcpy(strided_slice_param->ends_, (end->data()), end->size() * sizeof(int));
  }
  auto stride = strided_slice_prim->stride();
  if (stride != nullptr) {
    if (((size_t)stride->size()) > std::numeric_limits<size_t>::max() / sizeof(int)) {
      MS_LOG(ERROR) << "The value of stride.size() is too big";
      free(strided_slice_param);
      return nullptr;
    }
    memcpy(strided_slice_param->strides_, (stride->data()), stride->size() * sizeof(int));
  }
  strided_slice_param->begins_mask_ = strided_slice_prim->beginMask();
  strided_slice_param->ends_mask_ = strided_slice_prim->endMask();
  strided_slice_param->ellipsisMask_ = strided_slice_prim->ellipsisMask();
  strided_slice_param->newAxisMask_ = strided_slice_prim->newAxisMask();
  strided_slice_param->shrinkAxisMask_ = strided_slice_prim->shrinkAxisMask();

  return reinterpret_cast<OpParameter *>(strided_slice_param);
}

Registry g_stridedSliceV0ParameterRegistry(schema::v0::PrimitiveType_StridedSlice, PopulateStridedSliceParameterV0,
                                           SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
