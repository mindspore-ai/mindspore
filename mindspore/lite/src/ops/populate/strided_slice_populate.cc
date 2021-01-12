/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/populate/strided_slice_populate.h"
#include <limits>
#include "src/ops/strided_slice.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/strided_slice.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateStridedSliceParameter(const mindspore::lite::PrimitiveC *primitive) {
  StridedSliceParameter *strided_slice_param =
    reinterpret_cast<StridedSliceParameter *>(malloc(sizeof(StridedSliceParameter)));
  if (strided_slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc StridedSliceParameter failed.";
    return nullptr;
  }
  memset(strided_slice_param, 0, sizeof(StridedSliceParameter));
  strided_slice_param->op_parameter_.type_ = primitive->Type();
  auto n_dims = ((lite::StridedSlice *)primitive)->NDims();
  strided_slice_param->num_axes_ = n_dims;
  auto begin = ((lite::StridedSlice *)primitive)->GetBegins();
  if (begin.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of begin.size() is too big";
    free(strided_slice_param);
    return nullptr;
  }
  memcpy(strided_slice_param->begins_, (begin.data()), begin.size() * sizeof(int));
  auto end = ((lite::StridedSlice *)primitive)->GetEnds();
  if (end.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of end.size() is too big";
    free(strided_slice_param);
    return nullptr;
  }
  memcpy(strided_slice_param->ends_, (end.data()), end.size() * sizeof(int));
  auto stride = ((lite::StridedSlice *)primitive)->GetStrides();
  if (stride.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of stride.size() is too big";
    free(strided_slice_param);
    return nullptr;
  }
  memcpy(strided_slice_param->strides_, (stride.data()), stride.size() * sizeof(int));
  auto in_shape = ((lite::StridedSlice *)primitive)->GetInShape();
  if (in_shape.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of in_shape.size() is too big";
    free(strided_slice_param);
    return nullptr;
  }
  memcpy(strided_slice_param->in_shape_, (in_shape.data()), in_shape.size() * sizeof(int));
  strided_slice_param->in_shape_length_ = static_cast<int>(in_shape.size());
  return reinterpret_cast<OpParameter *>(strided_slice_param);
}

Registry StridedSliceParameterRegistry(schema::PrimitiveType_StridedSlice, PopulateStridedSliceParameter);

}  // namespace lite
}  // namespace mindspore
