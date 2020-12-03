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

#include "src/ops/slice.h"
#include "src/common/log_adapter.h"
#include "src/tensor.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/slice_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateSliceParameter(const mindspore::lite::PrimitiveC *primitive) {
  SliceParameter *slice_param = reinterpret_cast<SliceParameter *>(malloc(sizeof(SliceParameter)));
  if (slice_param == nullptr) {
    MS_LOG(ERROR) << "malloc SliceParameter failed.";
    return nullptr;
  }
  memset(slice_param, 0, sizeof(SliceParameter));
  auto param = reinterpret_cast<mindspore::lite::Slice *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  slice_param->op_parameter_.type_ = primitive->Type();
  auto param_begin = param->GetPostProcessBegin();
  auto param_size = param->GetPostProcessSize();
  if (param_begin.size() != param_size.size()) {
    free(slice_param);
    return nullptr;
  }
  slice_param->param_length_ = static_cast<int32_t>(param_begin.size());
  for (int32_t i = 0; i < slice_param->param_length_; ++i) {
    slice_param->begin_[i] = param_begin.at(i);
    slice_param->size_[i] = param_size.at(i);
  }
  return reinterpret_cast<OpParameter *>(slice_param);
}
Registry SliceParameterRegistry(schema::PrimitiveType_Slice, PopulateSliceParameter);

}  // namespace lite
}  // namespace mindspore
