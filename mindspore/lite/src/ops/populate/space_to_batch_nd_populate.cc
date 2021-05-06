/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "nnacl/fp32/space_to_batch_fp32.h"
using mindspore::schema::PrimitiveType_SpaceToBatchND;

namespace mindspore {
namespace lite {
OpParameter *PopulateSpaceToBatchNDParameter(const void *prim) {
  auto *primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_SpaceToBatchND();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<SpaceToBatchParameter *>(malloc(sizeof(SpaceToBatchParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToBatchParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(SpaceToBatchParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto block_shape = value->block_shape();
  if (block_shape == nullptr) {
    return reinterpret_cast<OpParameter *>(param);
  }
  auto block_shapes = std::vector<int64_t>(block_shape->begin(), block_shape->end());
  if (block_shapes.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of block_shapes.size() is too big";
    free(param);
    return nullptr;
  }
  param->m_ = block_shapes.size();

  auto param_paddings = value->paddings();
  if (param_paddings == nullptr) {
    MS_LOG(ERROR) << "param_paddings is nullptr";
    free(param);
    return nullptr;
  }
  auto fb_paddings = param_paddings->data();
  if (fb_paddings == nullptr) {
    MS_LOG(ERROR) << "fb_paddings is nullptr";
    free(param);
    return nullptr;
  }
  if (fb_paddings->size() == 0 ||
      ((*(fb_paddings->begin())) != nullptr && (*(fb_paddings->begin()))->data() != nullptr &&
       static_cast<uint64_t>(fb_paddings->size() * (*(fb_paddings->begin()))->data()->size()) >
         std::numeric_limits<size_t>::max() / sizeof(int64_t))) {
    MS_LOG(ERROR) << "The value of paddings.size() is zero or too big";
    free(param);
    return nullptr;
  }
  std::vector<int64_t> paddings;
  for (auto fb_padding : *fb_paddings) {
    auto paddings_data = fb_padding->data();
    if (paddings_data == nullptr) {
      MS_LOG(ERROR) << "paddings_data is nullptr";
      free(param);
      return nullptr;
    }
    auto paddings_vec = std::vector<int64_t>(paddings_data->begin(), paddings_data->end());
    paddings.insert(paddings.end(), paddings_vec.begin(), paddings_vec.end());
  }

  for (size_t i = 0; i < block_shapes.size(); ++i) {
    param->block_sizes_[i] = static_cast<int>(block_shapes[i]);
  }
  param->m_ = block_shapes.size();

  for (size_t i = 0; i < paddings.size(); ++i) {
    param->paddings_[i] = static_cast<int>(paddings[i]);
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_SpaceToBatchND, PopulateSpaceToBatchNDParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
