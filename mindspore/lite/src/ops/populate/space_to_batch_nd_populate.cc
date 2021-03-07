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

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateSpaceToBatchNDParameter(const void *prim) {
  auto *space_batch_param_nd = reinterpret_cast<SpaceToBatchParameter *>(malloc(sizeof(SpaceToBatchParameter)));
  if (space_batch_param_nd == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToBatchParameter failed.";
    return nullptr;
  }
  memset(space_batch_param_nd, 0, sizeof(SpaceToBatchParameter));
  const schema::Primitive *primitive = static_cast<const schema::Primitive *>(prim);
  space_batch_param_nd->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_SpaceToBatchND();
  if (param->block_shape() == nullptr) {
    return reinterpret_cast<OpParameter *>(space_batch_param_nd);
  }
  auto block_shapes = std::vector<int64_t>(param->block_shape()->begin(), param->block_shape()->end());
  if (block_shapes.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of block_shapes.size() is too big";
    free(space_batch_param_nd);
    return nullptr;
  }
  space_batch_param_nd->m_ = block_shapes.size();

  auto fb_paddings = param->paddings()->data();
  if (fb_paddings->size() == 0 ||
      static_cast<uint64_t>(fb_paddings->size() * (*(fb_paddings->begin()))->data()->size()) >
        std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
    MS_LOG(ERROR) << "The value of paddings.size() is zero or too big";
    free(space_batch_param_nd);
    return nullptr;
  }
  std::vector<int64_t> paddings;
  for (auto iter = fb_paddings->begin(); iter != fb_paddings->end(); ++iter) {
    auto paddings_data = (*iter)->data();
    auto paddings_vec = std::vector<int64_t>(paddings_data->begin(), paddings_data->end());
    paddings.insert(paddings.end(), paddings_vec.begin(), paddings_vec.end());
  }

  for (size_t i = 0; i < block_shapes.size(); ++i) {
    space_batch_param_nd->block_sizes_[i] = static_cast<int>(block_shapes[i]);
  }

  space_batch_param_nd->m_ = block_shapes.size();

  for (size_t i = 0; i < paddings.size(); ++i) {
    space_batch_param_nd->paddings_[i] = static_cast<int>(paddings[i]);
  }
  return reinterpret_cast<OpParameter *>(space_batch_param_nd);
}
}  // namespace
Registry g_spaceToBatchNDRegistry(schema::PrimitiveType_SpaceToBatchND, PopulateSpaceToBatchNDParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
