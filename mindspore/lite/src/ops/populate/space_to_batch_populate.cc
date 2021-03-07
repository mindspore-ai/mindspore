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
OpParameter *PopulateSpaceToBatchParameter(const void *prim) {
  SpaceToBatchParameter *space_batch_param =
    reinterpret_cast<SpaceToBatchParameter *>(malloc(sizeof(SpaceToBatchParameter)));
  if (space_batch_param == nullptr) {
    MS_LOG(ERROR) << "malloc SpaceToBatchParameter failed.";
    return nullptr;
  }
  memset(space_batch_param, 0, sizeof(SpaceToBatchParameter));
  const schema::Primitive *primitive = static_cast<const schema::Primitive *>(prim);
  space_batch_param->op_parameter_.type_ = primitive->value_type();
  auto param = primitive->value_as_SpaceToBatch();
  auto block_sizes = std::vector<int64_t>(param->block_size()->begin(), param->block_size()->end());
  if (block_sizes.size() > std::numeric_limits<size_t>::max() / sizeof(int)) {
    MS_LOG(ERROR) << "The value of block_sizes.size() is too big";
    free(space_batch_param);
    return nullptr;
  }
  space_batch_param->m_ = block_sizes.size();

  auto fb_paddings = param->paddings()->data();
  if (fb_paddings->size() == 0 ||
      static_cast<uint64_t>(fb_paddings->size() * (*(fb_paddings->begin()))->data()->size()) >
        std::numeric_limits<size_t>::max() / sizeof(int64_t)) {
    MS_LOG(ERROR) << "The value of paddings.size() is zero or too big";
    free(space_batch_param);
    return nullptr;
  }
  std::vector<int64_t> paddings;
  for (auto iter = fb_paddings->begin(); iter != fb_paddings->end(); ++iter) {
    auto paddings_data = (*iter)->data();
    auto paddings_vec = std::vector<int64_t>(paddings_data->begin(), paddings_data->end());
    paddings.insert(paddings.end(), paddings_vec.begin(), paddings_vec.end());
  }

  for (size_t i = 0; i < block_sizes.size(); ++i) {
    space_batch_param->block_sizes_[i] = static_cast<int>(block_sizes[i]);
  }

  for (size_t i = 0; i < paddings.size(); ++i) {
    space_batch_param->paddings_[i] = static_cast<int>(paddings[i]);
  }
  return reinterpret_cast<OpParameter *>(space_batch_param);
}
}  // namespace
Registry g_spaceToBatchRegistry(schema::PrimitiveType_SpaceToBatch, PopulateSpaceToBatchParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
