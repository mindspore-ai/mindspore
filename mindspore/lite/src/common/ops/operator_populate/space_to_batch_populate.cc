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
#include "nnacl/fp32/space_to_batch_fp32.h"
#include "ops/space_to_batch.h"
using mindspore::ops::kNameSpaceToBatch;
using mindspore::schema::PrimitiveType_SpaceToBatch;

namespace mindspore {
namespace lite {
OpParameter *PopulateSpaceToBatchOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SpaceToBatchParameter *>(PopulateOpParameter<SpaceToBatchParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::SpaceToBatch *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }

  auto block_sizes = op->get_block_size();
  if (block_sizes.size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "The value of block_sizes.size() is too big";
    free(param);
    return nullptr;
  }
  param->m_ = static_cast<int>(block_sizes.size());

  auto param_paddings = op->get_paddings();
  if (param_paddings.size() == 0) {
    MS_LOG(ERROR) << "paddings attr is nullptr";
    free(param);
    return nullptr;
  }
  if (param_paddings[0].size() == 0) {
    MS_LOG(ERROR) << "paddings attr is nullptr.";
    free(param);
    return nullptr;
  }
  if (param_paddings.size() > static_cast<size_t>(INT_MAX) || param_paddings[0].size() > static_cast<size_t>(INT_MAX)) {
    MS_LOG(ERROR) << "padding's size is too big.";
    free(param);
    return nullptr;
  }
  if (INT_MUL_OVERFLOW(static_cast<int>(param_paddings.size()), static_cast<int>(param_paddings[0].size()))) {
    MS_LOG(ERROR) << "padding's data length is too big.";
    free(param);
    return nullptr;
  }
  std::vector<int64_t> paddings;
  for (auto &paddings_vec : param_paddings) {
    paddings.insert(paddings.end(), paddings_vec.begin(), paddings_vec.end());
  }
  if (paddings.size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "Invalid paddings size " << paddings.size();
    free(param);
    return nullptr;
  }
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    param->block_sizes_[i] = static_cast<int>(block_sizes[i]);
  }

  for (size_t i = 0; i < paddings.size(); ++i) {
    param->paddings_[i] = static_cast<int>(paddings[i]);
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSpaceToBatch, PrimitiveType_SpaceToBatch, PopulateSpaceToBatchOpParameter)
}  // namespace lite
}  // namespace mindspore
