/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http:/
 * /www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/fp32/space_to_batch_fp32.h"
#include "ops/space_to_batch_nd.h"
using mindspore::ops::kNameSpaceToBatchND;
using mindspore::schema::PrimitiveType_SpaceToBatchND;

namespace mindspore {
namespace lite {
OpParameter *PopulateSpaceToBatchNDOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SpaceToBatchParameter *>(PopulateOpParameter<SpaceToBatchParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SpaceToBatchParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::SpaceToBatchND *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }

  auto block_shapes = op->get_block_shape();
  if (block_shapes.size() > COMM_SHAPE_SIZE) {
    MS_LOG(ERROR) << "The value of block_shapes.size() is too big";
    free(param);
    return nullptr;
  }

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
  for (size_t i = 0; i < block_shapes.size(); ++i) {
    param->block_sizes_[i] = static_cast<int>(block_shapes[i]);
  }
  param->m_ = block_shapes.size();

  for (size_t i = 0; i < paddings.size(); ++i) {
    param->paddings_[i] = static_cast<int>(paddings[i]);
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSpaceToBatchND, PrimitiveType_SpaceToBatchND, PopulateSpaceToBatchNDOpParameter)
}  // namespace lite
}  // namespace mindspore
