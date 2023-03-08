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
#include "nnacl/op_base.h"
#include "nnacl/splice_parameter.h"
#include "ops/splice.h"
using mindspore::ops::kNameSplice;
using mindspore::schema::PrimitiveType_Splice;

namespace mindspore {
namespace lite {
void DestroySpliceOpParameter(OpParameter *parameter) {
  MS_CHECK_PTR_IF_NULL(parameter);
  auto param = reinterpret_cast<SpliceParameter *>(parameter);
  if (param->context_ != nullptr) {
    free(param->context_);
    param->context_ = nullptr;
  }
  if (param->forward_indexes_ != nullptr) {
    free(param->forward_indexes_);
    param->forward_indexes_ = nullptr;
  }
}
OpParameter *PopulateSpliceOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SpliceParameter *>(PopulateOpParameter<SpliceParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SpliceParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::Splice *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }

  auto context = op->get_context();
  std::vector<int> primitive_context(context.begin(), context.end());
  if (primitive_context.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    MS_LOG(ERROR) << "size is too big.";
    free(param);
    return nullptr;
  }
  param->context_dim_ = static_cast<int>(primitive_context.size());

  // malloc && memset for context
  param->context_ = reinterpret_cast<int *>(malloc(primitive_context.size() * sizeof(int)));
  if (param->context_ == nullptr) {
    MS_LOG(ERROR) << "malloc param context_ error";
    free(param);
    return nullptr;
  }
  param->op_parameter_.destroy_func_ = DestroySpliceOpParameter;
  // src_to_dst_row_offset
  int src_to_dst_row_offset = INT32_MIN;
  (void)memset(param->context_, 0, primitive_context.size() * sizeof(int));
  for (size_t i = 0; i < primitive_context.size(); ++i) {
    param->context_[i] = primitive_context[i];
    src_to_dst_row_offset = std::max(src_to_dst_row_offset, std::abs(primitive_context.at(i)));
  }

  auto forward_indexes = op->get_forward_indexes();
  std::vector<int> primitive_forward_indexes(forward_indexes.begin(), forward_indexes.end());
  if (primitive_forward_indexes.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    MS_LOG(ERROR) << "size is too big.";
    DestroySpliceOpParameter(reinterpret_cast<OpParameter *>(param));
    free(param);
    return nullptr;
  }
  param->forward_indexes_dim_ = static_cast<int>(primitive_forward_indexes.size());

  // malloc && memset for forward_indexes
  param->forward_indexes_ = reinterpret_cast<int *>(malloc(primitive_forward_indexes.size() * sizeof(int)));
  if (param->forward_indexes_ == nullptr) {
    MS_LOG(ERROR) << "malloc param forward_indexes_ error";
    DestroySpliceOpParameter(reinterpret_cast<OpParameter *>(param));
    free(param);
    return nullptr;
  }
  (void)memset(param->forward_indexes_, 0, primitive_forward_indexes.size() * sizeof(int));
  (void)memcpy(param->forward_indexes_, primitive_forward_indexes.data(),
               primitive_forward_indexes.size() * sizeof(int));
  param->output_dim_ = op->get_output_dim();
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSplice, PrimitiveType_Splice, PopulateSpliceOpParameter)
}  // namespace lite
}  // namespace mindspore
