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
#include "nnacl/split_parameter.h"
#include "ops/split_with_overlap.h"
using mindspore::ops::kNameSplitWithOverlap;
using mindspore::schema::PrimitiveType_SplitWithOverlap;

namespace mindspore {
namespace lite {
OpParameter *PopulateSplitWithOverlapOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SplitWithOverlapParameter *>(PopulateOpParameter<SplitWithOverlapParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SplitWithOverlapParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::SplitWithOverlap *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not TopKFusion.";
    return nullptr;
  }

  param->num_split_ = static_cast<int>(op->get_number_split());
  param->split_dim_ = static_cast<int>(op->get_split_dim());

  if (param->num_split_ > SPLIT_MAX_SLICE_NUM) {
    MS_LOG(ERROR) << "SplitWithOverlap num_split_ error.";
    free(param);
    return nullptr;
  }

  auto ratio = op->get_ratio();
  auto extend_top = op->get_extend_top();
  auto extend_bottom = op->get_extend_bottom();
  if (static_cast<int>(ratio.size()) != param->num_split_ || static_cast<int>(extend_top.size()) != param->num_split_ ||
      static_cast<int>(extend_bottom.size()) != param->num_split_) {
    MS_LOG(ERROR) << "SplitWithOverlap parameter dat size error.";
    free(param);
    return nullptr;
  }

  for (size_t i = 0; i < ratio.size(); ++i) {
    param->ratio_[i] = static_cast<int>(ratio[i]);
    param->extend_top_[i] = static_cast<int>(extend_top[i]);
    param->extend_bottom_[i] = static_cast<int>(extend_bottom[i]);
  }

  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSplitWithOverlap, PrimitiveType_SplitWithOverlap, PopulateSplitWithOverlapOpParameter)
}  // namespace lite
}  // namespace mindspore
