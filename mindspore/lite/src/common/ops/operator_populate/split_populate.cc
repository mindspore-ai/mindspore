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
#include "nnacl/op_base.h"
#include "ops/split.h"
using mindspore::ops::kAxis;
using mindspore::ops::kNameSplit;
using mindspore::ops::kOutputNum;
using mindspore::ops::kSizeSplits;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore {
namespace lite {
void DestroySplitSizes(OpParameter *parameter) {
  MS_CHECK_PTR_IF_NULL(parameter);
  auto param = reinterpret_cast<SplitParameter *>(parameter);
  if (param->split_sizes_ != nullptr) {
    free(param->split_sizes_);
    param->split_sizes_ = nullptr;
  }
}

OpParameter *PopulateSplitOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<SplitParameter *>(PopulateOpParameter<SplitParameter>());
  if (param == nullptr) {
    MS_LOG(ERROR) << "new SplitParameter failed.";
    return nullptr;
  }

  mindspore::ValuePtr attr_output = base_operator->GetPrim()->GetAttr(kOutputNum);
  if (attr_output == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kOutputNum << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto output_num = GetValue<int64_t>(attr_output);
  if (output_num > std::numeric_limits<int>::max() / static_cast<int>(sizeof(int)) || output_num <= 0) {
    MS_LOG(ERROR) << "The value of param->num_split_ is not correct";
    free(param);
    return nullptr;
  }
  param->num_split_ = output_num;

  /* free split_sizes_ in split op base */
  param->split_sizes_ = reinterpret_cast<int *>(malloc(static_cast<size_t>(output_num) * sizeof(int)));
  if (param->split_sizes_ == nullptr) {
    MS_LOG(ERROR) << "malloc param split_sizes_ error";
    free(param);
    return nullptr;
  }
  param->op_parameter_.destroy_func_ = DestroySplitSizes;
  memset(param->split_sizes_, 0, static_cast<size_t>(param->num_split_) * sizeof(int));

  mindspore::ValuePtr attr_size_splits = base_operator->GetPrim()->GetAttr(kSizeSplits);
  if (attr_size_splits == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kSizeSplits << ") of operator(" << base_operator->name() << ") not exist";
    DestroySplitSizes(param);
    free(param);
    return nullptr;
  }
  auto split_sizes_vector = GetValue<std::vector<int64_t>>(attr_size_splits);

  if (split_sizes_vector.size() <= static_cast<uint32_t>(param->num_split_)) {
    int i = 0;
    for (auto iter : split_sizes_vector) {
      param->split_sizes_[i++] = iter;
    }
    param->split_count_ = param->num_split_;
  } else {
    param->split_count_ = 0;
  }

  mindspore::ValuePtr attr = base_operator->GetPrim()->GetAttr(kAxis);
  if (attr == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kAxis << ") of operator(" << base_operator->name() << ") not exist";
    DestroySplitSizes(param);
    free(param);
    return nullptr;
  }
  param->split_dim_ = GetValue<int64_t>(attr);
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameSplit, PrimitiveType_Split, PopulateSplitOpParameter)
}  // namespace lite
}  // namespace mindspore
