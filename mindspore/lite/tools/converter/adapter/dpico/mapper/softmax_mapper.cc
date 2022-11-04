/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "mapper/softmax_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "ops/softmax.h"
#include "op/softmax_operator.h"

namespace mindspore {
namespace dpico {
STATUS SoftmaxMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                          const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto softmax_prim = api::utils::cast<api::SharedPtr<ops::Softmax>>(prim);
  MS_ASSERT(softmax_prim != nullptr);

  auto softmax_operator = std::make_unique<mapper::SoftmaxOperator>();
  if (softmax_operator == nullptr) {
    MS_LOG(ERROR) << "softmax_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, softmax_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  softmax_operator->SetOpType(mapper::OpType::SOFTMAX);
  auto axes = softmax_prim->get_axis();
  if (axes.empty()) {
    MS_LOG(ERROR) << "axis attr is empty.";
    return RET_ERROR;
  } else {
    softmax_operator->SetRealAxes(static_cast<mapper::SoftmaxAxes>(axes.at(0)));
  }
  if (PushOfflineArgs(cnode, softmax_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(softmax_operator));
  return RET_OK;
}
REG_MAPPER(Softmax, SoftmaxMapper)
}  // namespace dpico
}  // namespace mindspore
