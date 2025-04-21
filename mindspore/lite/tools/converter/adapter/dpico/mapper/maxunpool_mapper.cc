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

#include "mapper/maxunpool_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/max_unpool_operator.h"
#include "parser/onnx/onnx_maxunpool_parser.h"

namespace mindspore {
namespace dpico {
STATUS MaxUnpoolMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                            const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto custom_prim = api::utils::cast<api::SharedPtr<ops::Custom>>(prim);
  MS_CHECK_TRUE_MSG(custom_prim != nullptr, RET_ERROR, "custom_prim is nullptr");
  auto maxunpool_operator = std::make_unique<mapper::MaxUnpoolOperator>();
  if (maxunpool_operator == nullptr) {
    MS_LOG(ERROR) << "maxunpool_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, maxunpool_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  vector<int32_t> kernels;
  vector<int32_t> strides;
  vector<int32_t> pads;
  maxunpool_operator->SetOpType(mapper::OpType::MAX_UNPOOL);
  if (prim->GetAttr(dpico::kKernelShape) != nullptr) {
    auto kernel = static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kKernelShape)));
    kernels = {kernel, kernel};
    maxunpool_operator->SetKernelShapeVec(kernels);
  }
  if (prim->GetAttr("strides") != nullptr) {
    auto stride = static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr("strides")));
    strides = {stride, stride};
    maxunpool_operator->SetStridesVec(strides);
  }
  if (prim->GetAttr("pads") != nullptr) {
    auto pad = static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr("pads")));
    pads = {pad, pad, pad, pad};
    maxunpool_operator->SetPadsVec(pads);
  }

  base_operators->push_back(std::move(maxunpool_operator));
  return RET_OK;
}
REG_MAPPER(MaxUnpool, MaxUnpoolMapper)
}  // namespace dpico
}  // namespace mindspore
