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

#include "mapper/detection_output_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include "parser/detection_output_param_holder.h"
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/detection_output_operator.h"

namespace mindspore {
namespace dpico {
STATUS DetectionOutputMapper::Map(const CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                                  const PrimitivePtr &prim, const CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto detection_output_operator = std::make_unique<mapper::DetectionOutputOperator>();
  if (detection_output_operator == nullptr) {
    MS_LOG(ERROR) << "detection_output_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, detection_output_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  detection_output_operator->SetOpType(mapper::OpType::DETECTION_OUTPUT);
  if (prim->GetAttr(kNumAnchors) != nullptr) {
    detection_output_operator->SetNumAnchors(GetValue<uint32_t>(prim->GetAttr(kNumAnchors)));
  }
  if (prim->GetAttr(kNumBboxesPerGrid) != nullptr) {
    detection_output_operator->SetNumBboxesPerGrid(GetValue<uint32_t>(prim->GetAttr(kNumBboxesPerGrid)));
  }
  if (prim->GetAttr(kNumCoords) != nullptr) {
    detection_output_operator->SetNumCoords(GetValue<uint32_t>(prim->GetAttr(kNumCoords)));
  }
  if (prim->GetAttr(kNumClasses) != nullptr) {
    detection_output_operator->SetNumClasses(GetValue<uint32_t>(prim->GetAttr(kNumClasses)));
  }
  if (prim->GetAttr(kNumGridsHeight) != nullptr) {
    detection_output_operator->SetNumGridsHeight(GetValue<uint32_t>(prim->GetAttr(kNumGridsHeight)));
  }
  if (prim->GetAttr(kNumGridsWidth) != nullptr) {
    detection_output_operator->SetNumGridsWidth(GetValue<uint32_t>(prim->GetAttr(kNumGridsWidth)));
  }
  if (prim->GetAttr(kDetectionOutputParam) != nullptr) {
    auto param_ptr_list =
      GetValue<std::vector<lite::DetectionOutputParamHolderPtr>>(prim->GetAttr(kDetectionOutputParam));
    std::vector<mapper::DetectionOutputParam> param_vec{};
    (void)std::transform(
      param_ptr_list.begin(), param_ptr_list.end(), std::back_inserter(param_vec),
      [](const lite::DetectionOutputParamHolderPtr &param_ptr) { return param_ptr->GetDetectionOutputParam(); });
    detection_output_operator->SetDetectionOutputParamVec(param_vec);
  }
  base_operators->push_back(std::move(detection_output_operator));
  return RET_OK;
}
REG_MAPPER(DetectionOutput, DetectionOutputMapper)
}  // namespace dpico
}  // namespace mindspore
