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

#include "mapper/bias_mapper.h"
#include <memory>
#include <utility>
#include <algorithm>
#include <vector>
#include "ops/bias_add.h"
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "common/op_enum.h"
#include "op/bias_operator.h"

namespace mindspore {
namespace dpico {
namespace {
STATUS SetBiasDataInfo(const api::CNodePtr &cnode, mapper::BiasOperator *bias_operator) {
  if (bias_operator == nullptr) {
    MS_LOG(ERROR) << "bias_operator is nullptr.";
    return RET_ERROR;
  }
  if (cnode->inputs().size() == kInputIndex3) {
    auto input_anode = cnode->input(kInputIndex2);
    MS_ASSERT(input_anode != nullptr);
    auto param_node = input_anode->cast<api::ParameterPtr>();
    if (param_node == nullptr || !param_node->has_default()) {
      MS_LOG(DEBUG) << "only parameter node needs to set BiasPtr";
      return RET_OK;
    }
    auto tensor_info = param_node->default_param()->cast<api::TensorPtr>();
    if (tensor_info != nullptr && tensor_info->DataSize() != 0) {
      auto data = reinterpret_cast<float *>(tensor_info->data());
      MS_CHECK_TRUE_MSG(data != nullptr, RET_ERROR, "data is nullptr.");
      bias_operator->SetBiasPtr(data);
      ShapeVector shape_vector;
      if (GetShapeVectorFromParameter(param_node, &shape_vector) != RET_OK) {
        MS_LOG(ERROR) << "get shape vector from parameter failed. " << param_node->fullname_with_scope();
        return RET_ERROR;
      }
      std::vector<int32_t> shape;
      (void)std::transform(shape_vector.begin(), shape_vector.end(), std::back_inserter(shape),
                           [](int64_t dim) { return static_cast<int32_t>(dim); });
      bias_operator->SetBiasShapeVec(shape);
    } else {
      MS_LOG(ERROR) << "param node's tensor info is invalid. " << param_node->fullname_with_scope();
      return RET_ERROR;
    }
  }

  return RET_OK;
}
}  // namespace
STATUS BiasMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                       const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }

  auto bias_operator = std::make_unique<mapper::BiasOperator>();
  if (bias_operator == nullptr) {
    MS_LOG(ERROR) << "bias_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, bias_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  if (prim->GetAttr(ops::kAxis) != nullptr) {
    bias_operator->SetAxis(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(ops::kAxis))));
  } else if (CheckPrimitiveType(cnode, api::MakeShared<ops::BiasAdd>())) {
    auto format = api::GetValue<int64_t>(prim->GetAttr(ops::kFormat));
    if (format == static_cast<int64_t>(mindspore::NCHW)) {
      bias_operator->SetAxis(1);
    } else if (format == static_cast<int64_t>(mindspore::NHWC)) {
      bias_operator->SetAxis(-1);
    } else {
      MS_LOG(ERROR) << "invalid format: " << format;
      return RET_ERROR;
    }
  }
  if (prim->GetAttr(dpico::kNumAxes) != nullptr) {
    bias_operator->SetBiasNumAxes(static_cast<int32_t>(api::GetValue<int64_t>(prim->GetAttr(dpico::kNumAxes))));
  }

  if (SetBiasDataInfo(cnode, bias_operator.get()) != RET_OK) {
    MS_LOG(ERROR) << "set bias data info failed.";
    return RET_ERROR;
  }

  bias_operator->SetOpType(mapper::OpType::BIAS);
  base_operators->push_back(std::move(bias_operator));
  return RET_OK;
}
REG_MAPPER(Bias, BiasMapper)
}  // namespace dpico
}  // namespace mindspore
