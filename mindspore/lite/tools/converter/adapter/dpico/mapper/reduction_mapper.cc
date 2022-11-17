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

#include "mapper/reduction_mapper.h"
#include <memory>
#include <set>
#include <utility>
#include <algorithm>
#include <vector>
#include "include/registry/converter_context.h"
#include "common/op_enum.h"
#include "common/fetch_content.h"
#include "ops/fusion/reduce_fusion.h"
#include "op/reduction_operator.h"

namespace mindspore {
namespace dpico {
namespace {
int GetReductionAxes(const api::CNodePtr &cnode, const api::SharedPtr<ops::ReduceFusion> &reduction_prim,
                     DataInfo *data_info, std::set<int> *axes) {
  if (data_info == nullptr || axes == nullptr) {
    MS_LOG(ERROR) << "input arg is nullptr." << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if ((cnode->inputs().size() > kInputIndex2 &&
       FetchDataFromParameterNode(cnode, kInputIndex2, data_info) == lite::RET_OK)) {
    if (data_info->data_type_ != kNumberTypeInt32) {
      MS_LOG(ERROR) << "data_type not correct";
      return RET_ERROR;
    }
    auto data = reinterpret_cast<int32_t *>(data_info->data_.data());
    int data_size;
    if (GetDataSizeFromTensor(data_info, &data_size) != RET_OK) {
      MS_LOG(ERROR) << "get data size from tensor failed.";
      return RET_ERROR;
    }
    (void)std::transform(data, data + data_size, std::inserter(*axes, (*axes).begin()),
                         [](const int32_t &value) { return static_cast<int32_t>(value); });
  } else if (reduction_prim->GetAttr(ops::kAxes) != nullptr) {
    auto axes_vec = api::GetValue<std::vector<int64_t>>(reduction_prim->GetAttr(ops::kAxes));
    (void)std::transform(axes_vec.begin(), axes_vec.end(), std::inserter(*axes, (*axes).begin()),
                         [](int64_t axis) { return static_cast<int32_t>(axis); });
    *axes = std::set<int32_t>(axes_vec.begin(), axes_vec.end());
  }
  return RET_OK;
}
}  // namespace
STATUS ReductionMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                            const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto reduction_prim = api::utils::cast<api::SharedPtr<ops::ReduceFusion>>(prim);
  MS_ASSERT(reduction_prim != nullptr);

  auto reduction_operator = std::make_unique<mapper::ReductionOperator>();
  if (reduction_operator == nullptr) {
    MS_LOG(ERROR) << "reduction_operator is nullptr.";
    return RET_ERROR;
  }

  if (SetCommonAttr(cnode, reduction_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  reduction_operator->SetOpType(mapper::OpType::REDUCTION);
  if (reduction_prim->GetAttr(ops::kMode) != nullptr) {
    auto reduce_mode = reduction_prim->get_mode();
    switch (reduce_mode) {
      case ReduceMode::Reduce_Sum:
        reduction_operator->SetReductionOp(mapper::ReductionOp::RD_SUM);
        break;
      case ReduceMode::Reduce_Sum_Square:
        reduction_operator->SetReductionOp(mapper::ReductionOp::RD_SUMSQ);
        break;
      case ReduceMode::Reduce_Mean:
        reduction_operator->SetReductionOp(mapper::ReductionOp::RD_MEAN);
        break;
      case ReduceMode::Reduce_ASum:
        reduction_operator->SetReductionOp(mapper::ReductionOp::RD_ASUM);
        break;
      case ReduceMode::Reduce_Max:
        reduction_operator->SetReductionOp(mapper::ReductionOp::RD_MAX);
        break;
      case ReduceMode::Reduce_Min:
        reduction_operator->SetReductionOp(mapper::ReductionOp::RD_MIN);
        break;
      case ReduceMode::Reduce_Prod:
        reduction_operator->SetReductionOp(mapper::ReductionOp::RD_PROD);
        break;
      default:
        MS_LOG(ERROR) << "Unsupported reduction op " << reduce_mode;
        return RET_ERROR;
    }
  }

  std::set<int32_t> axes;
  DataInfo data_info;
  if (GetReductionAxes(cnode, reduction_prim, &data_info, &axes) != RET_OK) {
    MS_LOG(ERROR) << "get reduction axes val failed." << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  reduction_operator->SetReduceAxesSet(axes);
  reduction_operator->SetAxis(*axes.begin());

  if (reduction_prim->GetAttr(ops::kCoeff) != nullptr) {
    reduction_operator->SetReductionCoeff(api::GetValue<float>(reduction_prim->GetAttr(ops::kCoeff)));
  }
  if (reduction_prim->GetAttr(ops::kKeepDims) != nullptr) {
    reduction_operator->SetReduceKeepDims(
      static_cast<int32_t>(api::GetValue<bool>(reduction_prim->GetAttr(ops::kKeepDims))));
  }
  if (PushOfflineArgs(cnode, reduction_operator.get(), 1) != RET_OK) {
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::move(reduction_operator));
  return RET_OK;
}
REG_MAPPER(ReduceFusion, ReductionMapper)
}  // namespace dpico
}  // namespace mindspore
