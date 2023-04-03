/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "parser/caffe/inputs_adjust.h"
#include <string>
#include <memory>
#include "common/op_enum.h"
#include "common/anf_util.h"
#include "common/check_base.h"
#include "ops/transpose.h"
#include "ops/reshape.h"
#include "ops/gather.h"
#include "ops/cast.h"
#include "ops/fusion/topk_fusion.h"
#include "ops/fusion/tile_fusion.h"
#include "ops/fusion/reduce_fusion.h"
#include "ops/fusion/pad_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/resize.h"

namespace mindspore::lite {
namespace {
constexpr int kBuildInputFlagTwo = 2;
constexpr int kBuildInputFlagThree = 3;
constexpr int kBuildInputFlagFour = 4;
}  // namespace
STATUS InputAdjust::AddAttrToInput(const api::FuncGraphPtr &func_graph, const api::CNodePtr &cnode, int input_num,
                                   const std::string &attr_name, int flag) {
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cnode is nullptr");
  if (!dpico::CheckInputs(cnode)) {
    MS_LOG(ERROR) << "input is invalid.";
    return lite::RET_INPUT_TENSOR_ERROR;
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(cnode->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr.";
    return lite::RET_ERROR;
  }
  auto value_ptr = primitive->GetAttr(attr_name);
  if (value_ptr == nullptr) {
    MS_LOG(DEBUG) << "there is no attr :" << attr_name;
    return lite::RET_NO_CHANGE;
  }
  auto inputs = cnode->inputs();
  if (inputs.empty()) {
    MS_LOG(ERROR) << "inputs is empty.";
    return lite::RET_ERROR;
  }
  if (static_cast<int>(inputs.size()) > input_num) {
    primitive->EraseAttr(attr_name);
    MS_LOG(DEBUG) << "input num has been meet, which is " << inputs.size();
    return lite::RET_OK;
  } else if (static_cast<int>(inputs.size()) < input_num) {
    MS_LOG(ERROR) << "input num is invalid.";
    return lite::RET_ERROR;
  }
  api::AnfNodePtr param_node;
  switch (flag) {
    case 1: {
      MS_CHECK_TRUE_MSG(!dpico::CastToInt(value_ptr).empty(), RET_ERROR, "value is empty");
      auto value_data_vec = dpico::CastToInt(value_ptr);
      auto value_data = value_data_vec.front();
      param_node =
        dpico::BuildIntValueParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    case kBuildInputFlagTwo: {
      auto value_data = dpico::CastToInt(value_ptr);
      param_node =
        dpico::BuildIntVecParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    case kBuildInputFlagThree: {
      auto value_data = dpico::CastToVec2DInt(value_ptr);
      param_node =
        dpico::BuildIntVec2DParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    case kBuildInputFlagFour: {
      auto value_data = api::GetValue<float>(value_ptr);
      param_node =
        dpico::BuildFloatValueParameterNode(func_graph, value_data, cnode->fullname_with_scope() + "_" + attr_name);
      break;
    }
    default: {
      MS_LOG(ERROR) << "Error attr flag";
      return lite::RET_ERROR;
    }
  }
  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->AddEdge(cnode, param_node);

  return lite::RET_OK;
}

bool InputAdjust::Run(const api::FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = api::FuncGraphManager::Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  auto node_list = api::FuncGraph::TopoSort(func_graph->get_return());
  STATUS status = lite::RET_OK;
  for (auto &node : node_list) {
    auto cnode = node->cast<api::CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::Transpose>())) {
      MS_LOG(INFO) << "Adjust Transpose";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "perm", kBuildInputFlagTwo);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::Reshape>())) {
      MS_LOG(INFO) << "Adjust Reshape";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "shape", kBuildInputFlagTwo);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::Gather>())) {
      MS_LOG(INFO) << "Adjust Gather";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex3, "axis", 1);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::Cast>())) {
      MS_LOG(INFO) << "Adjust Cast";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "to", 1);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::TopKFusion>())) {
      MS_LOG(INFO) << "Adjust TopKFusion";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "k", 1);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::TileFusion>())) {
      MS_LOG(INFO) << "Adjust TileFusion";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "multiples", kBuildInputFlagTwo);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::ReduceFusion>())) {
      MS_LOG(INFO) << "Adjust ReduceFusion";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "axes", kBuildInputFlagTwo);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::PadFusion>())) {
      MS_LOG(INFO) << "Adjust PadFusion";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "paddings", kBuildInputFlagThree);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::PowFusion>())) {
      MS_LOG(INFO) << "Adjust PowFuison";
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "power", kBuildInputFlagFour);
    } else if (dpico::CheckPrimitiveType(node, api::MakeShared<ops::Resize>())) {
      status = AddAttrToInput(func_graph, cnode, dpico::kInputIndex2, "zoom_factor", 1);
    }
    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "adjust input pass is failed.";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::lite
