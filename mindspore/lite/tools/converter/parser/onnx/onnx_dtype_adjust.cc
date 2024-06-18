/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "tools/converter/parser/onnx/onnx_dtype_adjust.h"
#include <string>
#include "include/errorcode.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/op_name.h"
#include "ops/array_ops.h"

namespace mindspore::lite {
namespace {
constexpr size_t kNameCastInputNum = 3;
}  // namespace

STATUS GetPrimFromCnode(const CNodePtr &cnode, PrimitivePtr *prim_ptr) {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(prim_ptr);
  CHECK_NULL_RETURN(cnode->input(0));

  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] is nullptr!";
    return lite::RET_ERROR;
  }
  *prim_ptr = GetValueNode<PrimitivePtr>(value_node);
  if (*prim_ptr == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] cast to primitive failed!";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS AdjustCast(CNodePtr cnode, const FuncGraphPtr &func_graph, bool keep_origin_dtype) {
  PrimitivePtr src_prim = nullptr;
  if (lite::GetPrimFromCnode(cnode, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed!";
    return lite::RET_ERROR;
  }
  if (!keep_origin_dtype) {
    int32_t dtype = kTypeUnknown;
    if (src_prim->HasAttr("to")) {
      dtype = GetValue<int32_t>(src_prim->GetAttr("to"));
    }
    if (dtype == kNumberTypeInt64) {
      if (cnode->size() != kNameCastInputNum) {
        MS_LOG(ERROR) << "Input size of cast must be " << kNameCastInputNum << ", real size: " << cnode->size() << "!";
        return lite::RET_ERROR;
      }
      auto to_input = cnode->input(kNameCastInputNum - 1);
      if (to_input == nullptr) {
        MS_LOG(ERROR) << "to_input is nullptr!";
        return lite::RET_ERROR;
      }
      if (!utils::isa<ParameterPtr>(to_input)) {
        return lite::RET_NO_CHANGE;
      }
      auto new_to_node = opt::BuildIntValueParameterNode(func_graph, (int32_t)kNumberTypeInt32,
                                                         cnode->fullname_with_scope() + "_value", true);
      if (new_to_node == nullptr) {
        MS_LOG(ERROR) << "new_to_node is null!";
        return lite::RET_ERROR;
      }
      cnode->set_input(kNameCastInputNum - 1, new_to_node);
    }
  }
  return lite::RET_OK;
}

bool OnnxDtypeAdjust::Adjust(const FuncGraphPtr &func_graph, const converter::ConverterParameters &flag) {
  MS_CHECK_TRUE_RET(func_graph != nullptr, false);
  auto manager = Manage(func_graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr!";
    return false;
  }
  auto node_list = TopoSort(func_graph->get_return());
  bool keep_origin_dtype = false;
  auto env_var = std::getenv("KEEP_ORIGIN_DTYPE");
  if (env_var != nullptr) {
    std::string env_var_value(env_var);
    keep_origin_dtype = env_var_value == "1";
  }
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      MS_LOG(DEBUG) << "node is not cnode.";
      continue;
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimShape)) {
      PrimitivePtr src_prim = nullptr;
      if (lite::GetPrimFromCnode(cnode, &src_prim) != lite::RET_OK) {
        MS_LOG(ERROR) << "Get primitive from cnode failed!";
        return false;
      }
      if (!src_prim->HasAttr(ops::kOutputDType) && keep_origin_dtype) {
        src_prim->AddAttr(ops::kOutputDType, MakeValue<int64_t>(kNumberTypeInt64));
      }
    } else if (opt::CheckPrimitiveType(node, prim::kPrimCast)) {
      auto ret = AdjustCast(cnode, func_graph, keep_origin_dtype);
      if (ret == lite::RET_ERROR) {
        return false;
      } else if (ret == lite::RET_NO_CHANGE) {
        continue;
      }
    } else if (opt::CheckPrimitiveType(node, prim::kPrimConstantOfShape)) {
      ValueNodePtr value_node = nullptr;
      PrimitivePtr src_prim = nullptr;
      if (GetPrimFromCnode(cnode, &src_prim) != lite::RET_OK) {
        MS_LOG(ERROR) << "Get primitive from cnode failed!";
        return lite::RET_ERROR;
      }
      if (!keep_origin_dtype && src_prim->HasAttr(ops::kDataType)) {
        auto dtype = GetValue<int64_t>(src_prim->GetAttr(ops::kDataType));
        if (dtype == kNumberTypeInt64) {
          src_prim->AddAttr(ops::kDataType, MakeValue<int64_t>(kNumberTypeInt32));
        }
      }
    }
  }
  return true;
}
}  // namespace mindspore::lite
