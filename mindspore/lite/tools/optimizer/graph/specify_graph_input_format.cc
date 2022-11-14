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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/specify_graph_input_format.h"
#include <memory>
#include <vector>
#include <map>
#include <utility>
#include "tools/optimizer/common/format_utils.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/transpose.h"

namespace mindspore {
namespace opt {
bool SpecifyGraphInputFormat::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  if (!update_input_format_) {
    MS_LOG(INFO) << "Export model type is MindIR, skip Pass SpecifyGraphInputFormat";
    return true;
  }
  if (exp_graph_input_format_ == cur_graph_input_format_) {
    return true;
  }
  if ((exp_graph_input_format_ != mindspore::NHWC && exp_graph_input_format_ != mindspore::NCHW) ||
      (cur_graph_input_format_ != mindspore::NHWC && cur_graph_input_format_ != mindspore::NCHW)) {
    MS_LOG(ERROR) << "this pass only support to transfer graph input format between nhwc with nchw.";
    return false;
  }
  auto manager = Manage(graph, true);
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr.";
    return false;
  }
  if (HandleGraphInput(graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "transfer graph input format from nhwc to nchw failed.";
    return false;
  }
  return true;
}

STATUS SpecifyGraphInputFormat::HandleGraphInput(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto graph_inputs = graph->get_inputs();
  for (const auto &input : graph_inputs) {
    auto input_node = input->cast<ParameterPtr>();
    MS_ASSERT(input_node != nullptr);
    auto abstract = input_node->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");

    ShapeVector shape;
    if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed." << input->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if (shape.size() != kInputSizeFour) {
      continue;
    }
    ShapeVector transfer_shape;
    if (exp_graph_input_format_ == mindspore::NCHW) {
      transfer_shape = {shape[0], shape[kInputIndexThree], shape[1], shape[kInputIndexTwo]};
    } else {
      transfer_shape = {shape[0], shape[kInputIndexTwo], shape[kInputIndexThree], shape[1]};
    }
    CNodePtr trans_cnode;
    if (exp_graph_input_format_ == mindspore::NCHW) {
      trans_cnode = opt::GenTransposeNode(graph, input, kNC2NH, input->fullname_with_scope() + "_nc2nh");
    } else {
      trans_cnode = opt::GenTransposeNode(graph, input, kNH2NC, input->fullname_with_scope() + "_nh2nc");
    }
    if (trans_cnode == nullptr) {
      MS_LOG(ERROR) << "create transpose cnode failed.";
      return lite::RET_ERROR;
    }
    auto trans_prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
    MS_CHECK_TRUE_MSG(trans_prim != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
    if (exp_graph_input_format_ == mindspore::NCHW) {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
    } else {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
    }
    trans_cnode->set_abstract(abstract->Clone());
    abstract->set_shape(std::make_shared<abstract::Shape>(transfer_shape));
    (void)manager->Replace(input, trans_cnode);
  }
  return lite::RET_OK;
}

bool SpecifyGraphInputFormat::GetCurGraphInputFormat(const FuncGraphPtr &func_graph, converter::FmkType fmk_type,
                                                     mindspore::Format *input_format) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(input_format != nullptr);

  std::vector<AnfNodePtr> inputs = func_graph->get_inputs();
  std::map<AnfNodePtr, std::vector<AnfNodePtr>> node_users;
  for (auto &input : inputs) {
    auto input_shape = opt::GetAnfNodeOutputShape(input, 0);
    if (input_shape.size() == DIMENSION_4D) {
      node_users[input] = {};
    }
  }

  auto format_ops = GetToNCHWOpMap();
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    MS_CHECK_TRUE_RET(node != nullptr, false);
    auto cnode = node->cast<CNodePtr>();
    if (!cnode) {
      continue;
    }
    bool is_input_user = false;
    for (size_t i = 1; i < cnode->size(); i++) {
      auto input = cnode->input(i);
      MS_CHECK_TRUE_RET(input != nullptr, false);
      auto it = node_users.find(input);
      if (it != node_users.end()) {
        it->second.push_back(cnode);
        is_input_user = true;
      }
    }
    if (!is_input_user) {
      continue;
    }
    (void)node_users.emplace(std::make_pair(cnode, std::vector<AnfNodePtr>{}));
    if (opt::CheckPrimitiveType(cnode, prim::kPrimTranspose)) {
      std::vector<int> perm;
      if (GetTransposePerm(cnode, &perm) != lite::RET_OK) {
        MS_LOG(ERROR) << "fetch transpose perm failed.";
        return false;
      }
      if (perm == kNC2NH) {
        *input_format = NCHW;
        return true;
      } else if (perm == kNH2NC) {
        *input_format = NHWC;
        return true;
      }
    }
    auto prim_node = cnode->input(0);
    auto prim = GetValueNode<PrimitivePtr>(prim_node);
    MS_CHECK_TRUE_RET(prim != nullptr, false);

    if (format_ops.find(prim->name()) == format_ops.end()) {
      continue;
    }
    auto format_attr = prim->GetAttr(ops::kFormat);
    if (format_attr == nullptr) {
      continue;
    }
    auto node_format = GetValue<int64_t>(format_attr);
    if (node_format == mindspore::NCHW) {
      *input_format = NCHW;
      return true;
    } else if (node_format == mindspore::NHWC) {
      *input_format = NHWC;
      return true;
    } else {
      MS_LOG(ERROR) << "Invalid node format " << node_format << ", node " << node->fullname_with_scope();
      return false;
    }
  }
  if (fmk_type == converter::kFmkTypeTf || fmk_type == converter::kFmkTypeTflite) {
    *input_format = NHWC;
  } else {
    *input_format = NCHW;
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
