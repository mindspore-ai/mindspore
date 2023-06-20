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

#define USE_DEPRECATED_API
#include "tools/optimizer/graph/specify_graph_output_format.h"
#include <memory>
#include <vector>
#include <map>
#include <stack>
#include "mindspore/core/ops/sequence_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "tools/lite_exporter/fetch_content.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/make_tuple.h"

namespace mindspore {
namespace opt {
bool SpecifyGraphOutputFormat::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  if (exp_graph_output_format_ != mindspore::NCHW && exp_graph_output_format_ != mindspore::NHWC) {
    MS_LOG(ERROR) << "this pass only support to transfer graph output format between nhwc to nchw.";
    return false;
  }
  auto manager = Manage(graph);
  MS_CHECK_TRUE_MSG(manager != nullptr, false, "manager is nullptr.");
  if (HandleGraphOutput(graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Specify graph output format failed.";
    return false;
  }
  return true;
}

STATUS SpecifyGraphOutputFormat::HandleGraphOutput(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto return_node = graph->get_return();
  auto inputs = return_node->inputs();
  std::stack<AnfNodePtr> nodes;
  // push make tuple cnode
  nodes.push(inputs[kInputIndexOne]);
  std::vector<AnfNodePtr> real_outputs;
  while (!nodes.empty()) {
    auto node = nodes.top();
    nodes.pop();
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cast node to cnode failed");
    if (opt::CheckPrimitiveType(cnode, prim::kPrimMakeTuple)) {
      for (size_t i = kInputIndexOne; i < cnode->inputs().size(); i++) {
        nodes.push(cnode->input(i));
      }
      continue;
    }
    if (opt::CheckPrimitiveType(cnode, prim::kPrimDepend)) {
      nodes.push(cnode->input(kInputIndexOne));
      continue;
    }
    real_outputs.emplace_back(cnode);
  }

  for (auto &node : real_outputs) {
    auto cnode = node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(cnode != nullptr, RET_ERROR, "cast node to cnode failed");
    Format format;
    auto ret = DetermineCertainOutputFormat(cnode, kIndex0, &format);
    if (ret != lite::RET_OK) {
      MS_LOG(ERROR) << "Determine " << cnode->fullname_with_scope() << " output format failed";
      return RET_ERROR;
    }

    ShapeVector shape;
    auto abstract = cnode->abstract();
    if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed." << cnode->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if ((format != mindspore::NCHW && format != mindspore::NHWC) || exp_graph_output_format_ == format) {
      MS_LOG(DEBUG) << "skip " << cnode->fullname_with_scope();
      continue;
    }

    if (shape.size() != kInputSizeFour) {
      continue;
    }

    ShapeVector transfer_shape;

    if (exp_graph_output_format_ == mindspore::NCHW) {
      transfer_shape = {shape[0], shape[kInputIndexThree], shape[1], shape[kInputIndexTwo]};
    } else {
      transfer_shape = {shape[0], shape[kInputIndexTwo], shape[kInputIndexThree], shape[1]};
    }
    CNodePtr trans_cnode;
    if (exp_graph_output_format_ == mindspore::NCHW) {
      trans_cnode = opt::GenTransposeNode(graph, cnode, kNH2NC, cnode->fullname_with_scope() + "_nh2nc");
    } else {
      trans_cnode = opt::GenTransposeNode(graph, cnode, kNC2NH, cnode->fullname_with_scope() + "_nc2nh");
    }
    if (trans_cnode == nullptr) {
      MS_LOG(ERROR) << "create transpose cnode failed.";
      return lite::RET_ERROR;
    }
    auto trans_prim = GetValueNode<PrimitivePtr>(trans_cnode->input(kIndex0));
    MS_CHECK_TRUE_MSG(trans_prim != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
    if (exp_graph_output_format_ == mindspore::NCHW) {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
    } else {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
    }
    auto trans_abstract = abstract->Clone();
    MS_CHECK_TRUE_MSG(trans_abstract != nullptr, lite::RET_NULL_PTR, "clone abstract failed");
    trans_abstract->set_shape(std::make_shared<abstract::Shape>(transfer_shape));
    trans_cnode->set_abstract(trans_abstract);
    (void)manager->Replace(cnode, trans_cnode);
  }
  return lite::RET_OK;
}
}  // namespace opt
}  // namespace mindspore
