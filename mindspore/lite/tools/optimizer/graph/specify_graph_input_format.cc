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

#include "tools/optimizer/graph/specify_graph_input_format.h"
#include <memory>
#include <vector>
#include "tools/optimizer/common/format_utils.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace opt {
bool SpecifyGraphInputFormat::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  if (format_ == mindspore::NHWC) {
    return true;
  }
  if (format_ != mindspore::NCHW) {
    MS_LOG(ERROR) << "this pass only support to transfer graph input format from nhwc to nchw.";
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
    if (format_ == mindspore::NCHW) {
      transfer_shape = {shape[0], shape[kInputIndexThree], shape[1], shape[kInputIndexTwo]};
    } else {
      transfer_shape = {shape[0], shape[kInputIndexTwo], shape[kInputIndexThree], shape[1]};
    }
    CNodePtr trans_cnode;
    if (format_ == mindspore::NCHW) {
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
    if (format_ == mindspore::NCHW) {
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
}  // namespace opt
}  // namespace mindspore
