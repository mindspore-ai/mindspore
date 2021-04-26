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

#include <unordered_set>
#include <unordered_map>
#include <memory>
#include "tools/optimizer/fisson/fisson_util.h"
#include "mindspore/core/base/core_ops.h"
#include "src/common/utils.h"
#include "tools/common/node_util.h"

namespace mindspore {
using lite::converter::FmkType;

namespace opt {
std::unordered_map<std::string, std::vector<AnfNodePtr>> g_graph_nodes_output = {};
std::unordered_map<std::string, std::vector<std::vector<ShapeVector>>> g_graph_nodes_out_shapes = {};

AnfNodePtr CreateOutputsOfConcat(const FuncGraphPtr &func_graph, const CNodePtr &conv_cnode,
                                 const std::vector<AnfNodePtr> &conv_outputs, const SplitInfo &split_info,
                                 const std::string &node_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(conv_cnode);

  int32_t nodes_num = conv_outputs.size();
  if (nodes_num != split_info.out_num) {
    MS_LOG(ERROR) << "Conv outputs has wrong input size";
    return nullptr;
  }

  // the inputs of concate are from the outputs of conv
  std::vector<AnfNodePtr> concate_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimConcat->name()))};
  for (int32_t i = 0; i < nodes_num; i++) {
    concate_inputs.push_back(conv_outputs[i]);
  }

  auto concate_cnode = func_graph->NewCNode(concate_inputs);
  MS_EXCEPTION_IF_NULL(concate_cnode);

  concate_cnode->set_fullname_with_scope(node_name + "_Concat");
  concate_cnode->set_scope(conv_cnode->scope());

  return concate_cnode;
}

int32_t GetCOutAxis(int32_t format) {
  switch (format) {
    case schema::Format_KHWC:
      return 0;
    case schema::Format_CHWK:
      return 3;
    case schema::Format_NCHW:
      return 0;
    default:
      MS_LOG(ERROR) << "Do not support format: " << format << " now.";
      return -1;
  }
}

int32_t GetCInAxis(int32_t format) {
  switch (format) {
    case schema::Format_KHWC:
      return 3;
    case schema::Format_CHWK:
      return 0;
    default:
      MS_LOG(ERROR) << "Do not support format: " << format << " now.";
      return -1;
  }
}

int32_t GetAxis(int32_t axis, int32_t format, const SplitInfo &split_info) {
  switch (split_info.primitive_type) {
    case mindspore::schema::PrimitiveType_Conv2DFusion:
      if (axis == CuttingStragedy::CUT_C_OUT) {
        return GetCOutAxis(format);
      } else if (axis == CuttingStragedy::CUT_C_IN) {
        return GetCInAxis(format);
      } else {
        MS_LOG(ERROR) << "Only channel_in and channel_out need to transform.";
      }
      break;
    default:
      MS_LOG(ERROR) << "Now, do not support the type : " << split_info.primitive_type;
  }
  return -1;
}

AnfNodePtr CreateOutputsOfAddN(const FuncGraphPtr &func_graph, const CNodePtr &conv_cnode,
                               const std::vector<AnfNodePtr> &conv_outputs, const SplitInfo &split_info,
                               const std::string &node_name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(conv_cnode);

  int32_t nodes_num = conv_outputs.size();
  if (nodes_num != split_info.out_num) {
    MS_LOG(ERROR) << "Conv outputs has wrong input size";
    return nullptr;
  }

  // the inputs of addn are from the outputs of conv
  std::vector<AnfNodePtr> addn_inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimAddN->name()))};
  for (int32_t i = 0; i < nodes_num; i++) {
    addn_inputs.push_back(conv_outputs[i]);
  }

  auto addn_cnode = func_graph->NewCNode(addn_inputs);
  MS_EXCEPTION_IF_NULL(addn_cnode);

  addn_cnode->set_fullname_with_scope(node_name + "_AddN");
  addn_cnode->set_scope(conv_cnode->scope());

  return addn_cnode;
}
}  // namespace opt
}  // namespace mindspore
