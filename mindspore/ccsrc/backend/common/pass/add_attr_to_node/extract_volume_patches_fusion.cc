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
#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "utils/ms_context.h"
#include "ops/array_ops.h"

namespace mindspore {
namespace opt {

AnfNodePtr InsertTransposeNode(const FuncGraphPtr &graph, const CNodePtr &cnode) {
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto input_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(cnode, kIndex0);
  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(cnode, kIndex0);

  // transpose input from NCDHW to NDHWC
  static std::vector<int64_t> perm_list{0, 2, 3, 4, 1};
  auto perm_node = CreateTensorInput(kernel_graph, NewValueNode(perm_list));
  auto transpose_node = graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimTranspose->name())), cnode->input(kIndex1), perm_node});
  MS_EXCEPTION_IF_NULL(transpose_node);
  std::vector<int64_t> output_shape;
  (void)std::transform(perm_list.begin(), perm_list.end(), std::back_inserter(output_shape),
                       [&input_shape](const int64_t &dim) { return input_shape[dim]; });

  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {output_shape}, transpose_node.get());

  // Create new ExtractVolumePatches
  auto extract_volume_patches_node = graph->NewCNode(
    {NewValueNode(std::make_shared<Primitive>(prim::kPrimExtractVolumePatches->name())), transpose_node});
  MS_EXCEPTION_IF_NULL(extract_volume_patches_node);
  auto ori_output_shape = common::AnfAlgo::GetOutputInferShape(cnode, kIndex0);
  std::vector<int64_t> new_output_shape;
  (void)std::transform(perm_list.begin(), perm_list.end(), std::back_inserter(new_output_shape),
                       [&ori_output_shape](const int64_t &dim) { return ori_output_shape[dim]; });
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {new_output_shape}, extract_volume_patches_node.get());
  common::AnfAlgo::CopyNodeAttrs(cnode, extract_volume_patches_node);
  common::AnfAlgo::SetNodeAttr("format", MakeValue(kOpFormat_NDHWC), extract_volume_patches_node);

  // transpose output from NDHWC to NCDHW
  static std::vector<int64_t> recover_perm_list{0, 4, 1, 2, 3};
  auto recover_perm_node = CreateTensorInput(kernel_graph, NewValueNode(recover_perm_list));
  auto recover_transpose_node =
    graph->NewCNode({NewValueNode(std::make_shared<Primitive>(prim::kPrimTranspose->name())),
                     extract_volume_patches_node, recover_perm_node});
  MS_EXCEPTION_IF_NULL(recover_transpose_node);
  common::AnfAlgo::SetOutputInferTypeAndShape({input_type}, {ori_output_shape}, recover_transpose_node.get());

  return recover_transpose_node;
}

const AnfNodePtr ExtractVolumePatchesFormatTranspose(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);

  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice ||
      common::AnfAlgo::HasNodeAttr("format", cnode)) {
    return cnode;
  }

  MS_LOG(DEBUG) << "ExtractVolumePatches transpose from format NCDHW to NDHWC.";

  std::vector<std::string> attr_list{"kernel_size", "strides"};
  for (auto &attr_name : attr_list) {
    if (common::AnfAlgo::HasNodeAttr(attr_name, cnode)) {
      constexpr size_t kRank = 5;
      auto attr_value = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(cnode, attr_name);
      if (attr_value.size() == kRank) {
        // transpose kernel_size and strides from NCDHW to NDHWC
        constexpr size_t c_idx = 1;
        auto value_c = attr_value[c_idx];
        (void)attr_value.erase(attr_value.begin() + c_idx);
        attr_value.emplace_back(value_c);
      }
      common::AnfAlgo::SetNodeAttr(attr_name, MakeValue(attr_value), cnode);
    }
  }

  return InsertTransposeNode(graph, cnode);
}
}  // namespace opt
}  // namespace mindspore
