/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "backend/common/optimizer/dynamic_shape/dynamic_shape_dtype_record.h"

#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt::dynamic_shape {
DynamicShapeDtypeManager &DynamicShapeDtypeManager::GetInstance() {
  static DynamicShapeDtypeManager instance{};
  return instance;
}

void DynamicShapeDtypeManager::Register(const AnfNodePtr &node, const TypePtrList &device_abs) {
  if (device_type_recorder_.find(node) == device_type_recorder_.end()) {
    (void)device_type_recorder_.emplace(node, device_abs);
  }
}

bool DynamicShapeDtypeManager::CheckDeviceType(const AnfNodePtr &node) const {
  return (device_type_recorder_.find(node) != device_type_recorder_.end());
}

TypePtrList DynamicShapeDtypeManager::GetDeviceType(const AnfNodePtr &node) {
  auto iter = device_type_recorder_.find(node);
  if (iter != device_type_recorder_.end()) {
    return iter->second;
  }
  return {};
}

bool DynamicShapeDtypeRecord::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto nodes = TopoSort(func_graph->get_return());
  for (const auto &node : nodes) {
    CNodePtr cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || !AnfUtils::IsRealKernel(cnode)) {
      continue;
    }

    auto kernel_info = node->kernel_info();
    if (kernel_info == nullptr || !kernel_info->has_build_info()) {
      continue;
    }

    auto out_num = common::AnfAlgo::GetOutputTensorNum(node);
    auto node_abs = node->abstract();
    if (node_abs->isa<abstract::AbstractTensor>()) {
      if (out_num != 1) {
        continue;
      }
      auto infer_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
      auto device_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
      if (infer_type != device_type) {
        TypePtrList new_abstract = {TypeIdToType(device_type)};
        DynamicShapeDtypeManager::GetInstance().Register(node, new_abstract);
      }
    } else if (node_abs->isa<abstract::AbstractTuple>()) {
      auto abstract_tuple = node_abs->cast<abstract::AbstractTuplePtr>();
      MS_EXCEPTION_IF_NULL(abstract_tuple);
      TypePtrList abstract_list;
      bool find_diff_element = false;
      for (size_t output_index = 0; output_index < out_num; ++output_index) {
        auto cur_element = abstract_tuple->elements()[output_index];
        MS_EXCEPTION_IF_NULL(cur_element);
        auto infer_type = common::AnfAlgo::GetOutputInferDataType(node, output_index);
        auto device_type = AnfAlgo::GetOutputDeviceDataType(node, output_index);
        if (infer_type != device_type) {
          find_diff_element = true;
        }
        abstract_list.push_back(TypeIdToType(device_type));
      }
      if (find_diff_element) {
        DynamicShapeDtypeManager::GetInstance().Register(node, abstract_list);
      }
    }
  }

  return true;
}
}  // namespace opt::dynamic_shape
}  // namespace mindspore
