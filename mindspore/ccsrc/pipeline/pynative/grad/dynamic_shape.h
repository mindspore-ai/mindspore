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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_DYNAMIC_SHAPE_H_
#define MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_DYNAMIC_SHAPE_H_

#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "pipeline/pynative/grad/top_cell.h"

namespace mindspore {
namespace pynative {
struct NodeInfo {
  NodeInfo() = default;
  explicit NodeInfo(const TensorGradType &grad_type, size_t op_index = 0, ValuePtr value = nullptr)
      : grad_type(grad_type), op_index(op_index), value(std::move(value)) {}
  TensorGradType grad_type;
  size_t op_index{};
  ValuePtr value;
};

struct DynamicDetectNodeInfo {
  DynamicDetectNodeInfo(PrimitivePtr op_prim, abstract::AbstractBasePtrList input_abs,
                        abstract::AbstractBasePtr out_abs)
      : op_prim(std::move(op_prim)), input_abs(std::move(input_abs)), out_abs(std::move(out_abs)) {}
  PrimitivePtr op_prim{nullptr};
  abstract::AbstractBasePtrList input_abs{};
  abstract::AbstractBasePtr out_abs{nullptr};
  bool is_graph_node{false};
  std::vector<std::pair<std::string, NodeInfo>> inputs;
  std::string graph_phase;
};
using DynamicDetectNodeInfoPtr = std::shared_ptr<DynamicDetectNodeInfo>;
using CellIdWithDynamicNodesMap =
  mindspore::HashMap<std::string, mindspore::HashMap<std::string, std::vector<DynamicDetectNodeInfoPtr>>>;

class NodeDynamicDetect {
 public:
  NodeDynamicDetect() = default;
  ~NodeDynamicDetect() = default;
  bool CheckNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                        const DynamicDetectNodeInfoPtr &node);
  bool IsNeedSaveDynamicDetectNodes(const TopCellInfoPtr &top_cell, bool use_dynamic_shape_process);

 private:
  bool IsNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs, const DynamicDetectNodeInfoPtr &node,
                     size_t node_idx);
  void SaveDynamicDetectNodeInfoInFirstTime(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                                            const DynamicDetectNodeInfoPtr &node, size_t node_idx);

  std::mutex async_mutex_;
  CellIdWithDynamicNodesMap cell_id_with_dynamic_detect_nodes_;
};
using NodeDynamicDetectPtr = std::shared_ptr<NodeDynamicDetect>;

class DynamicShape {
 public:
  DynamicShape() : dynamic_structure_ptr_(std::make_shared<NodeDynamicDetect>()) {}
  ~DynamicShape() = default;

  bool CheckNodeDynamic(const TopCellInfoPtr &top_cell, const ValuePtrList &inputs,
                        const DynamicDetectNodeInfoPtr &node) {
    return dynamic_structure_ptr_->CheckNodeDynamic(top_cell, inputs, node);
  }
  bool IsNeedSaveDynamicDetectNodes(const TopCellInfoPtr &top_cell, bool use_dynamic_shape_process) {
    return dynamic_structure_ptr_->IsNeedSaveDynamicDetectNodes(top_cell, use_dynamic_shape_process);
  }
  bool IsGraphDynamic() { return false; }

 private:
  NodeDynamicDetectPtr dynamic_structure_ptr_{nullptr};
};
using DynamicShapePtr = std::shared_ptr<DynamicShape>;
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_DYNAMIC_SHAPE_H_
