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
#include "backend/common/graph_kernel/reshape_reduce_for_cse.h"

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

#include "include/backend/optimizer/helper.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "backend/common/graph_kernel/graph_kernel_helper.h"
#include "ir/primitive.h"

namespace mindspore::graphkernel {
namespace {
/*
 *   before: keep_dims=False, axis=(1), out_shape=(a,b,c)
 * ------------------------------
 *   after: keep_dims=True, axis=(1) ,out_shape=(a,1,b,c)
 */
void ResetReduceAttrAndShape(const AnfNodePtr &node, const std::vector<TypeId> &target_output_types,
                             const std::vector<ShapeVector> &target_output_shapes) {
  common::AnfAlgo::SetNodeAttr(kAttrKeepDims, MakeValue(true), node);
  common::AnfAlgo::SetOutputInferTypeAndShape(target_output_types, target_output_shapes, node.get());
}

size_t ProcessTupleGetItem(const AnfNodePtr &node, const std::vector<TypeId> &target_output_types,
                           const std::vector<ShapeVector> &target_output_shapes) {
  size_t index = common::AnfAlgo::GetTupleGetItemOutIndex(node->cast<CNodePtr>());
  common::AnfAlgo::SetOutputInferTypeAndShape({target_output_types[index]}, {target_output_shapes[index]}, node.get());
  return index;
}

void InsertReshape(const FuncGraphPtr &graph, const AnfNodePtr &node, const TypeId &infer_type,
                   const ShapeVector &infer_shape, const TypeId &device_type) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimReshape->name())), node};
  auto reshape = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(reshape);
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(infer_shape), reshape);
  common::AnfAlgo::SetOutputInferTypeAndShape({infer_type}, {infer_shape}, reshape.get());
  reshape->set_kernel_info(std::make_shared<device::KernelInfo>());
  auto graph_sel_info =
    BuildSelectKernelBuildInfo({kOpFormat_DEFAULT}, {device_type}, {kOpFormat_DEFAULT}, {device_type});
  AnfAlgo::SetSelectKernelBuildInfo(graph_sel_info, reshape.get());
  (void)manager->Replace(node, reshape);
}

void InsertReshapeForMultiOutputs(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                  const std::vector<ShapeVector> &origin_output_shapes,
                                  const std::vector<ShapeVector> &target_output_shapes,
                                  const std::vector<TypeId> &target_output_types, const AnfNodePtr &target) {
  auto used_node_list = opt::GetRealNodeUsedList(graph, node);
  MS_EXCEPTION_IF_NULL(used_node_list);
  for (auto &output_info : (*used_node_list)) {
    auto used_node = output_info.first;
    if (IsPrimitiveCNode(used_node, prim::kPrimTupleGetItem)) {
      size_t index = ProcessTupleGetItem(used_node, target_output_types, target_output_shapes);
      InsertReshape(graph, used_node, target_output_types[index], origin_output_shapes[index],
                    AnfAlgo::GetAllOutputDeviceTypes(target)[index]);
    }
  }
}

bool IsSameAxis(const ValuePtr &main, const ValuePtr &other) {
  if (main == nullptr || other == nullptr) {
    return false;
  }
  if (main->isa<Int64Imm>()) {
    return GetValue<int64_t>(main) == GetValue<int64_t>(other);
  } else if (main->isa<ValueSequence>()) {
    return GetValue<std::vector<int64_t>>(main) == GetValue<std::vector<int64_t>>(other);
  }
  return false;
}

// Check if two reduce ops could be CSE after reshape
AnfNodePtr CanCSE(const FuncGraphPtr &graph, const CNodePtr &cnode) {
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  if (auto at = primitive->GetAttr(kAttrKeepDims); at != nullptr) {
    bool keep_dims = GetValue<bool>(at);
    if (keep_dims) {
      return nullptr;
    }
    auto axis = primitive->GetAttr(kAttrAxis);
    auto input = cnode->input(1);
    auto used_node_list = opt::GetRealNodeUsedList(graph, input);
    MS_EXCEPTION_IF_NULL(used_node_list);
    auto func = [&primitive, &axis](const std::pair<AnfNodePtr, int> &output) {
      if (IsPrimitiveCNode(output.first, primitive)) {
        auto target_primitive = common::AnfAlgo::GetCNodePrimitive(output.first->cast<CNodePtr>());
        if (target_primitive->HasAttr(kAttrKeepDims) && GetValue<bool>(target_primitive->GetAttr(kAttrKeepDims))) {
          return IsSameAxis(axis, target_primitive->GetAttr(kAttrAxis));
        }
      }
      return false;
    };
    auto iter = std::find_if(used_node_list->begin(), used_node_list->end(), func);
    return iter == used_node_list->end() ? nullptr : iter->first;
  }
  return nullptr;
}
}  // namespace

/*
 *   B = Reduce(A, keep_dims=True,axis=a)
 *   C = Reduce(A, keep_dims=False,axis=a)
 *   ------>
 *   B_ = Reduce(A, keep_dims=True,axis=a)
 *   C_ = Reduce(A, keep_dims=True,axis=a)
 *   D_ = Reshape(C_, C.shape)
 * ------------------------------
 *  B_,C_ will be optimized during further CSE
 */

bool ReshapeReduceForCSE::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  const PrimitiveSet prim_set{
    prim::kPrimReduceSum,  prim::kPrimReduceMax,       prim::kPrimReduceMin,
    prim::kPrimReduceMean, prim::kPrimArgMaxWithValue, prim::kPrimArgMinWithValue,
  };
  bool changed = false;
  for (auto node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsOneOfPrimitiveCNode(node, prim_set)) {
      if (auto target = CanCSE(graph, node->cast<CNodePtr>()); target != nullptr) {
        size_t output_num = AnfAlgo::GetOutputTensorNum(node);
        size_t target_output_num = AnfAlgo::GetOutputTensorNum(target);
        if (output_num != target_output_num) {
          MS_LOG(EXCEPTION) << "Node " << node->fullname_with_scope() << " and node " << target->fullname_with_scope()
                            << " can not CSE, because their output num is different: " << output_num << " vs "
                            << target_output_num << ".";
        }

        std::vector<ShapeVector> origin_output_shapes;
        for (size_t i = 0; i < output_num; i++) {
          (void)origin_output_shapes.emplace_back(common::AnfAlgo::GetOutputInferShape(node, i));
        }

        std::vector<ShapeVector> target_output_shapes;
        std::vector<TypeId> target_output_types;
        for (size_t i = 0; i < target_output_num; i++) {
          (void)target_output_shapes.emplace_back(common::AnfAlgo::GetOutputInferShape(target, i));
          (void)target_output_types.emplace_back(common::AnfAlgo::GetOutputInferDataType(target, i));
        }

        ResetReduceAttrAndShape(node, target_output_types, target_output_shapes);
        if (output_num == 1) {
          InsertReshape(graph, node, target_output_types[0], origin_output_shapes[0],
                        AnfAlgo::GetAllOutputDeviceTypes(target)[0]);
        } else {
          InsertReshapeForMultiOutputs(graph, node, origin_output_shapes, target_output_shapes, target_output_types,
                                       target);
        }
        changed = true;
      }
    }
  }
  return changed;
}
}  // namespace mindspore::graphkernel
