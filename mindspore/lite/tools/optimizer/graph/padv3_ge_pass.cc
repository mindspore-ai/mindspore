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
#include <memory>
#include <vector>
#include <string>
#include "ops/array_op_name.h"
#include "ops/auto_generate/gen_lite_ops.h"
#include "ops/make_tuple.h"
#include "ops/nn_ops.h"
#include "tools/optimizer/graph/padv3_ge_pass.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "utils/anf_utils.h"

namespace mindspore::opt {

/*
In MindSpore, padding order starts from the last dimension and goes backward (same as PyTorch), but GE padding order
starts from the first dimension and goes forward. So the purpose of this pass is to adapt MindSpore PadV3 op to Ascend
GE PadV3 op. Namely, reverse the padding order.

Main steps:
1. Slice according to padding length.
2. Create a concat vector in reverse order.
3. Set new concat op as the new padding input for PadV3.
*/
const ValueNodePtr PadV3GePass::GenerateDataValueTuple(const FuncGraphPtr &func_graph, int64_t value) {
  std::vector<int64_t> vec({value});
  auto tuple_value = MakeValue(vec);
  auto tuple_node = NewValueNode(tuple_value);
  tuple_node->set_abstract(tuple_value->ToAbstract());
  func_graph->AddValueNode(tuple_node);
  return tuple_node;
}

const ValueNodePtr PadV3GePass::GenerateDataValue(const FuncGraphPtr &func_graph, int64_t value) {
  auto scalar_value = MakeValue(value);
  auto scalar_node = NewValueNode(scalar_value);
  scalar_node->set_abstract(scalar_value->ToAbstract());
  func_graph->AddValueNode(scalar_node);
  return scalar_node;
}

const CNodePtr PadV3GePass::CreateStridedSlice(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node,
                                               int64_t index) {
  // set inputs
  auto begin_node = GenerateDataValueTuple(func_graph, index);
  MS_EXCEPTION_IF_NULL(begin_node);
  auto end_node = GenerateDataValueTuple(func_graph, index + kSizeOne);
  MS_EXCEPTION_IF_NULL(end_node);
  auto strides_node = GenerateDataValueTuple(func_graph, kSizeOne);
  MS_EXCEPTION_IF_NULL(strides_node);

  // set abstract
  ShapeVector tensor_shape = {1};
  auto tensor_shape_ptr = std::make_shared<abstract::Shape>(tensor_shape);
  MS_CHECK_TRUE_MSG(tensor_shape_ptr != nullptr, nullptr, "tensor_shape_ptr is nullptr.");
  TypeId infer_type;
  auto ret = GetDataTypeFromAnfNode(input_node, &infer_type);
  MS_CHECK_TRUE_MSG(ret == RET_OK, nullptr, "get data_type from node failed.");

  auto tmp_abstract = abstract::MakeAbstract(std::make_shared<abstract::Shape>(tensor_shape), TypeIdToType(infer_type));
  MS_CHECK_TRUE_MSG(tmp_abstract != nullptr, nullptr, "make AbstractTensor failed");

  auto begin_mask = GenerateDataValue(func_graph, 0);
  MS_CHECK_TRUE_MSG(begin_mask != nullptr, nullptr, "generate StridedSlice begin_mask node failed.");
  auto end_mask = GenerateDataValue(func_graph, 0);
  MS_CHECK_TRUE_MSG(end_mask != nullptr, nullptr, "generate StridedSlice end_mask node failed.");
  auto ellipsis_mask = GenerateDataValue(func_graph, 0);
  MS_CHECK_TRUE_MSG(ellipsis_mask != nullptr, nullptr, "generate StridedSlice ellipsis_mask node failed.");
  auto new_axis_mask = GenerateDataValue(func_graph, 0);
  MS_CHECK_TRUE_MSG(new_axis_mask != nullptr, nullptr, "generate StridedSlice new_axis_mask node failed.");
  auto shrink_axis_mask = GenerateDataValue(func_graph, 0);
  MS_CHECK_TRUE_MSG(shrink_axis_mask != nullptr, nullptr, "generate StridedSlice shrink_axis_mask node failed.");

  auto prim = NewValueNode(std::make_shared<Primitive>(kStridedSliceOpName));
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  AnfNodePtrList inputs = {prim,       input_node, begin_node,    end_node,      strides_node,
                           begin_mask, end_mask,   ellipsis_mask, new_axis_mask, shrink_axis_mask};
  CNodePtr strided_slice = func_graph->NewCNode(inputs);
  MS_CHECK_TRUE_RET(strided_slice != nullptr, nullptr);
  strided_slice->set_abstract(tmp_abstract);
  strided_slice->set_fullname_with_scope(input_node->fullname_with_scope() + "_strided_slice_" + std::to_string(index));

  // set attrs, all defaults to zero
  auto primitive = GetCNodePrimitive(strided_slice);
  MS_CHECK_TRUE_RET(primitive != nullptr, nullptr);
  return strided_slice;
}

const CNodePtr PadV3GePass::CreateConcatNode(const FuncGraphPtr &func_graph,
                                             const std::vector<AnfNodePtr> &concat_input_vec,
                                             std::string concat_node_name) {
  auto make_tuple_prim = std::make_shared<ops::MakeTuple>();
  MS_CHECK_TRUE_RET(make_tuple_prim != nullptr, nullptr);
  auto make_tuple_prim_c = make_tuple_prim->GetPrim();
  MS_CHECK_TRUE_RET(make_tuple_prim_c != nullptr, nullptr);

  auto make_tuple_cnode = func_graph->NewCNode(make_tuple_prim_c, concat_input_vec);
  AbstractBasePtrList abstract_list;
  for (size_t i = 0; i < concat_input_vec.size(); i++) {
    abstract_list.emplace_back(concat_input_vec.at(i)->abstract());
  }
  make_tuple_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  make_tuple_cnode->set_fullname_with_scope(concat_node_name + "_make_tuple_pre");

  auto concat_prim = std::make_shared<ops::Concat>();
  MS_CHECK_TRUE_RET(concat_prim != nullptr, nullptr);
  concat_prim->set_axis(0);
  const int input_num = concat_input_vec.size();
  (void)concat_prim->AddAttr("N", api::MakeValue(input_num));
  (void)concat_prim->AddAttr("inputNums", api::MakeValue(input_num));
  auto concat_prim_c = concat_prim->GetPrim();
  MS_CHECK_TRUE_RET(concat_prim_c != nullptr, nullptr);

  auto concat_cnode = func_graph->NewCNode(concat_prim_c, {make_tuple_cnode});
  auto concat_abstract =
    abstract::MakeAbstract(std::make_shared<abstract::Shape>(ShapeVector({input_num})), TypeIdToType(kNumberTypeInt32));
  concat_cnode->set_abstract(concat_abstract);
  concat_cnode->set_fullname_with_scope(concat_node_name);

  return concat_cnode;
}

const CNodePtr PadV3GePass::ProcessSliceNConcat(const FuncGraphPtr &func_graph, const AnfNodePtr &pad_node,
                                                const AnfNodePtr &input_node, int64_t fill_length,
                                                std::string concat_node_name) {
  std::vector<AnfNodePtr> concat_input_vec;
  for (int64_t i = 0; i < fill_length; i += 2) {
    // slice and insert to concat in reverse order
    auto slice_node_2 = CreateStridedSlice(func_graph, input_node, i + kSizeOne);
    slice_node_2->set_fullname_with_scope(pad_node->fullname_with_scope() + "_strided_slice_" +
                                          std::to_string(i + kSizeOne));
    concat_input_vec.insert(concat_input_vec.begin(), slice_node_2);

    auto slice_node_1 = CreateStridedSlice(func_graph, input_node, i);
    slice_node_1->set_fullname_with_scope(pad_node->fullname_with_scope() + "_strided_slice_" + std::to_string(i));
    concat_input_vec.insert(concat_input_vec.begin(), slice_node_1);
  }
  auto concat_node = CreateConcatNode(func_graph, concat_input_vec, concat_node_name);
  return concat_node;
}

const int64_t PadV3GePass::GetPaddingLength(const FuncGraphPtr &func_graph, const CNodePtr &pad_node) {
  auto padding_input_abstract = pad_node->input(kIndexOne)->abstract();
  MS_EXCEPTION_IF_NULL(padding_input_abstract);
  auto padding_input_shape_ptr = padding_input_abstract->GetShape();
  MS_EXCEPTION_IF_NULL(padding_input_shape_ptr);
  auto padding_input_shape_vec = padding_input_shape_ptr->GetShapeVector();
  MS_LOG(DEBUG) << "check padding_input_shape_vec: " << padding_input_shape_vec;
  int64_t dst_length = padding_input_shape_vec.size() * 2;
  return dst_length;
}

STATUS PadV3GePass::ProcessPadV3ForGE(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  auto node_list = TopoSort(func_graph->get_return());
  auto status = lite::RET_OK;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (CheckPrimitiveType(node, prim::kPrimPadV3)) {
      MS_LOG(INFO) << "Run flip pass on PadV3 node " << node->fullname_with_scope();
      auto cnode = node->cast<CNodePtr>();
      if (cnode->size() < kSizeThree) {
        MS_LOG(ERROR) << "PadV3 inputs size error";
        return RET_ERROR;
      }
      // Attr "ge_format" marks this PadV3 node has been processed by this pass, thus skipping.
      std::string ge_format_attr = "ge_format";
      if (cnode->HasAttr(ge_format_attr)) {
        auto ge_format_ptr = cnode->GetAttr(ge_format_attr);
        if (ge_format_ptr != nullptr) {
          auto ge_formatted = GetValue<bool>(ge_format_ptr);
          if (ge_formatted) {
            MS_LOG(DEBUG) << "skipping...padv3 pass has been run for node: "
                          << cnode->input(kIndexTwo)->fullname_with_scope();
            continue;
          }
        }
      }

      auto padding = cnode->input(kIndexTwo);
      MS_EXCEPTION_IF_NULL(padding);
      int64_t fill_length = GetPaddingLength(func_graph, cnode);
      auto concat_node = ProcessSliceNConcat(func_graph, cnode, padding, fill_length,
                                             cnode->fullname_with_scope() + "_pad_slice_concat");
      MS_EXCEPTION_IF_NULL(concat_node);

      // Set the final concat node as the PadV3 input
      // Add attr "ge_format" to PadV3 node to distinguish that this specific node has been processed
      bool pass_run = true;
      cnode->AddAttr("ge_format", MakeValue(pass_run));
      manager->SetEdge(cnode, kInputIndexTwo, concat_node);
    }

    if (status != lite::RET_OK && status != lite::RET_NO_CHANGE) {
      MS_LOG(ERROR) << "Failed to run flip PadV3 at cnode: " << node->fullname_with_scope();
      return lite::RET_ERROR;
    }
  }
  return lite::RET_OK;
}

bool PadV3GePass::Run(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, false);
  auto status = ProcessPadV3ForGE(func_graph, manager);
  MS_CHECK_TRUE_RET(status != lite::RET_ERROR, false);
  MS_LOG(INFO) << "run padv3 pass success!";
  return true;
}
}  // namespace mindspore::opt
