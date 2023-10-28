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

#include "plugin/device/ascend/optimizer/ge/print_to_stringformat_print.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include "ops/framework_ops.h"
#include "ops/sequence_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/dtype.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace opt {
namespace {
const char kSeparator[] = ", ";
const char kShapePrefix[] = "[";
const char kShapeSuffix[] = "]";
const char kEmptyShape[] = "[]";

std::string GetTensorShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::string shape_str = "shape=";
  auto abstract_ptr = node->abstract();
  MS_EXCEPTION_IF_NULL(abstract_ptr);
  auto shape_ptr = abstract_ptr->GetShapeTrack();
  if (shape_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "The shape of node " << node->fullname_with_scope() << " is nullptr";
  }
  auto shape_vec = shape_ptr->cast<abstract::ShapePtr>()->shape();
  if (shape_vec.empty()) {
    shape_str += kEmptyShape;
  } else {
    shape_str += kShapePrefix;
    for (auto &shape : shape_vec) {
      shape_str += std::to_string(shape);
      shape_str += " ";
    }
    shape_str.erase(shape_str.find_last_not_of(" ") + 1);
    shape_str += kShapeSuffix;
  }
  return shape_str;
}

std::string GetTensorDtype(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  std::string type_str = "dtype=";
  auto type = node->Type();
  MS_EXCEPTION_IF_NULL(type);
  MS_EXCEPTION_IF_NULL(dyn_cast<TensorType>(type));
  auto type_ptr = dyn_cast<TensorType>(type)->element();
  MS_EXCEPTION_IF_NULL(type_ptr);
  auto type_id = type_ptr->type_id();
  type_str += TypeIdToString(type_id);
  return type_str;
}

CNodePtr CreateNewMakeTuple(const FuncGraphPtr &graph, const std::vector<AnfNodePtr> &make_tuple_inputs) {
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> new_make_tuple_inputs{NewValueNode(std::make_shared<Primitive>(kMakeTupleOpName))};
  std::vector<AbstractBasePtr> abstract_list;
  for (auto &make_tuple_input_node : make_tuple_inputs) {
    MS_EXCEPTION_IF_NULL(make_tuple_input_node);
    new_make_tuple_inputs.emplace_back(make_tuple_input_node);
    abstract_list.emplace_back(make_tuple_input_node->abstract());
  }

  auto new_make_tuple_node = graph->NewCNode(new_make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(new_make_tuple_node);
  new_make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return new_make_tuple_node;
}

CNodePtr CreateNewPrint(const FuncGraphPtr &graph, const CNodePtr &string_format_node, const CNodePtr &print_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(string_format_node);
  MS_EXCEPTION_IF_NULL(print_node);

  std::vector<AnfNodePtr> new_print_inputs{NewValueNode(std::make_shared<Primitive>(kPrintOpName))};
  (void)new_print_inputs.emplace_back(string_format_node);
  // Add IOMonad.
  const CNodePtr &make_tuple_node = print_node->input(1)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_node);
  const std::vector<AnfNodePtr> &inputs = make_tuple_node->inputs();
  new_print_inputs.emplace_back(inputs.at(inputs.size() - 1));

  auto new_print_node = graph->NewCNode(new_print_inputs);
  MS_EXCEPTION_IF_NULL(new_print_node);
  new_print_node->set_abstract(string_format_node->abstract());

  auto primitive = GetCNodePrimitive(new_print_node);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive = primitive->Clone();
  MS_EXCEPTION_IF_NULL(primitive);
  (void)primitive->AddAttr("output_stream", MakeValue("stdout"));
  new_print_node->set_input(0, std::make_shared<ValueNode>(primitive));
  return new_print_node;
}

CNodePtr CreateShape(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  std::vector<AnfNodePtr> shape_inputs{NewValueNode(std::make_shared<Primitive>("TensorShape"))};
  (void)shape_inputs.emplace_back(node);
  auto shape_node = graph->NewCNode(shape_inputs);
  MS_EXCEPTION_IF_NULL(shape_node);
  abstract::AbstractBasePtr abs;
  auto node_abstract = node->abstract();
  MS_EXCEPTION_IF_NULL(node_abstract);
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(node_abstract->GetShapeTrack());
  auto shape = shape_map[kShape];
  ShapeVector tensor_shp({static_cast<int64_t>(shape.size())});
  if (IsDynamic(shape)) {
    if (IsDynamicRank(shape)) {
      abs = abstract::MakeAbstract(
        std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeDimAny}), kInt64);
    } else {
      auto elem = std::make_shared<abstract::AbstractScalar>(std::make_shared<ValueAny>(), std::make_shared<Int>(64));
      auto abs_tensor = std::make_shared<abstract::AbstractTensor>(elem, std::make_shared<abstract::Shape>(tensor_shp));
      auto shape_value = MakeValue(shape);
      abs_tensor->set_shape_value(shape_value);
      abs = abs_tensor;
    }
  } else {
    auto shp_buf_size = sizeof(int64_t) * shape.size();
    auto tensor = std::make_shared<tensor::Tensor>(kNumberTypeInt64, tensor_shp, shape.data(), shp_buf_size);
    abs = tensor->ToAbstract();
  }
  shape_node->set_abstract(abs);
  return shape_node;
}

CNodePtr CreateStringFormat(const FuncGraphPtr &graph, const CNodePtr &print_node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(print_node);
  const CNodePtr &make_tuple_node = print_node->input(1)->cast<CNodePtr>();
  const std::vector<AnfNodePtr> &inputs = make_tuple_node->inputs();
  constexpr auto placeholder = "{}";
  std::string str_template = "";
  constexpr auto summarize = -1;
  CNodePtr new_make_tuple_node;
  std::vector<AnfNodePtr> make_tuple_inputs;
  // Set node template attribute which StringFormat need.
  for (size_t input_index = 1; input_index < inputs.size() - 1; ++input_index) {
    auto input_node = inputs.at(input_index);
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsValueNode<StringImm>(input_node)) {
      auto valut_ptr = GetValueNode(input_node);
      str_template = str_template + GetValue<std::string>(valut_ptr) + "\n";
    } else {
      if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
        new_make_tuple_node = input_node->cast<CNodePtr>();
        break;
      }
      std::string str_dtype;
      auto abstract = input_node->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      auto shape_ptr = abstract->GetShapeTrack()->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(shape_ptr);
      auto shape = shape_ptr->shape();
      // For dynamic shape input tensor, insert TensorShape ops to get real shape.
      if (IsDynamic(shape)) {
        auto shape_node = CreateShape(graph, input_node);
        make_tuple_inputs.emplace_back(shape_node);
        str_template = str_template + "Tensor(shape=" + placeholder;
      } else {
        auto str_shape = GetTensorShape(input_node);
        str_template = str_template + "Tensor(" + str_shape;
      }
      str_dtype = GetTensorDtype(input_node);
      str_template = str_template + kSeparator + str_dtype + kSeparator + "value=\n" + placeholder + ")\n";
      make_tuple_inputs.emplace_back(input_node);
    }
  }
  if (!str_template.empty()) {
    str_template.pop_back();
  }
  if (new_make_tuple_node == nullptr) {
    new_make_tuple_node = CreateNewMakeTuple(graph, make_tuple_inputs);
  }
  std::vector<AnfNodePtr> string_format_inputs{NewValueNode(std::make_shared<Primitive>("StringFormat"))};
  string_format_inputs.emplace_back(new_make_tuple_node);
  auto string_format_node = graph->NewCNode(string_format_inputs);
  MS_EXCEPTION_IF_NULL(string_format_node);
  string_format_node->set_abstract(std::make_shared<abstract::AbstractScalar>(kString));
  auto primitive = GetCNodePrimitive(string_format_node);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive = primitive->Clone();
  MS_EXCEPTION_IF_NULL(primitive);
  (void)primitive->AddAttr("template", MakeValue(str_template));
  (void)primitive->AddAttr("placeholder", MakeValue(placeholder));
  (void)primitive->AddAttr("summarize", MakeValue(summarize));
  string_format_node->set_input(0, std::make_shared<ValueNode>(primitive));
  return string_format_node;
}
}  // namespace

const BaseRef PrintToStringFormatPrint::DefinePattern() const {
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({prim::kPrimPrint, Xs});
}

const AnfNodePtr PrintToStringFormatPrint::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                   const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // convert Print to StringFormat and PrintV2 to adapt CANN
  auto string_format_node = CreateStringFormat(func_graph, cnode);
  auto new_print_node = CreateNewPrint(func_graph, string_format_node, cnode);
  return new_print_node;
}
}  // namespace opt
}  // namespace mindspore
