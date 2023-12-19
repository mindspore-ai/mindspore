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

#include <vector>
#include <unordered_map>
#include "backend/common/graph_kernel/convert_input_and_attr.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ops/sequence_ops.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"

namespace mindspore::graphkernel {
namespace {
const std::set<std::string> &GetConvertInputAttrOps() {
  static const std::set<std::string> convert_input_attr_ops = {
    prim::kPrimSoftmax->name(),       prim::kPrimReduceSum->name(),   prim::kPrimReduceMax->name(),
    prim::kPrimReduceMin->name(),     prim::kPrimReduceMean->name(),  prim::kPrimOneHot->name(),
    prim::kPrimMinimumGrad->name(),   prim::kPrimMaximumGrad->name(), prim::kPrimGather->name(),
    prim::kPrimCumSum->name(),        prim::kPrimArgmin->name(),      prim::kPrimArgmax->name(),
    prim::kPrimBiasAdd->name(),       prim::kPrimBiasAddGrad->name(), prim::kPrimLayerNorm->name(),
    prim::kPrimLayerNormGrad->name(), prim::kPrimLogSoftmax->name(),  prim::kPrimLogSoftmaxGrad->name(),
  };
  return convert_input_attr_ops;
}

ValuePtr EnumToFormat(const ValuePtr &value) {
  if (!value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << value->ToString() << " is not Int64Imm.";
  }
  auto val = GetValue<int64_t>(value);
  if (val == 0) {
    return MakeValue("NCHW");
  } else if (val == 1) {
    return MakeValue("NHWC");
  } else {
    MS_LOG(EXCEPTION) << value->ToString() << " is unexpected.";
  }
}

ValuePtr FormatToEnum(const ValuePtr &value) {
  auto format = GetValue<std::string>(value);
  if (format == "NCHW") {
    return MakeValue<int64_t>(0);
  } else if (format == "NHWC") {
    return MakeValue<int64_t>(1);
  } else {
    MS_LOG(EXCEPTION) << value->ToString() << " value:" << format << " is unexpected.";
  }
}

ValuePtr EnumToDtype(const ValuePtr &value) {
  if (!value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << value->ToString() << " is not Int64Imm.";
  }
  auto val = GetValue<int64_t>(value);
  return TypeIdToType(static_cast<TypeId>(val));
}

ValuePtr DtypeToEnum(const ValuePtr &value) {
  if (!value->isa<Type>()) {
    MS_LOG(EXCEPTION) << value->ToString() << " is not Type.";
  }
  auto type_id = value->cast<TypePtr>()->type_id();
  return MakeValue<int64_t>(type_id);
}

using ArgHandlerFunc = std::function<ValuePtr(const ValuePtr &)>;

ArgHandlerFunc GetArgHandlerFunc(const std::string &arg_handler) {
  static const std::unordered_map<std::string, ArgHandlerFunc> arg_handler_funcs = {
    {"py_format_to_enum", EnumToFormat},
    {"dtype_to_enum", EnumToDtype},
  };
  if (arg_handler_funcs.find(arg_handler) != arg_handler_funcs.end()) {
    return arg_handler_funcs.at(arg_handler);
  } else {
    return nullptr;
  }
}

ArgHandlerFunc GetOppArgHandlerFunc(const std::string &arg_handler) {
  static const std::unordered_map<std::string, ArgHandlerFunc> opp_arg_handler_funcs = {
    {"py_format_to_enum", FormatToEnum},
    {"dtype_to_enum", DtypeToEnum},
  };
  if (opp_arg_handler_funcs.find(arg_handler) != opp_arg_handler_funcs.end()) {
    return opp_arg_handler_funcs.at(arg_handler);
  } else {
    return nullptr;
  }
}
}  // namespace

void ConvertInputToAttr::AddConstInputToAttr(const CNodePtr &cnode, const size_t input_index,
                                             const std::string &arg_name, const std::string &arg_handler,
                                             const PrimitivePtr &primitive) {
  if (input_index >= cnode->size() - 1) {
    MS_LOG(EXCEPTION) << "The index of args in op_def `" << input_index
                      << "` should less than the inputs size minus one `" << cnode->size() - 1 << "`.";
  }
  auto input_node = cnode->inputs()[input_index + 1];

  ValuePtr value = nullptr;
  if (input_node->isa<ValueNode>()) {
    auto value_node = input_node->cast<ValueNodePtr>();
    value = value_node->value();
  } else if (input_node->isa<Parameter>()) {
    auto parameter_node = input_node->cast<ParameterPtr>();
    value = parameter_node->abstract()->BuildValue();
  }
  if (value == nullptr) {
    MS_LOG(EXCEPTION) << cnode->ToString() << " is not Value.";
  }
  if (value->isa<ValueAny>()) {
    MS_LOG(EXCEPTION) << cnode->ToString() << " is ValueAny.";
  }
  if (!arg_handler.empty() && !value->isa<None>()) {
    auto arg_handler_func = GetArgHandlerFunc(arg_handler);
    MS_EXCEPTION_IF_NULL(arg_handler_func);
    value = arg_handler_func(value);
    primitive->AddAttr(arg_name, value);
    return;
  }

  if (!value->isa<tensor::Tensor>()) {
    primitive->AddAttr(arg_name, value);
    return;
  }
  auto value_vector = CheckAndConvertUtils::CheckTensorIntValue(arg_name, value, primitive->name());
  auto tensor = value->cast<tensor::TensorPtr>();
  auto tensor_shape = tensor->shape_c();
  MS_LOG(DEBUG) << cnode->ToString() << " 's input[" << input_index << "] is tensor.";
  if (tensor_shape.empty()) {
    primitive->AddAttr(arg_name, MakeValue(value_vector[0]));
  } else {
    primitive->AddAttr(arg_name, MakeValue(value_vector));
  }
}

bool ConvertInputToAttr::Process(const CNodePtr &cnode, const ops::OpDefPtr &op_def, const PrimitivePtr &primitive) {
  const auto &op_def_args = op_def->args_;
  const auto &op_def_indexes = op_def->indexes_;
  bool changed = false;
  if (op_def_args.size() != cnode->size() - 1) {
    MS_LOG(EXCEPTION) << "The size of args in op_def `" << op_def->args_.size()
                      << "` should be equal to the inputs size minus one `" << cnode->size() - 1 << "`.";
  }
  auto iter = op_def_args.crbegin();
  auto new_input_size = op_def_args.size();
  for (; iter != op_def_args.crend(); ++iter, --new_input_size) {
    // as_init_arg_ == 1 indicate the arg need convert, the arg need convert is at the tail of the list
    if (iter->as_init_arg_ != 1) {
      break;
    }
    const auto &arg_name = iter->arg_name_;
    const auto &arg_handler = iter->arg_handler_;
    MS_LOG(DEBUG) << cnode->ToString() << " convert input to attr: " << arg_name;
    if (auto index_iter = op_def_indexes.find(arg_name); index_iter != op_def_indexes.end()) {
      AddConstInputToAttr(cnode, index_iter->second, arg_name, arg_handler, primitive);
      changed = true;
    } else {
      MS_LOG(EXCEPTION) << primitive->name() << " not found index of attr[" << arg_name << "] in op def indexes.";
    }
  }
  auto inputs = cnode->inputs();
  if (changed) {
    // remainder args in op_def_arg is the size of new input args
    AnfNodePtrList new_inputs(inputs.begin(), inputs.begin() + new_input_size + 1);
    cnode->set_inputs(new_inputs);
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    cb->ResetKernelInfoInputs(cnode);
  }
  return changed;
}

bool ConvertInputToAttr::Run(const FuncGraphPtr &func_graph) {
  bool changed = false;
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto todos = TopoSort(func_graph->get_return());
  const auto &input_to_attr_ops = GetConvertInputAttrOps();
  for (auto &node : todos) {
    auto primitive = GetCNodePrimitive(node);
    if (primitive == nullptr) {
      continue;
    }
    const auto &op_name = primitive->name();
    if (input_to_attr_ops.count(op_name) == 0) {
      continue;
    }
    auto op_def = mindspore::ops::GetOpDef(op_name);
    if (op_def == nullptr) {
      MS_LOG(WARNING) << op_name << " not found in op def.";
      continue;
    }
    auto cnode = dyn_cast<CNode>(node);
    changed = Process(cnode, op_def, primitive) || changed;
  }
  if (changed) {
    auto mng = GkUtils::GetFuncGraphManager(func_graph);
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }
  return changed;
}

void ConvertAttrToInput::AddAttrToInput(const CNodePtr &cnode, const std::string &arg_name,
                                        const std::string &arg_handler, const PrimitivePtr &primitive) {
  auto value = primitive->GetAttr(arg_name);
  ValueNodePtr value_node;
  if (!arg_handler.empty()) {
    auto opp_arg_handler_func = GetOppArgHandlerFunc(arg_handler);
    MS_EXCEPTION_IF_NULL(opp_arg_handler_func);
    value = opp_arg_handler_func(value);
    value_node = std::make_shared<ValueNode>(value);
  } else if (value->isa<Int64Imm>()) {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(GetValue<int64_t>(value), kInt64);
    value_node = std::make_shared<ValueNode>(tensor_ptr);
  } else {
    value_node = std::make_shared<ValueNode>(value);
  }
  value_node->set_abstract(value->ToAbstract());
  auto cb = Callback::Instance();
  MS_EXCEPTION_IF_NULL(cb);
  cb->SetEmptyKernelInfo(value_node);
  cnode->add_input(value_node);
  primitive->DelAttr(arg_name);
}

bool ConvertAttrToInput::Process(const AnfNodePtr &node) {
  auto primitive = GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &op_name = primitive->name();
  auto op_def = mindspore::ops::GetOpDef(op_name);
  if (op_def == nullptr) {
    MS_LOG(WARNING) << op_name << " not found in op def.";
    return false;
  }
  const auto &op_def_args = op_def->args_;
  bool changed = false;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto ori_input_size = cnode->size() - 1;
  auto iter = op_def_args.cbegin() + ori_input_size;
  for (; iter != op_def_args.cend(); ++iter) {
    // as_init_arg_ == 1 indicate the arg need convert
    if (iter->as_init_arg_ != 1) {
      MS_LOG(EXCEPTION) << primitive->name() << "'s input:" << iter->arg_name_
                        << " must have as_init_arg_ when convert attr to input.";
    }
    MS_LOG(DEBUG) << cnode->DebugString() << " convert attr [" << iter->arg_name_ << "] to input.";
    ConvertAttrToInput::AddAttrToInput(cnode, iter->arg_name_, iter->arg_handler_, primitive);
    changed = true;
  }
  if (changed) {
    auto cb = Callback::Instance();
    MS_EXCEPTION_IF_NULL(cb);
    cb->ResetKernelInfoInputs(cnode);
  }
  return changed;
}

bool ConvertAttrToInput::Run(const FuncGraphPtr &func_graph) {
  bool changed = false;
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(func_graph->get_return());
  auto todos = TopoSort(func_graph->get_return());
  for (auto &node : todos) {
    if (NeedConvertInputAndAttr(node)) {
      changed = ConvertAttrToInput::Process(node) || changed;
    }
  }
  if (changed) {
    auto mng = GkUtils::GetFuncGraphManager(func_graph);
    GkUtils::UpdateFuncGraphManager(mng, func_graph);
  }
  return changed;
}

bool NeedConvertInputAndAttr(const AnfNodePtr &node) {
  return node->isa<CNode>() && GetConvertInputAttrOps().count(AnfUtils::GetCNodeName(node)) != 0;
}
}  // namespace mindspore::graphkernel
