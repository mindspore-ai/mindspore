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

#include "frontend/optimizer/irpass/vmap_eliminate.h"

#include <string>
#include <vector>
#include <set>
#include <regex>
#include "utils/hash_map.h"
#include "ir/func_graph_cloner.h"
#include "base/complex_storage.h"
#include "frontend/optimizer/irpass/gradient_eliminate.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "pipeline/pynative/pynative_execute.h"
#include "frontend/operator/composite/vmap.h"
#include "include/common/utils/convert_utils_py.h"

namespace mindspore {
namespace opt {
namespace irpass {
constexpr int kInvalidAxisSize = -1;
namespace internal {
// White list of primitives consistent before and after transformation.
const mindspore::HashSet<std::string> throughtout_op{prim::kPrimMakeTuple->name(),   prim::kPrimMakeList->name(),
                                                     prim::kPrimDepend->name(),      prim::kPrimReturn->name(),
                                                     prim::kPrimUpdateState->name(), prim::kPrimStopGradient->name()};
CNodePtr BuildBindInAxisSeqInput(const AnfNodePtr &input, const ValuePtr &in_axis, const FuncGraphPtr &fg) {
  auto input_abs = input->abstract();
  MS_EXCEPTION_IF_NULL(input_abs);
  auto input_abs_elements = dyn_cast<abstract::AbstractSequence>(input_abs);
  MS_EXCEPTION_IF_NULL(input_abs_elements);
  ValueSequencePtr in_axis_value_sequence = nullptr;
  if (in_axis->isa<ValueSequence>()) {
    in_axis_value_sequence = dyn_cast<ValueSequence>(in_axis);
    if (input_abs_elements->size() != in_axis_value_sequence->size()) {
      MS_EXCEPTION(ValueError) << "The length of input and in_axis should be the same but got input length: "
                               << input_abs_elements->size() << ", in_axis length: " << in_axis_value_sequence->size()
                               << ".";
    }
  }
  std::vector<AnfNodePtr> ret_inputs;
  if (input_abs->isa<abstract::AbstractList>()) {
    (void)ret_inputs.emplace_back(NewValueNode(prim::kPrimMakeList));
  } else {
    (void)ret_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  }

  for (unsigned int i = 0; i < input_abs_elements->size(); ++i) {
    std::vector<AnfNodePtr> seq_getitem_cnode_inputs;
    if (input_abs->isa<abstract::AbstractList>()) {
      (void)seq_getitem_cnode_inputs.emplace_back(NewValueNode(prim::kPrimListGetItem));
    } else {
      (void)seq_getitem_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    }
    (void)seq_getitem_cnode_inputs.emplace_back(input);
    (void)seq_getitem_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(i)));
    auto seq_getitem_cnode = fg->NewCNode(seq_getitem_cnode_inputs);
    MS_EXCEPTION_IF_NULL(seq_getitem_cnode);
    auto input_abs_element = (*input_abs_elements)[i];
    auto in_axis_value = in_axis_value_sequence == nullptr ? in_axis : (*in_axis_value_sequence)[i];
    CNodePtr cur_make_seq = nullptr;
    if (input_abs_element->isa<abstract::AbstractSequence>()) {
      seq_getitem_cnode->set_abstract(input_abs_element);
      cur_make_seq = BuildBindInAxisSeqInput(seq_getitem_cnode, in_axis_value, fg);
    } else {
      std::vector<AnfNodePtr> cur_make_tuple_inputs;
      (void)cur_make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      (void)cur_make_tuple_inputs.emplace_back(seq_getitem_cnode);
      (void)cur_make_tuple_inputs.emplace_back(NewValueNode(in_axis_value));
      cur_make_seq = fg->NewCNode(cur_make_tuple_inputs);
    }
    (void)ret_inputs.emplace_back(cur_make_seq);
  }
  return fg->NewCNode(ret_inputs);
}

AnfNodePtr UpdateParam(const FuncGraphPtr &vmap_fg, const AnfNodePtr &u_monad_node, bool is_feedback,
                       const ParamMappingVector &param_mapping_table) {
  MS_EXCEPTION_IF_NULL(u_monad_node);
  std::vector<AnfNodePtr> attach_tuple{NewValueNode(prim::kPrimMakeTuple)};
  for (auto &param_pair : param_mapping_table) {
    auto ref = param_pair.first;
    auto each_cell_params = param_pair.second;
    std::vector<AnfNodePtr> vmap_assign;
    if (is_feedback) {
      (void)vmap_assign.emplace_back(NewValueNode(prim::kPrimVmapUnstackAssign));
    } else {
      (void)vmap_assign.emplace_back(NewValueNode(prim::kPrimVmapStackAssign));
    }
    (void)vmap_assign.emplace_back(ref);
    (void)vmap_assign.insert(vmap_assign.end(), each_cell_params.begin(), each_cell_params.end());
    vmap_assign.push_back(u_monad_node);
    auto vmap_assign_cnode = vmap_fg->NewCNode(vmap_assign);
    attach_tuple.push_back(vmap_assign_cnode);
  }
  auto attach_cnode = vmap_fg->NewCNode(attach_tuple);
  auto update_state_prim = NewValueNode(prim::kPrimUpdateState);
  auto update_state_node = vmap_fg->NewCNode({update_state_prim, u_monad_node, attach_cnode});
  return update_state_node;
}

void GetMonadOffset(const std::vector<AnfNodePtr> &inputs, size_t *u_monad_offset, size_t *io_monad_offset) {
  // Check the last two (if exists) is monad input.
  if (*u_monad_offset != 0 || *io_monad_offset != 0) {
    MS_EXCEPTION(ValueError) << "The initial value of u_monad_offset and io_monad_offset should be 0, but we got "
                             << "u_monad_offset: " << *u_monad_offset << " and io_monad_offset: " << *io_monad_offset
                             << ".";
  }
  auto inputs_size = inputs.size();
  constexpr size_t max_monad_input_num = 2;
  if (HasAbstractMonad(inputs[inputs_size - 1])) {
    if (HasAbstractUMonad(inputs[inputs_size - 1])) {
      *u_monad_offset = 1;
    } else if (inputs_size >= max_monad_input_num && HasAbstractUMonad(inputs[inputs_size - max_monad_input_num])) {
      ++(*io_monad_offset);
      *u_monad_offset = *io_monad_offset + 1;
    } else {
      ++(*io_monad_offset);
    }
  }
}

void BindUMonad(const AnfNodePtr &u_monad_node, const FuncGraphPtr &vmap_fg, std::vector<AnfNodePtr> *outputs,
                ParamMappingVector *param_mapping_table) {
  MS_EXCEPTION_IF_NULL(u_monad_node);
  if (param_mapping_table == nullptr || param_mapping_table->empty()) {
    (void)outputs->emplace_back(u_monad_node);
  } else {
    auto update_state_node = UpdateParam(vmap_fg, u_monad_node, false, *param_mapping_table);
    (void)outputs->emplace_back(update_state_node);
  }
}

AnfNodePtr BindInAxis(const CNodePtr &vmap_app, const ValuePtr &in_axes, size_t *u_monad_offset,
                      size_t *io_monad_offset, ParamMappingVector *param_mapping_table) {
  FuncGraphPtr vmap_fg = vmap_app->func_graph();
  bool is_in_axes_value_sequence = in_axes->isa<ValueSequence>();
  ValueSequencePtr in_axes_to_value_sequence = dyn_cast<ValueSequence>(in_axes);

  auto inputs = vmap_app->inputs();
  auto inputs_size = inputs.size();
  if (inputs_size == 0) {
    MS_EXCEPTION(ValueError) << "The inputs number of CNode: " << vmap_app->DebugString()
                             << " should be positive but got : " << inputs_size << ".";
  }
  GetMonadOffset(inputs, u_monad_offset, io_monad_offset);
  size_t abstract_monad_count = *u_monad_offset > *io_monad_offset ? *u_monad_offset : *io_monad_offset;
  size_t real_params_size = inputs_size > abstract_monad_count ? inputs_size - abstract_monad_count : 0;
  if (is_in_axes_value_sequence && real_params_size - 1 != in_axes_to_value_sequence->size()) {
    MS_EXCEPTION(ValueError) << "The length of vmap_app inputs (except primitive input and monad input) is: "
                             << (real_params_size - 1)
                             << " and the length of in_axis is: " << in_axes_to_value_sequence->size()
                             << ". These two numbers should be equal.";
  }

  std::vector<AnfNodePtr> outputs;
  outputs.push_back(vmap_app->input(0));
  for (unsigned int i = 1; i < real_params_size; ++i) {
    auto input = inputs[i];
    auto in_axis = is_in_axes_value_sequence ? (*in_axes_to_value_sequence)[i - 1] : in_axes;
    auto input_abs = input->abstract();
    CNodePtr cur_make_seq_cnode = nullptr;
    if (input_abs->isa<abstract::AbstractSequence>()) {
      cur_make_seq_cnode = BuildBindInAxisSeqInput(input, in_axis, vmap_fg);
    } else {
      cur_make_seq_cnode = vmap_fg->NewCNode({NewValueNode(prim::kPrimMakeTuple), input, NewValueNode(in_axis)});
    }
    (void)outputs.emplace_back(cur_make_seq_cnode);
  }

  if (*u_monad_offset > 0 && inputs_size > *u_monad_offset) {
    AnfNodePtr u_monad_node = inputs[inputs_size - *u_monad_offset];
    BindUMonad(u_monad_node, vmap_fg, &outputs, param_mapping_table);
  }

  if (*io_monad_offset > 0 && inputs_size > 0) {
    (void)outputs.emplace_back(inputs.back());
  }

  return vmap_fg->NewCNode(outputs);
}

ValueSequencePtr GetInAxesSeq(const ValuePtr &in_axes, size_t parameters_size) {
  auto in_axes_seq = dyn_cast<ValueSequeue>(in_axes);
  if (in_axes_seq != nullptr || parameters_size <= 1) {
    return in_axes_seq;
  }

  // Even if the input parameter matches the same negative axis index, it may correspond to different positive indexes.
  // eg. ([A, B, C], -1) equivalent to ([A, B, C], 2), but ([A, B, C, D], -1) equivalent to ([A, B, C, D], 3) in vmap.
  // Therefore, when the in_axes is a negative integer, with multiple inputs, 'ValuePtr' need to be copied multi-times
  // to carry different positive index later.
  auto in_axes_int = dyn_cast<Int64Imm>(in_axes);

  // sub in_axes maybe a 'None'
  if (in_axes_int == nullptr) {
    return nullptr;
  }
  auto axis_nb = in_axes_int->value();
  if (axis_nb >= 0) {
    return nullptr;
  }
  std::vector<ValuePtr> elements;
  for (size_t i = 0; i < parameters_size; i++) {
    ValuePtr in_axis_copy = std::make_shared<Int64Imm>(axis_nb);
    elements.push_back(in_axis_copy);
  }
  return std::make_shared<ValueSequence>(elements);
}

void GetSubAxisSize(const AbstractBasePtr &sub_abs, ValuePtr *const sub_in_axes, int *axis_size,
                    std::vector<ValuePtr> *corrected_in_axes) {
  int sub_axis_size = GetAxisSizeByAbs(sub_abs, sub_in_axes);
  corrected_in_axes->push_back(*sub_in_axes);
  if (sub_axis_size == kInvalidAxisSize) {
    return;
  }
  if (*axis_size == kInvalidAxisSize) {
    *axis_size = sub_axis_size;
  } else if (*axis_size != sub_axis_size) {
    MS_EXCEPTION(ValueError) << "The 'axis_size' of each argument in the scope of 'vmap' should be equal, but got "
                             << *axis_size << " and " << sub_axis_size << ".";
  }
}

int GetAxisSizeByAbs(const AbstractBasePtr &abs, ValuePtr *const in_axes) {
  MS_EXCEPTION_IF_NULL(abs);
  MS_EXCEPTION_IF_NULL(*in_axes);
  int axis_size = kInvalidAxisSize;
  auto abs_sequence = dyn_cast<abstract::AbstractSequence>(abs);
  if (abs_sequence != nullptr) {
    std::vector<ValuePtr> corrected_in_axes;
    AbstractBasePtrList abs_list = abs_sequence->elements();
    size_t parameters_size = abs_sequence->size();
    auto in_axes_seq = GetInAxesSeq(*in_axes, parameters_size);
    int index = 0;
    for (auto sub_abs : abs_list) {
      if (sub_abs->isa<abstract::AbstractMonad>()) {
        break;
      }
      ValuePtr sub_in_axes = in_axes_seq != nullptr ? (*in_axes_seq)[index] : *in_axes;
      GetSubAxisSize(sub_abs, &sub_in_axes, &axis_size, &corrected_in_axes);
      index++;
    }
    *in_axes = std::make_shared<ValueSequence>(corrected_in_axes);
    return axis_size;
  }

  auto in_axes_int = dyn_cast<Int64Imm>(*in_axes);
  if (in_axes_int != nullptr) {
    int64_t axis = in_axes_int->value();
    if (!abs->isa<abstract::AbstractTensor>()) {
      // If got a AbstractScalar with a value 0 of type int32, it means that the input is not used later.
      auto abs_value = abs->BuildValue();
      MS_EXCEPTION_IF_NULL(abs_value);
      auto abs_int32_t = dyn_cast<Int32Imm>(abs_value);
      MS_EXCEPTION_IF_NULL(abs_int32_t);
      if (abs_int32_t->value() == 0) {
        MS_LOG(WARNING) << "There is a argument not used in the scope of vmap. Please check whether the inputs"
                        << " meet expectations.";
        return axis_size;
      }
      MS_EXCEPTION(ValueError) << "The abs should be AbstractTensor when axis is " << axis << ", but got a "
                               << abs->ToString() << ".";
    }
    auto shape = abs->BuildShape();
    MS_EXCEPTION_IF_NULL(shape);
    auto shape_ptr = dyn_cast<abstract::Shape>(shape);
    MS_EXCEPTION_IF_NULL(shape_ptr);
    ShapeVector orig_shape = shape_ptr->shape();
    int64_t shape_len = SizeToLong(orig_shape.size());
    if (axis < -shape_len || axis >= shape_len) {
      MS_EXCEPTION(ValueError) << "ValueError: axis " << axis << " is out of bounds for array of dimension ["
                               << -shape_len << "," << shape_len << ").";
    }
    axis = axis < 0 ? shape_len + axis : axis;
    *in_axes = std::make_shared<Int64Imm>(axis);
    axis_size = LongToInt(orig_shape[LongToSize(axis)]);
    return axis_size;
  }
  return axis_size;
}

// get the axis size of currently vmap scope, at the same time, the negative indexes in in_axes are converted to
// corresponding positive indexes.
int GetAxisSize(const CNodePtr &cnode, size_t cell_size, size_t parameters_size, ValuePtr *const in_axes) {
  MS_EXCEPTION_IF_NULL(cnode);
  // `axis_size` is unique within the scope of vmap, so we just need to get one of them.
  int axis_size = kInvalidAxisSize;
  auto in_axes_seq = GetInAxesSeq(*in_axes, parameters_size);
  std::vector<ValuePtr> corrected_in_axes;
  for (size_t i = 0; i < parameters_size; ++i) {
    auto sub_abs = cnode->input(i + 1)->abstract();
    MS_EXCEPTION_IF_NULL(sub_abs);
    if (sub_abs->isa<abstract::AbstractMonad>()) {
      break;
    }
    ValuePtr sub_in_axes = in_axes_seq != nullptr ? (*in_axes_seq)[i] : *in_axes;
    GetSubAxisSize(sub_abs, &sub_in_axes, &axis_size, &corrected_in_axes);
  }
  *in_axes = std::make_shared<ValueSequence>(corrected_in_axes);

  if (cell_size > 0) {
    if (axis_size == kInvalidAxisSize) {
      axis_size = SizeToLong(cell_size);
    } else if (SizeToLong(cell_size) != axis_size) {
      MS_EXCEPTION(ValueError) << "If you want to execute the model ensembling parallel training, please make sure "
                               << "the 'axis_size' in the scope of vmap consistent with the cell size of the input "
                               << "'CellList', otherwise, please do not enter 'CellList' as the first argument, "
                               << "but we get axis_size: " << axis_size << " and the cell size: " << cell_size << ".";
    }
  } else if (axis_size == kInvalidAxisSize) {
    MS_LOG(EXCEPTION) << "Failed to get 'axis_size' within the scope of vmap.";
  }
  return axis_size;
}

CNodePtr AttachToOutput(const FuncGraphPtr &func_graph, const CNodePtr &output, const AnfNodePtr &node) {
  TraceGuard guard(std::make_shared<TraceCopy>(output->debug_info()));
  auto depend = NewValueNode(prim::kPrimDepend);
  auto depend_cnode = func_graph->NewCNode({depend, output, node});
  MS_EXCEPTION_IF_NULL(depend_cnode);
  depend_cnode->set_abstract(output->abstract());
  return depend_cnode;
}

AnfNodePtr FeedBackParam(const FuncGraphPtr &vmap_post_fg, const AnfNodePtr &u_monad_node,
                         const AnfNodePtr &io_monad_node, const CNodePtr &output,
                         const ParamMappingVector &param_mapping_table) {
  auto update_state_after_assign = UpdateParam(vmap_post_fg, u_monad_node, true, param_mapping_table);
  MS_EXCEPTION_IF_NULL(update_state_after_assign);
  update_state_after_assign->set_abstract(u_monad_node->abstract());
  auto attach_output = AttachToOutput(vmap_post_fg, output, update_state_after_assign);
  if (io_monad_node) {
    attach_output = AttachToOutput(vmap_post_fg, attach_output, io_monad_node);
  }
  vmap_post_fg->set_output(attach_output);
  return NewValueNode(vmap_post_fg);
}

AnfNodePtr PostProcessVmap(const AnfNodePtr &expanded_vmap_node, const std::vector<size_t> &orig_fg_param_info,
                           const ValuePtr &out_axes, int axis_size, ParamMappingVector *param_mapping_table) {
  FuncGraphPtr vmap_post_fg = std::make_shared<FuncGraph>();
  std::vector<AnfNodePtr> exec_node;
  exec_node.push_back(expanded_vmap_node);
  AnfNodePtr u_monad_node = nullptr;
  AnfNodePtr io_monad_node = nullptr;
  size_t parameters_size = orig_fg_param_info[kParamSizeIndex];
  size_t u_monad_offset = orig_fg_param_info[kUMonadOffsetIndex];
  size_t io_monad_offset = orig_fg_param_info[kIOMonadOffsetIndex];
  size_t u_monad_index = parameters_size > u_monad_offset ? parameters_size - u_monad_offset : parameters_size;
  size_t io_monad_index = parameters_size > io_monad_offset ? parameters_size - io_monad_offset : parameters_size;
  for (size_t i = 0; i < parameters_size; ++i) {
    if (i == u_monad_index) {
      u_monad_node = vmap_post_fg->add_parameter();
      exec_node.push_back(u_monad_node);
      continue;
    } else if (i == io_monad_index) {
      io_monad_node = vmap_post_fg->add_parameter();
      exec_node.push_back(io_monad_node);
      continue;
    }
    exec_node.push_back(vmap_post_fg->add_parameter());
  }
  auto vmap_outputs = vmap_post_fg->NewCNode(exec_node);

  auto update_state_prim = NewValueNode(prim::kPrimUpdateState);
  if (u_monad_node) {
    auto update_state_cnode = vmap_post_fg->NewCNode({update_state_prim, u_monad_node, vmap_outputs});
    MS_EXCEPTION_IF_NULL(update_state_cnode);
    update_state_cnode->set_abstract(u_monad_node->abstract());
    u_monad_node = update_state_cnode;
  }
  if (io_monad_node) {
    auto update_state_cnode = vmap_post_fg->NewCNode({update_state_prim, io_monad_node, vmap_outputs});
    MS_EXCEPTION_IF_NULL(update_state_cnode);
    update_state_cnode->set_abstract(io_monad_node->abstract());
    io_monad_node = update_state_cnode;
  }

  // MatchOutAxis: Convert the outputs according to the out_axes to the specified physical perspective.
  auto match_out_axis_app =
    vmap_post_fg->NewCNode({NewValueNode(std::make_shared<prim::VmapMatchOutAxis>("VmapMatchOutAxis")), vmap_outputs,
                            NewValueNode(out_axes), NewValueNode(static_cast<int64_t>(axis_size))});

  if (param_mapping_table == nullptr || param_mapping_table->empty()) {
    auto output = match_out_axis_app;
    if (u_monad_node) {
      output = AttachToOutput(vmap_post_fg, output, u_monad_node);
    }
    if (io_monad_node) {
      output = AttachToOutput(vmap_post_fg, output, io_monad_node);
    }
    vmap_post_fg->set_output(output);
    return NewValueNode(vmap_post_fg);
  }

  // Feed parameters back to each cell in the model ensembling parallel training case.
  return FeedBackParam(vmap_post_fg, u_monad_node, io_monad_node, match_out_axis_app, *param_mapping_table);
}

ValuePtr CreatePrimtivePy(const mindspore::HashMap<std::string, ValuePtr> &attrs, const string &op_name) {
  const auto op_path = "mindspore.ops.primitive";
  const auto func = "_get_primitivec";
  py::dict attrs_py = py::dict();
  for (auto &v : attrs) {
    py::str name = v.first;
    attrs_py[name] = ValueToPyData(v.second);
  }
  py::object obj = python_adapter::CallPyFn(op_path, func, op_name, attrs_py);
  ValuePtr op_instance = nullptr;
  bool succ = parse::ConvertData(obj, &op_instance);
  if (!succ) {
    MS_LOG(ERROR) << "Failure:get Python op " << op_path << " from " << op_name << " fail";
    return nullptr;
  }
  return op_instance;
}

AnfNodePtr GetVmapRule(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &resource, int axis_size) {
  // Set a child scope named "vmap_'PrimitiveName'" for the vmap rule function,
  // and add "VmapRule" to the front.
  constexpr char vmap_rule_scope[] = "VmapRule/";
  constexpr char vmap_op_child_scope_prefix[] = "/vmap_";
  MS_EXCEPTION_IF_NULL(prim);
  auto scope = std::make_shared<Scope>(vmap_rule_scope + ScopeManager::GetInstance().GetCurrentScope()->name() +
                                       vmap_op_child_scope_prefix + prim->name());
  ScopeGuard scope_guard(scope);

  // Firstly we parse the python VmapRules function registered for specific primitive. If failed, get
  // the vmap general rule.
  FuncGraphPtr vmap_rule_fg = nullptr;
  AnfNodePtr vmap_rule_node = nullptr;
  py::function vmap_rule_fn;
  bool is_side_effect = false;
  if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_MEM)) {
    is_side_effect = true;
  } else if (GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_IO) && prim->name() != prim::kPrimPrint->name()) {
    MS_LOG(EXCEPTION) << prim->name() << " is a GRAPH_FLAG_SIDE_EFFECT_IO prim, vmap dont support currently.";
  }

  // Get vmap rule for specific primitive.
  if (prim->is_base()) {
    if (prim->attrs().empty()) {
      vmap_rule_fn = GetVmapRuleFunction(prim->name(), axis_size);
    } else {
      auto new_prim = CreatePrimtivePy(prim->attrs(), prim->name());
      vmap_rule_fn = new_prim->cast<PrimitivePyPtr>()->GetVmapRuleFunction(is_side_effect, axis_size);
    }
  } else {
    vmap_rule_fn = prim->cast<PrimitivePyPtr>()->GetVmapRuleFunction(is_side_effect, axis_size);
    if (py::isinstance<py::none>(vmap_rule_fn)) {
      vmap_rule_fn = GetVmapRuleFunction(prim->name(), axis_size);
    }
  }

  // If vmap rule for specific primitive not found, get vmap general rule.
  if (!vmap_rule_fn || py::isinstance<py::none>(vmap_rule_fn)) {
    MS_LOG(DEBUG) << "Fail to find vmap rule function for " << prim->name() << ", try to get the general vmap rule.";
    if (is_side_effect) {
      vmap_rule_fn = python_adapter::GetPyFn("mindspore.ops._vmap", "vmap_monad_rule")(prim->name(), axis_size);
    } else {
      vmap_rule_node =
        NewValueNode(std::make_shared<prim::VmapGeneralRule>("VmapGeneralRule", prim, static_cast<int64_t>(axis_size)));
    }
  }

  if (vmap_rule_node == nullptr) {
    vmap_rule_fg = parse::ParsePythonCode(vmap_rule_fn);
    if (vmap_rule_fg == nullptr) {
      MS_LOG(EXCEPTION) << "Fail to parse vmap rule function for " << prim->name() << ".";
    }
    auto vmap_rule_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_PROPAGATE);
    if (vmap_rule_flag) {
      vmap_rule_fg->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
    }
    pipeline::ResourceBasePtr res = (resource != nullptr) ? resource : std::make_shared<pipeline::Resource>();
    (void)parse::ResolveFuncGraph(vmap_rule_fg, res);
    vmap_rule_node = NewValueNode(vmap_rule_fg);
  }

  return vmap_rule_node;
}

AnfNodePtr ExpandVmapPrimitive(const AnfNodePtr &vnode, const pipeline::ResourceBasePtr &resource, int axis_size) {
  MS_EXCEPTION_IF_NULL(vnode);
  if (!IsValueNode<Primitive>(vnode)) {
    MS_LOG(EXCEPTION) << "Primitive node is not valid.";
  }
  auto prim = GetValueNode<PrimitivePtr>(vnode);
  MS_LOG(DEBUG) << "Overloading Primitive node " << vnode->DebugString() << ".";
  if (throughtout_op.count(prim->name()) > 0) {
    return vnode;
  }
  AnfNodePtr prim_vmap_rule = GetVmapRule(prim, resource, axis_size);
  if (prim_vmap_rule == nullptr) {
    MS_LOG(EXCEPTION) << "Primitive " << prim->name() << " transform to VmapRule failed.";
  }
  return prim_vmap_rule;
}

AnfNodePtr CopyNodeToVmap(const AnfNodePtr &node, const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng) {
  MS_EXCEPTION_IF_NULL(node);
  auto &node_user_map = mng->node_users();
  auto user = node_user_map.find(node);
  if (user != node_user_map.end() && !user->second.empty()) {
    auto user_set = user->second;
    if (user_set.size() > 1) {
      MS_LOG(DEBUG) << "The " << node->DebugString() << " is used in more than one place.";
      bool need_copy = false;
      // We assume that the nodes used in the unified graph are continuous in most cases, therefore, checking the
      // head and tail nodes can pick up the most scenes of that the ValueNode are used by multiple graphs, otherwise,
      // traverse the entire set.
      if (user_set.front().first->func_graph() != func_graph || user_set.back().first->func_graph() != func_graph) {
        need_copy = true;
      } else {
        for (auto pair : user_set) {
          if (pair.first->func_graph() != func_graph) {
            need_copy = true;
            break;
          }
        }
      }
      if (need_copy) {
        MS_LOG(DEBUG) << "Copy the " << node->DebugString() << " so that it can only be used in this graph.";
        auto value_node = dyn_cast<ValueNode>(node);
        MS_EXCEPTION_IF_NULL(value_node);
        auto value = value_node->value();
        MS_EXCEPTION_IF_NULL(value);
        auto copy_node = NewValueNode(value);
        for (auto pair : user_set) {
          if (pair.first->func_graph() == func_graph) {
            auto user_node = pair.first->cast<CNodePtr>();
            mng->SetEdge(user_node, pair.second, copy_node);
          }
        }
        return copy_node;
      }
    }
  }
  return node;
}

void BindFvAxis(const AnfNodePtr &node, const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &mng,
                const AnfNodePtr &stacked_param_node = nullptr) {
  MS_EXCEPTION_IF_NULL(node);
  auto &node_user_map = mng->node_users();
  auto user = node_user_map.find(node);
  if (user != node_user_map.end() && !user->second.empty()) {
    auto make_tuple = NewValueNode(prim::kPrimMakeTuple);
    CNodePtr replace_node = nullptr;
    if (stacked_param_node == nullptr) {
      replace_node = func_graph->NewCNode({make_tuple, node, NewValueNode(kNone)});
    } else {
      replace_node = func_graph->NewCNode({make_tuple, stacked_param_node, NewValueNode(SizeToLong(0))});
    }
    auto user_set = user->second;
    for (auto pair : user_set) {
      if (pair.first->func_graph() == func_graph) {
        auto user_node = pair.first->cast<CNodePtr>();
        mng->SetEdge(user_node, pair.second, replace_node);
      }
    }
  }
}

void BindParamAxis(const AnfNodePtr &node, const FuncGraphPtr &vmap_fg, const FuncGraphManagerPtr &manager,
                   mindspore::HashMap<std::string, ParameterPtr> *stacked_params) {
  if (stacked_params == nullptr || stacked_params->empty()) {
    BindFvAxis(node, vmap_fg, manager);
    return;
  }
  std::string param_name = dyn_cast<Parameter>(node)->name();
  std::regex match_prefix("^.*?\\d+\\.(.+)$");
  param_name = std::regex_replace(param_name, match_prefix, "vmap.$1");
  auto iter = stacked_params->find(param_name);
  if (iter != stacked_params->end()) {
    ParameterPtr stacked_param_node = iter->second;
    MS_EXCEPTION_IF_NULL(stacked_param_node);
    BindFvAxis(node, vmap_fg, manager, stacked_param_node);
  } else {
    BindFvAxis(node, vmap_fg, manager);
  }
}

void ExpandVmapValueNode(const FuncGraphPtr &vmap_fg, const pipeline::ResourceBasePtr &resource,
                         VisitedHashSetPair *visited_pair, int axis_size,
                         mindspore::HashMap<std::string, ParameterPtr> *stacked_params) {
  // Map ValueNode.
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto value_nodes = vmap_fg->value_nodes();

  auto visited_graph = &visited_pair->first;
  auto visited_node = &visited_pair->second;

  for (const auto &value_pair : value_nodes) {
    auto node = value_pair.first;
    // ValueNode may have been transformed when other graphs are expanded.
    if (visited_node->count(node) > 0) {
      MS_LOG(DEBUG) << node->DebugString() << " has been transformed.";
      continue;
    }
    node = CopyNodeToVmap(node, vmap_fg, manager);
    if (IsValueNode<FuncGraph>(node)) {
      MS_LOG(DEBUG) << "Map FuncGraph node " << node->DebugString() << ".";
      (void)visited_node->insert(node);
      auto sub_func_graph = GetValueNode<FuncGraphPtr>(node);
      if (visited_graph->count(sub_func_graph) > 0) {
        continue;
      }
      (void)visited_graph->insert(sub_func_graph);
      auto transformed_fg = ExpandVmapFunctor(sub_func_graph, resource, axis_size, visited_pair, stacked_params);
      auto replace_node = NewValueNode(transformed_fg);
      (void)visited_node->insert(replace_node);
      (void)manager->Replace(node, replace_node);
    } else if (IsValueNode<Primitive>(node)) {
      auto replace_node = ExpandVmapPrimitive(node, resource, axis_size);
      MS_EXCEPTION_IF_NULL(replace_node);
      (void)visited_node->insert(replace_node);
      (void)manager->Replace(node, replace_node);
    } else if (IsValueNode<Scalar>(node) || IsValueNode<tensor::Tensor>(node) || IsValueNode<None>(node) ||
               IsValueNode<ValueTuple>(node) || IsValueNode<ValueList>(node) || IsValueNode<Type>(node) ||
               IsValueNode<StringImm>(node)) {
      auto value_node_ptr = node->cast<ValueNodePtr>();
      ValuePtr node_value = value_node_ptr->value();
      std::vector<ValuePtr> elements;
      elements.push_back(node_value);
      elements.push_back(kNone);
      auto replace_value = std::make_shared<ValueTuple>(elements);
      auto replace_node = NewValueNode(replace_value);
      (void)visited_node->insert(replace_node);
      (void)manager->Replace(node, replace_node);
    } else if (IsValueNode<Monad>(node)) {
      continue;
    } else {
      MS_LOG(EXCEPTION) << "vmap do not support transform " << node->DebugString() << " right now.";
    }
  }
}

void ExpandVmapFreeVariable(const FuncGraphPtr &vmap_fg, const FuncGraphManagerPtr &manager,
                            const mindspore::HashSet<AnfNodePtr> &visited_node,
                            mindspore::HashMap<std::string, ParameterPtr> *stacked_params) {
  // Map free variable.
  auto free_variables_nodes = vmap_fg->free_variables();
  for (auto &pair : free_variables_nodes) {
    auto node = pair.first;
    if (visited_node.count(node) > 0 || node->isa<CNode>()) {
      continue;
    }
    if (IsValueNode<Scalar>(node) || IsValueNode<tensor::Tensor>(node) || IsValueNode<None>(node) ||
        IsValueNode<ValueTuple>(node) || IsValueNode<Type>(node)) {
      BindFvAxis(node, vmap_fg, manager);
    } else if (node->isa<Parameter>()) {
      BindParamAxis(node, vmap_fg, manager, stacked_params);
    } else {
      MS_LOG(EXCEPTION) << "vmap do not support transform " << node->DebugString() << " right now.";
    }
  }
}

FuncGraphPtr ExpandVmapFunctor(const FuncGraphPtr &vmap_fg, const pipeline::ResourceBasePtr &resource, int axis_size,
                               VisitedHashSetPair *visited_pair,
                               mindspore::HashMap<std::string, ParameterPtr> *stacked_params) {
  MS_EXCEPTION_IF_NULL(vmap_fg);
  auto manager = resource->manager();
  MS_EXCEPTION_IF_NULL(manager);
  manager->AddFuncGraph(vmap_fg);
  auto visited_node = &visited_pair->second;

  // The parameters of the current graph will be transformed in the upper graph, and recorded in
  // `visited_node` to avoid being repeatedly transformed refer as a free variable in other graph.
  auto parameter_nodes = vmap_fg->parameters();
  for (auto &node : parameter_nodes) {
    MS_LOG(DEBUG) << "parameter_nodes" << node->DebugString() << ".";
    (void)visited_node->insert(node);
  }

  ExpandVmapValueNode(vmap_fg, resource, visited_pair, axis_size, stacked_params);
  ExpandVmapFreeVariable(vmap_fg, manager, *visited_node, stacked_params);

  return vmap_fg;
}

// Entry to perform Vmap transformation.
AnfNodePtr ExpandVmap(const ValueNodePtr &vnode, const pipeline::ResourceBasePtr &resource, int axis_size,
                      mindspore::HashMap<std::string, ParameterPtr> *stacked_params) {
  MS_EXCEPTION_IF_NULL(vnode);
  if (IsValueNode<FuncGraph>(vnode)) {
    ScopeGuard scope_guard(vnode->scope());
    auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
    MS_EXCEPTION_IF_NULL(func_graph);
    MS_LOG(DEBUG) << "Funcgraph: " << func_graph->ToString() << " will perform the Vmap transformation.";

    // Record transformed FuncGraphs and other nodes to avoid repeatedly expanding and transforming.
    // Whose lifecycle is limited to the current extension.
    mindspore::HashSet<FuncGraphPtr> visited_graph;
    mindspore::HashSet<AnfNodePtr> visited_node;
    (void)visited_graph.insert(func_graph);
    (void)visited_node.insert(vnode);

    VisitedHashSetPair visited_pair(visited_graph, visited_node);
    auto tf_fg = ExpandVmapFunctor(func_graph, resource, axis_size, &visited_pair, stacked_params);
    visited_node.clear();

    return NewValueNode(tf_fg);
  }
  MS_LOG(EXCEPTION) << "Currently, the first argument in F.vmap only supports Cell, Python defined "
                       "function or @jit decorated function.";
}

std::string GetShapeString(const ShapeVector &tensor_shape) {
  std::ostringstream oss;
  oss << " Shape:";
  for (auto &dim : tensor_shape) {
    oss << " " << dim;
  }
  return oss.str();
}

void GenerateStackedParams(const FuncGraphPtr vmap_fg, size_t cell_size,
                           const std::vector<std::vector<AnfNodePtr>> &param_table,
                           mindspore::HashMap<std::string, ParameterPtr> *stacked_params,
                           ParamMappingVector *param_mapping_table) {
  MS_EXCEPTION_IF_NULL(vmap_fg);
  FuncGraphPtr top_fg = vmap_fg;
  while (top_fg->parent() != nullptr) {
    top_fg = top_fg->parent();
  }

  ShapeVector tensor_shape;
  TypeId tensor_type = kNumberTypeFloat32;
  std::string param_name = "";
  for (size_t i = 0; i < param_table[0].size(); ++i) {
    std::vector<AnfNodePtr> param;
    std::string orig_param_name = "";
    for (size_t j = 0; j < param_table.size(); ++j) {
      auto param_node = dyn_cast<Parameter>(param_table[j][i]);
      (void)param.emplace_back(param_node);
      MS_EXCEPTION_IF_NULL(param_node);
      auto default_param = param_node->default_param();
      MS_EXCEPTION_IF_NULL(default_param);
      auto param_tensor = dyn_cast<tensor::Tensor>(default_param);
      MS_EXCEPTION_IF_NULL(param_tensor);
      std::regex match_prefix("^.*?" + std::to_string(j) + "\\.(.+)$");
      if (j == 0) {
        tensor_shape = param_tensor->shape();
        tensor_type = param_tensor->data_type();
        orig_param_name = param_node->name();
        param_name = std::regex_replace(orig_param_name, match_prefix, "vmap.$1");
      } else {
        if (tensor_type != param_tensor->data_type()) {
          MS_LOG(EXCEPTION) << "The corresponding parameter's type in each cell should be consistent, but get "
                            << TypeIdToType(tensor_type)->ToString() << " and "
                            << TypeIdToType(param_tensor->data_type())->ToString() << " for the parameter "
                            << param_name << ".";
        }
        if (tensor_shape != param_tensor->shape()) {
          MS_LOG(EXCEPTION) << "The corresponding parameter's shape in each cell should be consistent, but get "
                            << GetShapeString(tensor_shape) << " and " << GetShapeString(param_tensor->shape())
                            << " for the parameter " << param_name << ".";
        }
        if (param_name != std::regex_replace(param_node->name(), match_prefix, "vmap.$1")) {
          MS_LOG(EXCEPTION) << "The corresponding parameter's postfix name in each cell should be consistent, but get "
                            << orig_param_name << " and " << param_node->name() << ".";
        }
      }
    }
    ParameterPtr param_node = nullptr;

    ShapeVector stacked_shape(tensor_shape);
    (void)stacked_shape.insert(stacked_shape.begin(), cell_size);
    tensor::TensorPtr stacked_param_tensor = std::make_shared<tensor::Tensor>(tensor_type, stacked_shape);
    MS_EXCEPTION_IF_NULL(stacked_param_tensor);

    ParamInfoPtr param_info = std::make_shared<ParamInfo>();
    param_info->set_name(param_name);
    stacked_param_tensor->set_param_info(param_info);

    param_node = top_fg->AddFvParameter(param_name, stacked_param_tensor);
    MS_LOG(DEBUG) << "Add new parameter " << param_node->ToString() << "to the top graph " << top_fg->ToString() << ".";

    (*stacked_params)[param_name] = param_node;
    std::pair<ParameterPtr, std::vector<AnfNodePtr>> param_mapping(param_node, param);
    (void)param_mapping_table->emplace_back(param_mapping);
  }
}

void GetCellParams(const FuncGraphPtr &vmap_fg, std::vector<AnfNodePtr> *param_nodes) {
  std::set<AnfNodePtr> memo;
  auto scan_fn = [&memo, param_nodes](const FuncGraphPtr &vmap_fg) {
    auto fv_nodes = vmap_fg->free_variables();
    for (auto &pair : fv_nodes) {
      auto node = pair.first;
      if (node->isa<Parameter>() && node->cast<ParameterPtr>()->has_default() && memo.emplace(node).second) {
        (void)param_nodes->emplace_back(node);
      }
    }
  };

  scan_fn(vmap_fg);
  auto used_fgs = vmap_fg->func_graphs_used_total();
  for (auto &fg : used_fgs) {
    scan_fn(fg);
  }
}

AnfNodePtr TraverseVmapNode(CNodePtr vmap_node, size_t cell_size,
                            mindspore::HashMap<std::string, ParameterPtr> *stacked_params,
                            ParamMappingVector *param_mapping_table) {
  AnfNodePtr vmap_fn_node = nullptr;
  auto cell_list_node = vmap_node->input(1);
  CNodePtr cnode = cell_list_node->cast<CNodePtr>();
  auto inputs_size = cnode->size();
  if (inputs_size != (cell_size + 1)) {
    MS_EXCEPTION(ValueError) << "The size of CellList Node should be equal to" << (cell_size + 1) << ", but get"
                             << inputs_size << ".";
  }
  std::vector<std::vector<AnfNodePtr>> param_table(cell_size, std::vector<AnfNodePtr>());
  FuncGraphPtr vmap_fg = nullptr;
  size_t param_size = 0;
  for (size_t i = 1; i < inputs_size; i++) {
    vmap_fn_node = cnode->input(i);
    vmap_fg = GetValueNode<FuncGraphPtr>(vmap_fn_node);
    MS_EXCEPTION_IF_NULL(vmap_fg);
    GetCellParams(vmap_fg, &param_table[i - 1]);
    if (param_size == 0) {
      param_size = param_table[i - 1].size();
    } else if (param_size != param_table[i - 1].size()) {
      MS_EXCEPTION(ValueError) << "Parameter size of each cell should be consistent, but get " << param_size << " and "
                               << param_table[i - 1].size() << ".";
    }
  }

  GenerateStackedParams(vmap_fg, cell_size, param_table, stacked_params, param_mapping_table);
  return vmap_fn_node;
}
}  // namespace internal

bool ExpandVmapPrim::CheckIfEmbedMetaFgPrim(const CNodePtr &node) const {
  MS_EXCEPTION_IF_NULL(node);
  AnfNodePtr value_node = node->input(1);
  if (IsPrimitiveCNode(value_node, prim::kPrimMakeTuple)) {
    CNodePtr cnode = value_node->cast<CNodePtr>();
    value_node = cnode->input(1);
  }
  if (IsValueNode<Primitive>(value_node)) {
    return false;
  }
  auto func_graph = GetValueNode<FuncGraphPtr>(value_node);
  if (func_graph == nullptr) {
    MS_LOG(EXCEPTION) << "Unexpected meta function graph node:" << node->DebugString();
  }
  if (parallel::IsEmbedShardNode(func_graph)) {
    MS_LOG(EXCEPTION)
      << "The usage of vmap nested shard (e.g vmap(shard)) is not supported currently. Current FuncGraph: "
      << func_graph->ToString();
  }

  auto func_graph_manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(func_graph_manager);
  return func_graph_manager->func_graph_meta_fg_prim_total(func_graph);
}

bool ExpandVmapPrim::operator()(const FuncGraphPtr &, const OptimizerPtr &optimizer) {
  // Expand vmap nodes that don't have embed j or vmap nodes.
  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &vmap_node : prim_nodes_) {
    auto VmapPrim = GetValueNode<PrimitivePtr>(vmap_node->input(0));
    MS_EXCEPTION_IF_NULL(VmapPrim);
    ValuePtr in_axes = VmapPrim->GetAttr("in_axes");
    MS_EXCEPTION_IF_NULL(in_axes);
    ValuePtr out_axes = VmapPrim->GetAttr("out_axes");
    MS_EXCEPTION_IF_NULL(out_axes);
    ValuePtr cell_size_value = VmapPrim->GetAttr("cell_size");
    MS_EXCEPTION_IF_NULL(cell_size_value);
    auto cell_size = cell_size_value->isa<UInt64Imm>() ? dyn_cast<UInt64Imm>(cell_size_value)->value() : 0;

    AnfNodePtr vmap_fn_node = nullptr;

    mindspore::HashMap<std::string, ParameterPtr> stacked_params;
    // Record the stacked parameters, and the corresponding origin parameters from each cell, preserved
    // for future feedback.
    ParamMappingVector param_mapping_table;

    if (cell_size > 0) {
      // This branch handles the model ensembling parallel training case. Get one function node in the 'CellList'
      // as the vmap function, meanwhile preprocess the cells parameters to get the stacked parameters and
      // the parameters mapping table.
      vmap_fn_node = internal::TraverseVmapNode(vmap_node, cell_size, &stacked_params, &param_mapping_table);
    } else {
      vmap_fn_node = vmap_node->input(1);
    }
    MS_EXCEPTION_IF_NULL(vmap_fn_node);
    FuncGraphPtr vmap_fg = GetValueNode<FuncGraphPtr>(vmap_fn_node);
    auto users = manager->node_users()[vmap_node];
    if (users.size() < 1) {
      MS_EXCEPTION(ValueError) << "vmap_node could used by at least one CNode, but got users.size() = " << users.size()
                               << ".";
    }

    for (auto &user : users) {
      // When `vmap_node` has more than one user or `fn` has more than one user, the original function graph
      // cannot be modified directly.
      MS_LOG(DEBUG) << "Funcgraph: " << vmap_fg->ToString() << " is also used outside the scope of vmap.";
      auto vmap_fg_copy = BasicClone(vmap_fg, true);
      manager->AddFuncGraph(vmap_fg_copy);
      vmap_fn_node = NewValueNode(vmap_fg_copy);

      if (parallel::IsPynativeParallel()) {
        auto func_graph = GetValueNode<FuncGraphPtr>(vmap_fn_node);
        func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, false);
      }

      // get axis size, simultaneous correction the negative in_axes.
      auto vmap_app = user.first->cast<CNodePtr>();
      int user_index = user.second;
      if (vmap_app->size() < 1) {
        MS_LOG(EXCEPTION) << "Something went wrong, CNode vmap_app's arguments is less than 1, CNode: "
                          << vmap_app->DebugString();
      }
      size_t parameters_size = vmap_app->size() - 1;
      std::vector<size_t> orig_fg_param_info;
      (void)orig_fg_param_info.emplace_back(parameters_size);
      int axis_size = internal::GetAxisSize(vmap_app, cell_size, parameters_size, &in_axes);

      // Step1: Bind the inputs with the corresponding in_axes.
      size_t u_monad_offset = 0;
      size_t io_monad_offset = 0;
      auto bind_axes_node =
        internal::BindInAxis(vmap_app, in_axes, &u_monad_offset, &io_monad_offset, &param_mapping_table);
      MS_EXCEPTION_IF_NULL(bind_axes_node);
      MS_EXCEPTION_IF_NULL(vmap_app->abstract());
      bind_axes_node->set_abstract(vmap_app->abstract());
      (void)manager->Replace(vmap_app, bind_axes_node);
      (void)orig_fg_param_info.emplace_back(u_monad_offset);
      (void)orig_fg_param_info.emplace_back(io_monad_offset);

      // Step2: Bind the variables with the corresponding axis, and overload the original
      // operation with the VmapRule operation meanwhile transfer the axis information.
      auto expanded_vmap =
        internal::ExpandVmap(vmap_fn_node->cast<ValueNodePtr>(), optimizer->resource(), axis_size, &stacked_params);
      MS_EXCEPTION_IF_NULL(expanded_vmap);

      // Step3: Post processing of converted vmap function graph, including: MatchOutAxis and Parameter feedback.
      auto match_out_axis =
        internal::PostProcessVmap(expanded_vmap, orig_fg_param_info, out_axes, axis_size, &param_mapping_table);
      MS_EXCEPTION_IF_NULL(match_out_axis);
      manager->SetEdge(bind_axes_node, user_index, match_out_axis);
    }
    change = true;
  }
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
