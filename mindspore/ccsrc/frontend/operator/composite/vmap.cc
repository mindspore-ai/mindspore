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

#include "frontend/operator/composite/vmap.h"

#include <memory>
#include <string>
#include "pybind11/pybind11.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "base/core_ops.h"
#include "abstract/abstract_value.h"
#include "abstract/abstract_function.h"
#include "pipeline/jit/parse/parse_base.h"
#include "pipeline/jit/parse/parse.h"
#include "pipeline/jit/parse/resolve.h"
#include "pipeline/jit/pipeline.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
// namespace to support composite operators definition
namespace prim {
CNodePtr VmapMatchOutAxis::GenerateFuncGraphInnerBroadcastAxis(
  const AnfNodePtr &inputs, const AnfNodePtr &out_axis, const AnfNodePtr &axis_size,
  const AbstractBasePtr &inputs_abstract_elements_begin) const {
  std::vector<AnfNodePtr> value_cnode_inputs;
  (void)value_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
  (void)value_cnode_inputs.emplace_back(inputs);
  (void)value_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
  auto value_cnode = fg_->NewCNode(value_cnode_inputs);
  std::vector<AnfNodePtr> dim_cnode_inputs;
  (void)dim_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
  (void)dim_cnode_inputs.emplace_back(inputs);
  (void)dim_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(1)));
  auto dim_cnode = fg_->NewCNode(dim_cnode_inputs);

  std::vector<AnfNodePtr> sub_inputs_cnode_inputs;
  (void)sub_inputs_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  auto inputs_abstract_elements_begin_tuple = dyn_cast<abstract::AbstractTuple>(inputs_abstract_elements_begin);
  auto inputs_abstract_elements_begin_tuple_elements = inputs_abstract_elements_begin_tuple->elements();
  // inputs: ((x, y), None) -> ((x, None), (y, None)).
  int64_t begin_tuple_size = static_cast<int64_t>(inputs_abstract_elements_begin_tuple_elements.size());
  for (int64_t i = 0; i < begin_tuple_size; ++i) {
    std::vector<AnfNodePtr> cur_tuple_getitem_inputs;
    (void)cur_tuple_getitem_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    (void)cur_tuple_getitem_inputs.emplace_back(value_cnode);
    (void)cur_tuple_getitem_inputs.emplace_back(NewValueNode(i));
    auto cur_value_cnode = fg_->NewCNode(cur_tuple_getitem_inputs);
    std::vector<AnfNodePtr> cur_make_tuple_cnode_inputs;
    (void)cur_make_tuple_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    (void)cur_make_tuple_cnode_inputs.emplace_back(cur_value_cnode);
    (void)cur_make_tuple_cnode_inputs.emplace_back(dim_cnode);
    auto cur_make_tuple_cnode = fg_->NewCNode(cur_make_tuple_cnode_inputs);
    (void)sub_inputs_cnode_inputs.emplace_back(cur_make_tuple_cnode);
  }
  auto sub_inputs_cnode = fg_->NewCNode(sub_inputs_cnode_inputs);
  std::vector<AnfNodePtr> out_cnode_inputs;
  (void)out_cnode_inputs.emplace_back(NewValueNode(std::make_shared<VmapMatchOutAxis>("VmapMatchOutAxis")));
  (void)out_cnode_inputs.emplace_back(sub_inputs_cnode);
  (void)out_cnode_inputs.emplace_back(out_axis);
  (void)out_cnode_inputs.emplace_back(axis_size);
  return fg_->NewCNode(out_cnode_inputs);
}

CNodePtr VmapMatchOutAxis::GenerateFuncGraphInnerSingleElement(
  const AnfNodePtr &inputs, const AnfNodePtr &out_axis, const AnfNodePtr &axis_size,
  const AbstractBasePtr &inputs_abstract_elements_end) const {
  std::vector<AnfNodePtr> value_cnode_inputs;
  (void)value_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
  (void)value_cnode_inputs.emplace_back(inputs);
  (void)value_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
  auto value_cnode = fg_->NewCNode(value_cnode_inputs);
  std::vector<AnfNodePtr> out_cnode_inputs;
  if (inputs_abstract_elements_end->isa<abstract::AbstractNone>()) {
    constexpr char kVmapFunctionModelName[] = "mindspore.ops._vmap";
    const py::function broadcast_by_axis = python_adapter::GetPyFn(kVmapFunctionModelName, "_broadcast_by_axis");
    auto broadcast_by_axis_fg = parse::ParsePythonCode(broadcast_by_axis);
    (void)out_cnode_inputs.emplace_back(NewValueNode(broadcast_by_axis_fg));
    (void)out_cnode_inputs.emplace_back(value_cnode);
    (void)out_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(0)));
    (void)out_cnode_inputs.emplace_back(axis_size);
  } else {
    std::vector<AnfNodePtr> dim_cnode_inputs;
    (void)dim_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    (void)dim_cnode_inputs.emplace_back(inputs);
    (void)dim_cnode_inputs.emplace_back(NewValueNode(static_cast<int64_t>(1)));
    auto dim_cnode = fg_->NewCNode(dim_cnode_inputs);
    constexpr char kVmapFunctionModelName[] = "mindspore.numpy";
    const py::function move_axis = python_adapter::GetPyFn(kVmapFunctionModelName, "moveaxis");
    auto move_axis_fg = parse::ParsePythonCode(move_axis);
    (void)out_cnode_inputs.emplace_back(NewValueNode(move_axis_fg));
    (void)out_cnode_inputs.emplace_back(value_cnode);
    (void)out_cnode_inputs.emplace_back(dim_cnode);
    (void)out_cnode_inputs.emplace_back(out_axis);
  }
  return fg_->NewCNode(out_cnode_inputs);
}

CNodePtr VmapMatchOutAxis::GenerateFuncGraphInnerAllTuple(const AnfNodePtr &inputs, const AnfNodePtr &out_axis,
                                                          const AnfNodePtr &axis_size,
                                                          const AbstractBasePtrList &inputs_abstract_elements,
                                                          const AbstractBasePtr &out_axes_abstract) const {
  bool is_out_axes_tuple = out_axes_abstract->isa<abstract::AbstractTuple>();
  abstract::AbstractTuplePtr out_axes_abstract_tuple = nullptr;
  AbstractBasePtrList out_axes_abstract_elements;
  auto inputs_abstract_elements_size = inputs_abstract_elements.size();
  if (is_out_axes_tuple) {
    out_axes_abstract_tuple = dyn_cast<abstract::AbstractTuple>(out_axes_abstract);
    out_axes_abstract_elements = out_axes_abstract_tuple->elements();
    if (out_axes_abstract_elements.size() != inputs_abstract_elements_size) {
      MS_LOG(EXCEPTION) << "The length of out_axes and inputs do not match. ";
    }
  }
  std::vector<AnfNodePtr> vals_out_tuple_cnode_inputs;
  (void)vals_out_tuple_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  constexpr size_t kEachInputsSize = 2;
  // inputs: (((x1, x1_axis), (x2, x2_axis)), ((y1, y2), y_axis), (z, z_axis))
  for (int64_t i = 0; i < static_cast<int64_t>(inputs_abstract_elements_size); ++i) {
    std::vector<AnfNodePtr> each_input_cnode_inputs;
    (void)each_input_cnode_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
    (void)each_input_cnode_inputs.emplace_back(inputs);
    (void)each_input_cnode_inputs.emplace_back(NewValueNode(i));
    auto each_input_cnode = fg_->NewCNode(each_input_cnode_inputs);
    AnfNodePtr dst_cnode = nullptr;
    if (is_out_axes_tuple) {
      dst_cnode = fg_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), out_axis, NewValueNode(i)});
    } else {
      dst_cnode = out_axis;
    }
    auto each_input_abstract = inputs_abstract_elements[i];
    AbstractBasePtr dst_abstract = is_out_axes_tuple ? out_axes_abstract_elements[i] : out_axes_abstract;
    auto each_input_abstract_tuple = dyn_cast<abstract::AbstractTuple>(each_input_abstract);
    MS_EXCEPTION_IF_NULL(each_input_abstract_tuple);
    auto each_inputs_abstract_elements = each_input_abstract_tuple->elements();
    auto each_inputs_abstract_elements_size = each_inputs_abstract_elements.size();
    if (each_inputs_abstract_elements_size == 0) {
      MS_LOG(EXCEPTION) << "each_inputs_abstract_elements_size is empty";
    }
    auto each_inputs_abstract_elements_begin = each_inputs_abstract_elements[0];
    if (each_inputs_abstract_elements_begin->isa<abstract::AbstractTuple>()) {
      auto each_inputs_abstract_elements_end = each_inputs_abstract_elements.back();
      if (each_inputs_abstract_elements_end->isa<abstract::AbstractTuple>()) {
        // current each input: ((x1, x1_axis), (x2, x2_axis)).
        std::vector<AnfNodePtr> out_cnode_inputs;
        (void)out_cnode_inputs.emplace_back(NewValueNode(std::make_shared<VmapMatchOutAxis>("VmapMatchOutAxis")));
        (void)out_cnode_inputs.emplace_back(each_input_cnode);
        (void)out_cnode_inputs.emplace_back(dst_cnode);
        (void)out_cnode_inputs.emplace_back(axis_size);
        (void)vals_out_tuple_cnode_inputs.emplace_back(fg_->NewCNode(out_cnode_inputs));
      } else {
        // current each input: ((y1, y2), y_axis).
        auto out_cnode = GenerateFuncGraphInnerBroadcastAxis(each_input_cnode, dst_cnode, axis_size,
                                                             each_inputs_abstract_elements_begin);
        (void)vals_out_tuple_cnode_inputs.emplace_back(out_cnode);
      }
    } else {
      // current each input: (z, z_axis).
      if (each_inputs_abstract_elements_size != kEachInputsSize) {
        MS_LOG(EXCEPTION) << "each input with no tuple should have only two elements.";
      }
      auto val_cnode =
        fg_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), each_input_cnode, NewValueNode(static_cast<int64_t>(0))});
      auto src_cnode =
        fg_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), each_input_cnode, NewValueNode(static_cast<int64_t>(1))});
      auto val_abstract = each_inputs_abstract_elements[0];
      auto src_abstract = each_inputs_abstract_elements[1];
      CNodePtr out_cnode = nullptr;
      if (src_abstract->isa<abstract::AbstractNone>() && !dst_abstract->isa<abstract::AbstractNone>()) {
        constexpr char kVmapFunctionModelName[] = "mindspore.ops._vmap";
        const py::function broadcast_by_axis = python_adapter::GetPyFn(kVmapFunctionModelName, "_broadcast_by_axis");
        auto broadcast_by_axis_fg = parse::ParsePythonCode(broadcast_by_axis);
        out_cnode = fg_->NewCNode({NewValueNode(broadcast_by_axis_fg), val_cnode, dst_cnode, axis_size});
      } else if (!src_abstract->isa<abstract::AbstractNone>() && dst_abstract->isa<abstract::AbstractNone>()) {
        MS_LOG(EXCEPTION) << "It is invalid that source is not None and dst is None.";
      } else if (src_abstract->isa<abstract::AbstractNone>() && dst_abstract->isa<abstract::AbstractNone>()) {
        out_cnode = val_cnode;
      } else {
        constexpr char kVmapFunctionModelName[] = "mindspore.numpy";
        const py::function move_axis = python_adapter::GetPyFn(kVmapFunctionModelName, "moveaxis");
        auto move_axis_fg = parse::ParsePythonCode(move_axis);
        out_cnode = fg_->NewCNode({NewValueNode(move_axis_fg), val_cnode, src_cnode, dst_cnode});
      }
      vals_out_tuple_cnode_inputs.emplace_back(out_cnode);
    }
  }
  return fg_->NewCNode(vals_out_tuple_cnode_inputs);
}

FuncGraphPtr VmapMatchOutAxis::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  auto args_spec_list_size = args_spec_list.size();
  constexpr size_t kMetaFGInputSize = 3;
  if (args_spec_list_size != kMetaFGInputSize) {
    MS_LOG(EXCEPTION) << "The number of inputs to VmapMatchOutAxis should be 3, but got " << args_spec_list_size << ".";
  }
  auto inputs_abstract = args_spec_list[0];
  auto out_axes_abstract = args_spec_list[1];
  auto axis_size_abstract = args_spec_list[2];
  if (!inputs_abstract->isa<abstract::AbstractTuple>()) {
    MS_LOG(EXCEPTION) << "The first input to VmapMatchOutAxis is vmap_inputs and should be a tuple but got "
                      << inputs_abstract->ToString() << ".";
  }
  auto out_axes_abstract_value = out_axes_abstract->BuildValue();
  if (out_axes_abstract_value == nullptr || out_axes_abstract_value == kAnyValue) {
    MS_LOG(EXCEPTION) << "The second input to VmapMatchOutAxis is out_axes and should be a constant value.";
  }
  auto axis_size_value = axis_size_abstract->BuildValue();
  if (axis_size_value == nullptr || !axis_size_value->isa<Int64Imm>()) {
    MS_LOG(EXCEPTION) << "The third input to VmapMatchOutAxis is axis size and should be a constant unsigned int64 "
                      << " value.";
  }
  auto inputs = fg_->add_parameter();
  auto out_axis = fg_->add_parameter();
  auto axis_size = fg_->add_parameter();

  auto inputs_abstract_tuple = dyn_cast<abstract::AbstractTuple>(inputs_abstract);
  auto inputs_abstract_elements = inputs_abstract_tuple->elements();
  auto inputs_abstract_elements_size = inputs_abstract_elements.size();
  if (inputs_abstract_elements_size == 0) {
    MS_LOG(EXCEPTION) << "The input to VmapMatchOutAxis is empty";
  }
  auto inputs_abstract_elements_begin = inputs_abstract_elements[0];
  auto inputs_abstract_elements_end = inputs_abstract_elements[inputs_abstract_elements_size - 1];
  CNodePtr out_cnode = nullptr;
  constexpr size_t kInputAbstractElementsSize = 2;
  if (inputs_abstract_elements_begin->isa<abstract::AbstractTuple>() &&
      inputs_abstract_elements_end->isa<abstract::AbstractTuple>()) {
    // All elements in inputs are tuple. The format of input is ((x, x_axis), (y, y_axis), (z, z_axis)).
    out_cnode =
      GenerateFuncGraphInnerAllTuple(inputs, out_axis, axis_size, inputs_abstract_elements, out_axes_abstract);
  } else if (inputs_abstract_elements_begin->isa<abstract::AbstractTuple>() &&
             !inputs_abstract_elements_end->isa<abstract::AbstractTuple>()) {
    // The last element of input is axis. The format is ((x, y), None).
    if (inputs_abstract_elements_size != kInputAbstractElementsSize) {
      MS_LOG(EXCEPTION) << "The length of elements should be 2 but got: " << inputs_abstract_elements_size << ".";
    }
    out_cnode = GenerateFuncGraphInnerBroadcastAxis(inputs, out_axis, axis_size, inputs_abstract_elements_begin);
  } else {
    // Single tuple element. (x, None)
    if (inputs_abstract_elements_size != kInputAbstractElementsSize) {
      MS_LOG(EXCEPTION) << "The length of elements should be 2 but got: " << inputs_abstract_elements_size << ".";
    }
    out_cnode = GenerateFuncGraphInnerSingleElement(inputs, out_axis, axis_size, inputs_abstract_elements_end);
  }
  fg_->set_output(out_cnode);
  return fg_;
}

FuncGraphPtr VmapGeneralPreprocess::GenerateFuncGraph(const AbstractBasePtrList &args_spec_list) {
  FuncGraphPtr fg = std::make_shared<FuncGraph>();
  auto prim = fg->add_parameter();
  auto args_size = args_spec_list.size();
  if (args_size <= 1) {
    MS_LOG(EXCEPTION) << "The length of input to VmapGeneralPreprocess must be greater than 1";
  }
  bool wrapped_tuple = false;
  int64_t inputs_size = args_size - 1;
  uint32_t offset = 1;
  auto get_tuple_elements = [args_size, &wrapped_tuple, &inputs_size,
                             &offset](const AbstractBasePtrList &args_spec_list) -> const AbstractBasePtrList & {
    if (args_size == 2) {
      auto arg = args_spec_list[1];
      if (!arg->isa<abstract::AbstractTuple>()) {
        MS_LOG(EXCEPTION) << "The second input to VmapGeneralPreprocess should be AbstractTuple but got: "
                          << arg->ToString() << ".";
      }
      auto arg_tuple = arg->cast<abstract::AbstractTuplePtr>();
      const auto &arg_tuple_elements = arg_tuple->elements();
      if (arg_tuple_elements.back()->isa<abstract::AbstractTuple>()) {
        // Operators with indefinite inputs length, such as `AddN`, whose inputs is wrapped
        // into a tuple. We need to process the internal elements separately and then re-wrap
        // them into tuple. Handle case such as args:(((A, 0), (B, 1), (C, None)),). Which
        // different from the case with single input parameter ((A, 0),).
        wrapped_tuple = true;
        inputs_size = arg_tuple_elements.size();
        offset = 0;
        return arg_tuple_elements;
      }
    }
    return args_spec_list;
  };
  auto tuple_elements = get_tuple_elements(args_spec_list);

  constexpr int64_t val_index = 0;
  constexpr int64_t dim_index = 1;
  bool is_all_none = true;
  constexpr size_t kCurTupleSize = 2;
  for (int64_t i = 0; i < inputs_size; ++i) {
    auto cur_arg = tuple_elements[i + offset];
    if (!cur_arg->isa<abstract::AbstractTuple>()) {
      MS_LOG(EXCEPTION) << "The " << i + offset
                        << "th input to VmapGeneralPreprocess should be AbstractTuple but got: " << cur_arg->ToString()
                        << ".";
    }
    auto cur_arg_tuple = cur_arg->cast<abstract::AbstractTuplePtr>();
    auto cur_arg_tuple_elements = cur_arg_tuple->elements();
    if (cur_arg_tuple_elements.size() != kCurTupleSize) {
      MS_LOG(EXCEPTION) << "The " << i + offset << "th input to VmapGeneralPreprocess should be a tuple with two "
                        << "elements but got " << cur_arg_tuple_elements.size() << " elements.";
    }
    if (!cur_arg_tuple_elements[dim_index]->isa<abstract::AbstractNone>()) {
      MS_LOG(INFO) << "The " << i + offset << "th input to VmapGeneralPreprocess has not None dim value.";
      is_all_none = false;
      break;
    }
  }

  std::vector<AnfNodePtr> output_cnode_inputs;
  (void)output_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  if (!is_all_none) {
    for (size_t i = 1; i < args_size; ++i) {
      (void)fg->add_parameter();
    }
    (void)output_cnode_inputs.emplace_back(NewValueNode(false));
    (void)output_cnode_inputs.emplace_back(NewValueNode(kNone));
  } else {
    std::vector<AnfNodePtr> prim_output_cnode_inputs;
    (void)prim_output_cnode_inputs.emplace_back(prim);
    if (wrapped_tuple) {
      auto val_in_param = fg->add_parameter();
      std::vector<AnfNodePtr> prim_inputs_cnode_inputs;
      (void)prim_inputs_cnode_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
      for (int64_t i = 0; i < inputs_size; ++i) {
        auto val_in_cnode = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), val_in_param, NewValueNode(i)});
        auto val_cnode = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), val_in_cnode, NewValueNode(val_index)});
        (void)prim_inputs_cnode_inputs.emplace_back(val_cnode);
      }
      auto prim_inputs_cnode = fg->NewCNode(prim_inputs_cnode_inputs);
      (void)prim_output_cnode_inputs.emplace_back(prim_inputs_cnode);
    } else {
      for (int64_t i = 0; i < inputs_size; ++i) {
        auto val_in_param = fg->add_parameter();
        auto val_cnode = fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), val_in_param, NewValueNode(val_index)});
        (void)prim_output_cnode_inputs.emplace_back(val_cnode);
      }
    }
    auto prim_output_cnode = fg->NewCNode(prim_output_cnode_inputs);
    const char kVmapFunctionModelName[] = "mindspore.ops._vmap";
    const py::function bind_all_none_fn = python_adapter::GetPyFn(kVmapFunctionModelName, "vmap_bind_all_none");
    auto bind_all_none_fg = parse::ParsePythonCode(bind_all_none_fn);
    auto bind_all_none_cnode = fg->NewCNode({NewValueNode(bind_all_none_fg), prim_output_cnode});
    (void)output_cnode_inputs.emplace_back(NewValueNode(true));
    (void)output_cnode_inputs.emplace_back(bind_all_none_cnode);
  }
  auto output_cnode = fg->NewCNode(output_cnode_inputs);
  fg->set_output(output_cnode);
  return fg;
}

REGISTER_PYBIND_DEFINE(VmapGeneralPreprocess_, ([](const py::module *m) {
                         (void)py::class_<VmapGeneralPreprocess, MetaFuncGraph, std::shared_ptr<VmapGeneralPreprocess>>(
                           *m, "VmapGeneralPreprocess_")
                           .def(py::init<std::string &>(), py::arg("fn"));
                       }));
}  // namespace prim
}  // namespace mindspore
