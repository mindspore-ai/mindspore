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

#include "include/common/expander/core/emitter.h"

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>
#include "ops/primitive_c.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace expander {
namespace {
std::pair<bool, std::vector<int64_t>> GetIntList(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->get());
  ValuePtr value_ptr = nullptr;
  if (node->isa<ValueNode>()) {
    value_ptr = node->get<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(value_ptr);
    if (value_ptr->isa<ValueSequence>() || value_ptr->isa<Scalar>()) {
      return std::make_pair(true, CheckAndConvertUtils::CheckIntOrTupleInt("value", value_ptr, "GetIntList"));
    }
  } else {
    auto abstract = node->get()->abstract();
    if (abstract != nullptr) {
      value_ptr = abstract->BuildValue();
    }
  }
  if (value_ptr != nullptr && value_ptr->isa<tensor::Tensor>()) {
    auto tensor = value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    // In pynative mode, need data sync before get tensor value, otherwise the tensor value may be undefined.
    tensor->data_sync();
    return std::make_pair(true, CheckAndConvertUtils::CheckTensorIntValue("value", value_ptr, "GetIntList"));
  }
  return std::make_pair(false, std::vector<int64_t>{});
}
}  // namespace

NodePtr Emitter::Emit(const std::string &op_name, const NodePtrList &inputs, const DAttr &attrs) const {
  const auto &op_primc_fns = ops::OpPrimCRegister::GetInstance().GetPrimCMap();
  const auto iter = op_primc_fns.find(op_name);
  PrimitivePtr primc = nullptr;
  if (iter == op_primc_fns.end()) {
    primc = std::make_shared<ops::PrimitiveC>(op_name);
    primc->SetAttrs(attrs);
  } else {
    primc = iter->second();
    if (!attrs.empty()) {
      for (auto &[k, v] : attrs) {
        primc->set_attr(k, v);
      }
    }
  }
  return EmitOp(primc, inputs, attrs);
}

NodePtr Emitter::EmitOp(const PrimitivePtr &prim, const NodePtrList &inputs, const DAttr &attrs) const {
  AnfNodePtrList cnode_inputs = {NewValueNode(prim)};
  cnode_inputs.reserve(inputs.size() + 1);
  (void)std::transform(inputs.cbegin(), inputs.cend(), std::back_inserter(cnode_inputs), [](const NodePtr &no) {
    MS_EXCEPTION_IF_NULL(no);
    return no->get();
  });
  auto cnode = func_graph_->NewCNode(cnode_inputs);
  if (scope_ != nullptr) {
    cnode->set_scope(scope_);
  }
  auto node = NewNode(cnode->cast<AnfNodePtr>());
  infer_->Infer(node);
  return node;
}

NodePtr Emitter::EmitValue(const ValuePtr &value) const {
  auto node = NewNode(NewValueNode(value));
  infer_->Infer(node);
  return node;
}

NodePtr Emitter::Exp(const NodePtr &x) const {
  return Emit(kExpOpName, {x},
              {{"base", MakeValue<float>(-1.0)}, {"scale", MakeValue<float>(1.0)}, {"shift", MakeValue<float>(0.0)}});
}

NodePtr Emitter::Log(const NodePtr &x) const {
  return Emit(kLogOpName, {x},
              {{"base", MakeValue<float>(-1.0)},
               {"scale", MakeValue<float>(1.0)},
               {"shift", MakeValue<float>(0.0)},
               {"cust_aicpu", MakeValue(kLogOpName)}});
}

NodePtr Emitter::Cast(const NodePtr &node, const TypePtr &type) const {
  // do not emit a node when the dst type is the same as src type
  if (node->dtype()->type_id() == type->type_id()) {
    return node;
  }
  return Emit("Cast", {node, EmitValue(type)});
}

NodePtr Emitter::Reshape(const NodePtr &node, const NodePtr &shape) const {
  MS_EXCEPTION_IF_NULL(node);
  auto [success, dst_shape] = GetIntList(shape);
  if (!success) {
    return Emit(prim::kReshape, {node, shape});
  }
  auto node_shape = node->shape();
  if (dst_shape.size() != node_shape.size()) {
    return Emit(prim::kReshape, {node, shape});
  }
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    if (dst_shape[i] != node_shape[i] && dst_shape[i] != -1) {
      return Emit(prim::kReshape, {node, shape});
    }
  }
  return node;
}

NodePtr Emitter::MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) const {
  return UnifyDtypeAndEmit(prim::kPrimMatMul->name(), a, b,
                           {{"transpose_x1", MakeValue(transpose_a)},
                            {"transpose_x2", MakeValue(transpose_b)},
                            {"transpose_a", MakeValue(transpose_a)},
                            {"transpose_b", MakeValue(transpose_b)}});
}

NodePtr Emitter::BatchMatMul(const NodePtr &a, const NodePtr &b, bool transpose_a, bool transpose_b) const {
  return UnifyDtypeAndEmit(prim::kPrimBatchMatMul->name(), a, b,
                           {{"adj_x1", MakeValue(transpose_a)},
                            {"adj_x2", MakeValue(transpose_b)},
                            {"transpose_a", MakeValue(transpose_a)},
                            {"transpose_b", MakeValue(transpose_b)}});
}

NodePtr Emitter::Transpose(const NodePtr &node, const NodePtr &perm) const {
  auto [success, perm_list] = GetIntList(perm);
  if (!success) {
    return Emit(kTransposeOpName, {node, perm});
  }
  // perm like [0, 1, 2, 3] does not need transpose.
  auto n = SizeToLong(perm_list.size());
  for (size_t i = 0; i < perm_list.size(); ++i) {
    // perm value may be negative, e.g. [0, -3, 2, 3] is equal to [0, 1, 2, 3]
    auto perm_i = perm_list[i] < 0 ? (perm_list[i] + n) : perm_list[i];
    if (perm_i != static_cast<int64_t>(i)) {
      return Emit(kTransposeOpName, {node, perm});
    }
  }
  return node;
}

NodePtr Emitter::Tile(const NodePtr &node, const NodePtr &multiples) const {
  auto [success, multiples_list] = GetIntList(multiples);
  if (!success) {
    return Emit(kTileOpName, {node, multiples});
  }
  bool is_all_one = std::all_of(multiples_list.begin(), multiples_list.end(), [](int64_t shp) { return shp == 1; });
  if (is_all_one && node->shape().size() >= multiples_list.size()) {
    return node;
  }
  return Emit(kTileOpName, {node, multiples});
}

NodePtr Emitter::ZerosLike(const NodePtr &node) const {
  if (node->isa<ValueNode>()) {
    if (node->dtype()->type_id() == kMetaTypeNone) {
      return Emit(prim::kZerosLike, {Tensor(0)});
    }
    auto value_node = node->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto v = value_node->value();
    MS_EXCEPTION_IF_NULL(v);
    if (v->isa<ValueSequence>()) {
      auto sh = GetValue<std::vector<int64_t>>(v);
      return Emit(prim::kZerosLike, {Tensor(sh)});
    } else if (v->isa<Scalar>() || v->isa<Type>()) {
      return Emit(prim::kZerosLike, {Tensor(0, v->type())});
    } else if (v->isa<Monad>()) {
      return Emit(prim::kZerosLike, {Tensor(0)});
    }
  }

  auto abs = node->abstract();
  MS_EXCEPTION_IF_NULL(abs);

  if (abs->isa<abstract::AbstractTensor>()) {
    return Emit(prim::kZerosLike, {node});
  } else if (abs->isa<abstract::AbstractMonad>() || abs->isa<abstract::AbstractType>() ||
             abs->isa<abstract::AbstractNone>()) {
    // To prevent return input directly, and return a operator for latter cnode link.
    return node;
  } else if (abs->isa<abstract::AbstractSequence>()) {
    auto sequence_abs = abs->cast<abstract::AbstractSequencePtr>();
    if (sequence_abs->empty()) {
      return node;
    }
    return Emit(prim::kSequenceZerosLike, {node});
  } else if (abs->isa<abstract::AbstractScalar>()) {
    return Emit(prim::kZerosLike, {Tensor(0, abs->BuildType())});
  }

  MS_LOG(EXCEPTION) << "Cannot emit ZerosLike for " << node->get()->ToString() << " with abstract " << abs;
}

NodePtr Emitter::Fill(double value, const ShapeVector &shape, TypeId data_type) const {
  size_t data_num = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
  std::vector<double> data(data_num, value);
  return Tensor(data_type, shape, &data[0], TypeId::kNumberTypeFloat64);
}

NodePtr Emitter::Fill(int64_t value, const ShapeVector &shape, TypeId data_type) const {
  size_t data_num = LongToSize(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
  std::vector<int64_t> data(data_num, value);
  return Tensor(data_type, shape, &data[0], TypeId::kNumberTypeInt64);
}

std::pair<bool, ShapeVector> Emitter::NeedReduce(const ShapeVector &shape, const std::vector<int64_t> &axis,
                                                 bool keep_dim) const {
  if (IsDynamic(shape)) {
    return std::make_pair(true, shape);
  }
  if (shape.empty()) {
    return std::make_pair(false, shape);
  }
  auto rank = SizeToLong(shape.size());
  auto real_axis = axis;
  if (real_axis.empty()) {
    // all reduce
    for (int64_t i = 0; i < rank; ++i) {
      real_axis.push_back(i);
    }
  }
  std::unordered_set<size_t> uniq_axis;
  for (size_t i = 0; i < real_axis.size(); ++i) {
    if (real_axis[i] < -rank || real_axis[i] >= rank) {
      MS_EXCEPTION(ValueError) << "Reduce axis[" << i << "] is " << real_axis[i] << ", which is out of range [-" << rank
                               << ", " << rank << ") for shape: " << shape;
    }
    auto axis_i = real_axis[i] < 0 ? real_axis[i] + rank : real_axis[i];
    (void)uniq_axis.insert(LongToSize(axis_i));
  }
  // Calc reduce output shape
  ShapeVector out_shape;
  bool need_reduce = false;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (uniq_axis.find(i) == uniq_axis.end()) {
      // not reduce axis
      out_shape.push_back(shape[i]);
    } else {
      // reduce axis
      if (shape[i] != 1) {
        need_reduce = true;
      }
      if (keep_dim) {
        out_shape.push_back(1);
      }
    }
  }
  return std::make_pair(need_reduce, out_shape);
}

std::pair<bool, ShapeVector> Emitter::NeedReduce(const NodePtr &shape, const NodePtr &axis, bool keep_dim) const {
  if (shape->isa<ValueNode>() && axis->isa<ValueNode>()) {
    auto shape_node = shape->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(shape_node);
    auto shape_v = shape_node->value();
    MS_EXCEPTION_IF_NULL(shape_v);
    auto shape_value = GetValue<ShapeVector>(shape_v);
    auto axis_node = axis->get<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(axis_node);
    auto axis_v = axis_node->value();
    MS_EXCEPTION_IF_NULL(axis_v);
    auto axis_value = GetValue<ShapeVector>(axis_v);
    return NeedReduce(shape_value, axis_value, keep_dim);
  }
  ShapeVector v;
  return std::make_pair(true, v);
}

NodePtr Emitter::ReduceSum(const NodePtr &x, const ShapeVector &axis, bool keep_dims) const {
  MS_EXCEPTION_IF_NULL(x);
  auto need_reduce = NeedReduce(x->shape(), axis, keep_dims);
  if (!need_reduce.first) {
    return Reshape(x, need_reduce.second);
  }
#ifdef WITH_BACKEND
  const auto &shape = x->shape();
  auto real_axis = axis;
  if (axis.empty()) {
    for (int64_t i = 0; i < SizeToLong(shape.size()); i++) {
      (void)real_axis.emplace_back(i);
    }
  }
  return Emit(prim::kPrimReduceSum->name(), {x, Value<ShapeVector>(real_axis)}, {{"keep_dims", MakeValue(keep_dims)}});
#endif
  return Emit(prim::kPrimReduceSum->name(), {x, Value<ShapeVector>(axis)}, {{"keep_dims", MakeValue(keep_dims)}});
}

NodePtrList Emitter::ShapeCalc(const NodePtrList &inputs, const ops::ShapeFunc &shape_func,
                               const ops::InferFunc &infer_func,
                               const std::vector<int64_t> &value_depend_indices) const {
  MS_EXCEPTION_IF_NULL(shape_func);
  MS_EXCEPTION_IF_NULL(infer_func);
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "ShapeCalc got empty inputs";
  }
  std::unordered_set<int64_t> indices(value_depend_indices.begin(), value_depend_indices.end());
  std::unordered_set<int64_t> const_args_indices;
  ShapeArray const_args(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    if (indices.find(static_cast<int64_t>(i)) == indices.end()) {
      // input[i]'s shape is used
      auto input_shape = inputs[i]->shape();
      if (!IsDynamic(input_shape)) {
        const_args_indices.insert(i);
        const_args[i] = input_shape;
      }
    } else {
      // input[i]'s value is used
      auto [success, vec] = GetIntList(inputs[i]);
      if (success) {
        const_args_indices.insert(i);
        const_args[i] = vec;
      }
    }
  }

  NodePtrList res;
  if (const_args_indices.size() == inputs.size()) {
    // Directly execute the lambda function only when all inputs are static
    auto out = shape_func(const_args);
    (void)std::transform(out.begin(), out.end(), std::back_inserter(res),
                         [this](const ShapeVector &sh) { return Value(sh); });
    return res;
  }

  NodePtrList args(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (const_args_indices.find(i) != indices.end()) {
      args[i] = Value(const_args[i]);
    } else if (indices.find(static_cast<int64_t>(i)) == indices.end()) {
      args[i] = Emit("TensorShape", {inputs[i]});
    } else {
      args[i] = inputs[i];
    }
  }
  auto out = Emit(ops::kNameShapeCalc, args,
                  {{ops::kAttrShapeFunc, std::make_shared<ops::ShapeFunction>(shape_func)},
                   {ops::kAttrInferFunc, std::make_shared<ops::InferFunction>(infer_func)},
                   {ops::kAttrValueDependIndices, MakeValue(value_depend_indices)},
                   {kAttrPrimitiveTarget, MakeValue("CPU")}});
  MS_EXCEPTION_IF_NULL(out);
  MS_EXCEPTION_IF_NULL(out->get());
  auto abs = out->abstract();
  MS_EXCEPTION_IF_NULL(abs);
  if (abs->isa<abstract::AbstractTuple>()) {
    auto abstract_tuple = abs->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abstract_tuple);
    for (size_t i = 0; i < abstract_tuple->size(); ++i) {
      res.push_back(TupleGetItem(out, i));
    }
  } else {
    res.push_back(out);
  }
  return res;
}  // namespace expander

std::tuple<NodePtr, NodePtr> Emitter::UnifyDtype2(const NodePtr &lhs, const NodePtr &rhs) const {
  auto it1 = type_map_.find(lhs->dtype()->type_id());
  auto it2 = type_map_.find(rhs->dtype()->type_id());
  if (it1 == type_map_.end() || it2 == type_map_.end() || it1->second == it2->second) {
    return {lhs, rhs};
  }
  if (it1->second < it2->second) {
    return {this->Cast(lhs, rhs->dtype()), rhs};
  }
  return {lhs, this->Cast(rhs, lhs->dtype())};
}

class Emitter::CtrlFlowBlock {
 public:
  explicit CtrlFlowBlock(const Emitter *emitter) : emitter_(emitter) { MS_EXCEPTION_IF_NULL(emitter); }
  ~CtrlFlowBlock() = default;
  NodePtr IfThenElse(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case) {
    auto tb = BuildSubgraph(true_case);
    auto fb = BuildSubgraph(false_case);
    auto s = emitter_->Emit("Switch", {cond, tb, fb});
    auto cnode = emitter_->func_graph_->NewCNode({s->get()});
    cnode->set_abstract(out_abstract_);
    auto node = emitter_->NewNode(cnode->cast<AnfNodePtr>());
    return node;
  }

  NodePtr While(const NodePtr &cond, const BlockFunc &while_body_func, const NodePtrList &init_list) {
    auto while_fg = std::make_shared<FuncGraph>();
    MS_EXCEPTION_IF_NULL(while_fg);
    auto cond_cnode = cond->get<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cond_cnode);
    cond_cnode->set_func_graph(while_fg);
    auto while_fg_emitter = std::make_unique<Emitter>(while_fg, std::make_shared<CppInferWithPartial>());
    AnfNodePtrList main_while_fg_inputs = {NewValueNode(while_fg)};
    std::map<AnfNodePtr, ParameterPtr> param_map;
    auto replace_by_param = [&main_while_fg_inputs, &param_map, &while_fg](const AnfNodePtr &inp) {
      auto &param = param_map[inp];
      if (param == nullptr) {
        param = while_fg->add_parameter();
        param->set_abstract(inp->abstract());
        (void)main_while_fg_inputs.emplace_back(inp);
      }
      return param;
    };

    auto empty_body_func = [&init_list](const Emitter *e) { return init_list; };
    auto empty_body_fg_with_inputs = BuildSubgraphOfPartial(empty_body_func);
    for (size_t i = 1; i < empty_body_fg_with_inputs.size(); i++) {
      auto inp = empty_body_fg_with_inputs[i]->get();
      empty_body_fg_with_inputs[i] = while_fg_emitter->NewNode(replace_by_param(inp));
    }
    for (size_t i = 1; i < cond_cnode->size(); i++) {
      auto inp = cond_cnode->input(i);
      if (!inp->isa<ValueNode>()) {
        cond_cnode->set_input(i, replace_by_param(inp));
      }
    }

    auto body_with_inputs = BuildSubgraphOfPartial(while_body_func);
    auto body_fg = body_with_inputs[0]->get<ValueNodePtr>()->value()->cast<FuncGraphPtr>();
    for (size_t i = 1; i < body_with_inputs.size(); i++) {
      body_with_inputs[i] = while_fg_emitter->NewNode(replace_by_param(body_with_inputs[i]->get()));
    }
    // replace the body's output to call the outside while-fg
    AnfNodePtrList body_while_fg_inputs{NewValueNode(while_fg)};
    if (IsPrimitiveCNode(body_fg->output(), prim::kPrimMakeTuple)) {
      auto mt = body_fg->output()->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(mt);
      (void)body_while_fg_inputs.insert(body_while_fg_inputs.end(), mt->inputs().begin() + 1, mt->inputs().end());
    } else {
      body_while_fg_inputs.push_back(body_fg->output());
    }
    if (body_while_fg_inputs.size() - 1 != init_list.size()) {
      MS_LOG(EXCEPTION) << "The while body's output size should be equal to init_list.size(), but got "
                        << (body_while_fg_inputs.size() - 1) << " vs " << init_list.size();
    }
    if (body_while_fg_inputs.size() < main_while_fg_inputs.size()) {
      for (size_t i = body_while_fg_inputs.size(); i < main_while_fg_inputs.size(); i++) {
        auto inp = while_fg->parameters()[i - 1];
        auto iter = std::find_if(body_with_inputs.begin(), body_with_inputs.end(),
                                 [&inp](const NodePtr &no) { return no->get() == inp; });
        if (iter != body_with_inputs.end()) {
          auto param_idx = iter - body_with_inputs.begin() - 1;
          body_while_fg_inputs.push_back(body_fg->parameters()[param_idx]);
        } else {
          body_with_inputs.push_back(while_fg_emitter->NewNode(inp));
          auto p = body_fg->add_parameter();
          p->set_abstract(inp->abstract());
          body_while_fg_inputs.push_back(p);
        }
      }
    }
    auto body_call_fg = body_fg->NewCNode(body_while_fg_inputs);
    body_call_fg->set_abstract(out_abstract_);
    body_fg->set_output(body_call_fg);

    auto tb = while_fg_emitter->Emit("Partial", body_with_inputs);
    auto fb = while_fg_emitter->Emit("Partial", empty_body_fg_with_inputs);
    auto s = while_fg_emitter->Emit("Switch", {cond, tb, fb});
    auto cnode = while_fg_emitter->func_graph_->NewCNode({s->get()});
    cnode->set_abstract(out_abstract_);
    while_fg->set_output(cnode);

    auto main_cnode = emitter_->func_graph_->NewCNode(main_while_fg_inputs);
    main_cnode->set_abstract(out_abstract_);
    return emitter_->NewNode(main_cnode);
  }

 protected:
  NodePtr BuildSubgraph(const BlockFunc &func) {
    auto fg = std::make_shared<FuncGraph>();
    MS_EXCEPTION_IF_NULL(fg);
    fg->set_switch_input(std::make_shared<bool>(true));
    auto e = std::make_unique<Emitter>(fg, emitter_->infer());
    auto outputs = func(e.get());
    if (outputs.empty()) {
      MS_LOG(EXCEPTION) << "The block function should not return empty list.";
    }
    if (output_num_ == 0) {
      output_num_ = outputs.size();
    } else if (output_num_ != outputs.size()) {
      MS_LOG(EXCEPTION) << "The count of outputs of each block function should be equal, but got " << output_num_
                        << " vs " << outputs.size() << ".";
    }
    NodePtr output;
    if (output_num_ > 1) {
      output = e->MakeTuple(outputs);
      SetSequenceNodeElementsUseFlags(output->get(), std::make_shared<std::vector<bool>>(output_num_, true));
    } else {
      output = outputs[0];
    }
    fg->set_output(output->get());
    if (out_abstract_ == nullptr) {
      out_abstract_ = output->abstract();
    }
    return emitter_->Value(fg);
  }

  NodePtrList BuildSubgraphOfPartial(const BlockFunc &func) {
    auto fg = std::make_shared<FuncGraph>();
    MS_EXCEPTION_IF_NULL(fg);
    fg->set_switch_input(std::make_shared<bool>(true));
    auto sub_emitter = std::make_unique<Emitter>(fg, emitter_->infer());
    auto output = func(sub_emitter.get());
    if (output.empty()) {
      MS_LOG(EXCEPTION) << "The block function should not return empty list.";
    }
    if (output_num_ == 0) {
      output_num_ = output.size();
    } else if (output_num_ != output.size()) {
      MS_LOG(EXCEPTION) << "The count of outputs of each block function should be equal, but got " << output_num_
                        << " vs " << output.size() << ".";
    }
    fg->set_output((output_num_ > 1) ? sub_emitter->MakeTuple(output)->get() : output[0]->get());
    if (out_abstract_ == nullptr) {
      out_abstract_ = fg->output()->abstract();
    }
    if (output_num_ > 1) {
      SetSequenceNodeElementsUseFlags(fg->output(), std::make_shared<std::vector<bool>>(output_num_, true));
    }

    // replace the captured inputs to parameter
    std::function<void(const CNodePtr &)> dfs;
    std::unordered_set<AnfNodePtr> visited;
    std::map<AnfNodePtr, ParameterPtr> param_map;
    NodePtrList fg_with_inputs = {emitter_->Value(fg)};
    dfs = [&visited, &dfs, &fg, &param_map, &fg_with_inputs, this](const CNodePtr &node) {
      (void)visited.insert(node);
      for (size_t i = 0; i < node->size(); i++) {
        auto inp = node->input(i);
        if (inp->func_graph() == nullptr) {
          continue;
        }
        if (inp->func_graph() == fg) {
          if (inp->isa<CNode>() && visited.count(inp) == 0) {
            dfs(inp->cast<CNodePtr>());
          }
        } else {
          auto &param = param_map[inp];
          if (param == nullptr) {
            param = fg->add_parameter();
            param->set_abstract(inp->abstract());
            (void)fg_with_inputs.emplace_back(emitter_->NewNode(inp));
          }
          node->set_input(i, param);
        }
      }
    };
    dfs(fg->get_return());
    return fg_with_inputs;
  }

  size_t output_num_{0};
  const Emitter *emitter_;
  abstract::AbstractBasePtr out_abstract_{nullptr};

  class CppInferWithPartial : public CppInfer {
   public:
    void Infer(const NodePtr &node) override {
      if (IsPrimitiveCNode(node->get(), prim::kPrimPartial) || IsPrimitiveCNode(node->get(), prim::kPrimSwitch)) {
        return;
      }
      CppInfer::Infer(node);
    }
  };
};

NodePtr Emitter::Conditional(const NodePtr &cond, const BlockFunc &true_case, const BlockFunc &false_case) const {
  return CtrlFlowBlock(this).IfThenElse(cond, true_case, false_case);
}

NodePtr Emitter::While(const NodePtr &cond, const BlockFunc &body, const NodePtrList &init_list) const {
  return CtrlFlowBlock(this).While(cond, body, init_list);
}

NodePtr operator+(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Add(lhs, rhs); }
NodePtr operator-(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Sub(lhs, rhs); }
NodePtr operator*(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->Mul(lhs, rhs); }
NodePtr operator/(const NodePtr &lhs, const NodePtr &rhs) { return lhs->emitter()->RealDiv(lhs, rhs); }
NodePtr operator-(const NodePtr &node) { return node->emitter()->Neg(node); }
}  // namespace expander
}  // namespace mindspore
