/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "transform/graph_ir/op_adapter.h"

namespace mindspore {
namespace transform {
static uint32_t CustomInferFunc(const Operator &) { return 0; }

bool OpAdapterImpl::IsCustomOp(const OperatorPtr &op) {
  MS_EXCEPTION_IF_NULL(op);
  auto it = cus_input_map_->find(op->GetOpType());
  if (it == cus_input_map_->end()) {
    return false;
  }
  return true;
}

Status OpAdapterImpl::GenerateCustomOpInputMap(const CusOperatorPtr &op, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(prim);
  // Create the map of custom op from input index to input name.
  std::unordered_map<int, std::string> input_map;
  auto value = prim->GetAttr("input_names");
  if (value == nullptr) {
    (*cus_output_map_)[prim->name()] = input_map;
    return NOT_FOUND;
  }

  auto input_names = GetValue<const std::vector<std::string>>(value);
  for (size_t i = 0; i < input_names.size(); ++i) {
    // input_map begin form 1
    input_map[i + 1] = input_names[i];
    op->CustomInputRegister(input_names[i]);
  }

  if (cus_input_map_->find(prim->name()) == cus_input_map_->end()) {
    (*cus_input_map_)[prim->name()] = input_map;
  }
  return SUCCESS;
}

Status OpAdapterImpl::GenerateCustomOpOutputMap(const CusOperatorPtr &op, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(prim);
  // Create the map of custom op from output index to output name.
  std::unordered_map<int, std::string> output_map;
  auto value = prim->GetAttr("output_names");
  if (value == nullptr) {
    // generate a empty output_map for it
    (*cus_output_map_)[prim->name()] = output_map;
    return NOT_FOUND;
  }

  auto output_names = GetValue<const std::vector<std::string>>(value);
  for (size_t i = 0; i < output_names.size(); ++i) {
    // output_map begin form 0
    output_map[i] = output_names[i];
    op->CustomOutputRegister(output_names[i]);
  }

  if (cus_output_map_->find(prim->name()) == cus_output_map_->end()) {
    (*cus_output_map_)[prim->name()] = output_map;
  }
  return SUCCESS;
}

OperatorPtr OpAdapterImpl::GenerateCustomOp(const AnfNodePtr anf) {
  MS_EXCEPTION_IF_NULL(anf);
  auto node = anf->cast<CNodePtr>();
  if (node == nullptr) {
    return nullptr;
  }

  if (node->inputs().empty()) {
    MS_LOG(EXCEPTION) << "length of node inputs is empty";
  }

  auto prim = GetValueNode<PrimitivePtr>(node->inputs()[0]);
  MS_EXCEPTION_IF_NULL(prim);
  auto op = std::make_shared<ge::CustomOperator>(node->fullname_with_scope(), prim->name());
  if (GenerateCustomOpInputMap(op, prim) != SUCCESS) {
    MS_LOG(WARNING) << "Custom op node has no input_names, op[" << prim->name() << "].";
  }

  if (GenerateCustomOpOutputMap(op, prim) != SUCCESS) {
    MS_LOG(WARNING) << "Custom op node has no output_names, op[" << prim->name() << "].";
  }

  op->CustomInferFuncRegister(CustomInferFunc);

  return op;
}

Status OpAdapterImpl::SetOpSubgraphFunc(const OperatorPtr &op, int index,
                                        const std::shared_ptr<std::vector<DfGraph>> &branches) {
  MS_EXCEPTION_IF_NULL(op);
  auto it = dyn_subgraph_map_.find(index);
  if (it != dyn_subgraph_map_.end()) {
    auto size = branches->size();
    it->second.create_dyn_subgraph(op, static_cast<unsigned int>(size));
    for (size_t i = 0; i < size; i++) {
      it->second.set_subgraph(op, static_cast<unsigned int>(i), std::make_shared<DfGraph>((*branches)[i]));
    }
    return SUCCESS;
  }
  return NOT_FOUND;
}

Status OpAdapterImpl::SetCustomOpInput(const CusOperatorPtr &op, int index, const OperatorPtr &input) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(input);
  auto it = cus_input_map_->find(op->GetOpType());
  if (it == cus_input_map_->end()) {
    return NOT_FOUND;
  }
  std::unordered_map<int, std::string> &input_map = it->second;

  if ((input_map.find(index) != input_map.end())) {
    MS_LOG(DEBUG) << "Link op " << input->GetName() << " to " << op->GetName() << ":" << input_map[index];
    (void)op->SetInput(input_map[index], *input);
    return SUCCESS;
  }
  return NOT_FOUND;
}

Status OpAdapterImpl::SetNormalOpInput(const OperatorPtr &op, int index, const OperatorPtr &input) {
  MS_EXCEPTION_IF_NULL(op);
  auto it = input_map_.find(index);
  if (it != input_map_.end()) {
    MS_EXCEPTION_IF_NULL(input);
    MS_LOG(DEBUG) << "Link op " << input->GetName() << " to " << op->GetName() << ":" << it->second.name;
    it->second.set_op(op, input);
    return SUCCESS;
  }
  return NOT_FOUND;
}

int OpAdapterImpl::setInput(const OperatorPtr &op, int index, const OperatorPtr &input) {
  if (IsCustomOp(op)) {
    auto cus_op = std::dynamic_pointer_cast<CustomOperator>(op);
    return static_cast<int>(SetCustomOpInput(cus_op, index, input));
  } else {
    return static_cast<int>(SetNormalOpInput(op, index, input));
  }
}

Status OpAdapterImpl::SetCustomOpInput(const CusOperatorPtr &op, int index, const OutHandler &handle) {
  MS_EXCEPTION_IF_NULL(op);
  auto it = cus_input_map_->find(op->GetOpType());
  if (it == cus_input_map_->end()) {
    return NOT_FOUND;
  }

  std::unordered_map<int, std::string> &input_map = it->second;
  if ((handle.op != nullptr) && (input_map.find(index) != input_map.end())) {
    if (handle.out.empty()) {
      MS_LOG(DEBUG) << "Link op " << handle.op->GetName() << " to " << op->GetName() << ":" << input_map[index];
      (void)op->SetInput(input_map[index], *(handle.op));
    } else {
      MS_LOG(DEBUG) << "Link op " << handle.op->GetName() << ":" << handle.out << " to " << op->GetName() << ":"
                    << input_map[index];
      (void)op->SetInput(input_map[index], *(handle.op), handle.out);
    }
    return SUCCESS;
  }
  return NOT_FOUND;
}

Status OpAdapterImpl::SetNormalOpInput(const OperatorPtr &op, int index, const OutHandler &handle) {
  MS_EXCEPTION_IF_NULL(op);
  auto it = input_map_.find(index);
  if ((handle.op != nullptr) && (it != input_map_.end())) {
    if (handle.out.empty()) {
      MS_LOG(DEBUG) << "Link op " << handle.op->GetName() << " to " << op->GetName() << ":" << it->second.name;
      it->second.set_op(op, handle.op);
    } else {
      MS_LOG(DEBUG) << "Link op " << handle.op->GetName() << ":" << handle.out << " to " << op->GetName() << ":"
                    << it->second.name;
      it->second.set_handle(op, handle);
    }
    return SUCCESS;
  }
  return NOT_FOUND;
}

int OpAdapterImpl::setInput(const OperatorPtr &op, int index, const OutHandler &handle) {
  if (IsCustomOp(op)) {
    auto cus_op = std::dynamic_pointer_cast<CustomOperator>(op);
    return static_cast<int>(SetCustomOpInput(cus_op, index, handle));
  } else {
    return static_cast<int>(SetNormalOpInput(op, index, handle));
  }
}

int OpAdapterImpl::setInput(const OperatorPtr &op, int index,
                            const std::shared_ptr<std::vector<OutHandler>> &handler_vec) {
  MS_EXCEPTION_IF_NULL(handler_vec);
  if (IsCustomOp(op)) {
    MS_LOG(ERROR) << "Custom Op do not support dynamic input";
    return static_cast<int>(FAILED);
  }
  MS_EXCEPTION_IF_NULL(op);
  auto it = dyn_input_map_.find(index);
  if (it != dyn_input_map_.end()) {
    it->second.create_dyn_input(op, static_cast<unsigned int>(handler_vec->size()));
    for (unsigned int i = 0; i < handler_vec->size(); ++i) {
      OutHandler h = (*handler_vec)[i];
      MS_EXCEPTION_IF_NULL(h.op);
      if (h.out.empty()) {
        MS_LOG(DEBUG) << "Link op " << h.op->GetName() << " to " << op->GetName() << ":" << it->second.name;
        it->second.set_op(op, (i) /* index start from 0 */, h.op);
      } else {
        MS_LOG(DEBUG) << "Link op " << h.op->GetName() << ":" << h.out << " to " << op->GetName() << ":"
                      << it->second.name;
        it->second.set_handle(op, i, h);
      }
    }
    return 0;
  }
  return static_cast<int>(NOT_FOUND);
}

OutHandler OpAdapterImpl::getOutput(const OperatorPtr &op, int index) {
  MS_EXCEPTION_IF_NULL(op);
  if (IsCustomOp(op)) {
    return getCustomOutput(op, index);
  }
  return getNormalOutput(op, index);
}

OutHandler OpAdapterImpl::getCustomOutput(const OperatorPtr &op, int index) {
  MS_EXCEPTION_IF_NULL(op);
  auto it = cus_output_map_->find(op->GetOpType());
  if (it == cus_output_map_->end()) {
    MS_LOG(ERROR) << "OpAdpator(" << op->GetName() << ") has both OUTPUT is not supported!";
    return OutHandler();
  }

  std::unordered_map<int, std::string> &output_map = it->second;

  if ((output_map.find(index) != output_map.end())) {
    return OutHandler(op, output_map[index]);
  }
  MS_LOG(ERROR) << "OpAdpator(" << op->GetName() << ") has no OUTPUT index(" << index << ")!";
  return OutHandler();
}

OutHandler OpAdapterImpl::getNormalOutput(const OperatorPtr &op, int index) {
  MS_EXCEPTION_IF_NULL(op);
  if (!dyn_output_map_.empty() && !output_map_.empty()) {
    MS_LOG(ERROR) << "OpAdpator(" << op->GetName() << ") has both OUTPUT and DYN_OUTPUT is not supported!";
    return OutHandler();
  }
  auto it = output_map_.find(index);
  if (it != output_map_.end()) {
    return OutHandler(op, it->second.name);
  } else if (!dyn_output_map_.empty()) {
    return OutHandler(op, dyn_output_map_.begin()->second.name + std::to_string(index));
  } else {
    MS_LOG(ERROR) << "OpAdpator(" << op->GetName() << ") has no OUTPUT and DYN_OUTPUT index(" << index << ")!";
    return OutHandler();
  }
}

Status OpAdapterImpl::UpdateSingleOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp,
                                             const TypePtr &type) {
  MS_EXCEPTION_IF_NULL(type);
  std::string format = "NCHW";
  if (op->GetOpType() == kExtractImagePatchesOpName) {
    format = "NHWC";
  }

  auto desc = CreateOutputDesc(dyn_cast<abstract::Shape>(shp), type, format);
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Update output descriptor failed!";
    return FAILED;
  }

  if (IsCustomOp(op)) {
    if (cus_output_map_->find(op->GetOpType()) == cus_output_map_->end() ||
        ((*cus_output_map_)[op->GetOpType()].empty())) {
      MS_LOG(ERROR) << "This op does not create custom output map";
      return FAILED;
    }
    auto cus_op = std::dynamic_pointer_cast<CustomOperator>(op);
    MS_EXCEPTION_IF_NULL(cus_op);
    std::unordered_map<int, std::string> output_map = (*cus_output_map_)[op->GetOpType()];
    (void)cus_op->UpdateOutputDesc(output_map[0], *desc);
  } else {
    if (output_map_.empty()) {
      MS_LOG(INFO) << "This op does not have output map";
      return FAILED;
    }
    output_map_.begin()->second.update_out_desc(op, *desc);
  }
  return SUCCESS;
}

size_t OpAdapterImpl::GetCustomOpOutputSize(const CusOperatorPtr &cus_op) {
  MS_EXCEPTION_IF_NULL(cus_op);
  if (cus_output_map_->find(cus_op->GetOpType()) == cus_output_map_->end()) {
    MS_LOG(ERROR) << "This op does not create custom output map";
    return 0;
  }
  size_t output_size = (*cus_output_map_)[cus_op->GetOpType()].size();
  return output_size;
}

std::shared_ptr<GeTensorDesc> OpAdapterImpl::CreateOutputDesc(const abstract::ShapePtr &shape_ptr, const TypePtr &type,
                                                              const std::string &format) {
  if (shape_ptr == nullptr) {
    MS_LOG(ERROR) << "Shape ptr is nullptr";
    return nullptr;
  }

  if (type == nullptr) {
    MS_LOG(ERROR) << "Type ptr is nullptr";
    return nullptr;
  }

  TypeId me_type = type->type_id();
  if (kObjectTypeTensorType == me_type) {
    me_type = dyn_cast<TensorType>(type)->element()->type_id();
  }
  auto desc = TransformUtil::GetGeTensorDesc(shape_ptr->shape(), me_type, format);
  return desc;
}

Status OpAdapterImpl::UpdateMultiOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp,
                                            const TypePtr &type) {
  auto tuple_shp = dyn_cast<abstract::TupleShape>(shp);
  MS_EXCEPTION_IF_NULL(tuple_shp);

  size_t output_size = 0;
  bool is_custom_op = IsCustomOp(op);
  if (is_custom_op) {
    output_size = GetCustomOpOutputSize(std::dynamic_pointer_cast<CustomOperator>(op));
  } else {
    output_size = output_map_.size();
  }

  if (output_size == 0) {
    MS_LOG(INFO) << "This op does not have output map";
    return FAILED;
  }

  if (output_size != tuple_shp->shape().size()) {
    MS_LOG(ERROR) << "output_map is not equal tuple_shape size";
    return FAILED;
  }
  std::string format = "NCHW";
  if (op->GetOpType() == kTopKOpName) {
    format = "NHWC";
  }
  for (size_t i = 0; i < tuple_shp->shape().size(); ++i) {
    auto tuple_type = dyn_cast<Tuple>(type);
    MS_EXCEPTION_IF_NULL(tuple_type);
    TypePtr type_elem = tuple_type->elements()[i];

    auto desc = CreateOutputDesc(dyn_cast<abstract::Shape>(tuple_shp->shape()[i]), type_elem, format);
    if (desc == nullptr) {
      MS_LOG(ERROR) << "Create output descriptor failed!";
      return FAILED;
    }

    if (is_custom_op) {
      (void)std::dynamic_pointer_cast<CustomOperator>(op)->UpdateOutputDesc((*cus_output_map_)[op->GetOpType()][i],
                                                                            *desc);
    } else {
      auto it = output_map_.find(i);
      if (it != output_map_.end()) {
        it->second.update_out_desc(op, *desc);
      }
    }
  }
  return SUCCESS;
}

std::shared_ptr<GeTensorDesc> OpAdapterImpl::CreateNodeDesc(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  TypeId me_type = node->Type()->type_id();
  if (kObjectTypeTensorType == me_type) {
    me_type = dyn_cast<TensorType>(node->Type())->element()->type_id();
  }
  if (me_type <= kNumberTypeBegin || me_type >= kNumberTypeEnd) {
    return nullptr;
  }

  std::vector<int> shape;
  auto shape_ptr = dyn_cast<abstract::Shape>(node->Shape());
  if (nullptr != shape_ptr) {
    shape = shape_ptr->shape();
  }

  auto desc = TransformUtil::GetGeTensorDesc(shape, me_type, "NCHW");
  if (desc == nullptr) {
    MS_LOG(ERROR) << "Update output descriptor failed!";
    return nullptr;
  }
  return desc;
}

void OpAdapterImpl::UpdateNormalOpInputDesc(const OperatorPtr &op, const AnfNodePtr &node) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is nullptr";
    return;
  }
  MS_EXCEPTION_IF_NULL(node);

  auto inputs = node->cast<CNodePtr>()->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    auto it = input_map_.find(i);
    if (it != input_map_.end()) {
      auto desc = CreateNodeDesc(inputs[i]);
      if (desc == nullptr) {
        continue;
      }
      if (op->GetOpType() == kExtractImagePatchesOpName) {
        desc->SetFormat(ge::Format::FORMAT_NHWC);
      }
      it->second.update_input_desc(op, *desc);
    }
  }
}

void OpAdapterImpl::UpdateCustomOpInputDesc(const CusOperatorPtr &op, const AnfNodePtr &node) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is nullptr";
    return;
  }
  MS_EXCEPTION_IF_NULL(node);

  if (cus_input_map_->find(op->GetOpType()) == cus_input_map_->end() || ((*cus_input_map_)[op->GetOpType()].empty())) {
    MS_LOG(ERROR) << "This op does not create custom input map";
    return;
  }

  std::unordered_map<int, std::string> &input_map = (*cus_input_map_)[op->GetOpType()];
  auto inputs = node->cast<CNodePtr>()->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (input_map.find(i) != input_map.end()) {
      auto desc = CreateNodeDesc(inputs[i]);
      if (desc == nullptr) {
        continue;
      }
      (void)op->UpdateInputDesc(input_map[i], *desc);
    }
  }
}

void OpAdapterImpl::updateInputDesc(const OperatorPtr &op, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(op);
  MS_EXCEPTION_IF_NULL(node);
  if (IsCustomOp(op)) {
    auto cus_op = std::dynamic_pointer_cast<CustomOperator>(op);
    UpdateCustomOpInputDesc(cus_op, node);
  } else {
    UpdateNormalOpInputDesc(op, node);
  }
}

void OpAdapterImpl::updateOutputDesc(const OperatorPtr &op, const abstract::BaseShapePtr &shp, const TypePtr &type,
                                     const AnfNodePtr &node) {
  if (op == nullptr) {
    MS_LOG(ERROR) << "op is nullptr";
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(INFO) << "Op name is " << op->GetName();

  auto normal_shape_ptr = dyn_cast<abstract::Shape>(shp);
  auto no_shape_ptr = dyn_cast<abstract::NoShape>(shp);

  if ((nullptr != normal_shape_ptr) || (nullptr != no_shape_ptr)) {
    if (UpdateSingleOutputDesc(op, shp, type) != SUCCESS) {
      return;
    }
  } else if (nullptr != dyn_cast<abstract::TupleShape>(shp)) {
    if (UpdateMultiOutputDesc(op, shp, type) != SUCCESS) {
      return;
    }
  } else {
    MS_LOG(WARNING) << "Update output desc failed, unknown output shape type";
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return;
  }

  // Need to update input_desc while the output_desc is updated
  updateInputDesc(op, node);
}

int OpAdapterImpl::setAttr(const OperatorPtr &op, const std::string &attr_key, const ValuePtr &attr_value) {
  auto it = attr_map_.find(attr_key);
  if (it != attr_map_.end()) {
    // switch case for each avalilable attribute type
    MS_LOG(INFO) << "Set attr: " << attr_key << "(" << it->second.name << "), value: " << attr_value->ToString();
    adpt_->AddAttrToDrawGraph(attr_key + std::string("=") + attr_value->ToString());
    it->second.set_attr(op, attr_value);
    return 0;
  }
  return static_cast<int>(NOT_FOUND);
}

int OpAdapterImpl::SetCustomOpAttr(const CusOperatorPtr &op, const PrimitivePtr &prim) {
  enum ValueType {
    SINGLE_VALUE = 0,
    SEQUEUE_VALUE,
    UNKNOWN_VALUE,
  };

  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(op);

  ValueType value_type = SINGLE_VALUE;
  for (auto item : prim->attrs()) {
    if (item.second->isa<Int32Imm>()) {
      (void)op->SetAttr(item.first, GetValue<int>(item.second));
    } else if (item.second->isa<StringImm>()) {
      (void)op->SetAttr(item.first, GetValue<std::string>(item.second));
    } else if (item.second->isa<BoolImm>()) {
      (void)op->SetAttr(item.first, GetValue<bool>(item.second));
    } else if (item.second->isa<FP32Imm>()) {
      (void)op->SetAttr(item.first, GetValue<float>(item.second));
    } else if (item.second->isa<ValueSequeue>()) {
      value_type = SEQUEUE_VALUE;
      auto val_seq = item.second->cast<ValueSequeuePtr>();
      if ((*val_seq)[0]->isa<StringImm>()) {
        (void)op->SetAttr(item.first, GetValue<const std::vector<std::string>>(item.second));
      } else if ((*val_seq)[0]->isa<FP32Imm>()) {
        (void)op->SetAttr(item.first, GetValue<const std::vector<float>>(item.second));
      } else if ((*val_seq)[0]->isa<Int32Imm>()) {
        (void)op->SetAttr(item.first, GetValue<const std::vector<int>>(item.second));
      } else if ((*val_seq)[0]->isa<BoolImm>()) {
        (void)op->SetAttr(item.first, GetValue<const std::vector<bool>>(item.second));
      } else {
        MS_LOG(EXCEPTION) << "Unsupported custom attribute type in adaptor, prim name: " << prim->name()
                          << ", attr name: " << item.first << ", value: " << item.second->ToString();
      }
    } else {
      MS_LOG(WARNING) << "Unsupported custom attribute type in adaptor, prim name: " << prim->name()
                      << ", attr name: " << item.first << ", value: " << item.second->ToString();
      return static_cast<int>(NOT_FOUND);
    }

    if (value_type == SINGLE_VALUE) {
      adpt_->AddAttrToDrawGraph(item.first + std::string("=") + item.second->ToString());
    } else if (value_type == SEQUEUE_VALUE) {
      adpt_->AddAttrToDrawGraph(item.first + std::string("=") + "[...]");
    }
  }
  return 0;
}

int OpAdapterImpl::SetNormalOpAttr(const OperatorPtr &op, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(op);
  for (auto &it : attr_map_) {
    auto value = prim->GetAttr(it.first);
    if (value != nullptr) {
      // set attr from primitive
      int ret = setAttr(op, it.first, value);
      if (ret) {
        return ret;
      }
    } else {
      // set attr from extra_attr
      auto it_extra = extra_attr_->find(it.first);
      if (it_extra != extra_attr_->end()) {
        int ret = setAttr(op, it.first, it_extra->second);
        if (ret) {
          return ret;
        }
      }
    }
  }
  return 0;
}

int OpAdapterImpl::setAttr(const OperatorPtr &op, const PrimitivePtr &prim) {
  int ret = 0;
  if (IsCustomPrim(prim)) {
    auto cus_op = std::dynamic_pointer_cast<CustomOperator>(op);
    ret = SetCustomOpAttr(cus_op, prim);
  } else {
    ret = SetNormalOpAttr(op, prim);
  }
  return ret;
}

int OpAdapterImpl::setAttr(const OperatorPtr &op, const AnfNodePtr &node) {
  // no attribute for lonely node
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return 0;
  }

  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return 0;
  }

  auto &inputs = cnode->inputs();
  if (inputs.empty()) {
    return 0;
  }

  // get Attr T from abstract of anfnode first,
  // if attr "T" appears in primitive, the primitive T will cover this one
  if (attr_map_.find("T") != attr_map_.end()) {
    // get dtype from inputs[1], if the node has no inputs, set the attr T with output dtype
    TypePtr type;
    if (inputs.size() > 1) {
      type = inputs[1]->Type();
    } else {
      type = node->Type();
    }
    if (type != nullptr) {
      (void)setAttr(op, "T", MakeValue(type));
    }
  }

  // set attr from primitive and ExtraAttr
  if (IsValueNode<Primitive>(inputs[0])) {
    // set attr from primitive
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(inputs[0]);
    int ret = setAttr(op, prim);
    if (ret != 0) {
      return ret;
    }
  }

  // set attr from const input
  for (auto &it : input_attr_map_) {
    if (inputs.size() <= it.first || !inputs[it.first]->isa<ValueNode>()) {
      continue;
    }
    auto const_value = GetValueNode(inputs[it.first]);
    MS_LOG(INFO) << "Set attr: input_" << it.first << "(" << it.second.name << "), value: " << const_value->ToString();
    if (const_value->isa<None>()) {
      continue;
    }
    adpt_->AddAttrToDrawGraph(it.second.name + std::string("=") + const_value->ToString());
    it.second.set_attr(op, const_value);
  }
  return 0;
}
}  // namespace transform
}  // namespace mindspore
