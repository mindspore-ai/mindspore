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
#include "plugin/device/ascend/optimizer/mindir/ascend_vm_op_adapter.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include "include/common/utils/utils.h"
#include "include/common/utils/anfalgo.h"
#include "plugin/device/ascend/optimizer/ascend_helper.h"
#include "plugin/device/ascend/optimizer/mindir/reg_ascend_vm_op_adaptation_info.h"
#include "plugin/device/ascend/kernel/tbe/tbe_dynamic_shape_util.h"

namespace mindspore::opt {
const AnfNodePtr AscendVmOpAdapter::Process(const FuncGraphPtr &, const AnfNodePtr &node, const EquivPtr &) const {
  if (node == nullptr || !AnfUtils::IsRealCNodeKernel(node)) {
    return nullptr;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(node);
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(node);
  auto op_adaptation_info =
    OpAdaptationInfoRegister::GetInstance().GetOpAdaptationInfo(op_name, kAscendDevice, is_dynamic);
  if (op_adaptation_info == nullptr) {
    return nullptr;
  }

  auto origin_op = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(origin_op);
  auto ret_node = ConvertToTargetOp(origin_op, op_adaptation_info);
  if ((ret_node != nullptr) && (ret_node != origin_op)) {
    MS_LOG(INFO) << "Replace op " << origin_op->fullname_with_scope() << " debug string:" << origin_op->DebugString()
                 << " with " << ret_node->fullname_with_scope() << " debug string:" << ret_node->DebugString()
                 << ", is dynamic shape:" << is_dynamic;
  }
  return ret_node;
}

CNodePtr AscendVmOpAdapter::ConvertNodeToCheck(const CNodePtr &origin_op,
                                               const OpAdaptationInfo &op_adaptation_info) const {
  MS_EXCEPTION_IF_NULL(origin_op);
  // check supported if the op need
  auto graph = origin_op->func_graph();
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  if (op_adaptation_info.NeedTBECheck()) {
    auto is_dynamic = common::AnfAlgo::IsDynamicShape(origin_op);
    // when cnode is a dynamic shape node, if origin op supported, use origin op
    if (is_dynamic) {
      auto ret = CheckAICoreSupported(origin_op);
      if (ret) {
        MS_LOG(DEBUG) << "Origin op " << origin_op->fullname_with_scope() << " is supported in this configuration";
        return origin_op;
      }
    }

    auto target_op = CreateTargetOp(origin_op, op_adaptation_info);
    if (target_op == nullptr) {
      MS_LOG(DEBUG) << "Create target op failed for node " << origin_op->fullname_with_scope();
      return origin_op;
    }

    auto ret = CheckAICoreSupported(target_op);
    if (!ret) {
      return origin_op;
    }

    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(origin_op, target_op);
    }
    return target_op;
  } else {
    auto target_op = CreateTargetOp(origin_op, op_adaptation_info);
    if (target_op == nullptr) {
      MS_LOG(DEBUG) << "Create target op failed for node " << origin_op->fullname_with_scope();
      return origin_op;
    }
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(origin_op, target_op);
    }
    return target_op;
  }
}

CNodePtr AscendVmOpAdapter::ConvertToTargetOp(const CNodePtr &origin_op, OpAdaptationInfo *op_adaptation_info) const {
  MS_EXCEPTION_IF_NULL(origin_op);
  MS_EXCEPTION_IF_NULL(op_adaptation_info);
  auto origin_op_name = op_adaptation_info->GetOriginOpName();
  auto target_op_name = op_adaptation_info->GetTargetOpName();
  auto pre_check_func = op_adaptation_info->GetPreCheckFunc();
  auto need_tbe_check = op_adaptation_info->NeedTBECheck();
  auto input_to_attr_map = op_adaptation_info->GetInputAttrInfoMap();
  auto attr_name_map = op_adaptation_info->GetAttrNameInfoMap();
  // Rename the attrs
  if (!attr_name_map.empty()) {
    auto origin_primitive = GetCNodePrimitive(origin_op);
    MS_EXCEPTION_IF_NULL(origin_primitive);
    for (const auto &iter : attr_name_map) {
      if (origin_primitive->HasAttr(iter.first)) {
        auto value = origin_primitive->GetAttr(iter.first);
        origin_primitive->set_attr(iter.second, value);
        origin_primitive->EraseAttr(iter.first);
        MS_LOG(INFO) << "Rename attr " << iter.first << " to " << iter.second << " for op "
                     << origin_op->fullname_with_scope();
      } else {
        MS_LOG(ERROR) << "Node " << origin_op->fullname_with_scope() << " has no attr " << iter.first;
        return origin_op;
      }
    }
  }

  // No need to check or const input to attr
  if ((!pre_check_func) && (!need_tbe_check) && (input_to_attr_map.empty())) {
    // Rename the op type
    if (target_op_name != origin_op_name) {
      auto origin_primitive = GetCNodePrimitive(origin_op);
      MS_EXCEPTION_IF_NULL(origin_primitive);
      origin_primitive->set_name(target_op_name);
      // reset full scope name
      origin_op->set_fullname_with_scope("");
      MS_LOG(INFO) << "Rename op type from " << origin_op << " to " << target_op_name << " for op "
                   << origin_op->fullname_with_scope();
      return origin_op;
    } else {
      return origin_op;
    }
  }

  // check through op custom pre-check function
  if (pre_check_func != nullptr) {
    auto ret = pre_check_func(origin_op);
    if (!ret) {
      MS_LOG(DEBUG) << "Pre check function return Not Change for op " << origin_op->fullname_with_scope();
      return origin_op;
    }
  }

  return ConvertNodeToCheck(origin_op, *op_adaptation_info);
}

template <typename T, typename Scalar>
ValuePtr GetTensorValue(const tensor::TensorPtr &tensor) {
  ValuePtr ret;
  auto tensor_value = TensorValueToVector<T>(tensor);
  if (tensor_value.size() == 1) {
    ret = std::make_shared<Scalar>(tensor_value[0]);
  } else {
    std::vector<ValuePtr> value_vec;
    for (const auto &elem : tensor_value) {
      auto value = std::make_shared<Scalar>(elem);
      MS_EXCEPTION_IF_NULL(value);
      value_vec.push_back(value);
    }
    ret = std::make_shared<ValueTuple>(value_vec);
  }
  return ret;
}

ValuePtr CreateValueFromTensor(const tensor::TensorPtr &tensor) {
  ValuePtr ret;
  if (tensor->has_user_data(kTensorValueIsType)) {
    ret = tensor->user_data<mindspore::Type>(kTensorValueIsType);
    return ret;
  }

  TypePtr data_type = tensor->Dtype();
  MS_EXCEPTION_IF_NULL(data_type);
  TypeId type_id = data_type->type_id();
  switch (type_id) {
    case kNumberTypeInt8: {
      ret = GetTensorValue<int8_t, Int8Imm>(tensor);
      break;
    }

    case kNumberTypeUInt8: {
      ret = GetTensorValue<uint8_t, UInt8Imm>(tensor);
      break;
    }

    case kNumberTypeInt16: {
      ret = GetTensorValue<int16_t, Int16Imm>(tensor);
      break;
    }

    case kNumberTypeUInt16: {
      ret = GetTensorValue<uint16_t, UInt16Imm>(tensor);
      break;
    }

    case kNumberTypeInt32: {
      ret = GetTensorValue<int32_t, Int32Imm>(tensor);
      break;
    }

    case kNumberTypeUInt32: {
      ret = GetTensorValue<uint32_t, UInt32Imm>(tensor);
      break;
    }

    case kNumberTypeInt64: {
      ret = GetTensorValue<int64_t, Int64Imm>(tensor);
      break;
    }

    case kNumberTypeUInt64: {
      ret = GetTensorValue<uint64_t, UInt64Imm>(tensor);
      break;
    }

    case kNumberTypeFloat32: {
      ret = GetTensorValue<float, FP32Imm>(tensor);
      break;
    }

    case kNumberTypeFloat64: {
      ret = GetTensorValue<double, FP64Imm>(tensor);
      break;
    }

    default:
      MS_LOG(EXCEPTION) << "Can't parse attr value :" << tensor->ToString() << ", Type:" << tensor->type_name();
  }
  return ret;
}

CNodePtr AscendVmOpAdapter::CreateTargetOp(const CNodePtr &origin_op,
                                           const OpAdaptationInfo &op_adaptation_info) const {
  MS_EXCEPTION_IF_NULL(origin_op);
  auto target_op_name = op_adaptation_info.GetTargetOpName();
  auto input_attr_info_map = op_adaptation_info.GetInputAttrInfoMap();

  auto origin_primitive = GetCNodePrimitive(origin_op);
  MS_EXCEPTION_IF_NULL(origin_primitive);
  auto target_primitive = std::make_shared<Primitive>(target_op_name);
  MS_EXCEPTION_IF_NULL(target_primitive);
  (void)target_primitive->SetAttrs(origin_primitive->attrs());
  std::vector<AnfNodePtr> target_inputs;
  auto inputs = origin_op->inputs();
  target_inputs.push_back(inputs[0]);

  auto input_names = origin_primitive->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(INFO) << "input_names are nullptr in cnode[" << origin_op->DebugString() << "]";
    return nullptr;
  }
  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);

  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPrimitiveCNode(input_node, prim::kPrimDepend)) {
      input_node = AnfUtils::VisitKernel(input_node, 0).first;
    }

    auto iter = input_attr_info_map.find(i);
    if (iter != input_attr_info_map.end() && input_node->isa<ValueNode>() && !HasAbstractMonad(input_node)) {
      auto ret = ConvertInputToAttr(origin_op, target_op_name, input_names_vec, i, input_node, iter, target_primitive);
      if (!ret) {
        return nullptr;
      }
    } else {
      target_inputs.push_back(inputs[i + 1]);
    }
  }

  // Update target_op's inputs
  target_inputs[0] = NewValueNode(target_primitive);
  auto graph = origin_op->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto target_op = opt::NewCNode(target_inputs, graph, {origin_op});
  MS_EXCEPTION_IF_NULL(target_op);
  target_op->set_abstract(origin_op->abstract());
  target_op->set_scope(origin_op->scope());
  target_op->set_primal_attrs(origin_op->primal_attrs());
  target_op->set_attrs(origin_op->attrs());
  auto is_dynamic = common::AnfAlgo::IsDynamicShape(origin_op);
  MS_LOG(DEBUG) << "Create op " << target_op->fullname_with_scope() << " debug string:" << target_op->DebugString()
                << " from " << origin_op->fullname_with_scope() << " debug string:" << origin_op->DebugString()
                << ", is dynamic shape:" << is_dynamic;
  return target_op;
}

bool AscendVmOpAdapter::ConvertInputToAttr(const CNodePtr &origin_op, const string &target_op_name,
                                           const std::vector<std::string> &input_names_vec, size_t i,
                                           const std::shared_ptr<AnfNode> &input_node,
                                           const std::map<size_t, InputAttrInfo>::iterator &iter,
                                           const std::shared_ptr<Primitive> &target_primitive) const {
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_LOG(DEBUG) << "start erase input[" << i
                << "] of cnode[" + origin_op->DebugString() + "], origin value:" << value_node->ToString()
                << ", Type:" << value_node->type_name();
  if (i >= input_names_vec.size()) {
    MS_LOG(INFO) << "Input index is invalid. input index: " << i << ", input name size " << input_names_vec.size();
    return false;
  }

  auto value = value_node->value();
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor->data().const_data() == nullptr) {
      MS_LOG(DEBUG) << "Const input data ptr is null from op " << origin_op->fullname_with_scope() << "'s input " << i;
      return false;
    }
    value = CreateValueFromTensor(tensor);
  }

  auto attr_name = GetAttrName(target_op_name, iter, input_names_vec[i]);
  value = UpdateAttrValue(origin_op, iter, value, attr_name);
  MS_LOG(DEBUG) << "new attr value:" << value_node->ToString() << ", Type:" << value_node->type_name();
  target_primitive->set_attr(attr_name, value);
  return true;
}

std::string AscendVmOpAdapter::GetAttrName(const string &target_op_name,
                                           const std::map<size_t, InputAttrInfo>::iterator &iter,
                                           const string &input_name) const {
  auto attr_name = iter->second.GetAttrName();
  if (attr_name.empty()) {
    MS_LOG(INFO) << "Attr name is empty for op " << target_op_name << ", use input name " << input_name << " instead.";
    attr_name = input_name;
  } else if (attr_name != input_name) {
    MS_LOG(INFO) << "Attr name not match input name: " << attr_name << " vs " << input_name;
  }
  return attr_name;
}

ValuePtr AscendVmOpAdapter::UpdateAttrValue(const CNodePtr &origin_op,
                                            const std::map<size_t, InputAttrInfo>::iterator &iter,
                                            const ValuePtr &value, const string &attr_name) const {
  ValuePtr ret = value;
  auto attr_dtype = iter->second.GetAttrDataType();
  if (attr_dtype.empty()) {
    auto op_name = common::AnfAlgo::GetCNodeName(origin_op);
    auto op_info_ptr = kernel::tbe::TbeDynamicShapeUtil::FindOp(op_name, origin_op);
    if (op_info_ptr) {
      auto op_info_attrs_ptr = op_info_ptr->attrs_ptr();
      for (const auto &op_info_attr_ptr : op_info_attrs_ptr) {
        std::string op_attr_name = op_info_attr_ptr->name();
        if (op_attr_name == attr_name) {
          attr_dtype = op_info_attr_ptr->type();
          break;
        }
      }
    }
  }
  if (!attr_dtype.empty()) {
    ret = UpdateAttrValueByDtype(value, attr_dtype);
  }
  return ret;
}

ValuePtr AscendVmOpAdapter::UpdateAttrValueByDtype(const ValuePtr &value, const std::string &attr_data_type) const {
  static std::set<std::string> kListDataType = {"listInt", "listStr", "listBool", "listFloat"};
  auto iter = kListDataType.find(attr_data_type);
  ValuePtr ret = value;
  if (iter != kListDataType.end()) {
    if (!value->isa<ValueSequence>()) {
      std::vector<ValuePtr> value_vec;
      value_vec.push_back(value);
      ret = std::make_shared<ValueTuple>(value_vec);
    }
  }
  return ret;
}
}  // namespace mindspore::opt
