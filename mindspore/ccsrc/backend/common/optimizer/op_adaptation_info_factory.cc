/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "include/backend/optimizer/op_adaptation_info_factory.h"

#include <memory>
#include "kernel/oplib/oplib.h"
#include "utils/log_adapter.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/convert_utils.h"
#include "include/backend/optimizer/helper.h"
#include "ops/framework_ops.h"

namespace mindspore::opt {
OpAdaptationInfo &OpAdaptationInfo::set_backend_op_name(const std::string &default_op_name) {
  backend_op_name_ = default_op_name;
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_target_op_name(const std::string &target_op_name) {
  target_op_name_ = target_op_name;
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_pre_check_func(std::function<bool(CNodePtr)> pre_check_func) {
  pre_check_func_ = std::move(pre_check_func);
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_need_tbe_check_supported(bool need_tbe_check_supported) {
  need_tbe_check_supported_ = need_tbe_check_supported;
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_input_attr_info(size_t input_index, const std::string &attr_data_type) {
  auto find = input_attr_map_.find(input_index);
  if (find != input_attr_map_.end()) {
    MS_LOG(ERROR) << "This input index (" << input_index << ")"
                  << " has been registered.";
    return *this;
  }
  input_attr_map_[input_index] = attr_data_type;
  return *this;
}

OpAdaptationInfo &OpAdaptationInfo::set_is_ascend_mindir() {
  is_ascend_mindir_ = true;
  return *this;
}

OpAdaptationInfoRegister &OpAdaptationInfoRegister::GetInstance() {
  static OpAdaptationInfoRegister inst;
  return inst;
}

std::string OpAdaptationInfoRegister::GenerateKey(const std::string &me_op_name, const std::string &device_name,
                                                  bool flag) {
  if (device_name != kCPUDevice && device_name != kGPUDevice && device_name != kAscendDevice) {
    MS_LOG(EXCEPTION) << "Backend type is invalid, should be one of [" << kCPUDevice << ", " << kGPUDevice << ", "
                      << kAscendDevice << "], but got " << device_name;
  }

  std::string flag_str = flag ? "true" : "false";
  return std::string(me_op_name + device_name + flag_str);
}

std::set<std::string> &OpAdaptationInfoRegister::GetOpName() {
  static std::set<std::string> op_names;
  return op_names;
}

std::map<std::string, OpAdaptationInfo *> &OpAdaptationInfoRegister::GetOpInfoMap() {
  static std::map<std::string, OpAdaptationInfo *> op_info_map;
  return op_info_map;
}

void OpAdaptationInfoRegister::RegOpAdaptationInfo(OpAdaptationInfo *reg_info) {
  MS_EXCEPTION_IF_NULL(reg_info);
  (void)GetOpName().insert(reg_info->me_op_name());
  auto key = GenerateKey(reg_info->me_op_name(), reg_info->device_name(), reg_info->flag());
  auto find = GetOpInfoMap().find(key);
  if (find != GetOpInfoMap().end()) {
    MS_LOG(DEBUG) << "This key (" << key << ") has been registered in me op info map.";
    return;
  }
  MS_LOG(DEBUG) << "Reg op adaptation info to factory, key: " << key;
  GetOpInfoMap()[key] = reg_info;
}

OpAdaptationInfo *OpAdaptationInfoRegister::GetOpAdaptationInfo(const std::string &me_op_name,
                                                                const std::string &device_name, bool flag) {
  auto name_iter = GetOpName().find(me_op_name);
  if (name_iter == GetOpName().end()) {
    return nullptr;
  }
  auto key = GenerateKey(me_op_name, device_name, flag);
  auto iter = GetOpInfoMap().find(key);
  if (iter == GetOpInfoMap().end()) {
    MS_LOG(DEBUG) << "Can't find op adaptation for op " << me_op_name << " on " << device_name << " when flag is "
                  << flag;
    return nullptr;
  }
  return iter->second;
}

CNodePtr OpAdaptationInfoRegister::CreateTargetOp(const CNodePtr &origin_op,
                                                  const OpAdaptationInfo &op_adaptation_info) {
  MS_EXCEPTION_IF_NULL(origin_op);
  auto target_op_name = op_adaptation_info.target_op_name();
  auto input_attr_info_map = op_adaptation_info.input_attr_map();

  auto origin_primitive = GetCNodePrimitive(origin_op);
  MS_EXCEPTION_IF_NULL(origin_primitive);
  auto target_primitive = std::make_shared<Primitive>(target_op_name);
  MS_EXCEPTION_IF_NULL(target_primitive);
  (void)target_primitive->SetAttrs(origin_primitive->attrs());
  std::vector<AnfNodePtr> target_inputs;
  auto inputs = origin_op->inputs();
  target_inputs.push_back(inputs[0]);
  bool ir_change = false;
  for (size_t i = 0; i < inputs.size() - 1; ++i) {
    auto input_node = inputs[i + 1];
    MS_EXCEPTION_IF_NULL(input_node);
    if (IsPrimitiveCNode(input_node, prim::kPrimDepend)) {
      input_node = AnfUtils::VisitKernel(input_node, 0).first;
    }

    auto iter = input_attr_info_map.find(i);
    if (iter != input_attr_info_map.end()) {
      auto is_value_node = input_node->isa<ValueNode>();
      auto is_monad = HasAbstractMonad(input_node);
      if (!is_value_node || is_monad) {
        MS_LOG(INFO) << "Convert " << origin_op->fullname_with_scope() << "'s input " << i
                     << " to attr failed. input is value node: " << is_value_node << ", is monad: " << is_monad;
        return nullptr;
      }

      auto ret = ConvertInputToAttr(origin_op, i, input_node, iter->second, target_primitive);
      if (!ret) {
        MS_LOG(INFO) << "Convert " << origin_op->fullname_with_scope() << "'s input " << i << " to attr failed.";
        return nullptr;
      }
      ir_change = true;
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
  target_op->set_primal_debug_infos(origin_op->primal_debug_infos());
  common::AnfAlgo::EraseNodeAttr(kAttrIsKernelDynamicImpl, target_op);
  if (common::AnfAlgo::HasNodeAttr(kAttrCustAicpu, origin_op)) {
    common::AnfAlgo::CopyNodeAttr(kAttrCustAicpu, origin_op, target_op);
  }

  common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), target_op);
  common::AnfAlgo::SetNodeAttr(kAttrMeOpName, MakeValue(op_adaptation_info.me_op_name()), target_op);
  common::AnfAlgo::SetNodeAttr(kAttrIRChange, MakeValue(ir_change), target_op);

  auto is_dynamic = common::AnfAlgo::IsDynamicShape(origin_op);
  MS_LOG(DEBUG) << "Create op " << target_op->fullname_with_scope() << ", debug string:" << target_op->DebugString()
                << ", attr text:" << target_primitive->GetAttrsText() << " from " << origin_op->fullname_with_scope()
                << ", debug string:" << origin_op->DebugString() << ", attr text:" << origin_primitive->GetAttrsText()
                << ", is dynamic shape:" << is_dynamic;
  return target_op;
}

bool OpAdaptationInfoRegister::ConvertInputToAttr(const CNodePtr &origin_op, size_t i,
                                                  const std::shared_ptr<AnfNode> &input_node,
                                                  const std::string &attr_data_type,
                                                  const std::shared_ptr<Primitive> &target_primitive) {
  auto value_node = input_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  MS_LOG(DEBUG) << "start erase input[" << i
                << "] of cnode[" + origin_op->DebugString() + "], origin value:" << value_node->ToString()
                << ", Type:" << value_node->type_name();

  auto value = value_node->value();
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    if (tensor->data().const_data() == nullptr && !tensor->has_user_data(kTensorValueIsEmpty)) {
      MS_LOG(DEBUG) << "Const input data ptr is null from op " << origin_op->fullname_with_scope() << "'s input " << i;
      return false;
    }
    value = CreateValueFromTensor(tensor);
    value = UpdateValueByAttrDataType(value, attr_data_type);
    MS_LOG(DEBUG) << "new attr value:" << value_node->ToString() << ", Type:" << value_node->type_name();
  }

  std::string attr_name = common::AnfAlgo::GetInputName(origin_op, i);
  if (attr_name.empty()) {
    MS_LOG(DEBUG) << "Attr name is empty.";
    return false;
  }

  if (origin_op->HasAttr(attr_name)) {
    auto origin_primitive = GetCNodePrimitive(origin_op);
    MS_EXCEPTION_IF_NULL(origin_primitive);
    MS_LOG(ERROR) << "Origin op already has this attr " << attr_name
                  << ". op attrs:" << origin_primitive->GetAttrsText() << ". DebugString:" << origin_op->DebugString();
    return false;
  }

  target_primitive->set_attr(attr_name, value);
  return true;
}

void OpAdaptationInfoRegister::RenamePrimitiveName(const CNodePtr &origin_op, const string &me_op_name,
                                                   const string &backend_op_name) {
  MS_EXCEPTION_IF_NULL(origin_op);
  if (backend_op_name == me_op_name) {
    return;
  }
  auto primitive = GetCNodePrimitive(origin_op);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive->set_name(backend_op_name);
  // reset full scope name
  origin_op->set_fullname_with_scope("");
  MS_LOG(INFO) << "Rename op type from " << me_op_name << " to " << backend_op_name << " for op "
               << origin_op->fullname_with_scope();
  if (me_op_name == kSparseGatherV2OpName) {
    common::AnfAlgo::SetNodeAttr(kAttrIsSparse, MakeValue(true), origin_op);
  }
  common::AnfAlgo::SetNodeAttr(kAttrOpAdaptationProcessed, MakeValue(true), origin_op);
}

RegisterHelper::RegisterHelper(const string &me_op_name, const string &device_name, bool flag, int len, ...) {
  mindspore::HashSet<size_t> input_to_attr;
  input_to_attr.reserve(static_cast<size_t>(IntToUint(len)));
  va_list var_ptr;
  va_start(var_ptr, len);
  for (int i = 0; i < len; ++i) {
    (void)input_to_attr.insert(static_cast<size_t>(IntToUint(va_arg(var_ptr, int))));
  }
  va_end(var_ptr);
  op_adaptation_info_ = std::make_shared<OpAdaptationInfo>(me_op_name, device_name, flag);
  MS_EXCEPTION_IF_NULL(op_adaptation_info_);
  for (auto &index : input_to_attr) {
    (void)op_adaptation_info_->set_input_attr_info(index);
  }
  opt::OpAdaptationInfoRegister::GetInstance().RegOpAdaptationInfo(op_adaptation_info_.get());
}

RegisterHelper::RegisterHelper(const OpAdaptationInfo &op_adaptation_info) {
  op_adaptation_info_ = std::make_shared<OpAdaptationInfo>(op_adaptation_info);
  MS_EXCEPTION_IF_NULL(op_adaptation_info_);
  opt::OpAdaptationInfoRegister::GetInstance().RegOpAdaptationInfo(op_adaptation_info_.get());
}
OpAdaptationInfo &OpAdaptationInfo::operator=(const OpAdaptationInfo &op_adaptation_info) {
  if (this == &op_adaptation_info) {
    return *this;
  }
  me_op_name_ = op_adaptation_info.me_op_name_;
  backend_op_name_ = op_adaptation_info.backend_op_name_;
  target_op_name_ = op_adaptation_info.target_op_name_;
  pre_check_func_ = op_adaptation_info.pre_check_func_;
  need_tbe_check_supported_ = op_adaptation_info.need_tbe_check_supported_;
  input_attr_map_ = op_adaptation_info.input_attr_map_;
  device_name_ = op_adaptation_info.device_name_;
  flag_ = op_adaptation_info.flag_;
  return *this;
}
}  // namespace mindspore::opt
