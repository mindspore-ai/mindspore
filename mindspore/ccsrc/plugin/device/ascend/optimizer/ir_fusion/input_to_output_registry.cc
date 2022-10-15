/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/optimizer/ir_fusion/input_to_output_registry.h"
#include "include/common/utils/utils.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
bool ApplyRMSPropPreCheck(const CNodePtr &node) {
  return !(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0) != kNumberTypeFloat32);
}

bool FusedMulApplyMomentumPreCheck(const CNodePtr &node) {
  TypeId data_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  return !(data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16);
}

bool SparseApplyRMSPropPreCheck(const CNodePtr &node) {
  return !(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0) != kNumberTypeFloat32);
}

bool ApplyAdagradV2PreCheck(const CNodePtr &node) {
  TypeId data_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  return !(data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16);
}

bool ApplyKerasMomentumPreCheck(const CNodePtr &node) {
  TypeId data_type = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0);
  return !(data_type != kNumberTypeFloat32 && data_type != kNumberTypeFloat16);
}

bool SparseApplyFtrlPreCheck(const CNodePtr &node) {
  return !(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0) != kNumberTypeFloat32);
}

bool SparseApplyFtrlV2PreCheck(const CNodePtr &node) {
  return !(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0) != kNumberTypeFloat32);
}

bool SparseApplyAdagradV2PreCheck(const CNodePtr &node) {
  return !(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0) != kNumberTypeFloat32);
}

bool SparseApplyAdadeltaPreCheck(const CNodePtr &node) {
  return !(common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 0) != kNumberTypeFloat32);
}
}  // namespace
InputToOutputRegistry::InputToOutputRegistry() {
  Register(kApplyRMSPropOpName, {1, 2}, ApplyRMSPropPreCheck);
  Register(kFusedMulApplyMomentumOpName, {1}, FusedMulApplyMomentumPreCheck);
  Register(kApplyAdagradOpName, {1});
  Register(kApplyAdagradDAOpName, {1, 2});
  Register(kApplyAdadeltaOpName, {1, 2});
  Register(kApplyPowerSignOpName, {1});
  Register(kApplyProximalAdagradOpName, {1});
  Register(kApplyAdaMaxOpName, {1, 2});
  Register(kApplyAdagradV2OpName, {1}, ApplyAdagradV2PreCheck);
  Register(kApplyKerasMomentumOpName, {1}, ApplyKerasMomentumPreCheck);
  Register(kSparseApplyFtrlOpName, {1, 2}, SparseApplyFtrlPreCheck);
  Register(kSparseApplyFtrlV2OpName, {1, 2}, SparseApplyFtrlV2PreCheck);
  Register(kSparseApplyAdagradV2OpName, {1}, SparseApplyAdagradV2PreCheck);
  Register(kSparseApplyProximalAdagradOpName, {1});
  Register(kSparseApplyAdagradOpName, {1});
  Register(kApplyFtrlV2OpName, {1, 2});
  Register(kApplyMomentumOpName, {1});
  Register(kApplyFtrlOpName, {1, 2});
  Register(kApplyAdamOpName, {1, 2});
  Register(kApplyCenteredRMSPropOpName, {1, 2, 3});
  Register(kApplyAddSignOpName, {1});
  Register(kSparseApplyRMSPropOpName, {1, 2}, SparseApplyRMSPropPreCheck);
  Register(kSparseApplyAdadeltaOpName, {1, 2}, SparseApplyAdadeltaPreCheck);
  Register(kApplyAdamWithAmsgradOpName, {1, 2});
}

InputToOutputRegistry &InputToOutputRegistry::Instance() {
  static InputToOutputRegistry instance{};
  return instance;
}

void InputToOutputRegistry::Register(const InputToOutputRegister &reg) {
  auto op_name = reg.op_name();
  if (op_input_to_output_map_.find(op_name) == op_input_to_output_map_.end()) {
    (void)op_input_to_output_map_.emplace(op_name, reg);
    MS_LOG(DEBUG) << op_name << " input2output register successfully!";
  }
}

void InputToOutputRegistry::Register(const std::string &op_name, const std::vector<size_t> &input_indices,
                                     const PreCheckFunc &pre_check_func) {
  if (op_input_to_output_map_.find(op_name) == op_input_to_output_map_.end()) {
    InputToOutputRegister reg(op_name, pre_check_func);
    reg.set_input_indices(input_indices);
    (void)op_input_to_output_map_.emplace(op_name, reg);
    MS_LOG(DEBUG) << op_name << " input2output register successfully!";
  }
}

bool InputToOutputRegistry::GetRegisterByOpName(const std::string &op_name, InputToOutputRegister *reg) const {
  if (op_input_to_output_map_.find(op_name) != op_input_to_output_map_.end()) {
    *reg = op_input_to_output_map_.at(op_name);
    MS_LOG(DEBUG) << op_name << " input2output find in registry.";
    return true;
  }
  return false;
}
}  // namespace opt
}  // namespace mindspore
