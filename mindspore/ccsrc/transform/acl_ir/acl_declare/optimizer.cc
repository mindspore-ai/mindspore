/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "transform/acl_ir/acl_adapter_info.h"

namespace mindspore {
namespace transform {
REGISTER_ACL_OP(AdamApplyOneWithDecay).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdaMaxD).set_run_mode(false);
REGISTER_ACL_OP(ApplyMomentum).set_run_mode(false);
REGISTER_ACL_OP(ApplyMomentumD).set_run_mode(false);
REGISTER_ACL_OP(ApplyKerasMomentumD).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdamWithAmsgradD).set_run_mode(false);
REGISTER_ACL_OP(ApplyPowerSignD).set_run_mode(false);
REGISTER_ACL_OP(ApplyProximalGradientDescent).set_run_mode(false);
REGISTER_ACL_OP(ApplyAddSignD).set_run_mode(false);
REGISTER_ACL_OP(ApplyCenteredRMSProp).set_run_mode(false);
REGISTER_ACL_OP(ApplyGradientDescent).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdagradD).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdagradV2D).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdagradDA).set_run_mode(false);
REGISTER_ACL_OP(SGD).set_run_mode(false);
REGISTER_ACL_OP(ApplyRMSProp).set_run_mode(false);
REGISTER_ACL_OP(ApplyRMSPropD).set_run_mode(false);
REGISTER_ACL_OP(ApplyProximalAdagrad).set_run_mode(false);
REGISTER_ACL_OP(ApplyProximalAdagradD).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyProximalAdagrad).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyProximalAdagradD).set_run_mode(false);
REGISTER_ACL_OP(ApplyFtrl).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdam).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdamD).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdamWithAmsgradV2).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdadelta).set_run_mode(false);
REGISTER_ACL_OP(ApplyAdadeltaD).set_run_mode(false);
REGISTER_ACL_OP(LarsV2Update).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyFtrl).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyFtrlD).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyFtrlV2).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyFtrlV2D).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyRMSProp).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyRMSPropD).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyAdadelta).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyAdagradD).set_run_mode(false);
REGISTER_ACL_OP(SparseApplyAdagradV2D).set_run_mode(false);
REGISTER_ACL_OP(LambApplyOptimizerAssign).set_run_mode(false);
}  // namespace transform
}  // namespace mindspore
