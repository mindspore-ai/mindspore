/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "checker/activation_checker.h"
#include <vector>
#include <unordered_set>
#include "mindapi/base/types.h"
#include "common/op_enum.h"

namespace mindspore {
namespace dpico {
namespace {
const std::unordered_set<ActivationType> kSupportedActivationTypes = {
  ActivationType::RELU, ActivationType::RELU6,  ActivationType::LEAKY_RELU, ActivationType::SIGMOID,
  ActivationType::TANH, ActivationType::HSWISH, ActivationType::HARD_TANH,  ActivationType::ELU};
}  // namespace
bool ActivationChecker::Check(api::CNodePtr op, int32_t output_num, mindspore::Format format) {
  if (!CheckInputW(op, 1, format, kMaxInputWOf4Dims)) {
    MS_LOG(INFO) << "input_w is not supported. " << op->fullname_with_scope();
    return false;
  }
  auto primitive = api::GetValueNode<api::PrimitivePtr>(op->input(0));
  if (primitive == nullptr) {
    MS_LOG(ERROR) << "primitive is nullptr";
    return false;
  }
  auto value_ptr = primitive->GetAttr(ops::kActivationType);
  if (value_ptr == nullptr) {
    MS_LOG(ERROR) << "kActivationType attr is nullptr.";
    return false;
  }
  auto activation_type = static_cast<mindspore::ActivationType>(api::GetValue<int64_t>(value_ptr));
  if (kSupportedActivationTypes.find(activation_type) == kSupportedActivationTypes.end()) {
    MS_LOG(WARNING) << "Not supported activation type: " << activation_type << ", will turn it to custom op. "
                    << op->fullname_with_scope();
    return false;
  }
  return true;
}

OpCheckerRegistrar g_ActivationChecker("Activation", new ActivationChecker());
}  // namespace dpico
}  // namespace mindspore
