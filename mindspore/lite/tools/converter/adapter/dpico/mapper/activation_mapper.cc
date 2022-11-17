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

#include "mapper/activation_mapper.h"
#include <memory>
#include <vector>
#include <utility>
#include <unordered_map>
#include "common/op_attr.h"
#include "ops/fusion/activation.h"
#include "op/relu_operator.h"
#include "op/sigmoid_operator.h"
#include "op/tanh_operator.h"
#include "op/hswish_operator.h"
#include "op/clip_operator.h"
#include "op/elu_operator.h"

namespace mindspore {
namespace dpico {
namespace {
constexpr float kNum6 = 6.0;
mapper::AfOperator *ReluMapFunc(const api::SharedPtr<ops::Activation> &activation_prim) {
  auto activation_operator = new (std::nothrow) mapper::ReluOperator();
  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "relu operator is nullptr. ";
    return nullptr;
  }
  activation_operator->SetOpType(mapper::OpType::RELUAF);
  if (activation_prim->GetAttr(ops::kAlpha) != nullptr) {
    static_cast<mapper::ReluOperator *>(activation_operator)->SetAlpha(activation_prim->get_alpha());
  }
  return activation_operator;
}

mapper::AfOperator *Relu6MapFunc(const api::SharedPtr<ops::Activation> &activation_prim) {
  auto activation_operator = new (std::nothrow) mapper::ClipOperator();

  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "relu operator is nullptr. ";
    return nullptr;
  }
  static_cast<mapper::ClipOperator *>(activation_operator)->SetClipCeil(kNum6);
  if (activation_prim->GetAttr(ops::kAlpha) != nullptr) {
    static_cast<mapper::ClipOperator *>(activation_operator)->SetAlpha(activation_prim->get_alpha());
  } else {
    static_cast<mapper::ClipOperator *>(activation_operator)->SetClipFloor(0.0);
  }
  return activation_operator;
}

mapper::AfOperator *HardTanhMapFunc(const api::SharedPtr<ops::Activation> &activation_prim) {
  auto activation_operator = new (std::nothrow) mapper::ClipOperator();
  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "relu operator is nullptr. ";
    return nullptr;
  }
  static_cast<mapper::ClipOperator *>(activation_operator)->SetClipFloor(activation_prim->get_min_val());
  static_cast<mapper::ClipOperator *>(activation_operator)->SetClipCeil(activation_prim->get_max_val());
  return activation_operator;
}

mapper::AfOperator *SigmoidMapFunc(const api::SharedPtr<ops::Activation> &) {
  auto activation_operator = new (std::nothrow) mapper::SigmoidOperator();
  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "sigmoid operator is nullptr. ";
    return nullptr;
  }
  activation_operator->SetOpType(mapper::OpType::SIGMOIDAF);
  return activation_operator;
}

mapper::AfOperator *TanhMapFunc(const api::SharedPtr<ops::Activation> &) {
  auto activation_operator = new (std::nothrow) mapper::TanhOperator();
  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "tanh operator is nullptr. ";
    return nullptr;
  }
  return activation_operator;
}

mapper::AfOperator *HswishMapFunc(const api::SharedPtr<ops::Activation> &activation_prim) {
  auto activation_operator = new (std::nothrow) mapper::HswishOperator();
  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "hswish operator is nullptr. ";
    return nullptr;
  }
  activation_operator->SetOpType(mapper::OpType::HSWISH);
  if (activation_prim->GetAttr(dpico::kNegativeSlope) != nullptr) {
    static_cast<mapper::HswishOperator *>(activation_operator)
      ->SetHswishSlope(api::GetValue<float>(activation_prim->GetAttr(dpico::kNegativeSlope)));
  }
  return activation_operator;
}

mapper::AfOperator *EluMapFunc(const api::SharedPtr<ops::Activation> &activation_prim) {
  auto activation_operator = new (std::nothrow) mapper::EluOperator();
  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "elu operator is nullptr. ";
    return nullptr;
  }
  activation_operator->SetOpType(mapper::OpType::ELU);
  if (activation_prim->GetAttr(ops::kAlpha) != nullptr) {
    static_cast<mapper::EluOperator *>(activation_operator)
      ->SetEluAlpha(api::GetValue<float>(activation_prim->GetAttr(ops::kAlpha)));
  }
  return activation_operator;
}

using ActivationMapFunc = mapper::AfOperator *(*)(const api::SharedPtr<ops::Activation> &activation_prim);
const std::unordered_map<ActivationType, ActivationMapFunc> kActivationMapFuncs = {
  {ActivationType::RELU, &ReluMapFunc},       {ActivationType::LEAKY_RELU, &ReluMapFunc},
  {ActivationType::RELU6, &Relu6MapFunc},     {ActivationType::HARD_TANH, &HardTanhMapFunc},
  {ActivationType::SIGMOID, &SigmoidMapFunc}, {ActivationType::TANH, &TanhMapFunc},
  {ActivationType::HSWISH, &HswishMapFunc},   {ActivationType::ELU, &EluMapFunc}};
}  // namespace

STATUS ActivationMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                             const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto activation_prim = api::utils::cast<api::SharedPtr<ops::Activation>>(prim);
  MS_ASSERT(activation_prim != nullptr);

  auto activation_type = activation_prim->get_activation_type();
  if (kActivationMapFuncs.find(activation_type) == kActivationMapFuncs.end()) {
    MS_LOG(ERROR) << "cur activation map is unsupported " << activation_type << ". node name is "
                  << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto activation_map_func = kActivationMapFuncs.at(activation_type);
  if (activation_map_func == nullptr) {
    MS_LOG(ERROR) << "activation_map_func is nullptr. " << activation_type << ". node name is "
                  << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  auto activation_operator = activation_map_func(activation_prim);
  if (activation_operator == nullptr) {
    MS_LOG(ERROR) << "map " << cnode->fullname_with_scope() << " to operator failed.";
    return RET_ERROR;
  }
  if (SetCommonAttr(cnode, activation_operator, output_cnodes) != RET_OK) {
    delete activation_operator;
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  if (PushOfflineArgs(cnode, activation_operator, 1) != RET_OK) {
    delete activation_operator;
    MS_LOG(ERROR) << "push offline args failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  base_operators->push_back(std::unique_ptr<mapper::AfOperator>(activation_operator));

  return RET_OK;
}
REG_MAPPER(Activation, ActivationMapper)
}  // namespace dpico
}  // namespace mindspore
