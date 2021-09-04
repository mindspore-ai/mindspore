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

#include "tools/converter/acl/mapper/activation_mapper.h"
#include <map>
#include <memory>
#include "tools/converter/acl/mapper/primitive_mapper_register.h"
#include "ops/elu.h"
#include "ops/gelu.h"
#include "ops/leaky_relu.h"
#include "ops/relu.h"
#include "ops/relu6.h"
#include "ops/sigmoid.h"
#include "ops/tanh.h"

namespace mindspore {
namespace lite {
STATUS ActivationMapper::Mapper(const CNodePtr &cnode) {
  static std::map<ActivationType, PrimitivePtr> activation_type_map = {
    {mindspore::ELU, std::make_shared<ops::Elu>()},
    {mindspore::GELU, std::make_shared<ops::GeLU>()},
    {mindspore::RELU, std::make_shared<ops::ReLU>()},
    {mindspore::RELU6, std::make_shared<ops::ReLU6>()},
    {mindspore::SIGMOID, std::make_shared<ops::Sigmoid>()},
    {mindspore::TANH, std::make_shared<ops::Tanh>()},
    {mindspore::LEAKY_RELU, std::make_shared<ops::LeakyRelu>()}};
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  auto activate_prim = dynamic_cast<ops::Activation *>(src_prim.get());
  if (activate_prim == nullptr) {
    MS_LOG(ERROR) << "Dynamic cast activation failed.";
    return lite::RET_ERROR;
  }
  PrimitivePtr dst_prim = nullptr;
  ActivationType type = activate_prim->get_activation_type();
  if (activation_type_map.find(type) != activation_type_map.end()) {
    dst_prim = activation_type_map[type];
  } else {
    MS_LOG(ERROR) << "Type " << static_cast<int>(type) << " is unsupported.";
    return lite::RET_ERROR;
  }
  MS_ASSERT(dst_prim != nullptr);
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

REGISTER_PRIMITIVE_MAPPER(kNameActivation, ActivationMapper)
}  // namespace lite
}  // namespace mindspore
