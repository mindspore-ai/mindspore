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

#include "mapper/custom_mapper.h"
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include "common/anf_util.h"
#include "common/op_attr.h"
#include "op/custom_operator.h"
#include "ops/custom.h"

namespace mindspore {
namespace dpico {
namespace {
custom::ExtendedParam GetParamFromAttrs(const api::SharedPtr<ops::Custom> &custom_prim, int index) {
  custom::ExtendedParam custom_param;
  if (custom_prim == nullptr) {
    MS_LOG(ERROR) << "custom_prim is nullptr.";
    return custom_param;
  }
  auto name_ptr = custom_prim->GetAttr("name" + std::to_string(index));
  if (name_ptr != nullptr) {
    custom_param.name = api::GetValue<string>(name_ptr);
  }
  auto type_ptr = custom_prim->GetAttr("type" + std::to_string(index));
  if (type_ptr != nullptr) {
    custom_param.type = static_cast<custom::AttributeType>(api::GetValue<int64_t>(type_ptr));
  }
  auto f_ptr = custom_prim->GetAttr("f" + std::to_string(index));
  if (f_ptr != nullptr) {
    custom_param.paramFloat = api::GetValue<float>(f_ptr);
  }
  auto i_ptr = custom_prim->GetAttr("i" + std::to_string(index));
  if (i_ptr != nullptr) {
    custom_param.paramInt = api::GetValue<int64_t>(i_ptr);
  }
  auto s_ptr = custom_prim->GetAttr("s" + std::to_string(index));
  if (s_ptr != nullptr) {
    custom_param.paramString = api::GetValue<std::string>(s_ptr);
  }
  auto floats_ptr = custom_prim->GetAttr("floats" + std::to_string(index));
  if (floats_ptr != nullptr) {
    custom_param.paramFloats = api::GetValue<std::vector<float>>(floats_ptr);
  }
  auto ints_ptr = custom_prim->GetAttr("ints" + std::to_string(index));
  if (ints_ptr != nullptr) {
    custom_param.paramInts = api::GetValue<std::vector<int64_t>>(ints_ptr);
  }
  auto strings_ptr = custom_prim->GetAttr("strings" + std::to_string(index));
  if (strings_ptr != nullptr) {
    custom_param.paramStrings = api::GetValue<std::vector<std::string>>(strings_ptr);
  }
  return custom_param;
}
}  // namespace

STATUS CustomMapper::Map(const api::CNodePtr &cnode, std::vector<BaseOperatorPtr> *base_operators,
                         const api::PrimitivePtr &prim, const api::CNodePtrList &output_cnodes) {
  if (base_operators == nullptr) {
    MS_LOG(ERROR) << "base_operators is nullptr.";
    return RET_ERROR;
  }
  auto custom_prim = api::utils::cast<api::SharedPtr<ops::Custom>>(prim);
  MS_CHECK_TRUE_MSG(custom_prim != nullptr, RET_ERROR, "custom_prim is nullptr");
  auto custom_operator = std::make_unique<mapper::CustomOperator>();
  if (SetCommonAttr(cnode, custom_operator.get(), output_cnodes) != RET_OK) {
    MS_LOG(ERROR) << "set common attr failed. " << cnode->fullname_with_scope();
    return RET_ERROR;
  }

  custom_operator->SetOpType(mapper::OpType::CUSTOM);
  if (prim->GetAttr(kExtendedOpType) != nullptr) {
    custom_operator->SetExtendedOpType(api::GetValue<std::string>(prim->GetAttr(kExtendedOpType)));
  }

  std::vector<custom::ExtendedAttr> attr_vec;
  custom::ExtendedAttr extended_attrs;

  int custom_param_size = 0;
  auto custom_param_size_attr = custom_prim->GetAttr(kCustomParamSize);
  if (custom_param_size_attr != nullptr) {
    custom_param_size = static_cast<int>(api::GetValue<int64_t>(custom_param_size_attr));
  }
  if (custom_param_size == 0) {
    MS_LOG(INFO) << "no custom param attr found.";
    return RET_OK;
  }
  for (int i = 0; i < custom_param_size; i++) {
    extended_attrs.SetExtendedParam(GetParamFromAttrs(custom_prim, i));
    attr_vec.emplace_back(extended_attrs);
  }
  custom_operator->SetExtendedAttrs(attr_vec);
  base_operators->push_back(std::move(custom_operator));
  return RET_OK;
}
REG_MAPPER(Custom, CustomMapper)
}  // namespace dpico
}  // namespace mindspore
