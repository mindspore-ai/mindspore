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

#include "tools/optimizer/parallel/operator_info_register.h"
#include <utility>
namespace mindspore {
namespace opt {

// find the only key of operator_info
bool SplitOpKey::operator<(const SplitOpKey &rhs) const {
  return std::tie(this->op_type_, this->data_type_, this->is_depth_wise_) <
         std::tie(rhs.op_type_, rhs.data_type_, rhs.is_depth_wise_);
}

std::string SplitOpKey::ToString() const {
  std::ostringstream split_info;
  split_info << "op_type: " << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_type_)) << "\t"
             << " data_type_: " << data_type_ << " is_depth_wise: " << is_depth_wise_ << "\t";
  return split_info.str();
}

OperatorInfoFactory *OperatorInfoFactory::GeInstance() {
  static OperatorInfoFactory factory;
  return &factory;
}

void OperatorInfoFactory::RegisterOperatorInfo(schema::PrimitiveType operator_type, TypeId type_id, bool is_depth_wise,
                                               const OperatorInfoCreatorFunc &creator_func) {
  // create a key to find the only create function
  SplitOpKey op_key(operator_type, type_id, is_depth_wise);
  if (operator_info_map_.find(op_key) != operator_info_map_.end()) {
    MS_LOG(ERROR) << " Operator already exist " << op_key.ToString();
    return;
  }
  this->operator_info_map_.insert(std::pair<SplitOpKey, OperatorInfoCreatorFunc>(op_key, creator_func));
}

OperatorInfoCreatorFunc OperatorInfoFactory::FindOperatorInfo(const SplitOpKey &op_key) {
  auto iterator = this->operator_info_map_.find(op_key);
  if (iterator != this->operator_info_map_.end()) {
    return iterator->second;
  }
  MS_LOG(ERROR) << "operator_info do not register" << op_key.ToString();
  return nullptr;
}

OperatorInfoRegister::OperatorInfoRegister(schema::PrimitiveType operator_type, TypeId type_id, bool is_depth_wise,
                                           const OperatorInfoCreatorFunc &creator_func) {
  OperatorInfoFactory::GeInstance()->RegisterOperatorInfo(operator_type, type_id, is_depth_wise, creator_func);
}
}  // namespace opt
}  // namespace mindspore
