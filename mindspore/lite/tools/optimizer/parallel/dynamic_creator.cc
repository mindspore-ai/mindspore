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

#include "tools/optimizer/parallel/dynamic_creator.h"
#include "tools/optimizer/parallel/conv2d_info.h"

namespace mindspore {
namespace opt {
// operator register
REGISTER(Conv2DInfo);
REGISTER(DepthwiseConv2DInfo);

std::string GetDisOpName(const std::string &prim_name) {
  std::string op_name = prim_name;
  if (!prim_name.empty() && (prim_name[0] == '_')) {
    op_name = prim_name.substr(1);
  }
  return op_name + "Info";
}

// create the OperatorInfo instance
OperatorInfoPtr OperatorInstance(const std::string &type_name, const std::string &orig_name,
                                 const SplitStrategy &strategy) {
  if (type_name.empty()) {
    MS_LOG(EXCEPTION) << "Length of name is zero!";
  }
  std::string distribute_opname = GetDisOpName(type_name);
  OperatorInfoPtr operator_ = (OperatorInfoPtr)DynCreator::Instance().Create(distribute_opname, strategy);
  if (operator_ == nullptr) {
    MS_LOG(INFO) << "Create " << type_name << " failed";
    return nullptr;
  }
  std::string origin_name = operator_->name();
  operator_->set_name(orig_name);
  MS_LOG(INFO) << "Successfully created operator " << origin_name;
  return operator_;
}

}  // namespace opt
}  // namespace mindspore
