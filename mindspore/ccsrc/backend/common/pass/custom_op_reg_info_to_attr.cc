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
#include "backend/common/pass/custom_op_reg_info_to_attr.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "kernel/oplib/opinfo.h"
#include "kernel/oplib/oplib.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
namespace {
constexpr auto kXs = "Xs";
constexpr auto kMCustom = "m_custom";
constexpr auto kRCustom = "r_custom";

void ParseAttrDefaultValue(const std::string &op_name, const std::string &attr_name, const std::string &attr_value,
                           const std::string &attr_type, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  try {
    if (attr_type == "int") {
      prim->set_attr(attr_name, std::make_shared<Int64Imm>(std::stoi(attr_value)));
    } else if (attr_type == "str") {
      prim->set_attr(attr_name, std::make_shared<StringImm>(attr_value));
    } else if (attr_type == "bool") {
      bool value = false;
      std::istringstream(attr_value) >> std::boolalpha >> value;
      prim->set_attr(attr_name, std::make_shared<BoolImm>(value));
    } else if (attr_type == "float") {
      prim->set_attr(attr_name, std::make_shared<FP32Imm>(std::stof(attr_value)));
    } else if (attr_type == "listInt") {
      std::stringstream ss(attr_value);
      std::string elem;
      std::vector<ValuePtr> value;
      while (std::getline(ss, elem, ',')) {
        value.push_back(std::make_shared<Int64Imm>(std::stoi(elem)));
      }
      prim->set_attr(attr_name, std::make_shared<ValueList>(value));
    } else if (attr_type == "listStr") {
      std::stringstream ss(attr_value);
      std::string elem;
      std::vector<ValuePtr> value;
      while (std::getline(ss, elem, ',')) {
        value.push_back(std::make_shared<StringImm>(elem));
      }
      prim->set_attr(attr_name, std::make_shared<ValueList>(value));
    } else if (attr_type == "listBool") {
      std::stringstream ss(attr_value);
      std::string elem;
      std::vector<ValuePtr> value;
      while (std::getline(ss, elem, ',')) {
        bool cur_value = false;
        std::istringstream(elem) >> std::boolalpha >> cur_value;
        value.push_back(std::make_shared<BoolImm>(cur_value));
      }
      prim->set_attr(attr_name, std::make_shared<ValueList>(value));
    } else if (attr_type == "listFloat") {
      std::stringstream ss(attr_value);
      std::string elem;
      std::vector<ValuePtr> value;
      while (std::getline(ss, elem, ',')) {
        value.push_back(std::make_shared<FP32Imm>(std::stof(elem)));
      }
      prim->set_attr(attr_name, std::make_shared<ValueList>(value));
    } else {
      MS_LOG(EXCEPTION) << "Unsupported attr type: " << attr_type;
    }
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Parse attr [" << attr_name << "] of op [" << op_name << "] failed! attr type: " << attr_type
                      << ", default value: " << attr_value << ", error message: " << e.what();
  }
}

void AddMissingAttrs(const CNodePtr &cnode, kernel::OpImplyType imply_type,
                     const std::unordered_set<std::string> &missing_attrs) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  primitive = primitive->Clone();
  auto op_name = common::AnfAlgo::GetCNodeName(cnode);
  auto op_info_ptr = mindspore::kernel::OpLib::FindOp(op_name, imply_type);
  MS_EXCEPTION_IF_NULL(op_info_ptr);
  auto all_attrs = op_info_ptr->attrs_ptr();
  bool need_update = false;
  for (const auto &attr : all_attrs) {
    auto attr_name = attr->name();
    if (missing_attrs.find(attr_name) == missing_attrs.end()) {
      continue;
    }
    // If attr's param_type is required, it should have default value.
    // If attr have default value, we should parse it no matter whether its param_type is required or not.
    auto default_value = attr->default_value();
    if (default_value.empty() && attr->param_type() != "required") {
      continue;
    }
    if (default_value.empty()) {
      MS_LOG(EXCEPTION) << "attr [" << attr_name << "] in the registration information of op [" << op_name
                        << "] does not have a value." << trace::DumpSourceLines(cnode);
    }
    ParseAttrDefaultValue(op_name, attr_name, default_value, attr->type(), primitive);
    need_update = true;
  }
  if (need_update) {
    cnode->set_input(kAnfPrimitiveIndex, NewValueNode(primitive));
  }
}

AnfNodePtr BuildCustom(const PatternMap &m, const AnfNodePtr &default_node) {
  auto cnode = m.Get(kMCustom)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto func_type = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrFuncType);
  auto attr_names = primitive->GetAttr(kAttrAttrNames);
  // Early return if all attr in reg info exist in the node's attr
  std::unordered_set<std::string> missing_attrs;
  auto attr_names_vec = GetValue<std::vector<std::string>>(attr_names);
  for (const auto &name : attr_names_vec) {
    if (!primitive->HasAttr(name)) {
      (void)missing_attrs.insert(name);
    }
  }
  if (missing_attrs.empty()) {
    return nullptr;
  }
  kernel::OpImplyType imply_type =
    func_type == kCustomTypeAICPU ? kernel::OpImplyType::kImplyAICPU : kernel::OpImplyType::kImplyAKG;
  AddMissingAttrs(cnode, imply_type, missing_attrs);

  return cnode;
}
}  // namespace

bool CustomOpRegInfoToAttr::CheckMatchedDAG(const PatternMap &, const FuncGraphPtr &, const AnfNodePtr &node) const {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto primitive = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(primitive);
  auto func_type = common::AnfAlgo::GetNodeAttr<std::string>(cnode, kAttrFuncType);
  // AKG/AICPU need to process attr, TBE will process later in the json creating phase.
  if (!IsOneOfCustomAkgType(func_type) || func_type == kCustomTypeAICPU) {
    return false;
  }
  // Early return if current node does not have attr
  auto attr_names = primitive->GetAttr(kAttrAttrNames);
  if (attr_names == nullptr) {
    return false;
  }
  return true;
}

void CustomOpRegInfoToAttr::DefineSrcPattern(SrcPattern *src_pattern) {
  (*src_pattern).AddSeqVar(kXs).AddCNode(kMCustom, {prim::kPrimCustom, kXs});
}

void CustomOpRegInfoToAttr::DefineDstPattern(DstPattern *dst_pattern) {
  (*dst_pattern).AddCNode(kRCustom, {prim::kPrimCustom, kXs}, BuildCustom);
}
}  // namespace opt
}  // namespace mindspore
