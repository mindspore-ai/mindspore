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

#include "tools/converter/adapter/acl/mapper/primitive_mapper.h"
#include <map>
#include <vector>
#include "tools/converter/adapter/acl/common/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/adapter/acl/mapper/tbe_op_def.h"
#include "ir/graph_utils.h"
#include "include/errorcode.h"
#include "include/registry/converter_context.h"
#include "ops/op_utils.h"
#include "ops/fusion/avg_pool_fusion.h"
#include "ops/fusion/max_pool_fusion.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "src/common/log_util.h"

namespace mindspore {
namespace lite {
namespace {
constexpr auto kCommonAttrValueNum = 2;
constexpr auto kNamePaddingMode = "padding_mode";
constexpr auto kNameCeilMode = "ceil_mode";
}  // namespace

STATUS PrimitiveMapper::Mapper(const CNodePtr &cnode) { return lite::RET_OK; }

STATUS PrimitiveMapper::GetValueNodeAndPrimFromCnode(const CNodePtr &cnode, ValueNodePtr *value_node,
                                                     PrimitivePtr *prim_ptr) const {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(value_node);
  CHECK_NULL_RETURN(prim_ptr);
  CHECK_NULL_RETURN(cnode->input(0));

  *value_node = cnode->input(0)->cast<ValueNodePtr>();
  if (*value_node == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] is nullptr.";
    return lite::RET_ERROR;
  }
  *prim_ptr = GetValueNode<PrimitivePtr>(*value_node);
  if (*prim_ptr == nullptr) {
    MS_LOG(ERROR) << "Value node[" << cnode->fullname_with_scope() << "] cast to primitive failed.";
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

STATUS PrimitiveMapper::AttrAdjust(const PrimitivePtr &prim, const std::string &name) const {
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "prim is nullptr.");
  auto value_ptr = prim->GetAttr(name);
  if (value_ptr == nullptr) {
    MS_LOG(WARNING) << prim->name() << " has no attr " << name;
    return lite::RET_OK;
  }
  if (utils::isa<ValueSequencePtr>(value_ptr)) {
    auto val_seq_ptr = value_ptr->cast<ValueSequencePtr>();
    CHECK_NULL_RETURN(val_seq_ptr);
    ValuePtr first_val = nullptr;
    if (!val_seq_ptr->value().empty()) {
      first_val = val_seq_ptr->value().front();
    }
    CHECK_NULL_RETURN(first_val);
    CHECK_NULL_RETURN(first_val->type());
    if (first_val->type()->number_type() != kNumberTypeInt64) {
      MS_LOG(ERROR) << "Value number type of name: " << prim->name() << " ,please check the attr name: " << name;
      return lite::RET_ERROR;
    }
  } else {
    CHECK_NULL_RETURN(value_ptr->type());
    if (value_ptr->type()->number_type() != kNumberTypeInt64) {
      MS_LOG(ERROR) << "Value number type of name: " << prim->name() << " ,please check the attr name: " << name;
      return lite::RET_ERROR;
    }
  }
  auto origin_value = opt::CastToInt(value_ptr);
  if (origin_value.size() != kCommonAttrValueNum) {
    MS_LOG(ERROR) << name << " Value num must be two.";
    return lite::RET_ERROR;
  }
  int64_t format = Format::NCHW;
  if (prim->GetAttr(ops::kFormat) != nullptr) {
    format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
  }
  std::vector<int64_t> new_value = {1, 1, static_cast<int64_t>(origin_value[0]), static_cast<int64_t>(origin_value[1])};
  if (format == Format::NHWC) {
    std::vector<int64_t> tmp = {1, static_cast<int64_t>(origin_value[0]), static_cast<int64_t>(origin_value[1]), 1};
    new_value.swap(tmp);
  }
  prim->AddAttr(name, MakeValue(new_value));
  return lite::RET_OK;
}

void PrimitiveMapper::AdjustCaffePoolAttr(const std::string &src_prim_name, const PrimitivePtr &dst_prim) const {
  int64_t mode = src_prim_name == ops::kNameAvgPoolFusion ? 1 : 0;
  dst_prim->AddAttr(ops::kMode, MakeValue(mode));

  auto run_mode_val = dst_prim->GetAttr(ops::kRoundMode);
  if (run_mode_val == nullptr) {
    MS_LOG(INFO) << "There is no attr run mode";
    return;
  }
  auto run_mode = GetValue<int64_t>(run_mode_val);
  int64_t run_mode_ge = run_mode == RoundMode::FLOOR ? 1 : 0;
  dst_prim->set_attr(ops::kRoundMode, MakeValue(run_mode_ge));
}

void PrimitiveMapper::AdjustOnnxPoolAttr(const std::string &src_prim_name, const PrimitivePtr &dst_prim) const {
  auto pad_mode_val = dst_prim->GetAttr(ops::kPadMode);
  static std::map<int64_t, std::string> kPadModToStrMap = {
    {PadMode::PAD, "CALCULATED"},
    {PadMode::SAME, "SAME"},
    {PadMode::VALID, "VALID"},
  };
  if (pad_mode_val) {
    auto pad_mode = GetValue<int64_t>(pad_mode_val);
    std::string padding_mode = "CALCULATED";
    if (kPadModToStrMap.find(pad_mode) != kPadModToStrMap.end()) {
      padding_mode = kPadModToStrMap[pad_mode];
    }
    if (src_prim_name == ops::kNameMaxPool && padding_mode == "CALCULATED") {
      padding_mode = "VALID";
    }
    std::string pad_mode_name = src_prim_name == acl::kNameMaxPoolV3 ? kNamePaddingMode : ops::kPadMode;
    dst_prim->AddAttr(pad_mode_name, MakeValue(padding_mode));
  } else {
    MS_LOG(INFO) << "There is no attr pad mode";
  }
  auto run_mode_val = dst_prim->GetAttr(ops::kRoundMode);
  if (run_mode_val) {
    int64_t run_mode = GetValue<int64_t>(run_mode_val);
    bool ceil_mode = run_mode == RoundMode::CEIL;
    dst_prim->AddAttr(kNameCeilMode, MakeValue(ceil_mode));
  } else {
    MS_LOG(INFO) << "There is no attr run mode";
  }
}

STATUS PrimitiveMapper::AdjustPoolAttr(int fmk_type, const std::string &src_prim_name,
                                       const PrimitivePtr &dst_prim) const {
  if (fmk_type == converter::kFmkTypeCaffe) {
    AdjustCaffePoolAttr(src_prim_name, dst_prim);
    return lite::RET_OK;
  } else if (fmk_type == converter::kFmkTypeOnnx) {
    AdjustOnnxPoolAttr(src_prim_name, dst_prim);
  }
  // adjust common attr
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, lite::RET_ERROR, "dst_prim is nullptr.");
  auto status = AttrAdjust(dst_prim, ops::kKernelSize);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "Adjust kernel size failed.";
    return status;
  }
  status = AttrAdjust(dst_prim, ops::kStrides);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "adjust strides failed.";
    return status;
  }
  return lite::RET_OK;
}

STATUS PrimitiveMapper::MoveAttrMap(const CNodePtr &cnode, const PrimitivePtr &dst_prim) const {
  ValueNodePtr value_node = nullptr;
  PrimitivePtr src_prim = nullptr;
  if (GetValueNodeAndPrimFromCnode(cnode, &value_node, &src_prim) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get primitive from cnode failed.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, lite::RET_ERROR, "dst_prim is nullptr.");
  dst_prim->SetAttrs(src_prim->attrs());
  value_node->set_value(dst_prim);
  return lite::RET_OK;
}

STATUS PrimitiveMapper::AddFloatAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                            const PrimitivePtr &dst_prim, const std::string &attr_name,
                                            bool empty_shape) const {
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, lite::RET_ERROR, "dst_prim is nullptr.");
  auto attr_val = dst_prim->GetAttr(attr_name);
  if (attr_val == nullptr) {
    MS_LOG(INFO) << "There is no attr: " << attr_name;
    return lite::RET_OK;
  }
  auto param_name = cnode->fullname_with_scope() + "_" + attr_name;
  auto inputs = cnode->inputs();
  auto value_data = GetValue<float>(attr_val);
  auto param_node = opt::BuildFloatValueParameterNode(func_graph, value_data, param_name, empty_shape);

  MS_CHECK_TRUE_MSG(param_node != nullptr, lite::RET_ERROR, "param_node is nullptr.");
  param_node->set_debug_info(std::make_shared<NodeDebugInfo>(param_name));
  inputs.push_back(param_node);
  cnode->set_inputs(inputs);
  return lite::RET_OK;
}

STATUS PrimitiveMapper::AddIntVecAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                             const PrimitivePtr &dst_prim, const std::string &attr_name) const {
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, lite::RET_ERROR, "dst_prim is nullptr.");
  auto attr_val = dst_prim->GetAttr(attr_name);
  if (attr_val == nullptr) {
    MS_LOG(INFO) << "There is no attr: " << attr_name;
    return lite::RET_OK;
  }
  auto param_name = cnode->fullname_with_scope() + "_" + attr_name;
  auto inputs = cnode->inputs();
  auto value_data = opt::CastToVec2DInt(attr_val);
  auto param_node = opt::BuildIntVec2DParameterNode(func_graph, value_data, param_name);

  MS_CHECK_TRUE_MSG(param_node != nullptr, lite::RET_ERROR, "param_node is nullptr.");
  param_node->set_debug_info(std::make_shared<NodeDebugInfo>(param_name));
  inputs.push_back(param_node);
  cnode->set_inputs(inputs);
  return lite::RET_OK;
}

STATUS PrimitiveMapper::AddIntAttrToInput(const FuncGraphPtr &func_graph, const CNodePtr &cnode,
                                          const PrimitivePtr &dst_prim, const std::string &attr_name,
                                          bool empty_shape) const {
  MS_CHECK_TRUE_MSG(dst_prim != nullptr, lite::RET_ERROR, "dst_prim is nullptr.");
  auto attr_val = dst_prim->GetAttr(attr_name);
  if (attr_val == nullptr) {
    MS_LOG(INFO) << "There is no attr: " << attr_name;
    return lite::RET_OK;
  }
  auto param_name = cnode->fullname_with_scope() + "_" + attr_name;
  auto inputs = cnode->inputs();
  auto value_data = opt::CastToInt(attr_val);
  if (value_data.size() < 1) {
    MS_LOG(ERROR) << "Invalid size: " << value_data.size();
    return lite::RET_ERROR;
  }
  auto param_node = opt::BuildIntValueParameterNode(func_graph, value_data[0], param_name, empty_shape);

  MS_CHECK_TRUE_MSG(param_node != nullptr, lite::RET_ERROR, "param_node is nullptr.");
  param_node->set_debug_info(std::make_shared<NodeDebugInfo>(param_name));
  inputs.push_back(param_node);
  cnode->set_inputs(inputs);
  return lite::RET_OK;
}

STATUS PrimitiveMapper::AddAttrForDynInputPrimitive(const CNodePtr &cnode, const std::string &attr_name) const {
  CHECK_NULL_RETURN(cnode);
  CHECK_NULL_RETURN(cnode->input(0));
  auto value_node = cnode->input(0)->cast<ValueNodePtr>();
  CHECK_NULL_RETURN(value_node);
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  CHECK_NULL_RETURN(prim);
  // add attr input num for dynamic input op
  int64_t num = static_cast<int64_t>(cnode->size());
  if (num > 1) {
    prim->AddAttr(attr_name, MakeValue(num - 1));
  }
  return lite::RET_OK;
}

STATUS PrimitiveMapper::AdjustAttrFormat(const PrimitivePtr &prim, const std::string &name) const {
  int64_t format = Format::NCHW;
  MS_CHECK_TRUE_MSG(prim != nullptr, lite::RET_ERROR, "prim is nullptr.");
  if (prim->GetAttr(ops::kFormat) != nullptr) {
    format = GetValue<int64_t>(prim->GetAttr(ops::kFormat));
  }
  std::string format_str = "NCHW";
  if (format == Format::NHWC) {
    format_str = "NHWC";
  }
  prim->AddAttr(name, MakeValue(format_str));
  return lite::RET_OK;
}

CNodePtr PrimitiveMapper::NewCNode(const CNodePtr &cnode, const PrimitivePtr &primitive,
                                   const std::vector<AnfNodePtr> &inputs, const abstract::AbstractBasePtr &abstract,
                                   const std::string &name) const {
  auto func_graph = cnode->func_graph();
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to NewCNode, funcGraph cannot be nullptr";
    return nullptr;
  }
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "Failed to NewCNode, FuncGraph manager cannot be nullptr";
    return nullptr;
  }
  auto new_node = func_graph->NewCNode(primitive, inputs);
  if (new_node == nullptr) {
    MS_LOG(ERROR) << "Failed to create node " << name << " for node " << cnode->fullname_with_scope();
    return nullptr;
  }
  new_node->set_fullname_with_scope(name);
  for (size_t i = 0; i < inputs.size(); i++) {
    manager->SetEdge(new_node, i + 1, inputs[i]);
  }
  new_node->set_abstract(abstract);
  return new_node;
}

CNodePtr PrimitiveMapper::NewCNode(const CNodePtr &cnode, const PrimitivePtr &primitive,
                                   const std::vector<AnfNodePtr> &inputs, const ShapeVector &shape, TypeId type_id,
                                   const std::string &name) const {
  auto abstract =
    std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), std::make_shared<abstract::Shape>(shape));
  return NewCNode(cnode, primitive, inputs, abstract, name);
}
}  // namespace lite
}  // namespace mindspore
