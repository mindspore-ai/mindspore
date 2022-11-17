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

#include "parser/caffe/caffe_custom_parser.h"
#include <memory>
#include <string>
#include "ops/custom.h"
#include "common/op_attr.h"
#include "./extended_attr.h"
#include "./pico_caffe.pb.h"
#include "op/custom_operator.h"

namespace mindspore {
namespace lite {
namespace {
int GetAttributeType(custom::AttributeType *attribue_type, const caffe::CustomAttribute_AttributeType &param_type) {
  if (attribue_type == nullptr) {
    MS_LOG(ERROR) << "input attribue_type is nullptr. ";
    return RET_ERROR;
  }
  switch (param_type) {
    case caffe::CustomAttribute_AttributeType_UNDEFINED:
      *attribue_type = custom::AttributeType::UNDEFINED;
      break;
    case caffe::CustomAttribute_AttributeType_FLOAT:
      *attribue_type = custom::AttributeType::FLOAT;
      break;
    case caffe::CustomAttribute_AttributeType_INT:
      *attribue_type = custom::AttributeType::INT;
      break;
    case caffe::CustomAttribute_AttributeType_STRING:
      *attribue_type = custom::AttributeType::STRING;
      break;
    case caffe::CustomAttribute_AttributeType_FLOATS:
      *attribue_type = custom::AttributeType::FLOATS;
      break;
    case caffe::CustomAttribute_AttributeType_INTS:
      *attribue_type = custom::AttributeType::INTS;
      break;
    case caffe::CustomAttribute_AttributeType_STRINGS:
      *attribue_type = custom::AttributeType::STRINGS;
      break;
    default:
      MS_LOG(ERROR) << "Unsupported Param Type: " << param_type;
      return RET_ERROR;
  }
  return RET_OK;
}
int SetAttrsByParam(const std::shared_ptr<ops::Custom> &custom_prim, const ::caffe::CustomAttribute &custom_param,
                    int index) {
  if (custom_prim == nullptr) {
    MS_LOG(ERROR) << "custom_prim is nullptr.";
    return RET_ERROR;
  }
  if (custom_param.has_name()) {
    (void)custom_prim->AddAttr("name" + std::to_string(index), api::MakeValue<std::string>(custom_param.name()));
  }
  if (custom_param.has_f()) {
    (void)custom_prim->AddAttr("f" + std::to_string(index), api::MakeValue<float>(custom_param.f()));
  }
  if (custom_param.has_i()) {
    (void)custom_prim->AddAttr("i" + std::to_string(index), api::MakeValue<int64_t>(custom_param.i()));
  }
  if (custom_param.has_s()) {
    (void)custom_prim->AddAttr("s" + std::to_string(index), api::MakeValue<std::string>(custom_param.s()));
  }
  if (custom_param.floats_size() > 0) {
    std::vector<float> floats;
    for (int i = 0; i < custom_param.floats_size(); i++) {
      (void)floats.emplace_back(custom_param.floats(i));
    }
    (void)custom_prim->AddAttr("floats" + std::to_string(index), api::MakeValue<std::vector<float>>(floats));
  }
  if (custom_param.ints_size() > 0) {
    std::vector<int64_t> ints;
    for (int i = 0; i < custom_param.ints_size(); i++) {
      (void)ints.emplace_back(custom_param.ints(i));
    }
    (void)custom_prim->AddAttr("ints" + std::to_string(index), api::MakeValue<std::vector<int64_t>>(ints));
  }
  if (custom_param.strings_size() > 0) {
    std::vector<string> strings;
    for (int i = 0; i < custom_param.strings_size(); i++) {
      (void)strings.emplace_back(custom_param.strings(i));
    }
    (void)custom_prim->AddAttr("strings" + std::to_string(index), api::MakeValue<std::vector<std::string>>(strings));
  }

  if (custom_param.has_type()) {
    custom::AttributeType attribute_type;
    if (GetAttributeType(&attribute_type, custom_param.type()) != RET_OK) {
      MS_LOG(ERROR) << "get custom param type failed.";
      return RET_ERROR;
    }
    (void)custom_prim->AddAttr("type" + std::to_string(index), api::MakeValue(static_cast<int64_t>(attribute_type)));
  }

  return RET_OK;
}
int SetAttrsByCustomParam(const std::shared_ptr<ops::Custom> &custom_prim, const caffe::LayerParameter &proto) {
  if (custom_prim == nullptr) {
    MS_LOG(ERROR) << "custom_prim is nullptr.";
    return RET_ERROR;
  }
  if (!proto.has_custom_param()) {
    MS_LOG(INFO) << "no custom param found";
    return RET_OK;
  }
  const auto &custom_param = proto.custom_param();
  int custom_param_size = custom_param.attribute_size();
  (void)custom_prim->AddAttr(dpico::kCustomParamSize, api::MakeValue(custom_param_size));
  if (custom_param_size == 0) {
    MS_LOG(INFO) << "no custom param found";
    return RET_OK;
  }
  for (int i = 0; i < custom_param_size; i++) {
    const auto &custom_param_by_index = custom_param.attribute(i);
    if (SetAttrsByParam(custom_prim, custom_param_by_index, i) != RET_OK) {
      MS_LOG(ERROR) << "set prim attrs from custom param failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace

BaseOperatorPtr CaffeCustomParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("Custom");
  if (proto.has_custom_param()) {
    const auto &custom_param = proto.custom_param();
    if (custom_param.has_extended_op_type()) {
      (void)prim->AddAttr(dpico::kExtendedOpType, api::MakeValue<std::string>(custom_param.extended_op_type()));
    }
    if (SetAttrsByCustomParam(prim, proto) != RET_OK) {
      MS_LOG(ERROR) << "set attrs by custom param failed.";
      return nullptr;
    }
  }
  return prim;
}

CaffeNodeRegistrar g_caffeCustomParser("Custom", new CaffeCustomParser());
}  // namespace lite
}  // namespace mindspore
