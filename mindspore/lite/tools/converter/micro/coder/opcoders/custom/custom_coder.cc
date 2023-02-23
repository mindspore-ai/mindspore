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

#include <string>
#include <map>
#include "tools/converter/micro/coder/opcoders/op_coder.h"
#include "tools/converter/micro/coder/opcoders/file_collector.h"
#include "tools/converter/micro/coder/opcoders/serializers/serializer.h"
#include "tools/converter/micro/coder/opcoders/custom/custom_coder.h"
#include "tools/converter/micro/coder/opcoders/op_coder_register.h"
#include "tools/converter/micro/coder/opcoders/kernel_registry.h"
#include "src/common/prim_util.h"
#include "nnacl/custom_parameter.h"

using mindspore::schema::PrimitiveType_Custom;

namespace mindspore::lite::micro {
std::map<Tensor *, void *> CustomCoder::const_tensor_map_;

void CustomCoder::Populate(const void *prim) {
  auto op = static_cast<const schema::Primitive *>(prim)->value_as_Custom();
  type_ = op->type()->str();
  for (size_t i = 0; i < op->attr()->size(); ++i) {
    auto attr = op->attr()->Get(i);
    std::string data;
    for (size_t j = 0; j < attr->data()->size(); ++j) {
      data.push_back(static_cast<char>(attr->data()->Get(j)));
    }
    attrs_[attr->name()->str()] = data;
  }
}

int CustomCoder::Prepare(CoderContext *const context) {
  if (GetPrimitiveType(node_->primitive_, schema_version_) != PrimitiveType_Custom) {
    MS_LOG(ERROR) << "Primitive type should be custom";
    return RET_ERROR;
  }
  Populate(node_->primitive_);
  for (const auto &tensor : input_tensors_) {
    if (tensor->category() == lite::Category::CONST_TENSOR) {
      if (!const_tensor_map_.count(tensor)) {
        auto buff = allocator_->Malloc(kNumberTypeUInt8, tensor->Size(), kOfflinePackWeight);
        memcpy_s(buff, tensor->Size(), tensor->data(), tensor->Size());
        const_tensor_map_[tensor] = buff;
      }
    }
  }

  return RET_OK;
}

int CustomCoder::TransformTensors(Serializer *code, std::string array_name, const std::vector<Tensor *> &tensors) {
  if (tensors.size() > 16) {
    MS_LOG(ERROR) << "The number of tensors is too large";
    return RET_ERROR;
  }
  (*code) << "\t\tTensorC " << array_name << "[" << tensors.size() << "];\n";
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->category() == lite::Category::CONST_TENSOR) {
      if (!const_tensor_map_.count(tensors[i])) {
        MS_LOG(ERROR) << "can't find the const tensor's runtime address";
        return RET_ERROR;
      }
      (*code) << "\t\t" << array_name << "[" << i
              << "].data_ = " << allocator_->GetRuntimeAddr(const_tensor_map_[tensors[i]]) << ";\n";
    } else {
      (*code) << "\t\t" << array_name << "[" << i << "].data_ = " << allocator_->GetRuntimeAddr(tensors[i]) << ";\n";
    }
    for (size_t j = 0; j < tensors[i]->shape().size(); ++j) {
      (*code) << "\t\t" << array_name << "[" << i << "].shape_[" << j << "] = " << tensors[i]->shape()[j] << ";\n";
    }
    (*code) << "\t\t" << array_name << "[" << i << "].shape_size_ = " << tensors[i]->shape().size() << ";\n";
    (*code) << "\t\t" << array_name << "[" << i << "].data_type_ = " << tensors[i]->data_type() << ";\n";
    (*code) << "\t\t" << array_name << "[" << i << "].format_ = " << tensors[i]->format() << ";\n";
    if (tensors[i]->tensor_name().size() > MAX_STR_LEN) {
      MS_LOG(ERROR) << "tensor name is too long: " << tensors[i]->tensor_name();
      return RET_ERROR;
    }
    (*code) << "\t\t" << array_name << "[" << i << "].name_ = "
            << "malloc(" << tensors[i]->tensor_name().length() + 1 << ");\n";
    (*code) << "\t\tstrcpy(" << array_name << "[" << i << "].name_, "
            << "\"" << tensors[i]->tensor_name() << "\""
            << ");\n";
  }

  return RET_OK;
}

int CustomCoder::TransformParams(Serializer *code, std::string var_name) {
  if (attrs_.size() > MAX_ATTR_NUM) {
    MS_LOG(ERROR) << "Attrs's number exceeds the maximum";
    return RET_ERROR;
  }

  (*code) << "\t\tCustomParameter " << var_name << ";\n";
  if (type_.size() > MAX_STR_LEN) {
    MS_LOG(ERROR) << "type name is too long: " << type_;
    return RET_ERROR;
  }
  (*code) << "\t\tstrcpy(" << var_name << ".type, "
          << "\"" << type_ << "\""
          << ");\n";
  int i = 0;
  for (auto iter = attrs_.begin(); iter != attrs_.end(); ++iter) {
    if (iter->first.size() > MAX_STR_LEN) {
      MS_LOG(ERROR) << "attr name is too long: " << iter->first;
      return RET_ERROR;
    }
    (*code) << "\t\tstrcpy(" << var_name << ".attr_name[" << i << "], "
            << "\"" << iter->first << "\""
            << ");\n";
    (*code) << "\t\t" << var_name << ".attr_data[" << i << "] = "
            << "malloc(" << iter->second.size() + 1 << ");\n";
    (*code) << "\t\tstrcpy(" << var_name << ".attr_data[" << i++ << "], "
            << "\"" << iter->second << "\""
            << ");\n";
  }
  (*code) << "\t\t" << var_name << ".attr_num = " << attrs_.size() << ";\n";
  return RET_OK;
}

void CustomCoder::FreeParams(Serializer *code, std::string var_name) {
  int i = 0;
  for (auto iter = attrs_.begin(); iter != attrs_.end(); ++iter) {
    (*code) << "\t\tfree(" << var_name << ".attr_data[" << i++ << "]);\n";
  }
}

void CustomCoder::FreeTensors(Serializer *code, std::string array_name, size_t tensors_num) {
  for (size_t i = 0; i < tensors_num; i++) {
    (*code) << "\t\tfree(" << array_name << "[" << i << "].name_);\n";
  }
}

int CustomCoder::DoCode(CoderContext *const context) {
  Collect(context, {"nnacl/custom_parameter.h", "nnacl/tensor_c.h", "src/registered_kernel.h"}, {});
  Serializer code;
  MS_CHECK_RET_CODE(TransformTensors(&code, "inputs", input_tensors_), "Transform input tensors error!");
  MS_CHECK_RET_CODE(TransformTensors(&code, "outputs", output_tensors_), "Transform output tensors error!");
  MS_CHECK_RET_CODE(TransformParams(&code, "param"), "Transform output tensors error!");
  code.CodeFunction(kCustomKernelName, "inputs", input_tensors_.size(), "outputs", output_tensors_.size(), "&param");
  FreeParams(&code, "param");
  FreeTensors(&code, "inputs", input_tensors_.size());
  FreeTensors(&code, "outputs", output_tensors_.size());
  context->AppendCode(code.str());
  return 0;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Custom, CPUOpCoderCreator<CustomCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeUInt8, PrimitiveType_Custom, CPUOpCoderCreator<CustomCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Custom, CPUOpCoderCreator<CustomCoder>)
}  // namespace mindspore::lite::micro
