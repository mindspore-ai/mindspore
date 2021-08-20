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
#include "coder/opcoders/op_coder.h"
#include "coder/opcoders/file_collector.h"
#include "coder/opcoders/serializers/serializer.h"
#include "coder/opcoders/custom/custom_coder.h"
#include "coder/opcoders/op_coder_register.h"
#include "coder/user_registry/user_kernel_register.h"
#include "src/common/prim_util.h"

using mindspore::schema::PrimitiveType_Custom;
#define MAX_TENSORS 16
#define MAX_STR_LEN 32
#define MAX_ATTR_NUM 8
#define MAX_TENSOR_NAME_LEN 32

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
  CoderKey key(target_, input_tensors_[0]->data_type(), PrimitiveType_Custom);
  auto result = UserKernelFactory::GetInstance()->FindUserKernel(key);
  if (result.empty()) {
    MS_LOG(ERROR) << "No user kernel register for custom op";
    return RET_ERROR;
  }
  header_ = result[0];
  function_ = result[1];
  Populate(node_->primitive_);
  for (const auto &tensor : input_tensors_) {
    if (tensor->category() == Tensor::Category::CONST_TENSOR) {
      if (!const_tensor_map_.count(tensor)) {
        auto buff = allocator_->Malloc(kNumberTypeUInt8, tensor->Size(), kOfflinePackWeight);
        memcpy_s(buff, tensor->Size(), tensor->data(), tensor->Size());
        const_tensor_map_[tensor] = buff;
      }
    }
  }
  Configurator::GetInstance()->SetCustomFlag();
  return RET_OK;
}

int CustomCoder::TransformTensors(Serializer *code, std::string array_name, const std::vector<Tensor *> &tensors) {
  if (tensors.size() > MAX_TENSORS) {
    MS_LOG(ERROR) << "The number of tensors is too large";
    return RET_ERROR;
  }
  (*code) << "\t\tCustomTensor " << array_name << "[" << tensors.size() << "];\n";
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->category() == Tensor::Category::CONST_TENSOR) {
      if (!const_tensor_map_.count(tensors[i])) {
        MS_LOG(ERROR) << "can't find the const tensor's runtime address";
        return RET_ERROR;
      }
      (*code) << "\t\t" << array_name << "[" << i
              << "].data = " << allocator_->GetRuntimeAddr(const_tensor_map_[tensors[i]]) << ";\n";
    } else {
      (*code) << "\t\t" << array_name << "[" << i << "].data = " << allocator_->GetRuntimeAddr(tensors[i]) << ";\n";
    }
    (*code) << "\t\t" << array_name << "[" << i << "].data_size = " << tensors[i]->Size() << ";\n";
    for (size_t j = 0; j < tensors[i]->shape().size(); ++j) {
      (*code) << "\t\t" << array_name << "[" << i << "].shape[" << j << "] = " << tensors[i]->shape()[j] << ";\n";
    }
    (*code) << "\t\t" << array_name << "[" << i << "].shape_size = " << tensors[i]->shape().size() << ";\n";
    (*code) << "\t\t" << array_name << "[" << i << "].data_type = " << tensors[i]->data_type() << ";\n";
    (*code) << "\t\t" << array_name << "[" << i << "].format = " << tensors[i]->format() << ";\n";
    if (tensors[i]->tensor_name().size() > MAX_TENSOR_NAME_LEN) {
      MS_LOG(ERROR) << "tensor name is too long: " << tensors[i]->tensor_name();
      return RET_ERROR;
    }
    (*code) << "\t\tstrcpy(" << array_name << "[" << i << "].name, "
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

  (*code) << "\t\tCustomParams " << var_name << ";\n";
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
    if (iter->second.size() > MAX_STR_LEN) {
      MS_LOG(ERROR) << "attr " << iter->first << " data is too long";
      return RET_ERROR;
    }
    (*code) << "\t\tstrcpy(" << var_name << ".attr_name[" << i << "], "
            << "\"" << iter->first << "\""
            << ");\n";
    (*code) << "\t\tstrcpy(" << var_name << ".attr_data[" << i++ << "], "
            << "\"" << iter->second << "\""
            << ");\n";
  }
  (*code) << "\t\t" << var_name << ".attr_num = " << attrs_.size() << ";\n";
  return RET_OK;
}

int CustomCoder::DoCode(CoderContext *const context) {
  Collect(context, {header_, "custom_params.h"}, {});
  Serializer code;
  MS_CHECK_RET_CODE(TransformTensors(&code, "inputs", input_tensors_), "Transform input tensors error!");
  MS_CHECK_RET_CODE(TransformTensors(&code, "outputs", output_tensors_), "Transform output tensors error!");
  MS_CHECK_RET_CODE(TransformParams(&code, "param"), "Transform output tensors error!");
  code.CodeFunction(function_, "inputs", input_tensors_.size(), "outputs", output_tensors_.size(), "&param");
  context->AppendCode(code.str());
  return 0;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Custom, CPUOpCoderCreator<CustomCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeUInt8, PrimitiveType_Custom, CPUOpCoderCreator<CustomCoder>)
REG_OPERATOR_CODER(kAllTargets, kNumberTypeFloat32, PrimitiveType_Custom, CPUOpCoderCreator<CustomCoder>)
}  // namespace mindspore::lite::micro
