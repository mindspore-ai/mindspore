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

#include "coder/opcoders/nnacl/int8/sigmoid_int8_coder.h"
#include "coder/opcoders/nnacl/int8/relux_int8_coder.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/activation_fp32.h"
#include "schema/model_generated.h"
#include "src/common/version_manager.h"

using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::lite::micro::nnacl {

std::unique_ptr<OperatorCoder> CPUActivationINT8CoderCreator(const std::vector<Tensor *> &in_tensors,
                                                             const std::vector<Tensor *> &out_tensors,
                                                             const Model::Node *node, size_t node_index,
                                                             Target target) {
  const void *primitive_c = node->primitive_;
  if (primitive_c == nullptr) {
    return nullptr;
  }
  int schema_version = VersionManager::GetInstance()->GetSchemaVersion();
  ParameterGen parameter_gen =
    PopulateRegistry::GetInstance()->GetParameterCreator(GetPrimitiveType(node->primitive_), schema_version);
  if (parameter_gen == nullptr) {
    MS_LOG(ERROR) << "parameter generator is nullptr";
    return nullptr;
  }
  OpParameter *parameter = parameter_gen(node->primitive_);
  if (parameter == nullptr) {
    MS_LOG(ERROR) << "PopulateParameter return nullptr, type: "
                  << schema::EnumNamePrimitiveType((schema::PrimitiveType)GetPrimitiveType(node->primitive_));
    return nullptr;
  }
  auto type = (reinterpret_cast<ActivationParameter *>(parameter))->type_;

  std::unique_ptr<OperatorCoder> coder;
  switch (static_cast<schema::ActivationType>(type)) {
    case schema::ActivationType_SIGMOID:
      coder = CPUOpCoderCreator<SigmodInt8Coder>(in_tensors, out_tensors, node, node_index, target);
      break;
    case schema::ActivationType_RELU:
      coder = CPUOpCoderCreator<ReluInt8Coder>(in_tensors, out_tensors, node, node_index, target);
      break;
    case schema::ActivationType_RELU6:
      coder = CPUOpCoderCreator<Relu6Int8Coder>(in_tensors, out_tensors, node, node_index, target);
      break;
    default:
      break;
  }

  if (coder == nullptr) {
    MS_LOG(ERROR) << "create conv2d int8 coder failed";
    return nullptr;
  }
  return coder;
}

REG_OPERATOR_CODER(kAllTargets, kNumberTypeInt8, PrimitiveType_Activation, CPUActivationINT8CoderCreator)
}  // namespace mindspore::lite::micro::nnacl
