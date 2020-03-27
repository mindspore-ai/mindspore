/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "predict/converter/lite_model/op_attr_packer.h"

namespace mindspore {
namespace predict {
namespace convert {
bool ActivationPacker(const CNodePtr &c_node_ptr, OpDefT *ms_op) {
  if (c_node_ptr == nullptr || ms_op == nullptr) {
    return false;
  }
  std::unique_ptr<ActivationT> attr(new ActivationT());
  MS_EXCEPTION_IF_NULL(attr);
  if (AnfAlgo::GetCNodeName(c_node_ptr) == "ReLU") {
    attr->type = predict::ActivationType::ActivationType_RELU;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "Sigmoid") {
    attr->type = predict::ActivationType::ActivationType_SIGMOID;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "ReLU6") {
    attr->type = predict::ActivationType::ActivationType_RELU6;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "ELU") {
    attr->type = predict::ActivationType::ActivationType_ELU;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "Leaky_ReLU") {
    attr->type = predict::ActivationType::ActivationType_LEAKY_RELU;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "ABS") {
    attr->type = predict::ActivationType::ActivationType_ABS;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "ReLU1") {
    attr->type = predict::ActivationType::ActivationType_RELU1;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "Softsign") {
    attr->type = predict::ActivationType::ActivationType_SOFTSIGN;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "Softplus") {
    attr->type = predict::ActivationType::ActivationType_SOFTPLUS;
  } else if (AnfAlgo::GetCNodeName(c_node_ptr) == "Tanh") {
    attr->type = predict::ActivationType::ActivationType_TANH;
  } else {
    attr->type = predict::ActivationType::ActivationType_UNKNOW;
    MS_LOG(WARNING) << "unknow Activation";
  }
  ms_op->name = c_node_ptr->fullname_with_scope();
  ms_op->attr.type = OpT_Activation;
  ms_op->attr.value = attr.release();
  return true;
}
}  // namespace convert
}  // namespace predict
}  // namespace mindspore
