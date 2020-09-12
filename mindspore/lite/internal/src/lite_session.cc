/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "internal/include/lite_session.h"
#include "internal/include/model.h"
#include "internal/include/ms_tensor.h"
#include "src/runtime/allocator.h"
#include "internal/include/errorcode.h"
#include "utils/log_adapter.h"
#include "internal/src/kernel/fp32/activation.h"
#include "internal/src/kernel/fp32/arithmetic_self.h"
#include "internal/src/kernel/fp32/matmul.h"
#include "internal/src/kernel/fp32_grad/arithmetic_self_grad.h"
#include "internal/src/kernel/fp32_grad/activation_grad.h"

static Context *g_Ctx;
static Model *g_Model;
static LiteSession g_Session;
static mindspore::lite::DefaultAllocator allocator;

LiteSession *LiteSession::CreateSession(Context *context) {
  g_Ctx = context;
  return &g_Session;
}

int LiteSession::CompileGraph(Model *model) {
  g_Model = model;
  for (auto in : g_Model->input_indices_) {
    g_Model->all_tensors_[in]->data_ = allocator.Malloc(g_Model->all_tensors_[in]->Size());
  }
  return 0;
}

TensorPtrVector LiteSession::GetInputs() const {
  TensorPtrVector in(g_Model->input_indices_.size());
  //    for(auto index : g_Model->input_indices_){
  //        in.emplace_back(g_Model->all_tensors_[index]);
  //    }
  return in;
}

TensorPtrVector LiteSession::GetInputsByName(const String &node_name) const { return TensorPtrVector(); }

TensorPtrVector LiteSession::GetOutputsByNodeName(const String &node_name) const { return TensorPtrVector(); }

TensorPtrVector LiteSession::GetOutputs() const {
  TensorPtrVector out(g_Model->output_indices_.size());
  //    for(auto index : g_Model->output_indices_){
  //        out.emplace_back(g_Model->all_tensors_[index]);
  //    }
  return out;
}

int LiteSession::RunGraph() {
  // invoke nnacl kernel
  NodePtrVector nodes = g_Model->nodes_;
  size_t nodes_size = nodes.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = nodes[i];
    if (node->primitive_ == nullptr) {
      MS_LOG(ERROR) << "node's primitive is NULL!";
      return RET_ERROR;
    }
    TensorPtrVector in_tensors;
    for (size_t j = 0; j < node->input_indices_.size(); ++j) {
      in_tensors.push_back(g_Model->all_tensors_[node->input_indices_[j]]);
    }
    TensorPtrVector out_tensors;
    for (size_t j = 0; j < node->output_indices_.size(); ++j) {
      out_tensors.push_back(g_Model->all_tensors_[node->output_indices_[j]]);
    }
    int type = node->primitive_->type_;
    int ret = RET_ERROR;
    switch (type) {
      case KernelType::MatMul:
        ret = DoMatMul(in_tensors, out_tensors, node, &allocator);
        break;
      case KernelType::Activation:
        ret = DoActivation(in_tensors, out_tensors, node, &allocator);
        break;
      case KernelType::Log:
      case KernelType::Neg:
        ret = DoArithmeticSelf(in_tensors, out_tensors, node, &allocator);
        break;
      case KernelType::LogGrad:
      case KernelType::NegGrad:
        ret = DoArithmeticGradSelf(in_tensors, out_tensors, node, &allocator);
        break;
      case KernelType::ActivationGrad:
        ret = DoActivationGrad(in_tensors, out_tensors, node, &allocator);
        break;
      default:
        MS_LOG(ERROR) << "Unsupport kernel type: " << type;
        return RET_PARAM_INVALID;
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "run kernel fail!ret: " << ret;
      return ret;
    }
  }
  return RET_OK;
}

StringVector LiteSession::GetOutputTensorNames() const { return StringVector(); }

MSTensor *LiteSession::GetOutputByTensorName(const String &tensor_name) const { return NULL; }

int LiteSession::Resize(const TensorPtrVector &inputs, Int32VectorVector dims) { return 0; }
