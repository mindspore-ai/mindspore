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
#include "internal/src/allocator.h"
#include "internal/include/errorcode.h"
#include "internal/src/lite_log.h"
#include "internal/src/kernel/fp32/activation.h"
#include "internal/src/kernel/fp32/arithmetic_self.h"
#include "internal/src/kernel/fp32/matmul.h"
#include "internal/src/kernel/fp32/arithmetic.h"
#include "internal/src/kernel/fp32/bias_add.h"
#ifdef SUPPORT_TRAIN
#include "internal/src/kernel/fp32_grad/arithmetic_self_grad.h"
#include "internal/src/kernel/fp32_grad/activation_grad.h"
#endif

static Context *g_ctx;
static Model *g_model;
static LiteSession g_session;
static mindspore::lite::Allocator g_allocator;
static bool g_infershape_interrupt = false;
static bool g_first_load = true;
typedef int (*InferShape)(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, OpParameter *param);
typedef int (*RunKernel)(const TensorPtrVector &in_tensors, const TensorPtrVector &out_tensors, Node *node,
                         mindspore::lite::Allocator *allocator);
static InferShape g_infershape_funcs[KernelType::KernelType_END];
static RunKernel g_runkernel_funcs[KernelType::KernelType_END];

static int ModelInferShape() {
  NodePtrVector nodes = g_model->nodes_;
  size_t nodes_size = nodes.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = nodes[i];
    if (node->primitive_ == NULL) {
      LITE_LOG_ERROR("node's primitive is NULL!");
      return RET_ERROR;
    }
    TensorPtrVector in_tensors;
    for (size_t j = 0; j < node->input_indices_.size(); ++j) {
      in_tensors.push_back(g_model->all_tensors_[node->input_indices_[j]]);
    }
    TensorPtrVector out_tensors;
    for (size_t j = 0; j < node->output_indices_.size(); ++j) {
      out_tensors.push_back(g_model->all_tensors_[node->output_indices_[j]]);
    }
    int type = node->primitive_->type_;
    InferShape infershape = g_infershape_funcs[type];
    if (infershape == NULL) {
      LITE_ERROR_LOG("Unsupport kernel type: %d", type);
      return RET_PARAM_INVALID;
    }
    int ret = (*infershape)(in_tensors, out_tensors, node->primitive_);
    if (ret == RET_INFER_INVALID) {
      g_infershape_interrupt = true;
      LITE_INFO_LOG("%s inferShape shouldn't be done before runtime, inferShape interrupt!", node->name_.c_str());
    }
    if (ret != RET_OK) {
      LITE_ERROR_LOG("Infer shape fail!ret: %d", ret);
      return ret;
    }
  }
  return RET_OK;
}

static void InitFuncs() {
  if (g_first_load) {
    g_infershape_funcs[KernelType::KernelType_MatMul] = DoMatMulInferShape;
    g_infershape_funcs[KernelType::KernelType_Activation] = DoActivationInferShape;
    g_infershape_funcs[KernelType::KernelType_Log] = DoArithmeticSelfInferShape;
    g_infershape_funcs[KernelType::KernelType_Neg] = DoArithmeticSelfInferShape;
    g_infershape_funcs[KernelType::KernelType_Mul] = DoArithmeticInferShape;
    g_infershape_funcs[KernelType::KernelType_BiasAdd] = DoBiasAddInferShape;

    g_runkernel_funcs[KernelType::KernelType_MatMul] = DoMatMul;
    g_runkernel_funcs[KernelType::KernelType_Activation] = DoActivation;
    g_runkernel_funcs[KernelType::KernelType_Log] = DoArithmeticSelf;
    g_runkernel_funcs[KernelType::KernelType_Neg] = DoArithmeticSelf;
    g_runkernel_funcs[KernelType::KernelType_Mul] = DoArithmetic;
    g_runkernel_funcs[KernelType::KernelType_BiasAdd] = DoBiasAdd;
#ifdef SUPPORT_TRAIN
    g_infershape_funcs[KernelType::KernelType_ActivationGrad] = DoActivationGradInferShape;
    g_infershape_funcs[KernelType::KernelType_NegGrad] = DoArithmeticSelfGradInferShape;
    g_infershape_funcs[KernelType::KernelType_LogGrad] = DoArithmeticSelfGradInferShape;

    g_runkernel_funcs[KernelType::KernelType_NegGrad] = DoArithmeticSelfGrad;
    g_runkernel_funcs[KernelType::KernelType_ActivationGrad] = DoActivationGrad;
    g_runkernel_funcs[KernelType::KernelType_LogGrad] = DoArithmeticSelfGrad;
#endif
    g_first_load = false;
  }
}

LiteSession *LiteSession::CreateSession(Context *context) {
  g_ctx = context;
  return &g_session;
}

int LiteSession::CompileGraph(Model *model) {
  InitFuncs();
  g_model = model;
  for (auto in : g_model->input_indices_) {
    if (in >= g_model->all_tensors_.size() || in < 0) {
      LITE_LOG_ERROR("Invalid input indices!");
      return RET_PARAM_INVALID;
    }
    g_model->all_tensors_[in]->data_ = g_allocator.Malloc(g_model->all_tensors_[in]->Size());
  }
  g_infershape_interrupt = false;
  int ret = ModelInferShape();
  if (ret != RET_OK && ret != RET_INFER_INVALID) {
    return ret;
  }
  return RET_OK;
}

TensorPtrVector LiteSession::GetInputs() const {
  TensorPtrVector in(g_model->input_indices_.size());
  for (size_t i = 0; i < g_model->input_indices_.size(); ++i) {
    auto index = g_model->input_indices_[i];
    if (index < 0 || index >= g_model->all_tensors_.size()) {
      LITE_ERROR_LOG("Invalid input index: %u", index);
      return TensorPtrVector();
    }
    in.at(i) = g_model->all_tensors_[index];
  }
  return in;
}

TensorPtrVector LiteSession::GetInputsByName(const String &node_name) const { return TensorPtrVector(); }

TensorPtrVector LiteSession::GetOutputsByNodeName(const String &node_name) const { return TensorPtrVector(); }

TensorPtrVector LiteSession::GetOutputs() const {
  TensorPtrVector out(g_model->output_indices_.size());
  for (size_t i = 0; i < g_model->output_indices_.size(); ++i) {
    auto index = g_model->output_indices_[i];
    if (index < 0 || index >= g_model->all_tensors_.size()) {
      LITE_ERROR_LOG("Invalid output index: %u", index);
      return TensorPtrVector();
    }
    out.at(i) = g_model->all_tensors_[index];
  }
  return out;
}

int LiteSession::RunGraph() {
  NodePtrVector nodes = g_model->nodes_;
  size_t nodes_size = nodes.size();
  for (size_t i = 0; i < nodes_size; ++i) {
    auto node = nodes[i];
    if (node->primitive_ == nullptr) {
      LITE_LOG_ERROR("node's primitive is NULL!");
      return RET_ERROR;
    }
    TensorPtrVector in_tensors;
    for (size_t j = 0; j < node->input_indices_.size(); ++j) {
      in_tensors.push_back(g_model->all_tensors_[node->input_indices_[j]]);
    }
    TensorPtrVector out_tensors;
    for (size_t j = 0; j < node->output_indices_.size(); ++j) {
      out_tensors.push_back(g_model->all_tensors_[node->output_indices_[j]]);
    }
    int type = node->primitive_->type_;
    if (g_infershape_interrupt) {
      InferShape infershape = g_infershape_funcs[type];
      if (infershape == NULL) {
        LITE_ERROR_LOG("Unsupport kernel type: %d", type);
        return RET_PARAM_INVALID;
      }
      int ret = (*infershape)(in_tensors, out_tensors, node->primitive_);
      if (ret != RET_OK) {
        LITE_ERROR_LOG("InferShape fail!ret: %d", ret);
        return ret;
      }
    }
    for (size_t j = 0; j < out_tensors.size(); ++j) {
      out_tensors[j]->data_ = g_allocator.Malloc(out_tensors[j]->Size());
      if (out_tensors[j]->data_ == NULL) {
        LITE_LOG_ERROR("Malloc data for out tensor fail!");
        return RET_NULL_PTR;
      }
    }
    RunKernel run_kernel = g_runkernel_funcs[type];
    if (run_kernel == NULL) {
      LITE_ERROR_LOG("Unsupport kernel type: %d", type);
      return RET_PARAM_INVALID;
    }

    int ret = (*run_kernel)(in_tensors, out_tensors, node, &g_allocator);
    if (ret != RET_OK) {
      LITE_ERROR_LOG("run kernel fail!ret: %d", ret);
      return ret;
    }
  }
  g_infershape_interrupt = false;
  return RET_OK;
}

StringVector LiteSession::GetOutputTensorNames() const { return StringVector(); }

MSTensor *LiteSession::GetOutputByTensorName(const String &tensor_name) const { return NULL; }

int LiteSession::Resize(const TensorPtrVector &inputs, const Int32VectorVector &dims) { return 0; }
