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

#include "src/train/train_session.h"
#include <sys/stat.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>
#include <fstream>
#include <memory>
#include <map>
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "src/tensor.h"
#include "src/lite_model.h"
#include "src/train/loss_kernel.h"
#include "src/train/optimizer_kernel.h"
#include "src/sub_graph_kernel.h"
#include "src/train/train_populate_parameter.h"
#include "src/train/train_populate_parameter_v0.h"
#include "src/executor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/convolution.h"
#include "src/runtime/kernel/arm/fp32/batchnorm_fp32.h"
#include "src/common/tensor_util.h"
#include "src/train/train_utils.h"
#include "src/train/train_export.h"
#include "src/common/prim_util.h"

namespace mindspore {
namespace lite {
const char *kGradName = "Gradients";
const char *kOptimizerName = "optimizer";

TrainSession::TrainSession() {
  is_train_session_ = true;
  InitCallBack();
}

int TrainSession::Init(InnerContext *context, const TrainCfg *train_cfg) {
  if (train_cfg != nullptr) {
    if (train_cfg->mix_precision_cfg_.loss_scale_ <= 0) {
      MS_LOG(ERROR) << "illegal loss scale configuration";
      return RET_NULL_PTR;
    }
    cfg_ = *train_cfg;
  }
  return lite::LiteSession::Init(context);
}

std::vector<CreatorOp> TrainSession::ReplaceOps() {
  const std::vector<CreatorOp> replace = {
    // currently no ops are Hijacked by TrainSession
  };
  mindspore::lite::KernelRegistry *reg = mindspore::lite::KernelRegistry::GetInstance();
  std::vector<CreatorOp> results;
  for (auto v : replace) {
    const CreatorOp cl = make_tuple(std::get<0>(v), reg->GetCreator(std::get<0>(v)));
    results.push_back(cl);
    reg->RegKernel(std::get<0>(v), std::get<1>(v));
  }
  return results;
}

void TrainSession::RestoreOps(const std::vector<CreatorOp> &restore) {
  mindspore::lite::KernelRegistry *reg = mindspore::lite::KernelRegistry::GetInstance();
  for (auto v : restore) {
    reg->RegKernel(std::get<0>(v), std::get<1>(v));
  }
}

int TrainSession::AllocWorkSpace() {
  size_t workspace_size = 0;

  for (auto kernel : this->train_kernels_) {
    if (workspace_size < static_cast<kernel::InnerKernel *>(kernel->kernel())->workspace_size()) {
      workspace_size = static_cast<kernel::InnerKernel *>(kernel->kernel())->workspace_size();
    }
  }
  workspace_ = malloc(workspace_size);
  if (workspace_ == nullptr) {
    MS_LOG(ERROR) << "cannot allocate " << workspace_size << " for workspace";
    return RET_ERROR;
  }
  for (auto kernel : this->train_kernels_) {
    static_cast<kernel::InnerKernel *>(kernel->kernel())->set_workspace(workspace_);
  }
  return RET_OK;
}

void TrainSession::FreeWorkSpace() {
  free(workspace_);
  for (auto kernel : this->train_kernels_) {
    static_cast<kernel::InnerKernel *>(kernel->kernel())->FreeWorkspace();
  }
}

int TrainSession::InitCallBack() {
  sched_mix_precision_callback_ = [&](const Model::Node *node) {
    if (!context_->IsCpuFloat16Enabled()) {
      return false;
    }
    auto node_type = GetPrimitiveType(node->primitive_, SCHEMA_VERSION::SCHEMA_CUR);
    if (node_type == schema::PrimitiveType_Cast) {
      return false;
    }
    auto in_size = node->input_indices_.size();
    bool force_fp16 = false;
    for (std::size_t k = 0; k < in_size; k++) {
      schema::Tensor *tensor = model_.get()->all_tensors_.at(node->input_indices_[k]);
      if ((tensor->dataType() == kNumberTypeFloat16) && (tensor->nodeType() == NodeType_ValueNode)) {
        force_fp16 = true;
        break;
      }
    }
    const auto &node_name = node->name_;
    bool is_fp16 = true;
    if (!force_fp16) {
      // optimizer runs in fp32
      if (node_name.find(kOptimizerName) != std::string::npos) {
        is_fp16 = false;
      }
      // loss function runs in fp32
      if ((node_name.find(get_loss_name()) != std::string::npos)) {
        is_fp16 = false;
      }
      // run bn according to user configuration
      if ((cfg_.mix_precision_cfg_.keep_batchnorm_fp32_) &&
          (node_type == schema::PrimitiveType_FusedBatchNorm || node_type == schema::PrimitiveType_BatchNorm ||
           node_type == schema::PrimitiveType_BatchNormGrad)) {
        is_fp16 = false;
      }
    }
    MS_LOG(DEBUG) << "Debug: " << node_name << ((is_fp16) ? " fp16" : " fp32");
    return is_fp16;
  };
  return RET_OK;
}

int TrainSession::CompileGraph(lite::Model *model) { return lite::RET_ERROR; }

int TrainSession::CompileTrainGraph(std::shared_ptr<Model> model) {
  model_ = model;
  auto restore = ReplaceOps();
  sched_cb_ = std::make_unique<SchedulerCb>(sched_mix_precision_callback_);
  if (sched_cb_ == nullptr) {
    MS_LOG(ERROR) << "Failed to create SchedulerCb node";
    return RET_ERROR;
  }

#ifdef ENABLE_V0
  if (reinterpret_cast<LiteModel *>(model_.get())->GetSchemaVersion() == SCHEMA_VERSION::SCHEMA_V0) {
    kernel::PopulateTrainV0Parameters();
  }
#endif
  if (reinterpret_cast<LiteModel *>(model_.get())->GetSchemaVersion() == SCHEMA_VERSION::SCHEMA_CUR) {
    kernel::PopulateTrainParameters();
  }

  auto ret = lite::LiteSession::CompileGraph(model_.get());
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to compile train model";
    return RET_ERROR;
  }
  orig_output_node_map_ = output_node_map_;
  orig_output_tensor_map_ = output_tensor_map_;
  orig_output_tensor_names_ = output_tensor_names_;
  for (auto inTensor : inputs_) inTensor->MutableData();
  RestoreOps(restore);
  CompileTrainKernels();      // Prepare a list of train kernels
  CompileOptimizedKernels();  // Prepare a list of kernels which are optimized (weight update step)
  CompileTrainOutputs();      // prepare outputs in train mode
  CompileEvalOutputs();       // prepare outputs in eval mode
  // Prepare a list of eval kernels
  if (CompileInferenceKernels() != RET_OK) {
    MS_LOG(ERROR) << "CompileInferenceKernels failed.";
    return RET_ERROR;
  }
  ret = AllocWorkSpace();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to allocate space";
    return RET_ERROR;
  }
  return RET_OK;
}

TrainSession::~TrainSession() { FreeWorkSpace(); }

int TrainSession::ExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                              const std::vector<kernel::LiteKernel *> &run_kernels) {
  for (auto *kernel : run_kernels) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Execute(before, after);
    if (RET_OK != ret) {
      MS_LOG(ERROR) << "Execute kernel failed, name: " << kernel->name();
      return ret;
    }
  }
  return RET_OK;
}

void TrainSession::RestoreTensorData() {
  for (auto &restored_origin_tensor : restored_origin_tensors_) {
    auto *origin_tensor = restored_origin_tensor.first;
    auto *restored_tensor = restored_origin_tensor.second;
    MS_ASSERT(origin_tensor != nullptr);
    MS_ASSERT(restored_tensor != nullptr);

    bool own_data = restored_tensor->own_data();
    if (origin_tensor->data_c() == nullptr) {
      restored_tensor->FreeData();
    } else {
      origin_tensor->FreeData();
    }
    origin_tensor->set_data_type(restored_tensor->data_type());
    origin_tensor->set_data(restored_tensor->data_c());
    origin_tensor->set_own_data(own_data);
  }
}

void TrainSession::FreeRestoreTensors() {
  for (auto &restored_origin_tensor : restored_origin_tensors_) {
    auto *restored_tensor = restored_origin_tensor.second;
    restored_tensor->set_data(nullptr);
    delete (restored_tensor);
  }
  restored_origin_tensors_.clear();
}

bool TrainSession::IsLossTensor(Tensor *tensor) {
  MS_ASSERT(tensor != nullptr);
  auto t_n = tensor->tensor_name();
  return (t_n.find(get_loss_name()) != std::string::npos);
}

bool TrainSession::AllInputsNeedScale(kernel::LiteKernel *kernel) {
  auto type = kernel->type();
  bool is_scale = false;
  switch (type) {
    case schema::PrimitiveType_AbsGrad:
    case schema::PrimitiveType_AddFusion:
    case schema::PrimitiveType_SubFusion:
    case schema::PrimitiveType_AddN:
      for (auto &tensor : kernel->in_tensors()) {
        is_scale = is_scale || tensor->IsScale();
      }
      return (is_scale);
    default:
      return false;
  }
  return false;
}

int TrainSession::MixPrecisionPreProcess(kernel::LiteKernel *kernel, float scale) {
  auto kernel_type = kernel->desc().data_type;
  auto all_scale = AllInputsNeedScale(kernel);

  for (auto &tensor : kernel->in_tensors()) {
    if ((tensor->IsScale() == false) && ((!IsLossKernel(kernel) && IsLossTensor(tensor)) || (all_scale == true))) {
      ScaleTensor(tensor, scale);
    }
    // adjust tensor data type
    if (tensor->data_type() != kernel_type) {
      auto restore_tensor = CastTensor(tensor, kernel_type, this->context_->device_and_pkg_support_fp16());
      if (restore_tensor != nullptr) {
        restored_origin_tensors_[tensor] = restore_tensor;
      }
    }
  }
  return RET_OK;
}

int TrainSession::MixPrecisionPostProcess(kernel::LiteKernel *kernel) {
  RestoreTensorData();
  FreeRestoreTensors();

  float scale = 1.0f;
  auto all_scale = AllInputsNeedScale(kernel);
  for (auto &tensor : kernel->in_tensors()) {
    if (tensor->IsScale()) {
      scale *= tensor->get_scale();
      if (all_scale) {
        break;
      }
    }
  }
  for (auto &tensor : kernel->out_tensors()) {
    tensor->set_scale(scale);
  }

  for (auto &tensor : kernel->in_tensors()) {
    if ((tensor->IsScale() == true) && ((!IsLossKernel(kernel) && IsLossTensor(tensor)) || (all_scale == true))) {
      ScaleTensor(tensor, 1.0f / scale);
    }
  }
  return RET_OK;
}

int TrainSession::MixPrecisionExecKernels(const KernelCallBack &before, const KernelCallBack &after,
                                          const std::vector<kernel::LiteKernel *> &run_kernels) {
  float scale = cfg_.mix_precision_cfg_.loss_scale_;
  for (auto *kernel : run_kernels) {
    MS_ASSERT(kernel != nullptr);
    MixPrecisionPreProcess(kernel, scale);
    auto ret = kernel->Execute(before, after);
    if (RET_OK != ret) {
      MixPrecisionPostProcess(kernel);
      // decrease loss scale in case of nan or inf
      if (ret == RET_OUT_OF_TENSOR_RANGE) {
        bool is_dynamic_scale = cfg_.mix_precision_cfg_.dynamic_loss_scale_;
        cfg_.mix_precision_cfg_.loss_scale_ = std::max(((is_dynamic_scale) ? (scale / 2.f) : scale), 1.0f);
        num_of_not_nan_iter_ = 0;
        return RET_OK;
      }
      MS_LOG(ERROR) << "Execute kernel failed, name: " << kernel->name();
      return ret;
    }
    MixPrecisionPostProcess(kernel);
  }
  // increase dynamic loss scale if pass pass threshold
  if (cfg_.mix_precision_cfg_.dynamic_loss_scale_) {
    num_of_not_nan_iter_++;
    if (num_of_not_nan_iter_ >= cfg_.mix_precision_cfg_.num_of_not_nan_iter_th_) {
      cfg_.mix_precision_cfg_.loss_scale_ = std::min((cfg_.mix_precision_cfg_.loss_scale_ * 2.0f), 65536.0f);
      num_of_not_nan_iter_ = 0;
    }
  }

  // cast output to FP32
  if (train_mode_ == false) {
    for (auto t : this->outputs_) {
      if (t->data_type() == kNumberTypeFloat16) {
        auto restore = CastTensor(t, kNumberTypeFloat32, this->context_->device_and_pkg_support_fp16());
        delete restore;
      }
    }
  }
  return RET_OK;
}

int TrainSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  // check inputs
  auto ret = CheckTensorsInvalid(inputs_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CheckInputs failed";
    return ret;
  }

  // build out tensor
  this->outputs_.clear();
  for (auto &ms_tensors : output_node_map_) {
    for (auto &ms_tensor : ms_tensors.second) {
      auto lite_tensor = static_cast<lite::Tensor *>(ms_tensor);
      this->outputs_.push_back(lite_tensor);
    }
  }

  if (this->context_ == nullptr) {
    MS_LOG(ERROR) << "context is null";
    return lite::RET_NULL_PTR;
  }
  auto &run_kernels = (train_mode_) ? train_kernels_ : inference_kernels_;
  if (context_->IsCpuFloat16Enabled()) {
    ret = MixPrecisionExecKernels(before, after, run_kernels);
  } else {
    ret = ExecKernels(before, after, run_kernels);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to run model kernels";
    return ret;
  }

  if (train_mode_ && virtual_batch_multiplier_) {
    virtual_batch_idx_++;
    if (virtual_batch_idx_ >= virtual_batch_multiplier_) {
      virtual_batch_idx_ = 0;
      ret = OptimizerStep();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "failed to optimize model weights";
        return ret;
      }
    }
  }
  return RET_OK;
}

int TrainSession::Train() {
  // shift kernels to train mode
  train_mode_ = true;
  virtual_batch_idx_ = 0;
  for (auto &kernel : this->train_kernels_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Train();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << kernel->name() << " failed to set train mode";
      return RET_ERROR;
    }
  }
  // set train outputs
  output_node_map_ = train_output_node_map_;
  output_tensor_map_ = train_output_tensor_map_;
  output_tensor_names_ = train_output_tensor_names_;
  return RET_OK;
}

int TrainSession::Eval() {
  // shift kernels to eval mode
  train_mode_ = false;
  virtual_batch_idx_ = 0;
  for (auto &kernel : this->train_kernels_) {
    MS_ASSERT(kernel != nullptr);
    auto ret = kernel->Eval();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << kernel->name() << " failed to set eval mode";
      return RET_ERROR;
    }
  }
  // set eval outputs
  output_node_map_ = eval_output_node_map_;
  output_tensor_map_ = eval_output_tensor_map_;
  output_tensor_names_ = eval_output_tensor_names_;
  return RET_OK;
}

void TrainSession::CompileEvalOutputs() {
  eval_output_node_map_.clear();
  eval_output_tensor_map_.clear();
  eval_output_tensor_names_.clear();
  for (auto kernel : this->train_kernels_) {
    if (IsLossKernel(kernel) && !(IsGradKernel(kernel))) {
      for (auto in_kernel : kernel->in_kernels()) {
        if (IsLossKernel(in_kernel) || IsGradKernel(in_kernel)) continue;
        // insert if not already in
        if (eval_output_node_map_.find(in_kernel->name()) == eval_output_node_map_.end()) {
          auto *ms_tensor = in_kernel->out_tensors().at(0);
          if (ms_tensor != nullptr) {
            ms_tensor->set_init_ref_count(ms_tensor->init_ref_count() + 1);
            eval_output_node_map_[in_kernel->name()].emplace_back(ms_tensor);
            auto index = TSFindTensor(tensors_, ms_tensor);
            if (index != tensors_.size()) {
              eval_output_tensor_map_.insert(std::make_pair(std::to_string(index), ms_tensor));
              if (!ms_tensor->tensor_name().empty()) {
                eval_output_tensor_names_.emplace_back(ms_tensor->tensor_name());
              } else {
                eval_output_tensor_names_.emplace_back(std::to_string(index));
              }
            }
          }
        }
      }
    }
  }
  if (eval_output_node_map_.size() == 0) eval_output_node_map_ = orig_output_node_map_;
  if (eval_output_tensor_map_.size() == 0) eval_output_tensor_map_ = orig_output_tensor_map_;
  if (eval_output_tensor_names_.size() == 0) eval_output_tensor_names_ = orig_output_tensor_names_;
}

void TrainSession::CompileTrainOutputs() {
  train_output_node_map_.clear();
  train_output_tensor_map_.clear();
  train_output_tensor_names_.clear();
  for (auto kernel : this->train_kernels_) {
    if (orig_output_node_map_.find(kernel->name()) == orig_output_node_map_.end()) continue;
    // Mask out optimizer out tensors
    if (IsMaskOutput(kernel)) continue;
    // insert if not already in
    if (train_output_node_map_.find(kernel->name()) == train_output_node_map_.end()) {
      auto *ms_tensor = kernel->out_tensors().at(0);
      if (ms_tensor != nullptr) {
        train_output_node_map_[kernel->name()].emplace_back(ms_tensor);
        auto index = TSFindTensor(tensors_, ms_tensor);
        if (index != tensors_.size()) {
          train_output_tensor_map_.insert(std::make_pair(std::to_string(index), ms_tensor));
          if (!ms_tensor->tensor_name().empty()) {
            train_output_tensor_names_.emplace_back(ms_tensor->tensor_name());
          } else {
            train_output_tensor_names_.emplace_back(std::to_string(index));
          }
        }
      }
    }
  }
  if (train_output_node_map_.size() == 0) train_output_node_map_ = orig_output_node_map_;
  if (train_output_tensor_map_.size() == 0) train_output_tensor_map_ = orig_output_tensor_map_;
  if (train_output_tensor_names_.size() == 0) train_output_tensor_names_ = orig_output_tensor_names_;
}

void TrainSession::BuildInferenceKernelsRecursive(kernel::LiteKernel *kernel, std::vector<kernel::LiteKernel *> *v) {
  MS_ASSERT(kernel != nullptr);
  MS_ASSERT(v != nullptr);
  if (std::find(v->begin(), v->end(), kernel) == v->end()) {  // kernel is not already in vector
    for (auto in_node : kernel->in_kernels()) {
      BuildInferenceKernelsRecursive(in_node, v);
    }
    if (!IsLossKernel(kernel)) v->push_back(kernel);
  }
}

void TrainSession::CompileTrainKernels() {
  train_kernels_.clear();
  for (auto ori_kernel : kernels_) {
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      train_kernels_.push_back(ori_kernel);
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      for (auto kernel : sub_graph->nodes()) {
        train_kernels_.push_back(kernel);
      }
    }
  }
}

int TrainSession::CompileInferenceKernels() {
  inference_kernels_.clear();
  for (auto item : eval_output_node_map_) {
    std::string kernel_name = item.first;
    auto kernel = TSFindKernel(train_kernels_, kernel_name);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "kernel is nullptr";
      return RET_ERROR;
    }
    BuildInferenceKernelsRecursive(kernel, &inference_kernels_);
  }
  if (inference_kernels_.size() == 0) {
    inference_kernels_ = this->train_kernels_;
  }
  return RET_OK;
}

void TrainSession::CompileOptimizedKernels() {
  std::vector<lite::Tensor *> out_tensor;
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      std::copy(kernel->in_tensors().begin(), kernel->in_tensors().end(), std::back_inserter(out_tensor));
    }
  }

  for (auto kernel : this->train_kernels_) {
    if (!IsOptimizer(kernel)) {
      for (auto it : kernel->in_tensors()) {
        if (std::find(out_tensor.begin(), out_tensor.end(), it) != out_tensor.end()) {
          kernel->SetTrainable(true);
          break;
        }
      }
    }
  }
}

int TrainSession::SetLearningRate(float learning_rate) {
  if (learning_rate < 0.0f) {
    MS_LOG(ERROR) << "learning rate should more than 0";
    return RET_ERROR;
  }
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      auto ret = optimizer->SetLearningRate(learning_rate);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set learning rate";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

float TrainSession::GetLearningRate() {
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      return optimizer->GetLearningRate();
    }
  }
  return 0.0;
}

int TrainSession::AdminSetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  auto mod = (virtual_batch_multiplier <= 1) ? kernel::OptimizerKernel::WeightUpdateMode::NORMAL
                                             : kernel::OptimizerKernel::WeightUpdateMode::VIRTUAL_BATCH;
  virtual_batch_multiplier_ = (virtual_batch_multiplier <= 1) ? 0 : virtual_batch_multiplier;
  virtual_batch_idx_ = 0;

  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      auto ret = optimizer->SetOptimizerMode(mod);
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set optimizer mode";
        return RET_ERROR;
      }
      if (mod == kernel::OptimizerKernel::WeightUpdateMode::VIRTUAL_BATCH) {
        lr = (lr < 0.0f) ? (optimizer->GetLearningRate() / static_cast<float>(virtual_batch_multiplier_)) : lr;
        ret = optimizer->SetLearningRate(lr);
      } else {
        ret = optimizer->RestoreDefaultLearningRate();
      }
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set learning rate";
        return RET_ERROR;
      }
    }

    if (IsBN(kernel) && kernel->IsTrainable()) {
      auto batchnorm = static_cast<kernel::BatchnormCPUKernel *>(kernel->kernel());
      auto ret = RET_OK;
      if (mod == kernel::OptimizerKernel::WeightUpdateMode::VIRTUAL_BATCH) {
        momentum = (momentum < 0.0f) ? (batchnorm->get_momentum() / virtual_batch_multiplier_) : momentum;
        ret = batchnorm->set_momentum(momentum);
      } else {
        ret = batchnorm->RestoreDefaultMomentum();
      }
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to set momentum";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}
int TrainSession::SetupVirtualBatch(int virtual_batch_multiplier, float lr, float momentum) {
  int tmp = (virtual_batch_multiplier <= 1) ? 0 : virtual_batch_multiplier;
  if (tmp != 0 && virtual_batch_multiplier_ != 0) {
    AdminSetupVirtualBatch(0, lr, momentum);
  }
  return AdminSetupVirtualBatch(virtual_batch_multiplier, lr, momentum);
}

int TrainSession::OptimizerStep() {
  for (auto kernel : this->train_kernels_) {
    if (IsOptimizer(kernel)) {
      auto optimizer = static_cast<kernel::OptimizerKernel *>(kernel->kernel());
      auto ret = optimizer->OptimizerStep();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << kernel->name() << " failed to do optimize step";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

bool TrainSession::IsLossKernel(const kernel::LiteKernel *kernel) const {
  return (kernel->type() == schema::PrimitiveType_SoftmaxCrossEntropyWithLogits ||
          kernel->type() == schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits ||
          kernel->type() == schema::PrimitiveType_SmoothL1Loss ||
          kernel->type() == schema::PrimitiveType_SmoothL1LossGrad ||
          kernel->type() == schema::PrimitiveType_SigmoidCrossEntropyWithLogits ||
          kernel->type() == schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad) ||
         kernel->name().find(cfg_.loss_name_) != std::string::npos;
}

bool TrainSession::IsGradKernel(const kernel::LiteKernel *kernel) const {
  return kernel->name().find(kGradName) != std::string::npos;
}

bool TrainSession::IsOptimizer(kernel::LiteKernel *kernel) const {
  return ((kernel->type() == schema::PrimitiveType_Adam) || (kernel->type() == schema::PrimitiveType_SGD) ||
          (kernel->type() == schema::PrimitiveType_ApplyMomentum));
}

bool TrainSession::IsMaskOutput(kernel::LiteKernel *kernel) const {
  return (IsOptimizer(kernel) || (kernel->type() == schema::PrimitiveType_Assign));
}

bool TrainSession::IsBN(kernel::LiteKernel *kernel) const {
  return ((kernel->type() == schema::PrimitiveType_BatchNorm) ||
          (kernel->type() == schema::PrimitiveType_FusedBatchNorm));
}

int TrainSession::Export(const std::string &file_name, ModelType model_type, QuantizationType quant_type,
                         FormatType format) {
  if (file_name.empty()) {
    MS_LOG(ERROR) << "File name cannot be empty";
    return RET_ERROR;
  }
  if (model_type > mindspore::lite::MT_INFERENCE || model_type < mindspore::lite::MT_TRAIN) {
    MS_LOG(ERROR) << "Export model type parameter error";
    return RET_ERROR;
  }
  if (quant_type < mindspore::lite::QT_DEFAULT || quant_type > mindspore::lite::QT_WEIGHT) {
    MS_LOG(ERROR) << "Export quant type parameter error";
    return RET_ERROR;
  }
  if (format != FT_FLATBUFFERS) {
    MS_LOG(ERROR) << "Currently only flatbuffer format is supported";
    return RET_ERROR;
  }

  bool orig_train_state = IsTrain();
  Eval();
  TrainExport texport(file_name);
  int status = texport.ExportInit(model_.get()->name_, model_.get()->version_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "cannot init export";
    return status;
  }
  status = texport.ExportNet((model_type == MT_TRAIN) ? train_kernels_ : inference_kernels_, tensors_,
                             (model_type == MT_TRAIN) ? train_output_tensor_names_ : eval_output_tensor_names_,
                             model_.get(), quant_type);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "cannot export Network";
    return status;
  }
  if (model_type == MT_INFERENCE) {
    status = texport.TrainModelDrop();
    if (status != RET_OK) {
      MS_LOG(ERROR) << "TrainModelDrop failed.";
      return status;
    }
  }
  status = texport.SaveToFile();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "failed to save to " << file_name;
    return status;
  }
  if (orig_train_state) Train();
  return status;
}
std::vector<tensor::MSTensor *> TrainSession::GetFeatureMaps() const {
  std::vector<tensor::MSTensor *> features;
  for (auto cur_tensor : this->tensors_) {
    if (cur_tensor->IsConst() && cur_tensor->data_type() == kNumberTypeFloat32) {
      features.push_back(cur_tensor);
    }
  }
  return features;
}

int TrainSession::UpdateFeatureMaps(const std::vector<tensor::MSTensor *> &features_map) {
  for (auto feature : features_map) {
    bool find = false;
    for (auto tensor : tensors_) {
      if (!tensor->IsConst() || tensor->data_type() != kNumberTypeFloat32) {
        continue;
      }
      if (feature->tensor_name() != tensor->tensor_name()) {
        continue;
      }
      if (feature->Size() != tensor->Size()) {
        MS_LOG(ERROR) << "feature name:" << feature->tensor_name() << ",len diff:"
                      << "old is:" << tensor->Size() << "new is:" << feature->Size();
        return RET_ERROR;
      }
      find = true;
      memcpy(tensor->data(), feature->data(), tensor->Size());
    }
    if (!find) {
      MS_LOG(ERROR) << "cannot find feature:" << feature->tensor_name() << ",update failed";
      return RET_ERROR;
    }
  }
  return RET_OK;
}
}  // namespace lite

session::LiteSession *session::TrainSession::CreateTrainSession(const std::string &fn, const lite::Context *context,
                                                                bool train_mode, const lite::TrainCfg *cfg) {
  auto session = std::make_unique<lite::TrainSession>();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }

  auto ret = session->Init(new (std::nothrow) mindspore::lite::InnerContext(context), cfg);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    return nullptr;
  }

  std::string filename = fn;
  if (filename.substr(filename.find_last_of(".") + 1) != "ms") {
    filename = filename + ".ms";
  }

  auto model = std::shared_ptr<lite::Model>(lite::Model::Import(filename.c_str()));
  if (model == nullptr) {
    MS_LOG(ERROR) << "create model for train session failed " << filename;
    return nullptr;
  }

  ret = session->CompileTrainGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Compiling Train Graph session failed";
    return nullptr;
  }

  if (train_mode) {
    ret = session->Train();
  } else {
    ret = session->Eval();
  }
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Could not switch to Train Modei " << train_mode;
    return nullptr;
  }
  return session.release();
}
}  // namespace mindspore
