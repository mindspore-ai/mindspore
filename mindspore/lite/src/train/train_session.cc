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
#include "include/errorcode.h"
#include "src/common/utils.h"
#include "src/tensor.h"
#include "src/train/loss_kernel.h"
#include "src/train/optimizer_kernel.h"
#include "src/sub_graph_kernel.h"
#include "src/train/train_populate_parameter.h"
#include "src/train/train_populate_parameter_v0.h"
#include "src/runtime/runtime_api.h"
#include "src/executor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/convolution.h"
#include "src/runtime/kernel/arm/fp32/batchnorm_fp32.h"

namespace mindspore {
namespace lite {

std::unique_ptr<char[]> ReadFileToBuf(const std::string &filename, size_t *size) {
  std::ifstream ifs(filename);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "File: " << filename << " does not exist";
    return std::unique_ptr<char[]>(nullptr);
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "File: " << filename << " open failed";
    return std::unique_ptr<char[]>(nullptr);
  }

  ifs.seekg(0, std::ios::end);
  auto tellg_ret = ifs.tellg();
  if (tellg_ret <= 0) {
    MS_LOG(ERROR) << "Could not read file " << filename;
    return std::unique_ptr<char[]>(nullptr);
  }
  size_t fsize = static_cast<size_t>(tellg_ret);

  std::unique_ptr<char[]> buf(new (std::nothrow) char[fsize]);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << filename;
    ifs.close();
    return std::unique_ptr<char[]>(nullptr);
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(buf.get(), fsize);
  if (!ifs) {
    MS_LOG(ERROR) << "only read " << ifs.gcount() << "bytes in " << filename;
    ifs.close();
    return std::unique_ptr<char[]>(nullptr);
  }
  ifs.close();
  if (size != nullptr) {
    *size = fsize;
  }
  return buf;
}

static size_t TSFindTensor(const std::vector<lite::Tensor *> &where, const lite::Tensor *searchParameter) {
  for (size_t i = 0; i < where.size(); i++) {
    if (where[i] == searchParameter) {
      return i;
    }
  }
  return where.size();
}

static kernel::LiteKernel *TSFindKernel(const std::vector<kernel::LiteKernel *> &where,
                                        const std::string &searchParameter) {
  auto it = std::find_if(where.begin(), where.end(),
                         [&searchParameter](const kernel::LiteKernel *k) { return (k->name() == searchParameter); });
  return *it;
}
TrainSession::TrainSession() {
#ifdef ENABLE_V0
  if (VersionManager::GetInstance()->CheckV0Schema()) {
    kernel::PopulateTrainV0Parameters();
  }
#endif
  if (!VersionManager::GetInstance()->CheckV0Schema()) {
    kernel::PopulateTrainParameters();
  }
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

void TrainSession::AllocWorkSpace() {
  size_t workspace_size = 0;
  for (auto kernel : this->train_kernels_) {
    if (workspace_size < kernel->workspace_size()) {
      workspace_size = kernel->workspace_size();
    }
  }
  mindspore::kernel::LiteKernel::AllocWorkspace(workspace_size);
}

int TrainSession::CompileGraph(lite::Model *model) { return lite::RET_ERROR; }

int TrainSession::CompileTrainGraph(mindspore::lite::TrainModel *model) {
  model_ = model;

  auto restore = ReplaceOps();
  auto ret = lite::LiteSession::CompileGraph(model);
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
  CompileInferenceKernels();  // Prepare a list of eval kernels
  AllocWorkSpace();

  return RET_OK;
}

TrainSession::~TrainSession() {
  mindspore::kernel::LiteKernel::FreeWorkspace();
  if (model_ != nullptr) {
    delete model_;
    model_ = nullptr;
  }
}

void *TrainSession::ExportToBuf(char *buf, size_t *len) const { return model_->ExportBuf(buf, len); }

int TrainSession::RunGraph(const KernelCallBack &before, const KernelCallBack &after) {
  this->outputs_.clear();

  // build out tensor
  for (auto ms_tensors : output_node_map_) {
    for (auto ms_tensor : ms_tensors.second) {
      this->outputs_.push_back((static_cast<lite::Tensor *>(ms_tensor)));
    }
  }

  if (this->context_ == nullptr) {
    MS_LOG(ERROR) << "context is null";
    return lite::RET_NULL_PTR;
  }
  auto run_kernel = (train_mode_) ? train_kernels_ : inference_kernels_;
  lite::CpuExecutor executor;
  auto ret = RET_OK;
  if (before == nullptr && after == nullptr) {
    ret = executor.Run(this->inputs_, this->outputs_, run_kernel, this->context_->allocator.get());
  } else {
    ret = executor.Run(this->inputs_, this->outputs_, run_kernel, this->context_->allocator.get(), before, after);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "failed to run model";
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

int TrainSession::SaveToFile(const std::string &filename) const {
  size_t fb_size = 0;
  const auto *buf = reinterpret_cast<char *>(model_->GetBuffer(&fb_size));
  if (buf == nullptr) {
    MS_LOG(ERROR) << "Could not Export Trained model";
    return lite::RET_NULL_PTR;
  }
  std::ofstream ofs(filename);
  if ((true != ofs.good()) || (true != ofs.is_open())) {
    MS_LOG(ERROR) << "Could not open file \"" << filename << "\" for writing";
    return RET_ERROR;
  }

  ofs.seekp(0, std::ios::beg);
  ofs.write(buf, fb_size);
  ofs.close();
  return chmod(filename.c_str(), S_IRUSR);
}

int TrainSession::Train() {
  // shift kernels to train mode
  train_mode_ = true;
  virtual_batch_idx_ = 0;
  for (auto kernel : this->train_kernels_) {
    MS_ASSERT(nullptr != kernel);
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
  for (auto kernel : this->train_kernels_) {
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

void TrainSession::CompileInferenceKernels() {
  inference_kernels_.clear();
  for (auto item : eval_output_node_map_) {
    std::string kernel_name = item.first;
    auto kernel = TSFindKernel(train_kernels_, kernel_name);
    BuildInferenceKernelsRecursive(kernel, &inference_kernels_);
  }
  if (inference_kernels_.size() == 0) {
    inference_kernels_ = this->train_kernels_;
  }
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
          kernel->set_trainable(true);
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
      auto optimizer = reinterpret_cast<kernel::OptimizerKernel *>(kernel);
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
      auto optimizer = reinterpret_cast<kernel::OptimizerKernel *>(kernel);
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
      auto optimizer = reinterpret_cast<kernel::OptimizerKernel *>(kernel);
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

    if (IsBN(kernel) && kernel->is_trainable()) {
      auto batchnorm = reinterpret_cast<kernel::BatchnormCPUKernel *>(kernel);
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
      auto optimizer = reinterpret_cast<kernel::OptimizerKernel *>(kernel);
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
  return (kernel->Type() == schema::PrimitiveType_SoftmaxCrossEntropyWithLogits ||
          kernel->Type() == schema::PrimitiveType_SparseSoftmaxCrossEntropyWithLogits ||
          kernel->Type() == schema::PrimitiveType_SmoothL1Loss ||
          kernel->Type() == schema::PrimitiveType_SmoothL1LossGrad ||
          kernel->Type() == schema::PrimitiveType_SigmoidCrossEntropyWithLogits ||
          kernel->Type() == schema::PrimitiveType_SigmoidCrossEntropyWithLogitsGrad) ||
         kernel->name().find(get_loss_name()) != std::string::npos;
}

bool TrainSession::IsGradKernel(const kernel::LiteKernel *kernel) const {
  return kernel->name().find("Gradients") != std::string::npos;
}

bool TrainSession::IsOptimizer(kernel::LiteKernel *kernel) const {
  return ((kernel->Type() == schema::PrimitiveType_Adam) || (kernel->Type() == schema::PrimitiveType_SGD) ||
          (kernel->Type() == schema::PrimitiveType_ApplyMomentum));
}

bool TrainSession::IsMaskOutput(kernel::LiteKernel *kernel) const {
  return (IsOptimizer(kernel) || (kernel->Type() == schema::PrimitiveType_Assign));
}

bool TrainSession::IsBN(kernel::LiteKernel *kernel) const {
  return ((kernel->Type() == schema::PrimitiveType_BatchNorm) ||
          (kernel->Type() == schema::PrimitiveType_FusedBatchNorm));
}

int TrainSession::SetLossName(std::string loss_name) {
  session::TrainSession::SetLossName(loss_name);
  CompileEvalOutputs();
  CompileInferenceKernels();
  if (IsEval()) {
    output_node_map_ = eval_output_node_map_;
    output_tensor_map_ = eval_output_tensor_map_;
    output_tensor_names_ = eval_output_tensor_names_;
  }
  return RET_OK;
}
}  // namespace lite

session::TrainSession *session::TrainSession::CreateSession(const char *model_buf, size_t size, lite::Context *context,
                                                            bool train_mode) {
  auto model = mindspore::lite::TrainModel::Import(model_buf, size);
  if (model == nullptr) {
    MS_LOG(ERROR) << "create model for  train session failed";
    return nullptr;
  }

  auto session = new (std::nothrow) lite::TrainSession();
  if (session == nullptr) {
    delete model;
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }
  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init session failed";
    delete session;
    delete model;
    return nullptr;
  }

  ret = session->CompileTrainGraph(model);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Compiling Train Graph session failed";
    delete session;
    return nullptr;
  }

  if (train_mode) {
    ret = session->Train();
  } else {
    ret = session->Eval();
  }
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "Could not switch to Train Modei " << train_mode;
    delete session;
    return nullptr;
  }

  return session;
}

session::TrainSession *session::TrainSession::CreateSession(const std::string &filename, lite::Context *context,
                                                            bool train_mode) {
  size_t size = -1;
  auto buf = lite::ReadFileToBuf(filename, &size);
  if (buf == nullptr) {
    return nullptr;
  }
  return session::TrainSession::CreateSession(buf.get(), size, context, train_mode);
}

}  // namespace mindspore
