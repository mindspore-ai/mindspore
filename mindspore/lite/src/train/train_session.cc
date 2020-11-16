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
#include "include/errorcode.h"
#include "include/train_model.h"
#include "src/common/utils.h"
#include "src/tensor.h"
#include "src/train/loss_kernel.h"
#include "src/sub_graph_kernel.h"
#include "src/train/train_populate_parameter.h"
#include "src/runtime/runtime_api.h"
#include "src/executor.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/fp32_grad/convolution.h"

namespace mindspore {
namespace lite {

static size_t TSFindTensor(const std::vector<lite::Tensor *> &where, const lite::Tensor *searchParameter) {
  for (size_t i = 0; i < where.size(); i++) {
    if (where[i] == searchParameter) {
      return i;
    }
  }
  return where.size();
}

TrainSession::TrainSession() { kernel::PopulateTrainParameters(); }

std::vector<CreatorOp> TrainSession::ReplaceOps() {
  const std::vector<CreatorOp> replace = {
    {{mindspore::kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, mindspore::schema::PrimitiveType_Conv2D},
     mindspore::kernel::CpuConvTrainFp32KernelCreator},
    {{mindspore::kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, mindspore::schema::PrimitiveType_DepthwiseConv2D},
     mindspore::kernel::CpuConvTrainFp32KernelCreator}};
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
  for (auto ori_kernel : kernels_) {
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (workspace_size < ori_kernel->GetWorkspaceSize()) {
        workspace_size = ori_kernel->GetWorkspaceSize();
      }
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      for (auto kernel : sub_graph->nodes()) {
        if (workspace_size < kernel->GetWorkspaceSize()) {
          workspace_size = kernel->GetWorkspaceSize();
        }
      }
    }
  }
  mindspore::kernel::LiteKernel::AllocWorkspace(workspace_size);
}

int TrainSession::CompileGraph(lite::Model *model) { return lite::RET_ERROR; }

int TrainSession::CompileTrainGraph(mindspore::lite::TrainModel *model) {
  model_ = model;

  auto restore = ReplaceOps();
  auto ret = lite::LiteSession::CompileGraph(model);
  orig_output_map_ = output_node_map_;
  orig_output_tensor_map_ = output_tensor_map_;
  for (auto inTensor : inputs_) inTensor->MutableData();
  RestoreOps(restore);
  AllocWorkSpace();
  MarkOptimizedKernels();
  CompileTrainKernels();
  if (train_mode_) {
    auto ret1 = Train();
    if (ret1 != RET_OK) {
      MS_LOG(ERROR) << "faild to initialize network in train mode";
      return RET_ERROR;
    }
  } else {
    auto ret1 = Eval();
    if (ret1 != RET_OK) {
      MS_LOG(ERROR) << "faild to initialize network in eval mode";
      return RET_ERROR;
    }
  }
  return ret;
}

TrainSession::~TrainSession() {
  mindspore::kernel::LiteKernel::FreeWorkspace();
  delete model_;
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
  lite::Executor executor;
  if (before == nullptr && after == nullptr) {
    return executor.Run(this->inputs_, this->outputs_, run_kernel, this->context_->allocator.get());
  } else {
    return executor.Run(this->inputs_, this->outputs_, run_kernel, this->context_->allocator.get(), before, after);
  }
}

int TrainSession::SaveToFile(const std::string &filename) const {
  size_t fb_size = 0;
  auto *buf = reinterpret_cast<char *>(ExportToBuf(nullptr, &fb_size));
  if (buf == NULL) {
    MS_LOG(ERROR) << "Could not Export Trained model";
    return lite::RET_NULL_PTR;
  }
  std::ofstream ofs(filename);
  if ((true != ofs.good()) || (true != ofs.is_open())) {
    MS_LOG(ERROR) << "Could not open file \"" << filename << "\" for writing";
    free(buf);
    return RET_ERROR;
  }

  ofs.seekp(0, std::ios::beg);
  ofs.write(buf, fb_size);
  ofs.close();
  free(buf);
  return chmod(filename.c_str(), S_IRUSR);
}

int TrainSession::Train() {
  for (auto ori_kernel : kernels_) {
    MS_ASSERT(nullptr != ori_kernel);
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      auto ret = ori_kernel->Train();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << ori_kernel->name() << " failed to set train mode";
        return RET_ERROR;
      }
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      MS_ASSERT(nullptr != sub_graph);
      for (auto kernel : sub_graph->nodes()) {
        MS_ASSERT(nullptr != kernel);
        auto ret = kernel->Train();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << kernel->name() << " failed to set train mode";
          return RET_ERROR;
        }
      }
    }
  }
  output_node_map_.clear();
  output_tensor_map_.clear();
  train_mode_ = true;
  for (auto ori_kernel : kernels_) {
    MS_ASSERT(nullptr != ori_kernel);
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      UpdateOutputMapByLossKernel(ori_kernel);
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      MS_ASSERT(nullptr != sub_graph);
      for (auto kernel : sub_graph->nodes()) {
        MS_ASSERT(nullptr != kernel);
        UpdateOutputMapByLossKernel(kernel);
      }
    }
  }
  return RET_OK;
}

void TrainSession::UpdateOutputMapByLossKernel(const kernel::LiteKernel *kernel) {
  if (IsLossKernel(kernel)) {
    auto *ms_tensor = kernel->out_tensors().at(0);
    if (ms_tensor != nullptr) {
      (void)ms_tensor->MutableData();
      output_node_map_[kernel->name()].emplace_back(ms_tensor);
      auto index = TSFindTensor(tensors_, ms_tensor);
      if (index != tensors_.size()) {
        output_tensor_map_.insert(std::make_pair(std::to_string(index), ms_tensor));
      }
    }
  }
}

void TrainSession::UpdateOutputMapByInKernel(const kernel::LiteKernel *kernel) {
  if (IsLossKernel(kernel)) {
    for (auto in_kernel : kernel->in_kernels()) {
      if (output_node_map_.find(in_kernel->name()) == output_node_map_.end()) {
        auto *ms_tensor = in_kernel->out_tensors().at(0);
        if (ms_tensor != nullptr) {
          output_node_map_[in_kernel->name()].emplace_back(ms_tensor);
          auto index = TSFindTensor(tensors_, ms_tensor);
          if (index != tensors_.size()) {
            output_tensor_map_.insert(std::make_pair(std::to_string(index), ms_tensor));
          }
        }
      }
    }
  }
}

int TrainSession::Eval() {
  for (auto ori_kernel : kernels_) {
    MS_ASSERT(nullptr != ori_kernel);
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      auto ret = ori_kernel->Eval();
      if (ret != RET_OK) {
        MS_LOG(ERROR) << ori_kernel->name() << " failed to set eval mode";
        return RET_ERROR;
      }
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      MS_ASSERT(nullptr != sub_graph);
      for (auto kernel : sub_graph->nodes()) {
        MS_ASSERT(nullptr != kernel);
        auto ret = kernel->Eval();
        if (ret != RET_OK) {
          MS_LOG(ERROR) << kernel->name() << " failed to set eval mode";
          return RET_ERROR;
        }
      }
    }
  }
  output_node_map_ = orig_output_map_;
  output_tensor_map_ = orig_output_tensor_map_;

  train_mode_ = false;
  for (auto ori_kernel : kernels_) {
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      UpdateOutputMapByInKernel(ori_kernel);
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      for (auto kernel : sub_graph->nodes()) {
        UpdateOutputMapByInKernel(kernel);
      }
    }
  }
  if (inference_kernels_.size() == 0) {
    BuildInferenceKernelsMap();
  }
  return RET_OK;
}

void TrainSession::BuildInferenceKernelsRecursive(kernel::LiteKernel *kernel, std::vector<kernel::LiteKernel *> *v) {
  if (std::find(v->begin(), v->end(), kernel) == v->end()) {  // kernel is not in vector
    v->push_back(kernel);
    for (auto in_node : kernel->in_kernels()) {
      BuildInferenceKernelsRecursive(in_node, v);
    }
  }
}

void TrainSession::BuildInferenceKernelsMap() {
  std::vector<kernel::LiteKernel *> req_kernels;
  for (auto kernel : this->kernels_) {
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (IsLossKernel(kernel)) {  // For each loss in the system add backward tree
        for (auto in_node : kernel->in_kernels()) {
          BuildInferenceKernelsRecursive(in_node, &req_kernels);
        }
      }
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      for (auto sb_kernel : sub_graph->nodes()) {
        if (IsLossKernel(sb_kernel)) {  // For each loss in the system add backward tree
          for (auto in_node : sb_kernel->in_kernels()) {
            BuildInferenceKernelsRecursive(in_node, &req_kernels);
          }
        }
      }
    }
  }

  inference_kernels_.clear();
  for (auto ori_kernel : kernels_) {
    if (ori_kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (std::find(req_kernels.begin(), req_kernels.end(), ori_kernel) != req_kernels.end()) {
        inference_kernels_.push_back(ori_kernel);
      }
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(ori_kernel);
      for (auto kernel : sub_graph->nodes()) {
        if (std::find(req_kernels.begin(), req_kernels.end(), kernel) != req_kernels.end()) {
          inference_kernels_.push_back(kernel);
        }
      }
    }
  }
  if (inference_kernels_.size() == 0) {
    inference_kernels_ = this->kernels_;
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

void TrainSession::MarkOptimizedKernels() {
  std::vector<lite::Tensor *> ot;
  for (auto kernel : this->kernels_) {
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (IsOptimizer(kernel)) {
        std::copy(kernel->in_tensors().begin(), kernel->in_tensors().end(), std::back_inserter(ot));
      }
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      for (auto sb_kernel : sub_graph->nodes()) {
        if (IsOptimizer(sb_kernel)) {
          std::copy(sb_kernel->in_tensors().begin(), sb_kernel->in_tensors().end(), std::back_inserter(ot));
        }
      }
    }
  }
  for (auto kernel : this->kernels_) {
    if (kernel->subgraph_type() == kernel::kNotSubGraph) {
      if (!IsOptimizer(kernel)) {
        for (auto it : kernel->in_tensors()) {
          if (std::find(ot.begin(), ot.end(), it) != ot.end()) {
            kernel->SetTrainable(true);
            break;
          }
        }
      }
    } else {
      auto sub_graph = reinterpret_cast<kernel::SubGraphKernel *>(kernel);
      for (auto sb_kernel : sub_graph->nodes()) {
        if (!IsOptimizer(sb_kernel)) {
          for (auto it : sb_kernel->in_tensors()) {
            if (std::find(ot.begin(), ot.end(), it) != ot.end()) {
              sb_kernel->SetTrainable(true);
              break;
            }
          }
        }
      }
    }
  }
}

bool TrainSession::IsLossKernel(const kernel::LiteKernel *kernel) const {
  return (kernel->Type() == schema::PrimitiveType_SoftmaxCrossEntropy);
}

bool TrainSession::IsOptimizer(kernel::LiteKernel *kernel) const {
  return ((kernel->Type() == schema::PrimitiveType_Adam) || (kernel->Type() == schema::PrimitiveType_Sgd) ||
          (kernel->Type() == schema::PrimitiveType_ApplyMomentum));
}

}  // namespace lite

session::TrainSession *session::TrainSession::CreateSession(lite::Context *context) {
  auto session = new (std::nothrow) lite::TrainSession();
  if (session == nullptr) {
    MS_LOG(ERROR) << "create session failed";
    return nullptr;
  }
  auto ret = session->Init(context);
  if (ret != mindspore::lite::RET_OK) {
    MS_LOG(ERROR) << "init sesssion failed";
    delete session;
    return nullptr;
  }
  return session;
}

}  // namespace mindspore
