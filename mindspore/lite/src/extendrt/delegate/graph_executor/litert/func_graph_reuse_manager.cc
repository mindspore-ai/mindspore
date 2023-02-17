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

#include "extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include <utility>
#include "src/common/common.h"
namespace mindspore {
std::mutex mtx_manager_;

FuncGraphReuseManager *FuncGraphReuseManager::GetInstance() {
  std::unique_lock<std::mutex> l(mtx_manager_);
  static FuncGraphReuseManager instance;
  return &instance;
}

FuncGraphPtr FuncGraphReuseManager::GetSharedFuncGraph(
  std::map<std::string, std::map<std::string, std::string>> config_info) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(mindspore::lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    if (id->second.find(lite::kInnerRunnerIDKey) != id->second.end()) {
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      if (all_func_graphs_.find(runner_id) != all_func_graphs_.end()) {
        auto func_graph = all_func_graphs_[runner_id];
        return func_graph;
      }
    } else {
      MS_LOG(ERROR) << "config info not find runner id.";
      return nullptr;
    }
  }
  MS_LOG(INFO) << "can not find model buf in all store function graphs";
  return nullptr;
}

Status FuncGraphReuseManager::StoreFuncGraph(FuncGraphPtr func_graph,
                                             std::map<std::string, std::map<std::string, std::string>> config_info) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    if (id->second.find(lite::kInnerRunnerIDKey) != id->second.end()) {
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      all_func_graphs_[runner_id] = func_graph;
      return kSuccess;
    }
  }
  return kSuccess;
}

std::pair<void *, std::shared_ptr<mindspore::infer::helper::InferHelpers>> FuncGraphReuseManager::GetFbModelBuf(
  size_t *data_size, bool *is_shared_fb_buf, std::map<std::string, std::map<std::string, std::string>> config_info) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    auto item_runner_id = id->second.find(lite::kInnerRunnerIDKey);
    if (item_runner_id != id->second.end()) {
      // get runner id
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      *is_shared_fb_buf = true;
      if (all_fb_model_buf_.find(runner_id) != all_fb_model_buf_.end() &&
          all_infer_helpers_.find(runner_id) != all_infer_helpers_.end()) {
        *data_size = all_fb_model_buf_[runner_id].buf_size;
        return std::make_pair(all_fb_model_buf_[runner_id].buf, all_infer_helpers_[runner_id]);
      }
    } else {
      MS_LOG(ERROR) << "config info not find runner id, numa id or worker id.";
      return std::make_pair(nullptr, nullptr);
    }
  }
  MS_LOG(INFO) << "can not find model buf in all store Pb model buf";
  return std::make_pair(nullptr, nullptr);
}

Status FuncGraphReuseManager::StoreFbModelBuf(void *model_buf, size_t data_size,
                                              std::shared_ptr<mindspore::infer::helper::InferHelpers> helper,
                                              std::map<std::string, std::map<std::string, std::string>> config_info) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    auto item_runner_id = id->second.find(lite::kInnerRunnerIDKey);
    if (item_runner_id != id->second.end()) {
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      ModelBufPair buf = {model_buf, data_size};
      all_fb_model_buf_[runner_id] = buf;
      all_infer_helpers_[runner_id] = helper;
      return kSuccess;
    }
  }
  return kLiteError;
}

KernelGraphPtr FuncGraphReuseManager::GetKernelGraph(
  std::map<std::string, std::map<std::string, std::string>> config_info) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(mindspore::lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    if (id->second.find(lite::kInnerRunnerIDKey) != id->second.end()) {
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      if (all_kernel_graph_.find(runner_id) != all_kernel_graph_.end()) {
        auto kernel_graph = all_kernel_graph_[runner_id];
        return kernel_graph;
      }
    } else {
      MS_LOG(ERROR) << "config info not find runner id.";
      return nullptr;
    }
  }
  MS_LOG(INFO) << "can not find model buf in all store function graphs";
  return nullptr;
}

Status FuncGraphReuseManager::StoreKernelGraph(std::map<std::string, std::map<std::string, std::string>> config_info,
                                               KernelGraphPtr kernel_graph) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    if (id->second.find(lite::kInnerRunnerIDKey) != id->second.end()) {
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      all_kernel_graph_[runner_id] = kernel_graph;
      return kSuccess;
    }
  }
  return kSuccess;
}

Status FuncGraphReuseManager::GetInOut(std::map<std::string, std::map<std::string, std::string>> config_info,
                                       std::vector<tensor::TensorPtr> *in_tensor,
                                       std::vector<tensor::TensorPtr> *out_tensor, std::vector<std::string> *in_name,
                                       std::vector<std::string> *out_name) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(mindspore::lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    if (id->second.find(lite::kInnerRunnerIDKey) != id->second.end()) {
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      if (all_in_tensors_.find(runner_id) != all_in_tensors_.end() &&
          all_out_tensors_.find(runner_id) != all_out_tensors_.end() &&
          all_in_names_.find(runner_id) != all_in_names_.end() &&
          all_out_names_.find(runner_id) != all_out_names_.end()) {
        *in_tensor = all_in_tensors_[runner_id];
        *out_tensor = all_out_tensors_[runner_id];
        *in_name = all_in_names_[runner_id];
        *out_name = all_out_names_[runner_id];
        return kSuccess;
      }
    } else {
      MS_LOG(ERROR) << "config info not find runner id.";
      return kLiteError;
    }
  }
  MS_LOG(INFO) << "can not find model buf in all store function graphs";
  return kLiteError;
}

Status FuncGraphReuseManager::StoreInOut(std::map<std::string, std::map<std::string, std::string>> config_info,
                                         std::vector<tensor::TensorPtr> in_tensor,
                                         std::vector<tensor::TensorPtr> out_tensor, std::vector<std::string> in_name,
                                         std::vector<std::string> out_name) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  auto id = config_info.find(lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    if (id->second.find(lite::kInnerRunnerIDKey) != id->second.end()) {
      auto runner_id = id->second[lite::kInnerRunnerIDKey];
      all_in_tensors_[runner_id] = in_tensor;
      all_out_tensors_[runner_id] = out_tensor;
      all_in_names_[runner_id] = in_name;
      all_out_names_[runner_id] = out_name;
      return kSuccess;
    }
  }
  return kSuccess;
}

void FuncGraphReuseManager::ReleaseSharedFuncGraph(
  std::map<std::string, std::map<std::string, std::string>> config_info) {
  std::unique_lock<std::mutex> l(mtx_manager_);
  MS_LOG(INFO) << "ReleaseSharedFuncGraph begin.";
  std::string runner_id = "";
  auto id = config_info.find(lite::kInnerModelParallelRunnerSection);
  if (id != config_info.end()) {
    runner_id = id->second[lite::kInnerRunnerIDKey];
  }
  if (all_func_graphs_.find(runner_id) != all_func_graphs_.end()) {
    MS_LOG(INFO) << "release shared function graph of runner id: " << runner_id;
    all_func_graphs_.erase(runner_id);
  }
  if (all_kernel_graph_.find(runner_id) != all_kernel_graph_.end()) {
    MS_LOG(INFO) << "release shared kernel graph of runner id: " << runner_id;
    all_kernel_graph_.erase(runner_id);
  }
  if (all_infer_helpers_.find(runner_id) != all_infer_helpers_.end()) {
    MS_LOG(INFO) << "release shared infer helpers of runner id: " << runner_id;
    all_infer_helpers_.erase(runner_id);
  }
  if (all_in_names_.find(runner_id) != all_in_names_.end() && all_out_names_.find(runner_id) != all_out_names_.end() &&
      all_in_tensors_.find(runner_id) != all_in_tensors_.end() &&
      all_out_tensors_.find(runner_id) != all_out_tensors_.end()) {
    MS_LOG(INFO) << "release shared input/output of runner id: " << runner_id;
    all_in_names_.erase(runner_id);
    all_out_names_.erase(runner_id);
    all_in_tensors_.erase(runner_id);
    all_out_tensors_.erase(runner_id);
  }
  if (all_fb_model_buf_.find(runner_id) != all_fb_model_buf_.end()) {
    void *fb_model_buf = all_fb_model_buf_[runner_id].buf;
    if (fb_model_buf != nullptr) {
      free(fb_model_buf);
      fb_model_buf = nullptr;
    }
    all_fb_model_buf_.erase(runner_id);
  }
  MS_LOG(INFO) << "ReleaseSharedFuncGraph end.";
}

FuncGraphReuseManager::~FuncGraphReuseManager() {
  std::unique_lock<std::mutex> l(mtx_manager_);
  MS_LOG(INFO) << "~FuncGraphReuseManager() begin.";
  all_func_graphs_.clear();
  all_kernel_graph_.clear();
  all_infer_helpers_.clear();
  all_in_tensors_.clear();
  all_out_tensors_.clear();
  all_in_names_.clear();
  all_out_names_.clear();
  for (auto &item : all_fb_model_buf_) {
    auto &buf = item.second.buf;
    free(buf);
    buf = nullptr;
  }
  all_fb_model_buf_.clear();
  MS_LOG(INFO) << "~FuncGraphReuseManager() end.";
}
}  // namespace mindspore
