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
 * Limitations under the License.
 */

#include "transform/graph_runner.h"
#include <algorithm>
#include <string>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/config_manager.h"
#include "sys/time.h"
#include "utils/callbacks.h"
#include "utils/utils.h"
#include "./common.h"

#ifdef NO_GE_CLIENT
namespace ge {
Session::Session(const std::map<std::string, std::string>& options) {
  if (options.empty()) {
    MS_LOG(ERROR) << "session input options is empty";
  }
  sessionId_ = 0;
}
Session::~Session() {}
}  // namespace ge
#endif

namespace mindspore {
namespace transform {
std::shared_ptr<ge::Session> GraphRunner::NewSession(const SessionOptions& sess_options) {
  std::shared_ptr<ge::Session> ret = std::make_shared<ge::Session>(sess_options);
  if (ret == nullptr) {
    MS_LOG(ERROR) << "Create GE session failed";
    return nullptr;
  }
  MS_LOG(INFO) << "Create new GE session success";
  return ret;
}

GraphRunner::GraphRunner(const GraphRunnerOptions& options)
    : options_(options), graph_manager_(DfGraphManager::GetInstance()) {
  if (ConfigManager::GetInstance().parallel_strategy() == ParallelStrategy::ONE_DEVICE) {
    MS_LOG(INFO) << "ME run in ONE_DEVICE strategy mode";
  }

  if (options.sess_ptr != nullptr) {
    sess_ = options.sess_ptr;
  } else {
    sess_ = NewSession(options.options);
    if (sess_ == nullptr) {
      MS_LOG(EXCEPTION) << "GraphRunner initialize failed!!";
      return;
    }
  }

#if (defined ENABLE_GE)
  // register the callback function
  if (sess_->RegisterCallBackFunc(callbacks::kCheckPoint, callbacks::CheckpointSaveCallback) != ge::GRAPH_SUCCESS) {
    MS_LOG(EXCEPTION) << "register callback failed!";
    return;
  }

  if (sess_->RegisterCallBackFunc(callbacks::kSummary, callbacks::SummarySaveCallback) != ge::GRAPH_SUCCESS) {
    MS_LOG(EXCEPTION) << "register summary callback failed!";
    return;
  }
#endif

  std::vector<DfGraphWrapperPtr> wrappers = graph_manager_.GetAllGraphs();
  if (wrappers.empty()) {
    MS_LOG(INFO) << "The GraphManager is empty!!";
    return;
  }

#ifdef ENABLE_GE
  for (auto& it : wrappers) {
    MS_LOG(INFO) << "Add the graph " << (*it).name_ << " to GE, it's id is: " << (*it).id_;
    (void)sess_->AddGraph((*it).id_, *((*it).graph_ptr_));
  }
#endif
}

Status GraphRunner::RunGraph(const RunOptions& options, const std::vector<GeTensorPtr>& inputs,
                             std::vector<GeTensorPtr>* outputs) {
  std::string name = options.name;
  if (name.empty()) {
    MS_LOG(ERROR) << "The graph name is null";
    return Status::INVALID_ARGUMENT;
  }

  DfGraphWrapperPtr wrap_ptr = graph_manager_.GetGraphByName(name);
  if (wrap_ptr == nullptr) {
    MS_LOG(ERROR) << "Get graph form DfGraphManager failed!";
    return Status::NOT_FOUND;
  }

  if (wrap_ptr->graph_ptr_ == nullptr) {
    MS_LOG(WARNING) << "The graph is null";
    return Status::NOT_FOUND;
  }

  // call ge::RunGraph() to exec a graph;
  std::vector<GeTensor> ge_inputs;
  std::vector<GeTensor> ge_outputs;

  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(ge_inputs),
                       [](const GeTensorPtr& i) { return *i; });

  MS_LOG(INFO) << "Run the graph in GE with " << ge_inputs.size() << " inputs";

  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);

#ifdef ENABLE_GE
  if (sess_ == nullptr) {
    MS_LOG(ERROR) << "The GE session is null, can't run the graph!";
    return Status::FAILED;
  }

  ge::Status ret = sess_->RunGraph(wrap_ptr->id_, ge_inputs, ge_outputs);
  if (ret != ge::GRAPH_SUCCESS) {
    MS_LOG(ERROR) << "Call GE RunGraph Failed, ret is: " << ret;
    return Status::FAILED;
  }
#else
  ge_outputs.swap(ge_inputs);
#endif

  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Call GE RunGraph Success in " << cost << " us, the GE outputs num is: " << ge_outputs.size();

  (void)std::transform(ge_outputs.begin(), ge_outputs.end(), std::back_inserter(*outputs),
                       [](const GeTensor& ge_tensor) { return std::make_shared<GeTensor>(ge_tensor); });

  return Status::SUCCESS;
}

Status GraphRunner::RunGraph(const RunOptions& options, const std::vector<MeTensorPtr>& inputs,
                             std::vector<MeTensorPtr>* const outputs) {
  std::vector<GeTensorPtr> ge_inputs;
  for (auto it : inputs) {
    MS_LOG(INFO) << "inputs tensor's data size is: " << (*it).DataSize();
    auto shape = (*it).shape();
    std::string shape_str;
    for (const auto& elem : shape) {
      shape_str += std::to_string(elem);
      shape_str += " ";
    }
    MS_LOG(INFO) << "inputs tensor's shape is: { " << shape_str << "}";

    auto ge_tensor_ptr = TransformUtil::ConvertTensor(it, kOpFormat_NCHW);
    if (ge_tensor_ptr != nullptr) {
      ge_inputs.emplace_back(ge_tensor_ptr);
    } else {
      MS_LOG(INFO) << "Convert input Me tensor to Ge tensor failed. Abort this graph";
      return Status::FAILED;
    }
  }

  std::vector<GeTensorPtr> ge_outputs;
  Status ret;
  {
    // Release GIL before calling into (potentially long-running) C++ code
    py::gil_scoped_release release;
    ret = RunGraph(options, ge_inputs, &ge_outputs);
  }
  if (ret != Status::SUCCESS) {
    return ret;
  } else {
    // conver GeTensor to MeTensor
    for (auto& it : ge_outputs) {
      auto tensor = TransformUtil::ConvertGeTensor(it);
      if (tensor != nullptr) {
        outputs->emplace_back(tensor);
      }
    }
    MS_LOG(INFO) << "Return Me tensor outputs num is: " << outputs->size();
    return Status::SUCCESS;
  }
}
}  // namespace transform
}  // namespace mindspore
