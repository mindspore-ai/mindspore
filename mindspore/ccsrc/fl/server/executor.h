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

#ifndef MINDSPORE_CCSRC_FL_SERVER_EXECUTOR_H_
#define MINDSPORE_CCSRC_FL_SERVER_EXECUTOR_H_

#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include "fl/server/common.h"
#include "fl/server/parameter_aggregator.h"
#ifdef ENABLE_ARMOUR
#include "fl/armour/cipher/cipher_unmask.h"
#endif

namespace mindspore {
namespace fl {
namespace server {
// Executor is the entrance for server to handle aggregation, optimizing, model querying, etc. It handles
// logics relevant to kernel launching.
class Executor {
 public:
  static Executor &GetInstance() {
    static Executor instance;
    return instance;
  }

  // FuncGraphPtr func_graph is the graph compiled by the frontend. aggregation_count is the number which will
  // be used for aggregators.
  // As noted in header file parameter_aggregator.h, we create aggregators by trainable parameters, which is the
  // optimizer cnode's input. So we need to initialize server executor using func_graph.
  void Initialize(const FuncGraphPtr &func_graph, size_t aggregation_count);

  // Reinitialize parameter aggregators after scaling operations are done.
  bool ReInitForScaling();

  // After hyper-parameters are updated, some parameter aggregators should be reinitialized.
  bool ReInitForUpdatingHyperParams(size_t aggr_threshold);

  // Called in parameter server training mode to do Push operation.
  // For the same trainable parameter, HandlePush method must be called aggregation_count_ times before it's considered
  // as completed.
  bool HandlePush(const std::string &param_name, const UploadData &upload_data);

  // Called in parameter server training mode to do Pull operation.
  // Returns the value of parameter param_name.
  // HandlePull method must be called the same times as HandlePush is called before it's considered as
  // completed.
  AddressPtr HandlePull(const std::string &param_name);

  // Called in federated learning training mode. Update value for parameter param_name.
  bool HandleModelUpdate(const std::string &param_name, const UploadData &upload_data);

  // Called in asynchronous federated learning training mode. Update current model with the new feature map
  // asynchronously.
  bool HandleModelUpdateAsync(const std::map<std::string, UploadData> &feature_map);

  // Overwrite the weights in server using pushed feature map.
  bool HandlePushWeight(const std::map<std::string, Address> &feature_map);

  // Returns multiple trainable parameters passed by weight_names.
  std::map<std::string, AddressPtr> HandlePullWeight(const std::vector<std::string> &param_names);

  // Reset the aggregation status for all aggregation kernels in the server.
  void ResetAggregationStatus();

  // Judge whether aggregation processes for all weights/gradients are completed.
  bool IsAllWeightAggregationDone();

  // Judge whether the aggregation processes for the given param_names are completed.
  bool IsWeightAggrDone(const std::vector<std::string> &param_names);

  // Returns whole model in key-value where key refers to the parameter name.
  std::map<std::string, AddressPtr> GetModel();

  // Returns whether the executor singleton is already initialized.
  bool initialized() const;

  const std::vector<std::string> &param_names() const;

  // The unmasking method for pairwise encrypt algorithm.
  bool Unmask();

  // The setter and getter for unmasked flag to judge whether the unmasking is completed.
  void set_unmasked(bool unmasked);
  bool unmasked() const;

 private:
  Executor() : initialized_(false), aggregation_count_(0), param_names_({}), param_aggrs_({}), unmasked_(false) {}
  ~Executor() = default;
  Executor(const Executor &) = delete;
  Executor &operator=(const Executor &) = delete;

  // Returns the trainable parameter name parsed from this cnode.
  std::string GetTrainableParamName(const CNodePtr &cnode);

  // Server's graph is basically the same as Worker's graph, so we can get all information from func_graph for later
  // computations. Including forward and backward propagation, aggregation, optimizing, etc.
  bool InitParamAggregator(const FuncGraphPtr &func_graph);

  bool initialized_;
  size_t aggregation_count_;
  std::vector<std::string> param_names_;

  // The map for trainable parameter names and its ParameterAggregator, as noted in the header file
  // parameter_aggregator.h
  std::map<std::string, std::shared_ptr<ParameterAggregator>> param_aggrs_;

  // The mutex ensures that the operation on whole model is threadsafe.
  // The whole model is constructed by all trainable parameters.
  std::mutex model_mutex_;

  // Because ParameterAggregator is not threadsafe, we have to create mutex for each ParameterAggregator so we can
  // acquire lock before calling its method.
  std::map<std::string, std::mutex> parameter_mutex_;

#ifdef ENABLE_ARMOUR
  armour::CipherUnmask cipher_unmask_;
#endif

  // The flag represents the unmasking status.
  std::atomic<bool> unmasked_;
};
}  // namespace server
}  // namespace fl
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FL_SERVER_EXECUTOR_H_
