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

#ifndef MINDSPORE_LITE_SRC_CXX_API_MODEL_MODEL_IMPL_H_
#define MINDSPORE_LITE_SRC_CXX_API_MODEL_MODEL_IMPL_H_

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <unordered_map>
#include "include/api/model.h"
#include "include/api/context.h"
#include "include/api/cell.h"
#include "include/lite_session.h"
#include "src/cxx_api/graph/graph_data.h"
#include "src/inner_context.h"
#include "src/lite_session.h"

template <class T>
void clearVectorOfPointers(std::vector<T> *v) {
  if (v != nullptr) {
    for (typename std::vector<T>::iterator it = v->begin(); it != v->end(); ++it) {
      delete (*it);
    }
    v->clear();
  }
}

namespace mindspore {

typedef std::shared_ptr<lite::LiteSession>(CreateTrainSessionProto)(std::shared_ptr<Graph::GraphData> graph_data,
                                                                    std::shared_ptr<TrainCfg> cfg,
                                                                    lite::InnerContext *context);
CreateTrainSessionProto *CreateTrainSessionCallbackHolder(CreateTrainSessionProto *proto = nullptr);

namespace session {
class Metrics;
class TrainLoopCallBack;
}  // namespace session

class ModelImpl {
 public:
  ModelImpl() : graph_(nullptr), session_(nullptr), context_(nullptr) {}
  ~ModelImpl() = default;

  Status Build();
  Status Build(const void *model_data, size_t data_size, ModelType model_type,
               const std::shared_ptr<Context> &model_context);
  Status Build(const std::string &model_path, ModelType model_type, const std::shared_ptr<Context> &model_context);
  Status Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims);

  Status Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs, const MSKernelCallBack &before,
                 const MSKernelCallBack &after);

  lite::LiteSession *CreateLiteSession(lite::InnerContext *context);

  Status LoadConfig(const std::string &config_path);
  std::vector<MSTensor> GetInputs();
  std::vector<MSTensor> GetOutputs();
  std::vector<MSTensor> GetGradients() const;
  Status ApplyGradients(const std::vector<MSTensor> &gradients);
  std::vector<MSTensor> GetOptimizerParams() const;
  Status SetOptimizerParams(const std::vector<MSTensor> &params);
  MSTensor GetInputByTensorName(const std::string &name);
  std::vector<std::string> GetOutputTensorNames();
  MSTensor GetOutputByTensorName(const std::string &name);
  std::vector<MSTensor> GetOutputsByNodeName(const std::string &name);

  static bool CheckModelSupport(const std::string &device_type, ModelType model_type);
  bool IsTrainModel();

  Status InitMetrics(std::vector<Metrics *> metrics) {
    metrics_ = metrics;
    return kSuccess;
  }
  std::vector<Metrics *> GetMetrics() { return metrics_; }
  const session::LiteSession *GetSession() const { return session_.get(); }

 protected:
  // Utility methods
  Status ConvertCallbacks(Model *model, std::vector<TrainCallBack *> *i_cbs,
                          std::vector<session::TrainLoopCallBack *> *o_cbs,
                          std::vector<session::TrainLoopCallBack *> *adapter_cbs);
  Status PrepareMetrics(Model *model, std::vector<session::Metrics *> *o_ms,
                        std::vector<session::Metrics *> *adapter_ms);

 private:
  friend class Model;
  friend class Serialization;
  std::shared_ptr<Graph> graph_ = nullptr;
  std::shared_ptr<lite::LiteSession> session_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;
  std::shared_ptr<TrainCfg> cfg_ = nullptr;
  std::vector<Metrics *> metrics_;
  void SetGraph(const std::shared_ptr<Graph> &graph) { graph_ = graph; }
  void SetContext(const std::shared_ptr<Context> &context) { context_ = context; }
  void SetConfig(const std::shared_ptr<TrainCfg> cfg) { cfg_ = cfg; }
  Status RunGraph(const MSKernelCallBack &before, const MSKernelCallBack &after);
  std::map<std::string, TypeId> execution_plan_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_CXX_API_MODEL_MODEL_IMPL_H_
