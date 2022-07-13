/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_RUNTIME_CONTEXT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_RUNTIME_CONTEXT_H_

#include <memory>
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"

namespace mindspore::dataset {
class TreeConsumer;
/// Class that represents single runtime instance which can consume data from a data pipeline
class RuntimeContext {
 public:
  /// Default constructor
  RuntimeContext() = default;

  /// Initialize the runtime, for now we just call the global init
  /// \return Status error code
  Status Init() const;

  /// Set the tree consumer
  /// \param tree_consumer to be assigned
  void AssignConsumer(std::shared_ptr<TreeConsumer> tree_consumer);

  /// Get the tree consumer
  /// \return Raw pointer to the tree consumer.
  TreeConsumer *GetConsumer();

  /// Method to terminate the runtime, this will not release the resources
  /// \return Status error code
  virtual Status Terminate() = 0;

  virtual ~RuntimeContext() = default;

  std::shared_ptr<TreeConsumer> tree_consumer_;
};

/// Class that represents C++ single runtime instance which can consume data from a data pipeline
class NativeRuntimeContext : public RuntimeContext {
 public:
  /// Method to terminate the runtime, this will not release the resources
  /// \return Status error code
  Status Terminate() override;

  ~NativeRuntimeContext() override;

 private:
  /// Internal function to perform the termination
  /// \return Status error code
  Status TerminateImpl();
};

}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_RUNTIME_CONTEXT_H_
