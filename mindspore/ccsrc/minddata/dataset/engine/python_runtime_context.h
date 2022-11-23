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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PYTHON_RUNTIME_CONTEXT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PYTHON_RUNTIME_CONTEXT_H_

#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/consumers/python_tree_consumer.h"
#include "minddata/dataset/engine/runtime_context.h"

namespace mindspore::dataset {
class NativeRuntimeContext;

/// Class that represents Python single runtime instance which can consume data from a data pipeline
class PythonRuntimeContext : public RuntimeContext {
 public:
  /// Method to terminate the runtime, this will not release the resources
  /// \return Status error code
  Status Terminate() override;

  /// Safe destructing the tree that includes python objects
  ~PythonRuntimeContext() override;

  TreeConsumer *GetPythonConsumer();

 private:
  /// Internal function to perform the termination
  /// \return Status error code
  Status TerminateImpl();
};

}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_PYTHON_RUNTIME_CONTEXT_H_
