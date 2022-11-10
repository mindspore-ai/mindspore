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

#include "minddata/dataset/engine/python_runtime_context.h"
#include "pybind11/pybind11.h"

namespace mindspore::dataset {
Status PythonRuntimeContext::Terminate() {
  MS_LOG(INFO) << "Terminating a Dataset PythonRuntime.";
  if (tree_consumer_ != nullptr) {
    return TerminateImpl();
  }
  MS_LOG(INFO) << "Dataset TreeConsumer was not initialized.";
  return Status::OK();
}

Status PythonRuntimeContext::TerminateImpl() {
  CHECK_FAIL_RETURN_UNEXPECTED(tree_consumer_ != nullptr, "Dataset TreeConsumer is not initialized.");
  // Release GIL before joining all threads
  py::gil_scoped_release gil_release;
  return tree_consumer_->Terminate();
}

PythonRuntimeContext::~PythonRuntimeContext() {
  Status rc = PythonRuntimeContext::Terminate();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error while terminating the consumer. Message:" << rc;
  }
  if (tree_consumer_) {
    tree_consumer_.reset();
  }
}

TreeConsumer *PythonRuntimeContext::GetPythonConsumer() {
  if (GlobalContext::config_manager()->get_debug_mode()) {
    return dynamic_cast<PythonPullBasedIteratorConsumer *>(tree_consumer_.get());
  } else {
    return dynamic_cast<PythonIteratorConsumer *>(tree_consumer_.get());
  }
}
}  // namespace mindspore::dataset
