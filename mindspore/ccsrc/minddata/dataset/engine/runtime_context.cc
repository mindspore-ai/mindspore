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

#include "minddata/dataset/engine/runtime_context.h"
namespace mindspore::dataset {
void RuntimeContext::AssignConsumer(std::shared_ptr<TreeConsumer> tree_consumer) {
  tree_consumer_ = std::move(tree_consumer);
}
Status NativeRuntimeContext::Terminate() {
  MS_LOG(INFO) << "Terminating a Dataset NativeRuntime.";
  if (tree_consumer_ != nullptr) {
    return TerminateImpl();
  }
  MS_LOG(INFO) << "Dataset TreeConsumer was not initialized.";
  return Status::OK();
}

Status NativeRuntimeContext::TerminateImpl() {
  CHECK_FAIL_RETURN_UNEXPECTED(tree_consumer_ != nullptr, "Dataset TreeConsumer is not initialized.");
  return tree_consumer_->Terminate();
}

NativeRuntimeContext::~NativeRuntimeContext() {
  Status rc = NativeRuntimeContext::Terminate();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Error while terminating the consumer. Message:" << rc;
  }
}

TreeConsumer *RuntimeContext::GetConsumer() { return tree_consumer_.get(); }

Status RuntimeContext::Init() const { return GlobalInit(); }
}  // namespace mindspore::dataset
