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
#include "minddata/dataset/include/iterator.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/include/datasets.h"

namespace mindspore {
namespace dataset {
namespace api {

// Get the next row from the data pipeline.
bool Iterator::GetNextRow(TensorMap *row) {
  Status rc = consumer_->GetNextAsMap(row);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetNextRow: Failed to get next row. Error status: " << rc;
    row->clear();
    return false;
  }
  return true;
}

// Get the next row from the data pipeline.
bool Iterator::GetNextRow(TensorVec *row) {
  Status rc = consumer_->GetNextAsVector(row);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetNextRow: Failed to get next row. Error status: " << rc;
    row->clear();
    return false;
  }
  return true;
}

// Shut down the data pipeline.
void Iterator::Stop() { runtime_context->Terminate(); }

// Function to build and launch the execution tree.
Status Iterator::BuildAndLaunchTree(std::shared_ptr<Dataset> ds) {
  runtime_context = std::make_unique<RuntimeContext>();
  RETURN_IF_NOT_OK(runtime_context->Init());
  auto consumer = std::make_unique<IteratorConsumer>();
  consumer_ = consumer.get();
  RETURN_IF_NOT_OK(consumer->Init(ds));
  runtime_context->AssignConsumer(std::move(consumer));
  return Status::OK();
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
