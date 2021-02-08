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
#include "minddata/dataset/engine/runtime_context.h"
#include "minddata/dataset/include/datasets.h"

namespace mindspore {
namespace dataset {

Iterator::Iterator() : consumer_(nullptr) {}
Iterator::~Iterator() { Stop(); }

// Get the next row from the data pipeline.
Status Iterator::GetNextRow(MSTensorMap *row) {
  // Clean data buffer
  row->clear();
  std::unordered_map<std::string, std::shared_ptr<dataset::Tensor>> md_map;
  Status rc = consumer_->GetNextAsMap(&md_map);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetNextRow: Failed to get next row. Error status: " << rc;
    row->clear();
    return rc;
  }
  for (auto de_tensor : md_map) {
    CHECK_FAIL_RETURN_UNEXPECTED(de_tensor.second->HasData(), "Apply transform failed, output tensor has no data");
    row->insert(std::make_pair(de_tensor.first, mindspore::MSTensor(std::make_shared<DETensor>(de_tensor.second))));
  }

  return Status::OK();
}

// Get the next row from the data pipeline.
Status Iterator::GetNextRow(MSTensorVec *row) {
  // Clean data buffer
  row->clear();
  // create a dataset tensor row and fetch. Then we convert the output to MSTensor
  std::vector<std::shared_ptr<dataset::Tensor>> md_row;
  Status rc = consumer_->GetNextAsVector(&md_row);
  if (rc.IsError()) {
    row->clear();
    return rc;
  }
  for (auto de_tensor : md_row) {
    CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(), "Apply transform failed, output tensor has no data");
    row->push_back(mindspore::MSTensor(std::make_shared<DETensor>(de_tensor)));
  }
  return Status::OK();
}

// Shut down the data pipeline.
void Iterator::Stop() {
  Status rc = runtime_context_->Terminate();
  if (rc.IsError()) {
    MS_LOG(ERROR) << rc.ToString();
  }
}

// Function to build and launch the execution tree.
Status Iterator::BuildAndLaunchTree(std::shared_ptr<Dataset> ds, int32_t num_epochs) {
  runtime_context_ = std::make_unique<NativeRuntimeContext>();
  RETURN_IF_NOT_OK(runtime_context_->Init());
  auto consumer = std::make_unique<IteratorConsumer>(num_epochs);
  consumer_ = consumer.get();
  RETURN_IF_NOT_OK(consumer->Init(ds->IRNode()));
  runtime_context_->AssignConsumer(std::move(consumer));
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
