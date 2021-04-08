/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/consumers/pull_based_tree_consumer.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/runtime_context.h"
#include "minddata/dataset/include/datasets.h"

namespace mindspore {
namespace dataset {

Iterator::Iterator() : consumer_(nullptr) {}
Iterator::~Iterator() { Stop(); }

// Get the next row from the data pipeline.
Status Iterator::GetNextRowCharIF(MSTensorMapChar *row) {
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
    std::vector<char> col_name(de_tensor.first.begin(), de_tensor.first.end());
    row->insert(std::make_pair(col_name, mindspore::MSTensor(std::make_shared<DETensor>(de_tensor.second))));
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
  std::transform(md_row.begin(), md_row.end(), std::back_inserter(*row),
                 [](auto t) { return mindspore::MSTensor(std::make_shared<DETensor>(t)); });
  return Status::OK();
}

// Shut down the data pipeline.
void Iterator::Stop() {
  if (runtime_context_ != nullptr) {
    Status rc = runtime_context_->Terminate();
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc.ToString();
    }
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

PullIterator::PullIterator() : pull_consumer_(nullptr) {}
// Get the next row from the data pipeline.
Status PullIterator::GetRows(int32_t num_rows, std::vector<MSTensorVec> *const row) {
  for (int i = 0; i < num_rows; i++) {
    std::vector<std::shared_ptr<dataset::Tensor>> md_row;
    Status rc = pull_consumer_->GetNextAsVector(&md_row);

    if (rc.IsError()) {
      row->clear();
      MS_LOG(ERROR) << "GetNextRow: Failed to get next row. Error status: " << rc;
      return rc;
    }

    MSTensorVec ms_row = {};
    for (auto de_tensor : md_row) {
      CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(), "Apply transform failed, output tensor has no data");
      ms_row.push_back(mindspore::MSTensor(std::make_shared<DETensor>(de_tensor)));
    }
    row->push_back(ms_row);
  }
  return Status::OK();
}

Status PullIterator::GetNextRow(MSTensorVec *const row) {
  CHECK_FAIL_RETURN_UNEXPECTED(pull_consumer_ != nullptr, "Consumer is nullptr.");
  std::vector<std::shared_ptr<dataset::Tensor>> md_row;
  Status rc = pull_consumer_->GetNextAsVector(&md_row);
  if (rc.IsError()) {
    row->clear();
    MS_LOG(ERROR) << "GetNextRow: Failed to get next row. Error status: " << rc;
    return rc;
  }

  for (auto de_tensor : md_row) {
    CHECK_FAIL_RETURN_UNEXPECTED(de_tensor->HasData(), "Apply transform failed, output tensor has no data");
    row->push_back(mindspore::MSTensor(std::make_shared<DETensor>(de_tensor)));
  }
  return Status::OK();
}

// Function to build and launch the execution tree. This function kicks off a different type of consumer
// for the tree, the reason why this is the case is due to the fact that PullBasedIterator does not need
// to instantiate threads for each op. As such, the call to the consumer will by pass the execution tree.
Status PullIterator::BuildAndLaunchTree(std::shared_ptr<Dataset> ds) {
  if (pull_consumer_ == nullptr) pull_consumer_ = std::make_unique<PullBasedIteratorConsumer>();
  RETURN_IF_NOT_OK(pull_consumer_->Init(std::move(ds->IRNode())));
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
