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

#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

std::shared_ptr<DatasetNode> RandomNode::Copy() {
  std::shared_ptr<RandomNode> node;
  if (schema_ != nullptr) {
    node = std::make_shared<RandomNode>(total_rows_, schema_, columns_list_, cache_);
  } else {
    node = std::make_shared<RandomNode>(total_rows_, schema_path_, columns_list_, cache_);
  }
  return node;
}

void RandomNode::Print(std::ostream &out) const { out << Name() + "(num_row:" + std::to_string(total_rows_) + ",...)"; }

// ValidateParams for RandomNode
Status RandomNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (total_rows_ < 0) {
    std::string err_msg =
      "RandomNode: total_rows must be greater than or equal 0, now get " + std::to_string(total_rows_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!columns_list_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("RandomNode", "columns_list", columns_list_));
  }

  // allow total_rows == 0 for now because RandomOp would generate a random row when it gets a 0
  CHECK_FAIL_RETURN_UNEXPECTED(total_rows_ == 0 || total_rows_ >= num_workers_,
                               "RandomNode needs total_rows >= num_workers, total_rows=" + std::to_string(total_rows_) +
                                 ", num_workers=" + std::to_string(num_workers_) + ".");

  return Status::OK();
}

int32_t RandomNode::GenRandomInt(int32_t min, int32_t max) {
  std::uniform_int_distribution<int32_t> uniDist(min, max);
  return uniDist(rand_gen_);
}

// Build for RandomNode
Status RandomNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  rand_gen_.seed(GetSeed());  // seed the random generator
  // If total rows was not given, then randomly pick a number
  std::shared_ptr<SchemaObj> schema_obj;
  if (!schema_path_.empty()) {
    schema_obj = Schema(schema_path_);
    if (schema_obj == nullptr) {
      std::string err_msg = "RandomNode::Build : Invalid schema path";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
  }

  std::string schema_json_string, schema_file_path;
  if (schema_ != nullptr) {
    schema_->set_dataset_type("Random");
    if (total_rows_ != 0) {
      schema_->set_num_rows(total_rows_);
    }
    schema_json_string = schema_->to_json();
  } else {
    schema_file_path = schema_path_;
  }

  std::vector<std::string> columns_to_load;
  if (columns_list_.size() > 0) {
    columns_to_load = columns_list_;
  }
  if (!schema_file_path.empty() || !schema_json_string.empty()) {
    data_schema_ = std::make_unique<DataSchema>();
    if (!schema_file_path.empty()) {
      data_schema_->LoadSchemaFile(schema_file_path, columns_to_load);
    } else if (!schema_json_string.empty()) {
      data_schema_->LoadSchemaString(schema_json_string, columns_to_load);
    }
  }

  std::shared_ptr<RandomDataOp> op;
  op = std::make_shared<RandomDataOp>(num_workers_, connector_que_size_, rows_per_buffer_, total_rows_,
                                      std::move(data_schema_));
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Get the shard id of node
Status RandomNode::GetShardId(int32_t *shard_id) {
  // RandomDataset doesn't support multiple shards
  *shard_id = 0;
  return Status::OK();
}

// Get Dataset size
Status RandomNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows;
  num_rows = total_rows_ != 0 ? total_rows_ : data_schema_->num_rows();
  *dataset_size = num_rows;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

// RandomDataset by itself is a non-mappable dataset that does not support sampling.
// However, if a cache operator is injected at some other place higher in the tree, that cache can
// inherit this sampler from the leaf, providing sampling support from the caching layer.
// That is why we setup the sampler for a leaf node that does not use sampling.
Status RandomNode::SetupSamplerForCache(std::shared_ptr<SamplerObj> *sampler) {
  // RandomOp doesn't support sampler, should not support sharding, select sampler should just be sequential.
  *sampler = SelectSampler(total_rows_, false, 1, 0);
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status RandomNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<RandomNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status RandomNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<RandomNode>(), modified);
}
}  // namespace dataset
}  // namespace mindspore
