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

#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
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
  if (total_rows_ < 0) {
    std::string err_msg =
      "RandomNode: total_rows must be greater than or equal 0, now get " + std::to_string(total_rows_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!columns_list_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("RandomNode", "columns_list", columns_list_));
  }

  return Status::OK();
}

int32_t RandomNode::GenRandomInt(int32_t min, int32_t max) {
  std::uniform_int_distribution<int32_t> uniDist(min, max);
  return uniDist(rand_gen_);
}

// Build for RandomNode
std::vector<std::shared_ptr<DatasetOp>> RandomNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  rand_gen_.seed(GetSeed());  // seed the random generator
  // If total rows was not given, then randomly pick a number
  std::shared_ptr<SchemaObj> schema_obj;
  if (!schema_path_.empty()) {
    schema_obj = Schema(schema_path_);
    if (schema_obj == nullptr) {
      return {};
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

  std::unique_ptr<DataSchema> data_schema;
  std::vector<std::string> columns_to_load;
  if (columns_list_.size() > 0) {
    columns_to_load = columns_list_;
  }
  if (!schema_file_path.empty() || !schema_json_string.empty()) {
    data_schema = std::make_unique<DataSchema>();
    if (!schema_file_path.empty()) {
      data_schema->LoadSchemaFile(schema_file_path, columns_to_load);
    } else if (!schema_json_string.empty()) {
      data_schema->LoadSchemaString(schema_json_string, columns_to_load);
    }
  }

  // RandomOp by itself is a non-mappable dataset that does not support sampling.
  // However, if a cache operator is injected at some other place higher in the tree, that cache can
  // inherit this sampler from the leaf, providing sampling support from the caching layer.
  // That is why we save the sampler here in a leaf node that does not use sampling.
  // RandomOp doesn't support sampler, should not support sharding, select sampler should just be sequential.
  std::shared_ptr<SamplerObj> sampler_ = SelectSampler(total_rows_, false, 1, 0);

  std::shared_ptr<RandomDataOp> op;
  op = std::make_shared<RandomDataOp>(num_workers_, connector_que_size_, rows_per_buffer_, total_rows_,
                                      std::move(data_schema), std::move(sampler_->Build()));
  build_status = AddCacheOp(&node_ops);  // remove me after changing return val of Build()
  RETURN_EMPTY_IF_ERROR(build_status);

  node_ops.push_back(op);

  return node_ops;
}

// Get the shard id of node
Status RandomNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
