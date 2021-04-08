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

#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/source/generator_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

GeneratorNode::GeneratorNode(py::function generator_function, const std::vector<std::string> &column_names,
                             const std::vector<DataType> &column_types, int64_t source_len,
                             std::shared_ptr<SamplerObj> sampler)
    : MappableSourceNode(),
      generator_function_(generator_function),
      column_names_(column_names),
      column_types_(column_types),
      reset_ancestor_(nullptr),
      sampler_(std::move(sampler)),
      source_len_(source_len) {}

GeneratorNode::GeneratorNode(py::function generator_function, const std::shared_ptr<SchemaObj> &schema,
                             int64_t source_len, std::shared_ptr<SamplerObj> sampler)
    : MappableSourceNode(),
      generator_function_(generator_function),
      schema_(schema),
      reset_ancestor_(nullptr),
      sampler_(std::move(sampler)),
      source_len_(source_len) {}

std::shared_ptr<DatasetNode> GeneratorNode::Copy() {
  std::shared_ptr<GeneratorNode> node;
  if (schema_ == nullptr) {
    node = std::make_shared<GeneratorNode>(generator_function_, column_names_, column_types_, source_len_, sampler_);
  } else {
    node = std::make_shared<GeneratorNode>(generator_function_, schema_, source_len_, sampler_);
  }
  return node;
}

void GeneratorNode::Print(std::ostream &out) const {
  out << Name() + "(<func>:" + ",columns:" + PrintColumns(column_names_) + ",<col_types>) ";
}

Status GeneratorNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  std::unique_ptr<DataSchema> data_schema = std::make_unique<DataSchema>();

  if (schema_ != nullptr) {
    column_names_.clear();
    column_types_.clear();
    std::string schema_json_string = schema_->to_json();
    RETURN_IF_NOT_OK(data_schema->LoadSchemaString(schema_json_string, {}));

    for (int32_t i = 0; i < data_schema->NumColumns(); i++) {
      ColDescriptor col = data_schema->column(i);
      column_names_.push_back(col.name());
      column_types_.push_back((col.type()));
    }
  }
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  if (sampler_) RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  // GeneratorOp's constructor takes in a prefetch_size, which isn't being set by user nor is it being used by
  // GeneratorOp internally. Here it is given a zero which is the default in generator builder
  std::shared_ptr<GeneratorOp> op = std::make_shared<GeneratorOp>(generator_function_, column_names_, column_types_, 0,
                                                                  rows_per_buffer_, connector_que_size_, sampler_rt);
  // set the number of rows from source length
  op->SetNumRows(source_len_);

  // Add this GeneratorOp to its RepeatOp/EpochCtrlOp ancestor's EOE list.
  // When the ancestor reaches an end-of-epoch boundary, it will send a "reset" signal to all the ops in the EOE list.
  // The ancestor is updated by GeneratorNodePass post pass.
  // Assumption:
  //   We build the run-time ops from IR nodes from top to bottom. Hence Repeat/EpochCtrl ancestor ops are built
  //   before this leaf Generator op is built.
  if (reset_ancestor_ != nullptr) {
    reset_ancestor_->op_->AddToEoeList(op);
  }
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// no validation is needed for generator op.
Status GeneratorNode::ValidateParams() {
  if (source_len_ == 0) {
    std::string err_msg = "GeneratorNode: data row of input source must not be 0, got: " + std::to_string(source_len_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status GeneratorNode::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = 0;
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status GeneratorNode::Accept(IRNodePass *p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<GeneratorNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status GeneratorNode::AcceptAfter(IRNodePass *p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<GeneratorNode>(), modified);
}

Status GeneratorNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                     int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  if (source_len_ == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), dataset_size));
    dataset_size_ = *dataset_size;
    return Status::OK();
  } else {
    int64_t sample_size;
    int64_t num_rows;
    num_rows = source_len_;
    std::shared_ptr<SamplerRT> sampler_rt = nullptr;
    if (sampler_) RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
    sample_size = sampler_ ? sampler_rt->CalculateNumSamples(num_rows) : num_rows;
    *dataset_size = sample_size;
    dataset_size_ = *dataset_size;
    return Status::OK();
  }
}
}  // namespace dataset
}  // namespace mindspore
