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

#include "minddata/dataset/engine/ir/datasetops/source/generator_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/generator_op.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {

GeneratorNode::GeneratorNode(py::function generator_function, const std::vector<std::string> &column_names,
                             const std::vector<DataType> &column_types)
    : MappableSourceNode(),
      generator_function_(generator_function),
      column_names_(column_names),
      column_types_(column_types) {}

GeneratorNode::GeneratorNode(py::function generator_function, const std::shared_ptr<SchemaObj> &schema)
    : generator_function_(generator_function), schema_(schema) {}

std::shared_ptr<DatasetNode> GeneratorNode::Copy() {
  std::shared_ptr<GeneratorNode> node;
  if (schema_ == nullptr) {
    node = std::make_shared<GeneratorNode>(generator_function_, column_names_, column_types_);
  } else {
    node = std::make_shared<GeneratorNode>(generator_function_, schema_);
  }
  return node;
}

void GeneratorNode::Print(std::ostream &out) const {
  out << Name() + "(<func>:" + ",columns:" + PrintColumns(column_names_) + ",<col_types>)";
}

Status GeneratorNode::Build(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
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

  // GeneratorOp's constructor takes in a prefetch_size, which isn't being set by user nor is it being used by
  // GeneratorOp internally. Here it is given a zero which is the default in generator builder
  std::shared_ptr<GeneratorOp> op = std::make_shared<GeneratorOp>(generator_function_, column_names_, column_types_, 0,
                                                                  rows_per_buffer_, connector_que_size_);

  // Init() is called in builder when generator is built. Here, since we are getting away from the builder class, init
  // needs to be called when the op is built. The caveat is that Init needs to be made public (before it is private).
  // This method can be privatized once we move Init() to Generator's functor. However, that is a bigger change which
  // best be delivered when the test cases for this api is ready.
  RETURN_IF_NOT_OK(op->Init());

  node_ops->push_back(op);

  return Status::OK();
}

// no validation is needed for generator op.
Status GeneratorNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  return Status::OK();
}

Status GeneratorNode::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = 0;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
