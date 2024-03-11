/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/opt/pre/insert_map_pass.h"

#include <string>
#include <vector>

#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#endif
#include "minddata/dataset/kernels/ir/data/transforms_ir.h"

namespace mindspore::dataset {
#ifndef ENABLE_ANDROID
Status InsertMapPass::Visit(std::shared_ptr<TFRecordNode> node, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(modified);

#if !defined(_WIN32) && !defined(_WIN64)
  // construct schema from the inputs of TFRecordNode
  auto data_schema = DataSchema();
  RETURN_IF_NOT_OK(node->CreateDataSchema(&data_schema));

  // get the output column list
  std::vector<std::string> output_columns;
  RETURN_IF_NOT_OK(data_schema.GetColumnName(&output_columns));
  if (output_columns.empty()) {
    if (!node->ColumnsList().empty()) {
      output_columns = node->ColumnsList();
    } else {
      // Unable to fetch output columns, degraded to do parsing directly in TFRecordOp
      MS_LOG(WARNING)
        << "If both schema and column list are not set, the performance of TFRecordDataset may be degraded.";
      *modified = false;
      return Status::OK();
    }
  }

  // not to parse the protobuf in TFRecordOp
  node->SetDecode(false);

  // if the next node is batch, do parallel parsing in ParseExampleOp
  bool parallel_parse = node->Parent()->Name() == kBatchNode;
  const auto parse_example =
    std::make_shared<transforms::ParseExampleOperation>(data_schema, node->ColumnsList(), parallel_parse);
  auto map_node = std::make_shared<MapNode>(std::vector<std::shared_ptr<TensorOperation>>{parse_example},
                                            std::vector<std::string>{"proto"}, output_columns);
  if (parallel_parse) {
    // parallel parsing use a thread pool inside ParseExampleOp, so we only need 1 worker for map
    (void)map_node->SetNumWorkers(1);
  }

  if (node->Parent()->Name() == kBatchNode) {
    MS_LOG(INFO) << "Insert a Map node after Batch to parse protobuf in parallel.";
    RETURN_IF_NOT_OK(node->Parent()->InsertAbove(map_node));
  } else {
    MS_LOG(INFO) << "Insert a Map node after TFRecord to parse protobuf one by one.";
    RETURN_IF_NOT_OK(node->InsertAbove(map_node));
  }
  *modified = true;
#endif
  return Status ::OK();
}
#endif
}  // namespace mindspore::dataset
