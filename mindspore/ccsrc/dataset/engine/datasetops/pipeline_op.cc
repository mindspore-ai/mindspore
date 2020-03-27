/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include <iostream>
#include "dataset/engine/datasetops/pipeline_op.h"

namespace mindspore {
namespace dataset {
// Constructor
PipelineOp::PipelineOp(int32_t op_connector_size) : DatasetOp(op_connector_size) {}

// A print method typically used for debugging
void PipelineOp::Print(std::ostream &out, bool show_all) const {
  // Call base class printer first
  DatasetOp::Print(out, show_all);

  // Then display our own stuff for the pipeline op
  // out << "This is a pipeline op print.  nothing to display here at the moment.\n";
}
}  // namespace dataset
}  // namespace mindspore
