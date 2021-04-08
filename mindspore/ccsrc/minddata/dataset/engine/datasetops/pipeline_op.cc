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
#include "minddata/dataset/engine/datasetops/pipeline_op.h"
#include <iostream>

namespace mindspore {
namespace dataset {
// Constructor
PipelineOp::PipelineOp(int32_t op_connector_size, std::shared_ptr<SamplerRT> sampler)
    : DatasetOp(op_connector_size, sampler) {}

// A print method typically used for debugging
void PipelineOp::Print(std::ostream &out, bool show_all) const {
  // Summary 1-liner print
  if (!show_all) {
    // Call super class printer
    DatasetOp::Print(out, show_all);
    out << " [workers: ";
    if (this->inlined()) {
      out << "0 (inlined)]";
    } else {
      out << "1]";  // Pipeline ops only have 1 worker
    }
  } else {
    // Detailed print
    DatasetOp::Print(out, show_all);
    out << "\nNum workers: ";
    if (this->inlined()) {
      out << "0 (inlined)";
    } else {
      out << "1";  // Pipeline ops only have 1 worker
    }
  }
}
}  // namespace dataset
}  // namespace mindspore
