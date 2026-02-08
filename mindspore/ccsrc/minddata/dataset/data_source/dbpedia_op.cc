/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/data_source/dbpedia_op.h"

#include "include/utils/common.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
DBpediaOp::DBpediaOp(const std::vector<std::string> &dataset_files_list, char field_delim,
                     const std::vector<std::shared_ptr<BaseRecord>> &column_default,
                     const std::vector<std::string> &column_name, int32_t num_workers, int64_t num_samples,
                     int32_t worker_connector_size, int32_t op_connector_size, bool shuffle_files, int32_t num_devices,
                     int32_t device_id)
    : CsvOp(dataset_files_list, field_delim, column_default, column_name, num_workers, num_samples,
            worker_connector_size, op_connector_size, shuffle_files, num_devices, device_id) {}

void DBpediaOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nSample count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nDBpedia files list:\n";
    for (int i = 0; i < csv_files_list_.size(); ++i) {
      out << " " << csv_files_list_[i];
    }
    out << "\n\n";
  }
}
}  // namespace dataset
}  // namespace mindspore
