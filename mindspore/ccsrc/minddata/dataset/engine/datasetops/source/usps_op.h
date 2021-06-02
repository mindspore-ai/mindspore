/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_USPS_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_USPS_OP_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/nonmappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"
#include "minddata/dataset/engine/jagged_connector.h"

namespace mindspore {
namespace dataset {
class USPSOp : public NonMappableLeafOp {
 public:
  // Constructor.
  // @param const std::string &dataset_dir - dir directory of USPS data file.
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test' or 'all'.
  // @param std::unique_ptr<DataSchema> data_schema - the schema of the USPS dataset.
  // @param num_workers - number of worker threads reading data from tf_file files.
  // @param worker_connector_size - size of each internal queue.
  // @param num_samples - number of samples to read.
  // @param op_connector_size - size of each queue in the connector that the child operator pulls from.
  // @param shuffle_files - whether to shuffle the files before reading data.
  // @param num_devices - number of devices.
  // @param device_id - device id.
  USPSOp(const std::string &dataset_dir, const std::string &usage, std::unique_ptr<DataSchema> data_schema,
         int32_t num_workers, int32_t worker_connector_size, int64_t num_samples, int32_t op_connector_size,
         bool shuffle_files, int32_t num_devices, int32_t device_id);

  // Destructor.
  ~USPSOp() = default;

  // Op name getter.
  // @return std::string - Name of the current Op.
  std::string Name() const override { return "USPSOp"; }

  // A print method typically used for debugging.
  // @param std::ostream &out - out stream.
  // @param bool show_all - whether to show all information.
  void Print(std::ostream &out, bool show_all) const override;

  // Instantiates the internal queues and connectors
  // @return Status - the error code returned.
  Status Init() override;

  // Function to count the number of samples in the USPS dataset.
  // @param const std::string &dir - path to the USPS directory.
  // @param const std::string &usage - Usage of this dataset, can be 'train', 'test' or 'all'.
  // @param int64_t *count - output arg that will hold the minimum of the actual dataset size and numSamples.
  // @return Status - the error coed returned.
  static Status CountTotalRows(const std::string &dir, const std::string &usage, int64_t *count);

  // File names getter.
  // @return Vector of the input file names.
  std::vector<std::string> FileNames() { return data_files_list_; }

 private:
  // Function to count the number of samples in one data file.
  // @param const std::string &data_file - path to the data file.
  // @return int64_t - the count result.
  int64_t CountRows(const std::string &data_file);

  // Reads a data file and loads the data into multiple TensorRows.
  // @param data_file - the data file to read.
  // @param start_offset - the start offset of file.
  // @param end_offset - the end offset of file.
  // @param worker_id - the id of the worker that is executing this function.
  // @return Status - the error code returned.
  Status LoadFile(const std::string &data_file, int64_t start_offset, int64_t end_offset, int32_t worker_id) override;

  // Parses a single row and puts the data into a tensor table.
  // @param line - the content of the row.
  // @param trow - image & label read into this tensor row.
  // @return Status - the error code returned.
  Status LoadTensor(std::string *line, TensorRow *trow);

  // Calculate number of rows in each shard.
  // @return Status - the error code returned.
  Status CalculateNumRowsPerShard() override;

  // Fill the IOBlockQueue.
  // @param i_keys - keys of file to fill to the IOBlockQueue.
  // @return Status - the error code returned.
  Status FillIOBlockQueue(const std::vector<int64_t> &i_keys) override;

  // Get all files in the dataset_dir_.
  // @return Status - The status code returned.
  Status GetFiles();

  // Parse a line to image and label.
  // @param line - the content of the row.
  // @param images_buffer - image destination.
  // @param labels_buffer - label destination.
  // @return Status - the status code returned.
  Status ParseLine(std::string *line, const std::unique_ptr<unsigned char[]> &images_buffer,
                   const std::unique_ptr<uint32_t[]> &labels_buffer);

  // Private function for computing the assignment of the column name map.
  // @return Status - the error code returned.
  Status ComputeColMap() override;

  const std::string usage_;  // can be "all", "train" or "test".
  std::string dataset_dir_;  // directory of data files.
  std::unique_ptr<DataSchema> data_schema_;

  std::vector<std::string> data_files_list_;
};
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_USPS_OP_H_
