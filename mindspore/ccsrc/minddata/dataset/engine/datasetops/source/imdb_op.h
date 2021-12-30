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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IMDB_OP_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IMDB_OP_H_

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/data_schema.h"
#include "minddata/dataset/engine/datasetops/parallel_op.h"
#include "minddata/dataset/engine/datasetops/source/mappable_leaf_op.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/queue.h"
#include "minddata/dataset/util/services.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/wait_post.h"

namespace mindspore {
namespace dataset {
/// Forward declares
template <typename T>
class Queue;

class IMDBOp : public MappableLeafOp {
 public:
  /// \brief Constructor.
  /// \param[in] int32_t num_workers - num of workers reading texts in parallel.
  /// \param[in] std::string dataset_dir - dir directory of IMDB dataset.
  /// \param[in] int32_t queue_size - connector queue size.
  /// \param[in] std::string usage - the type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[in] DataSchema data_schema - the schema of each column in output data.
  /// \param[in] std::unique_ptr<Sampler> sampler - sampler tells Folder what to read.
  IMDBOp(int32_t num_workers, const std::string &dataset_dir, int32_t queue_size, const std::string &usage,
         std::unique_ptr<DataSchema> data_schema, std::shared_ptr<SamplerRT> sampler);

  /// \brief Destructor.
  ~IMDBOp() = default;

  /// \brief Parse IMDB data.
  /// \return Status - The status code returned.
  Status PrepareData() override;

  /// \brief Method derived from RandomAccess Op, enable Sampler to get all ids for each class
  /// \param[in] map cls_ids - key label, val all ids for this class
  /// \return Status - The status code returned.
  Status GetClassIds(std::map<int32_t, std::vector<int64_t>> *cls_ids) const override;

  /// \brief A print method typically used for debugging.
  /// \param[out] out The output stream to write output to.
  /// \param[in] show_all A bool to control if you want to show all info or just a summary.
  void Print(std::ostream &out, bool show_all) const override;

  /// \brief This function return the num_rows.
  /// \param[in] std::string path - dir directory of IMDB dataset.
  /// \param[in] std::string usage - the type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[out] int64_t *num_rows - output arg that will hold the actual dataset size.
  /// \return Status - The status code returned.
  static Status CountRows(const std::string &path, const std::string &usage, int64_t *num_rows);

  /// \brief Op name getter.
  /// \return Name of the current Op.
  std::string Name() const override { return "IMDBOp"; }

  /// \brief Dataset name getter.
  /// \param[in] upper Whether to get upper name.
  /// \return Dataset name of the current Op.
  virtual std::string DatasetName(bool upper = false) const { return upper ? "IMDB" : "imdb"; }

  /// \brief Base-class override for GetNumClasses
  /// \param[out] int64_t *num_classes - the number of classes
  /// \return Status - The status code returned.
  Status GetNumClasses(int64_t *num_classes) override;

 private:
  /// \brief Load a tensor row according to a pair.
  /// \param[in] uint64_t row_id - row_id need to load.
  /// \param[out] TensorRow *row - text & task read into this tensor row.
  /// \return Status - The status code returned.
  Status LoadTensorRow(row_id_type row_id, TensorRow *row) override;

  /// \brief Parses a single row and puts the data into a tensor table.
  /// \param[in] string line - the content of the row.
  /// \param[out] Tensor *out_row - the id of the row filled in the tensor table.
  /// \return Status - The status code returned.
  Status LoadTensor(const std::string &line, std::shared_ptr<Tensor> *out_row);

  /// \brief Reads a text file and loads the data into Tensor.
  /// \param[in] string file - the file to read.
  /// \param[out] Tensor *out_row - the id of the row filled in the tensor table.
  /// \return Status - The status code returned.
  Status LoadFile(const std::string &file, std::shared_ptr<Tensor> *out_row);

  /// \brief Called first when function is called
  /// \param[in] string folder - the folder include files.
  /// \param[in] string label - the name of label.
  /// \return Status - The status code returned.
  Status GetDataByUsage(const std::string &folder, const std::string &label);

  /// \brief function for computing the assignment of the column name map.
  /// \return Status - The status code returned.
  Status ComputeColMap() override;

  std::string folder_path_;  // directory of text folder
  std::string usage_;
  int64_t sampler_ind_;
  std::unique_ptr<DataSchema> data_schema_;
  std::vector<std::pair<std::string, int32_t>> text_label_pairs_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_DATASETOPS_SOURCE_IMDB_OP_H_
