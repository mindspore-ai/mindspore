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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_

#include <sys/stat.h>
#include <unistd.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "include/iterator.h"
#include "include/samplers.h"

namespace mindspore {
namespace dataset {

class Tensor;
class TensorShape;
class TreeGetters;

class DatasetCache;
class DatasetNode;
class Iterator;
class TensorOperation;
class SchemaObj;
class SamplerObj;

// Dataset classes (in alphabetical order)
class BatchDataset;
class MapDataset;
class ProjectDataset;
class ShuffleDataset;
class DSCallback;

/// \class Dataset datasets.h
/// \brief A base class to represent a dataset in the data pipeline.
class Dataset : public std::enable_shared_from_this<Dataset> {
 public:
  // need friend class so they can access the children_ field
  friend class Iterator;
  friend class TransferNode;

  /// \brief Constructor
  Dataset();

  /// \brief Destructor
  ~Dataset() = default;

  /// \brief Gets the dataset size
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \return dataset size. If failed, return -1
  int64_t GetDatasetSize(bool estimate = false);

  //  /// \brief Gets the output type
  //  /// \return a vector of DataType. If failed, return an empty vector
  //  std::vector<DataType> GetOutputTypes();

  /// \brief Gets the output shape
  /// \return a vector of TensorShape. If failed, return an empty vector
  std::vector<TensorShape> GetOutputShapes();

  /// \brief Gets the batch size
  /// \return int64_t
  int64_t GetBatchSize();

  /// \brief Gets the repeat count
  /// \return int64_t
  int64_t GetRepeatCount();

  /// \brief Gets the number of classes
  /// \return number of classes. If failed, return -1
  int64_t GetNumClasses();

  /// \brief Gets the column names
  /// \return Names of the columns. If failed, return an empty vector
  std::vector<std::string> GetColumnNames();

  /// \brief Gets the class indexing
  /// \return a map of ClassIndexing. If failed, return an empty map
  std::vector<std::pair<std::string, std::vector<int32_t>>> GetClassIndexing();

  /// \brief Setter function for runtime number of workers
  /// \param[in] num_workers The number of threads in this operator
  /// \return Shared pointer to the original object
  std::shared_ptr<Dataset> SetNumWorkers(int32_t num_workers);

  /// \brief Function to create an Iterator over the Dataset pipeline
  /// \param[in] columns List of columns to be used to specify the order of columns
  /// \param[in] num_epochs Number of epochs to run through the pipeline, default -1 which means infinite epochs.
  ///     An empty row is returned at the end of each epoch
  /// \return Shared pointer to the Iterator
  std::shared_ptr<Iterator> CreateIterator(std::vector<std::string> columns = {}, int32_t num_epochs = -1);

  /// \brief Function to create a BatchDataset
  /// \notes Combines batch_size number of consecutive rows into batches
  /// \param[in] batch_size The number of rows each batch is created with
  /// \param[in] drop_remainder Determines whether or not to drop the last possibly incomplete
  ///     batch. If true, and if there are less than batch_size rows
  ///     available to make the last batch, then those rows will
  ///     be dropped and not propagated to the next node
  /// \return Shared pointer to the current BatchDataset
  std::shared_ptr<BatchDataset> Batch(int32_t batch_size, bool drop_remainder = false);

  /// \brief Function to create a MapDataset
  /// \notes Applies each operation in operations to this dataset
  /// \param[in] operations Vector of operations to be applied on the dataset. Operations are
  ///     applied in the order they appear in this list
  /// \param[in] input_columns Vector of the names of the columns that will be passed to the first
  ///     operation as input. The size of this list must match the number of
  ///     input columns expected by the first operator. The default input_columns
  ///     is the first column
  /// \param[in] output_columns Vector of names assigned to the columns outputted by the last operation
  ///     This parameter is mandatory if len(input_columns) != len(output_columns)
  ///     The size of this list must match the number of output columns of the
  ///     last operation. The default output_columns will have the same
  ///     name as the input columns, i.e., the columns will be replaced
  /// \param[in] project_columns A list of column names to project
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  /// \return Shared pointer to the current MapDataset
  std::shared_ptr<MapDataset> Map(std::vector<std::shared_ptr<TensorOperation>> operations,
                                  const std::vector<std::string> &input_columns = {},
                                  const std::vector<std::string> &output_columns = {},
                                  const std::vector<std::string> &project_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  std::vector<std::shared_ptr<DSCallback>> callbacks = {}) {
    return std::make_shared<MapDataset>(shared_from_this(), operations, input_columns, output_columns, project_columns,
                                        cache, callbacks);
  }

  /// \brief Function to create a Project Dataset
  /// \notes Applies project to the dataset
  /// \param[in] columns The name of columns to project
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ProjectDataset> Project(const std::vector<std::string> &columns) {
    return std::make_shared<ProjectDataset>(shared_from_this(), columns);
  }

  /// \brief Function to create a Shuffle Dataset
  /// \notes Randomly shuffles the rows of this dataset
  /// \param[in] buffer_size The size of the buffer (must be larger than 1) for shuffling
  /// \return Shared pointer to the current ShuffleDataset
  std::shared_ptr<ShuffleDataset> Shuffle(int32_t buffer_size) {
    return std::make_shared<ShuffleDataset>(shared_from_this(), buffer_size);
  }

  std::shared_ptr<DatasetNode> IRNode() { return ir_node_; }

 protected:
  std::shared_ptr<TreeGetters> tree_getters_;
  std::shared_ptr<DatasetNode> ir_node_;
};

class BatchDataset : public Dataset {
 public:
  BatchDataset(std::shared_ptr<Dataset> input, int32_t batch_size, bool drop_remainder = false);
  ~BatchDataset() = default;
};

class MapDataset : public Dataset {
 public:
  MapDataset(std::shared_ptr<Dataset> input, std::vector<std::shared_ptr<TensorOperation>> operations,
             const std::vector<std::string> &input_columns, const std::vector<std::string> &output_columns,
             const std::vector<std::string> &project_columns, const std::shared_ptr<DatasetCache> &cache,
             std::vector<std::shared_ptr<DSCallback>> callbacks);
  ~MapDataset() = default;
};

class ProjectDataset : public Dataset {
 public:
  ProjectDataset(std::shared_ptr<Dataset> input, const std::vector<std::string> &columns);
  ~ProjectDataset() = default;
};

class ShuffleDataset : public Dataset {
 public:
  ShuffleDataset(std::shared_ptr<Dataset> input, int32_t buffer_size);
  ~ShuffleDataset() = default;
};

/// \brief Function to create a SchemaObj
/// \param[in] schema_file Path of schema file
/// \return Shared pointer to the current schema
std::shared_ptr<SchemaObj> Schema(const std::string &schema_file = "");

class AlbumDataset : public Dataset {
 public:
  AlbumDataset(const std::string &dataset_dir, const std::string &data_schema,
               const std::vector<std::string> &column_names = {}, bool decode = false,
               const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
               const std::shared_ptr<DatasetCache> &cache = nullptr);
  ~AlbumDataset() = default;
};

/// \brief Function to create an AlbumDataset
/// \notes The generated dataset is specified through setting a schema
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] data_schema Path to dataset schema file
/// \param[in] column_names Column names used to specify columns to load, if empty, will read all columns.
///     (default = {})
/// \param[in] decode the option to decode the images in dataset (default = false)
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
std::shared_ptr<AlbumDataset> Album(const std::string &dataset_dir, const std::string &data_schema,
                                    const std::vector<std::string> &column_names = {}, bool decode = false,
                                    const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                    const std::shared_ptr<DatasetCache> &cache = nullptr);

class MnistDataset : public Dataset {
 public:
  explicit MnistDataset(const std::string &dataset_dir, const std::string &usage = "all",
                        const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                        const std::shared_ptr<DatasetCache> &cache = nullptr);
  ~MnistDataset() = default;
};

/// \brief Function to create a MnistDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of MNIST, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Object used to choose samples from the dataset. If sampler is not given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current MnistDataset
std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir, const std::string &usage = "all",
                                    const std::shared_ptr<SamplerObj> &sampler = RandomSampler(),
                                    const std::shared_ptr<DatasetCache> &cache = nullptr);

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
