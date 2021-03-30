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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/types.h"
#include "include/iterator.h"
#include "include/samplers.h"
#include "include/transforms.h"

namespace mindspore {
namespace dataset {

class Tensor;
class TensorShape;
class TreeAdapter;
class TreeAdapterLite;
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

  /// \brief Gets the output type
  /// \return a vector of DataType. If failed, return an empty vector
  std::vector<mindspore::DataType> GetOutputTypes();

  /// \brief Gets the output shape
  /// \return a vector of TensorShape. If failed, return an empty vector
  std::vector<std::vector<int64_t>> GetOutputShapes();

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
  std::vector<std::string> GetColumnNames() { return VectorCharToString(GetColumnNamesCharIF()); }

  /// \brief Gets the class indexing
  /// \return a map of ClassIndexing. If failed, return an empty map
  std::vector<std::pair<std::string, std::vector<int32_t>>> GetClassIndexing() {
    return ClassIndexCharToString(GetClassIndexingCharIF());
  }

  /// \brief Setter function for runtime number of workers
  /// \param[in] num_workers The number of threads in this operator
  /// \return Shared pointer to the original object
  std::shared_ptr<Dataset> SetNumWorkers(int32_t num_workers);

  /// \brief Function to create an PullBasedIterator over the Dataset
  /// \param[in] columns List of columns to be used to specify the order of columns
  /// \return Shared pointer to the Iterator
  std::shared_ptr<PullIterator> CreatePullBasedIterator(std::vector<std::vector<char>> columns = {});

  /// \brief Function to create an Iterator over the Dataset pipeline
  /// \param[in] columns List of columns to be used to specify the order of columns
  /// \param[in] num_epochs Number of epochs to run through the pipeline, default -1 which means infinite epochs.
  ///     An empty row is returned at the end of each epoch
  /// \return Shared pointer to the Iterator
  std::shared_ptr<Iterator> CreateIterator(std::vector<std::string> columns = {}, int32_t num_epochs = -1) {
    return CreateIteratorCharIF(VectorStringToChar(columns), num_epochs);
  }

  /// \brief Function to transfer data through a device.
  /// \notes If device is Ascend, features of data will be transferred one by one. The limitation
  ///     of data transmission per time is 256M.
  /// \param[in] queue_name Channel name (default="", create new unique name).
  /// \param[in] device_type Type of device (default="", get from MSContext).
  /// \param[in] device_id id of device (default=1, get from MSContext).
  /// \param[in] num_epochs Number of epochs (default=-1, infinite epochs).
  /// \param[in] send_epoch_end Whether to send end of sequence to device or not (default=true).
  /// \param[in] total_batches Number of batches to be sent to the device (default=0, all data).
  /// \param[in] create_data_info_queue Whether to create queue which stores types and shapes
  ///     of data or not(default=false).
  /// \return Returns true if no error encountered else false.
  bool DeviceQueue(std::string queue_name = "", std::string device_type = "", int32_t device_id = 0,
                   int32_t num_epochs = -1, bool send_epoch_end = true, int32_t total_batches = 0,
                   bool create_data_info_queue = false) {
    return DeviceQueueCharIF(StringToChar(queue_name), StringToChar(device_type), device_id, num_epochs, send_epoch_end,
                             total_batches, create_data_info_queue);
  }

  /// \brief Function to create a Saver to save the dynamic data processed by the dataset pipeline
  /// \note Usage restrictions:
  ///     1. Supported dataset formats: 'mindrecord' only
  ///     2. To save the samples in order, set dataset's shuffle to false and num_files to 1.
  ///     3. Before calling the function, do not use batch operator, repeat operator or data augmentation operators
  ///        with random attribute in map operator.
  ///     4. Mindrecord does not support bool, uint64, multi-dimensional uint8(drop dimension) nor
  ///        multi-dimensional string.
  /// \param[in] file_name Path to dataset file
  /// \param[in] num_files Number of dataset files (default=1)
  /// \param[in] file_type Dataset format (default="mindrecord")
  /// \return Returns true if no error encountered else false
  bool Save(std::string dataset_path, int32_t num_files = 1, std::string dataset_type = "mindrecord") {
    return SaveCharIF(StringToChar(dataset_path), num_files, StringToChar(dataset_type));
  }

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
  /// \param[in] operations Vector of raw pointers to TensorTransform objects to be applied on the dataset. Operations
  ///     are applied in the order they appear in this list
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
  std::shared_ptr<MapDataset> Map(std::vector<TensorTransform *> operations,
                                  const std::vector<std::string> &input_columns = {},
                                  const std::vector<std::string> &output_columns = {},
                                  const std::vector<std::string> &project_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  std::vector<std::shared_ptr<DSCallback>> callbacks = {}) {
    std::vector<std::shared_ptr<TensorOperation>> transform_ops;
    (void)std::transform(
      operations.begin(), operations.end(), std::back_inserter(transform_ops),
      [](TensorTransform *op) -> std::shared_ptr<TensorOperation> { return op != nullptr ? op->Parse() : nullptr; });
    return std::make_shared<MapDataset>(shared_from_this(), transform_ops, VectorStringToChar(input_columns),
                                        VectorStringToChar(output_columns), VectorStringToChar(project_columns), cache,
                                        callbacks);
  }

  /// \brief Function to create a MapDataset
  /// \notes Applies each operation in operations to this dataset
  /// \param[in] operations Vector of shared pointers to TensorTransform objects to be applied on the dataset.
  ///     Operations are applied in the order they appear in this list
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
  std::shared_ptr<MapDataset> Map(std::vector<std::shared_ptr<TensorTransform>> operations,
                                  const std::vector<std::string> &input_columns = {},
                                  const std::vector<std::string> &output_columns = {},
                                  const std::vector<std::string> &project_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  std::vector<std::shared_ptr<DSCallback>> callbacks = {}) {
    std::vector<std::shared_ptr<TensorOperation>> transform_ops;
    (void)std::transform(operations.begin(), operations.end(), std::back_inserter(transform_ops),
                         [](std::shared_ptr<TensorTransform> op) -> std::shared_ptr<TensorOperation> {
                           return op != nullptr ? op->Parse() : nullptr;
                         });
    return std::make_shared<MapDataset>(shared_from_this(), transform_ops, VectorStringToChar(input_columns),
                                        VectorStringToChar(output_columns), VectorStringToChar(project_columns), cache,
                                        callbacks);
  }

  /// \brief Function to create a MapDataset
  /// \notes Applies each operation in operations to this dataset
  /// \param[in] operations Vector of TensorTransform objects to be applied on the dataset. Operations are applied in
  ///     the order they appear in this list
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
  std::shared_ptr<MapDataset> Map(const std::vector<std::reference_wrapper<TensorTransform>> operations,
                                  const std::vector<std::string> &input_columns = {},
                                  const std::vector<std::string> &output_columns = {},
                                  const std::vector<std::string> &project_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  std::vector<std::shared_ptr<DSCallback>> callbacks = {}) {
    std::vector<std::shared_ptr<TensorOperation>> transform_ops;
    (void)std::transform(operations.begin(), operations.end(), std::back_inserter(transform_ops),
                         [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
    return std::make_shared<MapDataset>(shared_from_this(), transform_ops, VectorStringToChar(input_columns),
                                        VectorStringToChar(output_columns), VectorStringToChar(project_columns), cache,
                                        callbacks);
  }

  /// \brief Function to create a Project Dataset
  /// \notes Applies project to the dataset
  /// \param[in] columns The name of columns to project
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ProjectDataset> Project(const std::vector<std::string> &columns) {
    return std::make_shared<ProjectDataset>(shared_from_this(), VectorStringToChar(columns));
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

 private:
  // Char interface(CharIF) of GetColumnNames
  std::vector<std::vector<char>> GetColumnNamesCharIF();

  // Char interface(CharIF) of GetClassIndexing
  std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> GetClassIndexingCharIF();

  // Char interface(CharIF) of CreateIterator
  std::shared_ptr<Iterator> CreateIteratorCharIF(std::vector<std::vector<char>> columns, int32_t num_epochs);

  // Char interface(CharIF) of DeviceQueue
  bool DeviceQueueCharIF(const std::vector<char> &queue_name, const std::vector<char> &device_type, int32_t device_id,
                         int32_t num_epochs, bool send_epoch_end, int32_t total_batches, bool create_data_info_queue);

  // Char interface(CharIF) of Save
  bool SaveCharIF(const std::vector<char> &dataset_path, int32_t num_files, const std::vector<char> &dataset_type);
};

class SchemaObj {
 public:
  /// \brief Constructor
  explicit SchemaObj(const std::string &schema_file = "") : SchemaObj(StringToChar(schema_file)) {}

  /// \brief Destructor
  ~SchemaObj() = default;

  /// \brief SchemaObj Init function
  /// \return bool true if schema initialization is successful
  Status Init();

  /// \brief Add new column to the schema with unknown shape of rank 1
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(mindspore::DataType).
  /// \return Status code
  Status add_column(const std::string &name, mindspore::DataType ms_type) {
    return add_column_char(StringToChar(name), ms_type);
  }

  /// \brief Add new column to the schema with unknown shape of rank 1
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(std::string).
  /// \param[in] shape Shape of the column.
  /// \return Status code
  Status add_column(const std::string &name, const std::string &ms_type) {
    return add_column_char(StringToChar(name), StringToChar(ms_type));
  }

  /// \brief Add new column to the schema
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(mindspore::DataType).
  /// \param[in] shape Shape of the column.
  /// \return Status code
  Status add_column(const std::string &name, mindspore::DataType ms_type, const std::vector<int32_t> &shape) {
    return add_column_char(StringToChar(name), ms_type, shape);
  }

  /// \brief Add new column to the schema
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(std::string).
  /// \param[in] shape Shape of the column.
  /// \return Status code
  Status add_column(const std::string &name, const std::string &ms_type, const std::vector<int32_t> &shape) {
    return add_column_char(StringToChar(name), StringToChar(ms_type), shape);
  }

  /// \brief Get a JSON string of the schema
  /// \return JSON string of the schema
  std::string to_json() { return CharToString(to_json_char()); }

  /// \brief Get a JSON string of the schema
  std::string to_string() { return to_json(); }

  /// \brief Set a new value to dataset_type
  void set_dataset_type(std::string dataset_type);

  /// \brief Set a new value to num_rows
  void set_num_rows(int32_t num_rows);

  /// \brief Get the current num_rows
  int32_t get_num_rows() const;

  /// \brief Get schema file from JSON file
  /// \param[in] json_string Name of JSON file to be parsed.
  /// \return Status code
  Status FromJSONString(const std::string &json_string) { return FromJSONStringCharIF(StringToChar(json_string)); }

  /// \brief Parse and add column information
  /// \param[in] json_string Name of JSON string for column dataset attribute information, decoded from schema file.
  /// \return Status code
  Status ParseColumnString(const std::string &json_string) {
    return ParseColumnStringCharIF(StringToChar(json_string));
  }

 private:
  // Char constructor of SchemaObj
  explicit SchemaObj(const std::vector<char> &schema_file);

  // Char interface of add_column
  Status add_column_char(const std::vector<char> &name, mindspore::DataType ms_type);

  Status add_column_char(const std::vector<char> &name, const std::vector<char> &ms_type);

  Status add_column_char(const std::vector<char> &name, mindspore::DataType ms_type, const std::vector<int32_t> &shape);

  Status add_column_char(const std::vector<char> &name, const std::vector<char> &ms_type,
                         const std::vector<int32_t> &shape);

  // Char interface of to_json
  const std::vector<char> to_json_char();

  // Char interface of FromJSONString
  Status FromJSONStringCharIF(const std::vector<char> &json_string);

  // Char interface of ParseColumnString
  Status ParseColumnStringCharIF(const std::vector<char> &json_string);

  struct Data;
  std::shared_ptr<Data> data_;
};

class BatchDataset : public Dataset {
 public:
  BatchDataset(std::shared_ptr<Dataset> input, int32_t batch_size, bool drop_remainder = false);
  ~BatchDataset() = default;
};

class MapDataset : public Dataset {
 public:
  MapDataset(std::shared_ptr<Dataset> input, std::vector<std::shared_ptr<TensorOperation>> operations,
             const std::vector<std::vector<char>> &input_columns, const std::vector<std::vector<char>> &output_columns,
             const std::vector<std::vector<char>> &project_columns, const std::shared_ptr<DatasetCache> &cache,
             std::vector<std::shared_ptr<DSCallback>> callbacks);
  ~MapDataset() = default;
};

class ProjectDataset : public Dataset {
 public:
  ProjectDataset(std::shared_ptr<Dataset> input, const std::vector<std::vector<char>> &columns);
  ~ProjectDataset() = default;
};

class ShuffleDataset : public Dataset {
 public:
  ShuffleDataset(std::shared_ptr<Dataset> input, int32_t buffer_size);
  ~ShuffleDataset() = default;
};

/// \brief Function to create a SchemaObj
/// \param[in] schema_file Path of schema file
/// \note This api exists because std::string will constrained by ABI compile macro but char don't.
/// \return Shared pointer to the current schema
std::shared_ptr<SchemaObj> SchemaCharIF(const std::vector<char> &schema_file);

/// \brief Function to create a SchemaObj
/// \param[in] schema_file Path of schema file
/// \return Shared pointer to the current schema
inline std::shared_ptr<SchemaObj> Schema(const std::string &schema_file = "") {
  return SchemaCharIF(StringToChar(schema_file));
}

class AlbumDataset : public Dataset {
 public:
  AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
               const std::vector<std::vector<char>> &column_names, bool decode, const std::shared_ptr<Sampler> &sampler,
               const std::shared_ptr<DatasetCache> &cache);
  AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
               const std::vector<std::vector<char>> &column_names, bool decode, const Sampler *sampler,
               const std::shared_ptr<DatasetCache> &cache);
  AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
               const std::vector<std::vector<char>> &column_names, bool decode,
               const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache);
  ~AlbumDataset() = default;
};

/// \brief Function to create an AlbumDataset
/// \notes The generated dataset is specified through setting a schema
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] data_schema Path to dataset schema file
/// \param[in] column_names Column names used to specify columns to load, if empty, will read all columns.
///     (default = {})
/// \param[in] decode the option to decode the images in dataset (default = false)
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<AlbumDataset> Album(const std::string &dataset_dir, const std::string &data_schema,
                                           const std::vector<std::string> &column_names = {}, bool decode = false,
                                           const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AlbumDataset>(StringToChar(dataset_dir), StringToChar(data_schema),
                                        VectorStringToChar(column_names), decode, sampler, cache);
}
/// \brief Function to create an AlbumDataset
/// \notes The generated dataset is specified through setting a schema
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] data_schema Path to dataset schema file
/// \param[in] column_names Column names used to specify columns to load
/// \param[in] decode the option to decode the images in dataset
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<AlbumDataset> Album(const std::string &dataset_dir, const std::string &data_schema,
                                           const std::vector<std::string> &column_names, bool decode,
                                           const Sampler *sampler,
                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AlbumDataset>(StringToChar(dataset_dir), StringToChar(data_schema),
                                        VectorStringToChar(column_names), decode, sampler, cache);
}
/// \brief Function to create an AlbumDataset
/// \notes The generated dataset is specified through setting a schema
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] data_schema Path to dataset schema file
/// \param[in] column_names Column names used to specify columns to load
/// \param[in] decode the option to decode the images in dataset
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<AlbumDataset> Album(const std::string &dataset_dir, const std::string &data_schema,
                                           const std::vector<std::string> &column_names, bool decode,
                                           const std::reference_wrapper<Sampler> sampler,
                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AlbumDataset>(StringToChar(dataset_dir), StringToChar(data_schema),
                                        VectorStringToChar(column_names), decode, sampler, cache);
}

class MnistDataset : public Dataset {
 public:
  explicit MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                        const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);
  explicit MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                        const std::shared_ptr<DatasetCache> &cache);
  explicit MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                        const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache);
  ~MnistDataset() = default;
};

/// \brief Function to create a MnistDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of MNIST, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current MnistDataset
inline std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir, const std::string &usage = "all",
                                           const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a MnistDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of MNIST, can be "train", "test" or "all"
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current MnistDataset
inline std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir, const std::string &usage,
                                           const Sampler *sampler,
                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a MnistDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of MNIST, can be "train", "test" or "all"
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current MnistDataset
inline std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir, const std::string &usage,
                                           const std::reference_wrapper<Sampler> sampler,
                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
