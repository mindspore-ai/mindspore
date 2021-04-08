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
#include <functional>
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
#include "minddata/dataset/include/iterator.h"
#include "minddata/dataset/include/json_fwd.hpp"
#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/include/text.h"

namespace mindspore {
namespace dataset {

class Tensor;
class TensorShape;
class TreeAdapter;
class TreeAdapterLite;
class TreeGetters;
class Vocab;

class DatasetCache;
class DatasetNode;

class Iterator;
class PullBasedIterator;

class TensorOperation;
class SchemaObj;
class SamplerObj;
class CsvBase;

// Dataset classes (in alphabetical order)
class BatchDataset;
class MapDataset;
class ProjectDataset;
class ShuffleDataset;
class BucketBatchByLengthDataset;
class FilterDataset;
class CSVDataset;
class TransferDataset;
class ConcatDataset;
class RenameDataset;

class SentencePieceVocab;
enum class SentencePieceModel;

class DSCallback;

class RepeatDataset;
class SkipDataset;
class TakeDataset;
class ZipDataset;

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

  /// \brief Function to create a BucketBatchByLengthDataset
  /// \notes Bucket elements according to their lengths. Each bucket will be padded and batched when
  ///    they are full.
  /// \param[in] column_names Columns passed to element_length_function
  /// \param[in] bucket_boundaries A list consisting of the upper boundaries of the buckets.
  ///    Must be strictly increasing. If there are n boundaries, n+1 buckets are created: One bucket for
  ///    [0, bucket_boundaries[0]), one bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
  ///    0<i<n, and one bucket for [bucket_boundaries[n-1], inf).
  /// \param[in] bucket_batch_sizes A list consisting of the batch sizes for each bucket.
  ///    Must contain elements equal to the size of bucket_boundaries + 1.
  /// \param[in] element_length_function A function pointer that takes in MSTensorVec and outputs a MSTensorVec.
  ///    The output must contain a single tensor containing a single int32_t. If no value is provided,
  ///    then size of column_names must be 1, and the size of the first dimension of that column will be taken
  ///    as the length (default=nullptr)
  /// \param[in] pad_info Represents how to batch each column. The key corresponds to the column name, the value must
  ///    be a tuple of 2 elements.  The first element corresponds to the shape to pad to, and the second element
  ///    corresponds to the value to pad with. If a column is not specified, then that column will be padded to the
  ///    longest in the current batch, and 0 will be used as the padding value. Any unspecified dimensions will be
  ///    padded to the longest in the current batch, unless if pad_to_bucket_boundary is true. If no padding is
  ///    wanted, set pad_info to None (default=empty dictionary).
  /// \param[in] pad_to_bucket_boundary If true, will pad each unspecified dimension in pad_info to the
  ///    bucket_boundary minus 1. If there are any elements that fall into the last bucket,
  ///    an error will occur (default=false).
  /// \param[in] drop_remainder If true, will drop the last batch for each bucket if it is not a full batch
  ///    (default=false).
  /// \return Shared pointer to the current BucketBatchByLengthDataset
  std::shared_ptr<BucketBatchByLengthDataset> BucketBatchByLength(
    const std::vector<std::string> &column_names, const std::vector<int32_t> &bucket_boundaries,
    const std::vector<int32_t> &bucket_batch_sizes,
    std::function<MSTensorVec(MSTensorVec)> element_length_function = nullptr,
    const std::map<std::string, std::pair<std::vector<int64_t>, MSTensor>> &pad_info = {},
    bool pad_to_bucket_boundary = false, bool drop_remainder = false) {
    return std::make_shared<BucketBatchByLengthDataset>(
      shared_from_this(), VectorStringToChar(column_names), bucket_boundaries, bucket_batch_sizes,
      element_length_function, PadInfoStringToChar(pad_info), pad_to_bucket_boundary, drop_remainder);
  }

  /// \brief Function to create a SentencePieceVocab from source dataset
  /// \notes Build a SentencePieceVocab from a dataset.
  /// \param[in] col_names Column names to get words from. It can be a vector of column names
  /// \param[in] vocab_size Vocabulary size.
  /// \param[in] character_coverage Percentage of characters covered by the model, must be between
  ///     0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
  ///     Japanese or Chinese character sets, and 1.0 for other languages with small character sets.
  /// \param[in] model_type Model type. Choose from unigram (default), bpe, char, or word.
  ///     The input sentence must be pretokenized when using word type.
  /// \param[in] params A vector contains more option parameters of sentencepiece library
  std::shared_ptr<SentencePieceVocab> BuildSentencePieceVocab(
    const std::vector<std::string> &col_names, int32_t vocab_size, float character_coverage,
    SentencePieceModel model_type, const std::unordered_map<std::string, std::string> &params) {
    return BuildSentencePieceVocabCharIF(VectorStringToChar(col_names), vocab_size, character_coverage, model_type,
                                         UnorderedMapStringToChar(params));
  }

  /// \brief Function to create a Vocab from source dataset
  /// \notes Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
  ///    which contains top_k most frequent words (if top_k is specified)
  /// \param[in] columns Column names to get words from. It can be a vector of column names
  /// \param[in] freq_range A tuple of integers (min_frequency, max_frequency). Words within the frequency
  ///    range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
  ///    can be set to default, which corresponds to 0/total_words separately
  /// \param[in] top_k Number of words to be built into vocab. top_k most frequent words are
  ///    taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken
  /// \param[in] special_tokens A list of strings, each one is a special token
  /// \param[in] special_first Whether special_tokens will be prepended/appended to vocab, If special_tokens
  ///    is specified and special_first is set to default, special_tokens will be prepended
  /// \return Shared pointer to the current Vocab
  std::shared_ptr<Vocab> BuildVocab(const std::vector<std::string> &columns = {},
                                    const std::pair<int64_t, int64_t> &freq_range = {0, kDeMaxFreq},
                                    int64_t top_k = kDeMaxTopk, const std::vector<std::string> &special_tokens = {},
                                    bool special_first = true) {
    return BuildVocabCharIF(VectorStringToChar(columns), freq_range, top_k, VectorStringToChar(special_tokens),
                            special_first);
  }

  /// \brief Function to create a ConcatDataset
  /// \notes Concat the datasets in the input
  /// \param[in] datasets List of shared pointers to the dataset that should be concatenated together
  /// \return Shared pointer to the current ConcatDataset
  std::shared_ptr<ConcatDataset> Concat(const std::vector<std::shared_ptr<Dataset>> &datasets) {
    std::vector<std::shared_ptr<Dataset>> all_datasets{shared_from_this()};
    all_datasets.insert(std::end(all_datasets), std::begin(datasets), std::end(datasets));
    return std::make_shared<ConcatDataset>(all_datasets);
  }

  /// \brief Function to filter dataset by predicate
  /// \notes If input_columns is not provided or empty, all columns will be used
  /// \param[in] predicate Function callable which returns a boolean value. If false then filter the element
  /// \param[in] input_columns List of names of the input columns to filter
  /// \return Shared pointer to the current FilterNode
  std::shared_ptr<FilterDataset> Filter(std::function<MSTensorVec(MSTensorVec)> predicate,
                                        const std::vector<std::string> &input_columns = {}) {
    return std::make_shared<FilterDataset>(shared_from_this(), predicate, VectorStringToChar(input_columns));
  }

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

  /// \brief Function to create a Rename Dataset
  /// \notes Renames the columns in the input dataset
  /// \param[in] input_columns List of the input columns to rename
  /// \param[in] output_columns List of the output columns
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<RenameDataset> Rename(const std::vector<std::string> &input_columns,
                                        const std::vector<std::string> &output_columns) {
    return std::make_shared<RenameDataset>(shared_from_this(), VectorStringToChar(input_columns),
                                           VectorStringToChar(output_columns));
  }
  /// \brief Function to create a RepeatDataset
  /// \notes Repeats this dataset count times. Repeat indefinitely if count is -1
  /// \param[in] count Number of times the dataset should be repeated
  /// \return Shared pointer to the current Dataset
  /// \note Repeat will return shared pointer to `Dataset` instead of `RepeatDataset`
  ///     due to a limitation in the current implementation
  std::shared_ptr<RepeatDataset> Repeat(int32_t count = -1) {
    return std::make_shared<RepeatDataset>(shared_from_this(), count);
  }
  /// \brief Function to create a Shuffle Dataset
  /// \notes Randomly shuffles the rows of this dataset
  /// \param[in] buffer_size The size of the buffer (must be larger than 1) for shuffling
  /// \return Shared pointer to the current ShuffleDataset
  std::shared_ptr<ShuffleDataset> Shuffle(int32_t buffer_size) {
    return std::make_shared<ShuffleDataset>(shared_from_this(), buffer_size);
  }

  /// \brief Function to create a SkipDataset
  /// \notes Skips count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be skipped.
  /// \return Shared pointer to the current SkipDataset
  std::shared_ptr<SkipDataset> Skip(int32_t count) { return std::make_shared<SkipDataset>(shared_from_this(), count); }

  /// \brief Function to create a TakeDataset
  /// \notes Takes count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be taken.
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<TakeDataset> Take(int32_t count = -1) {
    return std::make_shared<TakeDataset>(shared_from_this(), count);
  }

  /// \brief Function to create a Zip Dataset
  /// \notes Applies zip to the dataset
  /// \param[in] datasets A list of shared pointers to the datasets that we want to zip
  /// \return Shared pointer to the current Dataset
  std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
    std::vector<std::shared_ptr<Dataset>> all_datasets = datasets;
    all_datasets.push_back(shared_from_this());
    return std::make_shared<ZipDataset>(all_datasets);
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

  // Char interface(CharIF) of BuildSentencePieceVocab
  std::shared_ptr<SentencePieceVocab> BuildSentencePieceVocabCharIF(
    const std::vector<std::vector<char>> &col_names, int32_t vocab_size, float character_coverage,
    SentencePieceModel model_type, const std::map<std::vector<char>, std::vector<char>> &params);

  // Char interface(CharIF) of BuildVocab
  std::shared_ptr<Vocab> BuildVocabCharIF(const std::vector<std::vector<char>> &columns,
                                          const std::pair<int64_t, int64_t> &freq_range, int64_t top_k,
                                          const std::vector<std::vector<char>> &special_tokens, bool special_first);
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
  /// \brief Parse the columns and add them to columns
  /// \param[in] columns Dataset attribution information, decoded from schema file.
  ///    Support both nlohmann::json::value_t::array and nlohmann::json::value_t::onject.
  /// \return Status code
  Status parse_column(nlohmann::json columns);

  /// \brief Get schema file from JSON file
  /// \param[in] json_obj parsed JSON object
  /// \return Status code
  Status from_json(nlohmann::json json_obj);

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

class BucketBatchByLengthDataset : public Dataset {
 public:
  BucketBatchByLengthDataset(
    std::shared_ptr<Dataset> input, const std::vector<std::vector<char>> &column_names,
    const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
    std::function<MSTensorVec(MSTensorVec)> element_length_function = nullptr,
    const std::map<std::vector<char>, std::pair<std::vector<int64_t>, MSTensor>> &pad_info = {},
    bool pad_to_bucket_boundary = false, bool drop_remainder = false);
  ~BucketBatchByLengthDataset() = default;
};

class ConcatDataset : public Dataset {
 public:
  explicit ConcatDataset(const std::vector<std::shared_ptr<Dataset>> &input);
  ~ConcatDataset() = default;
};

class FilterDataset : public Dataset {
 public:
  FilterDataset(std::shared_ptr<Dataset> input, std::function<MSTensorVec(MSTensorVec)> predicate,
                const std::vector<std::vector<char>> &input_columns);
  ~FilterDataset() = default;
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

class RenameDataset : public Dataset {
 public:
  RenameDataset(std::shared_ptr<Dataset> input, const std::vector<std::vector<char>> &input_columns,
                const std::vector<std::vector<char>> &output_columns);
  ~RenameDataset() = default;
};

class RepeatDataset : public Dataset {
 public:
  RepeatDataset(std::shared_ptr<Dataset> input, int32_t count);
  ~RepeatDataset() = default;
};

class ShuffleDataset : public Dataset {
 public:
  ShuffleDataset(std::shared_ptr<Dataset> input, int32_t buffer_size);
  ~ShuffleDataset() = default;
};

class SkipDataset : public Dataset {
 public:
  SkipDataset(std::shared_ptr<Dataset> input, int32_t count);
  ~SkipDataset() = default;
};

class TakeDataset : public Dataset {
 public:
  TakeDataset(std::shared_ptr<Dataset> input, int32_t count);
  ~TakeDataset() = default;
};

class ZipDataset : public Dataset {
 public:
  explicit ZipDataset(const std::vector<std::shared_ptr<Dataset>> &inputs);
  ~ZipDataset() = default;
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

class CelebADataset : public Dataset {
 public:
  explicit CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                         const std::shared_ptr<Sampler> &sampler, bool decode,
                         const std::set<std::vector<char>> &extensions, const std::shared_ptr<DatasetCache> &cache);
  explicit CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                         bool decode, const std::set<std::vector<char>> &extensions,
                         const std::shared_ptr<DatasetCache> &cache);
  explicit CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                         const std::reference_wrapper<Sampler> sampler, bool decode,
                         const std::set<std::vector<char>> &extensions, const std::shared_ptr<DatasetCache> &cache);
  ~CelebADataset() = default;
};

/// \brief Function to create a CelebADataset
/// \notes The generated dataset has two columns ['image', 'attr'].
///      The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train", "valid" or "test" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<CelebADataset> CelebA(
  const std::string &dataset_dir, const std::string &usage = "all",
  const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(), bool decode = false,
  const std::set<std::string> &extensions = {}, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CelebADataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, decode,
                                         SetStringToChar(extensions), cache);
}

/// \brief Function to create a CelebADataset
/// \notes The generated dataset has two columns ['image', 'attr'].
///      The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train", "valid" or "test"
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<CelebADataset> CelebA(const std::string &dataset_dir, const std::string &usage,
                                             const Sampler *sampler, bool decode = false,
                                             const std::set<std::string> &extensions = {},
                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CelebADataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, decode,
                                         SetStringToChar(extensions), cache);
}

/// \brief Function to create a CelebADataset
/// \notes The generated dataset has two columns ['image', 'attr'].
///      The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train", "valid" or "test"
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<CelebADataset> CelebA(const std::string &dataset_dir, const std::string &usage,
                                             const std::reference_wrapper<Sampler> sampler, bool decode = false,
                                             const std::set<std::string> &extensions = {},
                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CelebADataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, decode,
                                         SetStringToChar(extensions), cache);
}

class Cifar10Dataset : public Dataset {
 public:
  explicit Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                          const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);
  explicit Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                          const std::shared_ptr<DatasetCache> &cache);
  explicit Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                          const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache);
  ~Cifar10Dataset() = default;
};

/// \brief Function to create a Cifar10 Dataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR10, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<Cifar10Dataset> Cifar10(
  const std::string &dataset_dir, const std::string &usage = "all",
  const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
  const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar10 Dataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR10, can be "train", "test" or "all"
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir, const std::string &usage,
                                               const Sampler *sampler,
                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar10 Dataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR10, can be "train", "test" or "all"
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir, const std::string &usage,
                                               const std::reference_wrapper<Sampler> sampler,
                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

class Cifar100Dataset : public Dataset {
 public:
  explicit Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);
  explicit Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                           const std::shared_ptr<DatasetCache> &cache);
  explicit Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache);
  ~Cifar100Dataset() = default;
};

/// \brief Function to create a Cifar100 Dataset
/// \notes The generated dataset has three columns ["image", "coarse_label", "fine_label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR100, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<Cifar100Dataset> Cifar100(
  const std::string &dataset_dir, const std::string &usage = "all",
  const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
  const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar100Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar100 Dataset
/// \notes The generated dataset has three columns ["image", "coarse_label", "fine_label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR100, can be "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<Cifar100Dataset> Cifar100(const std::string &dataset_dir, const std::string &usage,
                                                 const Sampler *sampler,
                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar100Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar100 Dataset
/// \notes The generated dataset has three columns ["image", "coarse_label", "fine_label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of CIFAR100, can be "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<Cifar100Dataset> Cifar100(const std::string &dataset_dir, const std::string &usage,
                                                 const std::reference_wrapper<Sampler> sampler,
                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar100Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

class CLUEDataset : public Dataset {
 public:
  explicit CLUEDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &task,
                       const std::vector<char> &usage, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                       int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);
  ~CLUEDataset() = default;
};

/// \brief Function to create a CLUEDataset
/// \notes The generated dataset has a variable number of columns depending on the task and usage
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] task The kind of task, one of "AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC" and "CSL" (default="AFQMC").
/// \param[in] usage Be used to "train", "test" or "eval" data (default="train").
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified. (Default = 0)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current CLUEDataset
inline std::shared_ptr<CLUEDataset> CLUE(const std::vector<std::string> &dataset_files,
                                         const std::string &task = "AFQMC", const std::string &usage = "train",
                                         int64_t num_samples = 0, ShuffleMode shuffle = ShuffleMode::kGlobal,
                                         int32_t num_shards = 1, int32_t shard_id = 0,
                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CLUEDataset>(VectorStringToChar(dataset_files), StringToChar(task), StringToChar(usage),
                                       num_samples, shuffle, num_shards, shard_id, cache);
}

class CocoDataset : public Dataset {
 public:
  CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
              const std::vector<char> &task, const bool &decode, const std::shared_ptr<Sampler> &sampler,
              const std::shared_ptr<DatasetCache> &cache);
  CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
              const std::vector<char> &task, const bool &decode, const Sampler *sampler,
              const std::shared_ptr<DatasetCache> &cache);
  CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
              const std::vector<char> &task, const bool &decode, const std::reference_wrapper<Sampler> sampler,
              const std::shared_ptr<DatasetCache> &cache);
  ~CocoDataset() = default;
};

/// \brief Function to create a CocoDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                  ['iscrowd', dtype=uint32]].
///     - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///     - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                 ['num_keypoints', dtype=uint32]].
///     - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                 ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] annotation_file Path to the annotation json
/// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<CocoDataset> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                         const std::string &task = "Detection", const bool &decode = false,
                                         const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CocoDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), StringToChar(task),
                                       decode, sampler, cache);
}

/// \brief Function to create a CocoDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                  ['iscrowd', dtype=uint32]].
///     - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///     - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                 ['num_keypoints', dtype=uint32]].
///     - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                 ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] annotation_file Path to the annotation json
/// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<CocoDataset> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                         const std::string &task, const bool &decode, const Sampler *sampler,
                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CocoDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), StringToChar(task),
                                       decode, sampler, cache);
}

/// \brief Function to create a CocoDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                  ['iscrowd', dtype=uint32]].
///     - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///     - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                 ['num_keypoints', dtype=uint32]].
///     - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                 ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] annotation_file Path to the annotation json
/// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<CocoDataset> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                         const std::string &task, const bool &decode,
                                         const std::reference_wrapper<Sampler> sampler,
                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CocoDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), StringToChar(task),
                                       decode, sampler, cache);
}

class CSVDataset : public Dataset {
 public:
  explicit CSVDataset(const std::vector<std::vector<char>> &dataset_files, char field_delim,
                      const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                      const std::vector<std::vector<char>> &column_names, int64_t num_samples, ShuffleMode shuffle,
                      int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);
  ~CSVDataset() = default;
};

/// \brief Function to create a CSVDataset
/// \notes The generated dataset has a variable number of columns
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///    will be sorted in a lexicographical order.
/// \param[in] field_delim A char that indicates the delimiter to separate fields (default=',').
/// \param[in] column_defaults List of default values for the CSV field (default={}). Each item in the list is
///    either a valid type (float, int, or string). If this is not provided, treats all columns as string type.
/// \param[in] column_names List of column names of the dataset (default={}). If this is not provided, infers the
///    column_names from the first row of CSV file.
/// \param[in] num_samples The number of samples to be included in the dataset.
///    (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode::kGlobal)
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///    specified only when num_shards is also specified. (Default = 0)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<CSVDataset> CSV(const std::vector<std::string> &dataset_files, char field_delim = ',',
                                       const std::vector<std::shared_ptr<CsvBase>> &column_defaults = {},
                                       const std::vector<std::string> &column_names = {}, int64_t num_samples = 0,
                                       ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                       int32_t shard_id = 0, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CSVDataset>(VectorStringToChar(dataset_files), field_delim, column_defaults,
                                      VectorStringToChar(column_names), num_samples, shuffle, num_shards, shard_id,
                                      cache);
}

class ImageFolderDataset : public Dataset {
 public:
  explicit ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode,
                              const std::shared_ptr<Sampler> &sampler, const std::set<std::vector<char>> &extensions,
                              const std::map<std::vector<char>, int32_t> &class_indexing,
                              const std::shared_ptr<DatasetCache> &cache);
  explicit ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
                              const std::set<std::vector<char>> &extensions,
                              const std::map<std::vector<char>, int32_t> &class_indexing,
                              const std::shared_ptr<DatasetCache> &cache);
  explicit ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode,
                              const std::reference_wrapper<Sampler> sampler,
                              const std::set<std::vector<char>> &extensions,
                              const std::map<std::vector<char>, int32_t> &class_indexing,
                              const std::shared_ptr<DatasetCache> &cache);
  ~ImageFolderDataset() = default;
};

/// \brief Function to create an ImageFolderDataset
/// \notes A source dataset that reads images from a tree of directories
///     All images within one folder have the same label
///     The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] decode A flag to decode in ImageFolder
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] extensions File extensions to be read
/// \param[in] class_indexing a class name to label map
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current ImageFolderDataset
inline std::shared_ptr<ImageFolderDataset> ImageFolder(
  const std::string &dataset_dir, bool decode = false,
  const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
  const std::set<std::string> &extensions = {}, const std::map<std::string, int32_t> &class_indexing = {},
  const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ImageFolderDataset>(StringToChar(dataset_dir), decode, sampler, SetStringToChar(extensions),
                                              MapStringToChar(class_indexing), cache);
}

/// \brief Function to create an ImageFolderDataset
/// \notes A source dataset that reads images from a tree of directories
///     All images within one folder have the same label
///     The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] decode A flag to decode in ImageFolder
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] extensions File extensions to be read
/// \param[in] class_indexing a class name to label map
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current ImageFolderDataset
inline std::shared_ptr<ImageFolderDataset> ImageFolder(const std::string &dataset_dir, bool decode,
                                                       const Sampler *sampler,
                                                       const std::set<std::string> &extensions = {},
                                                       const std::map<std::string, int32_t> &class_indexing = {},
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ImageFolderDataset>(StringToChar(dataset_dir), decode, sampler, SetStringToChar(extensions),
                                              MapStringToChar(class_indexing), cache);
}

/// \brief Function to create an ImageFolderDataset
/// \notes A source dataset that reads images from a tree of directories
///     All images within one folder have the same label
///     The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] decode A flag to decode in ImageFolder
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] extensions File extensions to be read
/// \param[in] class_indexing a class name to label map
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current ImageFolderDataset
inline std::shared_ptr<ImageFolderDataset> ImageFolder(const std::string &dataset_dir, bool decode,
                                                       const std::reference_wrapper<Sampler> sampler,
                                                       const std::set<std::string> &extensions = {},
                                                       const std::map<std::string, int32_t> &class_indexing = {},
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ImageFolderDataset>(StringToChar(dataset_dir), decode, sampler, SetStringToChar(extensions),
                                              MapStringToChar(class_indexing), cache);
}

class ManifestDataset : public Dataset {
 public:
  explicit ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                           const std::shared_ptr<Sampler> &sampler,
                           const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
                           const std::shared_ptr<DatasetCache> &cache);
  explicit ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                           const Sampler *sampler, const std::map<std::vector<char>, int32_t> &class_indexing,
                           bool decode, const std::shared_ptr<DatasetCache> &cache);
  explicit ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                           const std::reference_wrapper<Sampler> sampler,
                           const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
                           const std::shared_ptr<DatasetCache> &cache);
  ~ManifestDataset() = default;
};

/// \brief Function to create a ManifestDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_file The dataset file to be read
/// \param[in] usage Need "train", "eval" or "inference" data (default="train")
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current ManifestDataset
inline std::shared_ptr<ManifestDataset> Manifest(
  const std::string &dataset_file, const std::string &usage = "train",
  const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
  const std::map<std::string, int32_t> &class_indexing = {}, bool decode = false,
  const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ManifestDataset>(StringToChar(dataset_file), StringToChar(usage), sampler,
                                           MapStringToChar(class_indexing), decode, cache);
}

/// \brief Function to create a ManifestDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_file The dataset file to be read
/// \param[in] usage Need "train", "eval" or "inference" data
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current ManifestDataset
inline std::shared_ptr<ManifestDataset> Manifest(const std::string &dataset_file, const std::string &usage,
                                                 const Sampler *sampler,
                                                 const std::map<std::string, int32_t> &class_indexing = {},
                                                 bool decode = false,
                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ManifestDataset>(StringToChar(dataset_file), StringToChar(usage), sampler,
                                           MapStringToChar(class_indexing), decode, cache);
}

/// \brief Function to create a ManifestDataset
/// \notes The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_file The dataset file to be read
/// \param[in] usage Need "train", "eval" or "inference" data
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current ManifestDataset
inline std::shared_ptr<ManifestDataset> Manifest(const std::string &dataset_file, const std::string &usage,
                                                 const std::reference_wrapper<Sampler> sampler,
                                                 const std::map<std::string, int32_t> &class_indexing = {},
                                                 bool decode = false,
                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ManifestDataset>(StringToChar(dataset_file), StringToChar(usage), sampler,
                                           MapStringToChar(class_indexing), decode, cache);
}

class MindDataDataset : public Dataset {
 public:
  explicit MindDataDataset(const std::vector<char> &dataset_file, const std::vector<std::vector<char>> &columns_list,
                           const std::shared_ptr<Sampler> &sampler, const nlohmann::json *padded_sample,
                           int64_t num_padded);
  explicit MindDataDataset(const std::vector<char> &dataset_file, const std::vector<std::vector<char>> &columns_list,
                           const Sampler *sampler, const nlohmann::json *padded_sample, int64_t num_padded);
  explicit MindDataDataset(const std::vector<char> &dataset_file, const std::vector<std::vector<char>> &columns_list,
                           const std::reference_wrapper<Sampler> sampler, const nlohmann::json *padded_sample,
                           int64_t num_padded);
  explicit MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                           const std::vector<std::vector<char>> &columns_list, const std::shared_ptr<Sampler> &sampler,
                           const nlohmann::json *padded_sample, int64_t num_padded);
  explicit MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                           const std::vector<std::vector<char>> &columns_list, const Sampler *sampler,
                           const nlohmann::json *padded_sample, int64_t num_padded);
  explicit MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                           const std::vector<std::vector<char>> &columns_list,
                           const std::reference_wrapper<Sampler> sampler, const nlohmann::json *padded_sample,
                           int64_t num_padded);
  ~MindDataDataset() = default;
};

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
///     in the same path will be found and loaded automatically.
/// \param[in] columns_list List of columns to be read (default={})
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()),
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
inline std::shared_ptr<MindDataDataset> MindData(
  const std::string &dataset_file, const std::vector<std::string> &columns_list = {},
  const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(), nlohmann::json *padded_sample = nullptr,
  int64_t num_padded = 0) {
  return std::make_shared<MindDataDataset>(StringToChar(dataset_file), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded);
}

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
///     in the same path will be found and loaded automatically.
/// \param[in] columns_list List of columns to be read
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
inline std::shared_ptr<MindDataDataset> MindData(const std::string &dataset_file,
                                                 const std::vector<std::string> &columns_list, const Sampler *sampler,
                                                 nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0) {
  return std::make_shared<MindDataDataset>(StringToChar(dataset_file), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded);
}

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
///     in the same path will be found and loaded automatically.
/// \param[in] columns_list List of columns to be read
/// \param[in] sampler Sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
inline std::shared_ptr<MindDataDataset> MindData(const std::string &dataset_file,
                                                 const std::vector<std::string> &columns_list,
                                                 const std::reference_wrapper<Sampler> sampler,
                                                 nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0) {
  return std::make_shared<MindDataDataset>(StringToChar(dataset_file), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded);
}

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_files List of dataset files to be read directly.
/// \param[in] columns_list List of columns to be read (default={})
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()),
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
inline std::shared_ptr<MindDataDataset> MindData(
  const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list = {},
  const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(), nlohmann::json *padded_sample = nullptr,
  int64_t num_padded = 0) {
  return std::make_shared<MindDataDataset>(VectorStringToChar(dataset_files), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded);
}

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_files List of dataset files to be read directly.
/// \param[in] columns_list List of columns to be read
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
inline std::shared_ptr<MindDataDataset> MindData(const std::vector<std::string> &dataset_files,
                                                 const std::vector<std::string> &columns_list, const Sampler *sampler,
                                                 nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0) {
  return std::make_shared<MindDataDataset>(VectorStringToChar(dataset_files), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded);
}

/// \brief Function to create a MindDataDataset
/// \param[in] dataset_files List of dataset files to be read directly.
/// \param[in] columns_list List of columns to be read
/// \param[in] sampler Sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \return Shared pointer to the current MindDataDataset
inline std::shared_ptr<MindDataDataset> MindData(const std::vector<std::string> &dataset_files,
                                                 const std::vector<std::string> &columns_list,
                                                 const std::reference_wrapper<Sampler> sampler,
                                                 nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0) {
  return std::make_shared<MindDataDataset>(VectorStringToChar(dataset_files), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded);
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

/// \brief Function to create a ConcatDataset
/// \notes Reload "+" operator to concat two datasets
/// \param[in] datasets1 Shared pointer to the first dataset to be concatenated
/// \param[in] datasets2 Shared pointer to the second dataset to be concatenated
/// \return Shared pointer to the current ConcatDataset
inline std::shared_ptr<ConcatDataset> operator+(const std::shared_ptr<Dataset> &datasets1,
                                                const std::shared_ptr<Dataset> &datasets2) {
  return std::make_shared<ConcatDataset>(std::vector({datasets1, datasets2}));
}

class RandomDataDataset : public Dataset {
 public:
  RandomDataDataset(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema,
                    const std::vector<std::vector<char>> &columns_list, std::shared_ptr<DatasetCache> cache);

  RandomDataDataset(const int32_t &total_rows, const std::vector<char> &schema_path,
                    const std::vector<std::vector<char>> &columns_list, std::shared_ptr<DatasetCache> cache);

  ~RandomDataDataset() = default;
};

/// \brief Function to create a RandomDataset
/// \param[in] total_rows Number of rows for the dataset to generate (default=0, number of rows is random)
/// \param[in] schema SchemaObj to set column type, data type and data shape
/// \param[in] columns_list List of columns to be read (default={}, read all columns)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
template <typename T = std::shared_ptr<SchemaObj>>
std::shared_ptr<RandomDataDataset> RandomData(const int32_t &total_rows = 0, const T &schema = nullptr,
                                              const std::vector<std::string> &columns_list = {},
                                              const std::shared_ptr<DatasetCache> &cache = nullptr) {
  std::shared_ptr<RandomDataDataset> ds;
  if constexpr (std::is_same<T, std::nullptr_t>::value || std::is_same<T, std::shared_ptr<SchemaObj>>::value) {
    std::shared_ptr<SchemaObj> schema_obj = schema;
    ds =
      std::make_shared<RandomDataDataset>(total_rows, std::move(schema_obj), VectorStringToChar(columns_list), cache);
  } else {
    ds = std::make_shared<RandomDataDataset>(total_rows, StringToChar(schema), VectorStringToChar(columns_list), cache);
  }
  return ds;
}

class TextFileDataset : public Dataset {
 public:
  explicit TextFileDataset(const std::vector<std::vector<char>> &dataset_files, int64_t num_samples,
                           ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                           const std::shared_ptr<DatasetCache> &cache);
  ~TextFileDataset() = default;
};

/// \brief Function to create a TextFileDataset
/// \notes The generated dataset has one column ['text']
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0 means all samples.)
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///     Can be any of:
///     ShuffleMode.kFalse - No shuffling is performed.
///     ShuffleMode.kFiles - Shuffle files only.
///     ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified. (Default = 0)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current TextFileDataset
inline std::shared_ptr<TextFileDataset> TextFile(const std::vector<std::string> &dataset_files, int64_t num_samples = 0,
                                                 ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                                 int32_t shard_id = 0,
                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<TextFileDataset>(VectorStringToChar(dataset_files), num_samples, shuffle, num_shards,
                                           shard_id, cache);
}

class TFRecordDataset : public Dataset {
 public:
  TFRecordDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &schema,
                  const std::vector<std::vector<char>> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, bool shard_equal_rows, std::shared_ptr<DatasetCache> cache);

  /// \brief Constructor
  /// \note Parameter 'schema' is shared pointer to Schema object
  TFRecordDataset(const std::vector<std::vector<char>> &dataset_files, std::shared_ptr<SchemaObj> schema,
                  const std::vector<std::vector<char>> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, bool shard_equal_rows, std::shared_ptr<DatasetCache> cache);

  ~TFRecordDataset() = default;
};

/// \brief Function to create a TFRecordDataset
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] schema SchemaObj or string to schema path. (Default = nullptr, which means that the
///     meta data from the TFData file is considered the schema.)
/// \param[in] columns_list List of columns to be read. (Default = {}, read all columns)
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0 means all samples.)
///     If num_samples is 0 and numRows(parsed from schema) does not exist, read the full dataset;
///     If num_samples is 0 and numRows(parsed from schema) is greater than 0, read numRows rows;
///     If both num_samples and numRows(parsed from schema) are greater than 0, read num_samples rows.
/// \param[in] shuffle The mode for shuffling data every epoch. (Default = ShuffleMode::kGlobal)
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be specified only
///     when num_shards is also specified. (Default = 0)
/// \param[in] shard_equal_rows Get equal rows for all shards. (Default = False, number of rows of
///     each shard may be not equal)
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current TFRecordDataset
template <typename T = std::shared_ptr<SchemaObj>>
std::shared_ptr<TFRecordDataset> TFRecord(const std::vector<std::string> &dataset_files, const T &schema = nullptr,
                                          const std::vector<std::string> &columns_list = {}, int64_t num_samples = 0,
                                          ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                          int32_t shard_id = 0, bool shard_equal_rows = false,
                                          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  std::shared_ptr<TFRecordDataset> ds = nullptr;
  if constexpr (std::is_same<T, std::nullptr_t>::value || std::is_same<T, std::shared_ptr<SchemaObj>>::value) {
    std::shared_ptr<SchemaObj> schema_obj = schema;
    ds = std::make_shared<TFRecordDataset>(VectorStringToChar(dataset_files), std::move(schema_obj),
                                           VectorStringToChar(columns_list), num_samples, shuffle, num_shards, shard_id,
                                           shard_equal_rows, cache);
  } else {
    std::string schema_path = schema;
    if (!schema_path.empty()) {
      struct stat sb;
      int rc = stat(schema_path.c_str(), &sb);
      if (rc != 0) {
        return nullptr;
      }
    }
    ds = std::make_shared<TFRecordDataset>(VectorStringToChar(dataset_files), StringToChar(schema_path),
                                           VectorStringToChar(columns_list), num_samples, shuffle, num_shards, shard_id,
                                           shard_equal_rows, cache);
  }
  return ds;
}

class VOCDataset : public Dataset {
 public:
  explicit VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                      const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                      bool decode, const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);
  explicit VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                      const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                      bool decode, const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);
  explicit VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                      const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                      bool decode, const std::reference_wrapper<Sampler> sampler,
                      const std::shared_ptr<DatasetCache> &cache);
  ~VOCDataset() = default;
};

/// \brief Function to create a VOCDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                  ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///     - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection"
/// \param[in] usage The type of data list text file to be read (default = "train").
/// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
/// given,
///     a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler())
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<VOCDataset> VOC(const std::string &dataset_dir, const std::string &task = "Segmentation",
                                       const std::string &usage = "train",
                                       const std::map<std::string, int32_t> &class_indexing = {}, bool decode = false,
                                       const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<VOCDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      MapStringToChar(class_indexing), decode, sampler, cache);
}

/// \brief Function to create a VOCDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                  ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///     - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection"
/// \param[in] usage The type of data list text file to be read.
/// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<VOCDataset> VOC(const std::string &dataset_dir, const std::string &task,
                                       const std::string &usage, const std::map<std::string, int32_t> &class_indexing,
                                       bool decode, const Sampler *sampler,
                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<VOCDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      MapStringToChar(class_indexing), decode, sampler, cache);
}

/// \brief Function to create a VOCDataset
/// \notes The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                  ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///     - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection"
/// \param[in] usage The type of data list text file to be read.
/// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task
/// \param[in] decode Decode the images after reading
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<VOCDataset> VOC(const std::string &dataset_dir, const std::string &task,
                                       const std::string &usage, const std::map<std::string, int32_t> &class_indexing,
                                       bool decode, const std::reference_wrapper<Sampler> sampler,
                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<VOCDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      MapStringToChar(class_indexing), decode, sampler, cache);
}

std::shared_ptr<DatasetCache> CreateDatasetCacheCharIF(session_id_type id, uint64_t mem_sz, bool spill,
                                                       std::optional<std::vector<char>> hostname = std::nullopt,
                                                       std::optional<int32_t> port = std::nullopt,
                                                       std::optional<int32_t> num_connections = std::nullopt,
                                                       std::optional<int32_t> prefetch_sz = std::nullopt);

/// \brief Function the create a cache to be attached to a dataset
/// \param id A user assigned session id for the current pipeline.
/// \param mem_sz Size of the memory set aside for the row caching (default=0 which means unlimited,
///     note that it might bring in the risk of running out of memory on the machine).
/// \param spill Spill to disk if out of memory (default=False).
/// \param hostname optional host name (default="127.0.0.1").
/// \param port optional port (default=50052).
/// \param num_connections optional number of connections (default=12).
/// \param prefetch_sz optional prefetch size (default=20).
/// \return Shared pointer to DatasetCache. If error, nullptr is returned.
inline std::shared_ptr<DatasetCache> CreateDatasetCache(session_id_type id, uint64_t mem_sz, bool spill,
                                                        std::optional<std::string> hostname = std::nullopt,
                                                        std::optional<int32_t> port = std::nullopt,
                                                        std::optional<int32_t> num_connections = std::nullopt,
                                                        std::optional<int32_t> prefetch_sz = std::nullopt) {
  return CreateDatasetCacheCharIF(id, mem_sz, spill, OptionalStringToChar(hostname), port, num_connections,
                                  prefetch_sz);
}

/// \brief Function to create a ZipDataset
/// \notes Applies zip to the dataset
/// \param[in] datasets List of shared pointers to the datasets that we want to zip
/// \return Shared pointer to the current Dataset
inline std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  return std::make_shared<ZipDataset>(datasets);
}
}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASETS_H_
