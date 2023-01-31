/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_DATASETS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_DATASETS_H_

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
#include "include/dataset/iterator.h"
#include "nlohmann/json_fwd.hpp"
#include "include/dataset/samplers.h"
#include "include/dataset/text.h"

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
class DATASET_API Dataset : public std::enable_shared_from_this<Dataset> {
 public:
  // need friend class so they can access the children_ field
  friend class Iterator;
  friend class DataQueueNode;

  /// \brief Constructor
  Dataset();

  /// \brief Destructor
  virtual ~Dataset() = default;

  /// \brief Get the dataset size
  /// \param[in] estimate This is only supported by some of the ops and it's used to speed up the process of getting
  ///     dataset size at the expense of accuracy.
  /// \return Dataset size. If failed, return -1.
  int64_t GetDatasetSize(bool estimate = false);

  /// \brief Get the output type
  /// \return A vector contains output DataType of dataset. If failed, return an empty vector.
  std::vector<mindspore::DataType> GetOutputTypes();

  /// \brief Get the output shape
  /// \return A vector contains output TensorShape of dataset. If failed, return an empty vector.
  std::vector<std::vector<int64_t>> GetOutputShapes();

  /// \brief Get the batch size
  /// \return Batch size configuration of dataset.
  int64_t GetBatchSize();

  /// \brief Get the repeat count
  /// \return Repeat count configuration of dataset.
  int64_t GetRepeatCount();

  /// \brief Get the number of classes
  /// \return Number of classes of dataset. If failed, return -1.
  int64_t GetNumClasses();

  /// \brief Get the column names
  /// \return A vector contains all column names of dataset. If failed, return an empty vector.
  std::vector<std::string> GetColumnNames() { return VectorCharToString(GetColumnNamesCharIF()); }

  /// \brief Get the class indexing
  /// \return A map of ClassIndexing of dataset. If failed, return an empty map.
  std::vector<std::pair<std::string, std::vector<int32_t>>> GetClassIndexing() {
    return ClassIndexCharToString(GetClassIndexingCharIF());
  }

  /// \brief Function to set runtime number of workers.
  /// \param[in] num_workers The number of threads in this operation.
  /// \return Shared pointer to the original object.
  /// \par Example
  /// \code
  ///      /* Set number of workers(threads) to process the dataset in parallel */
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true);
  ///      ds = ds->SetNumWorkers(16);
  /// \endcode
  std::shared_ptr<Dataset> SetNumWorkers(int32_t num_workers);

  /// \brief A Function to create an PullBasedIterator over the Dataset.
  /// \return Shared pointer to the Iterator.
  /// \par Example
  /// \code
  ///      /* dataset is an instance of Dataset object */
  ///      std::shared_ptr<Iterator> = dataset->CreatePullBasedIterator();
  ///      std::unordered_map<std::string, mindspore::MSTensor> row;
  ///      iter->GetNextRow(&row);
  /// \endcode
  std::shared_ptr<PullIterator> CreatePullBasedIterator();

  /// \brief Function to create an Iterator over the Dataset pipeline.
  /// \param[in] num_epochs Number of epochs to run through the pipeline (default=-1, which means infinite epochs).
  ///     An empty row is returned at the end of each epoch.
  /// \return Shared pointer to the Iterator.
  /// \par Example
  /// \code
  ///      /* dataset is an instance of Dataset object */
  ///      std::shared_ptr<Iterator> = dataset->CreateIterator();
  ///      std::unordered_map<std::string, mindspore::MSTensor> row;
  ///      iter->GetNextRow(&row);
  /// \endcode
  std::shared_ptr<Iterator> CreateIterator(int32_t num_epochs = -1) { return CreateIteratorCharIF(num_epochs); }

  /// \brief Function to transfer data through a device.
  /// \note If device is Ascend, features of data will be transferred one by one. The limitation
  ///     of data transmission per time is 256M.
  /// \param[in] queue_name Channel name (default="", create new unique name).
  /// \param[in] device_type Type of device (default="", get from MSContext).
  /// \param[in] device_id id of device (default=1, get from MSContext).
  /// \param[in] num_epochs Number of epochs (default=-1, infinite epochs).
  /// \param[in] send_epoch_end Whether to send end of sequence to device or not (default=true).
  /// \param[in] total_batches Number of batches to be sent to the device (default=0, all data).
  /// \param[in] create_data_info_queue Whether to create queue which stores types and shapes
  ///     of data or not (default=false).
  /// \return Returns true if no error encountered else false.
  bool DeviceQueue(const std::string &queue_name = "", const std::string &device_type = "", int32_t device_id = 0,
                   int32_t num_epochs = -1, bool send_epoch_end = true, int32_t total_batches = 0,
                   bool create_data_info_queue = false) {
    return DeviceQueueCharIF(StringToChar(queue_name), StringToChar(device_type), device_id, num_epochs, send_epoch_end,
                             total_batches, create_data_info_queue);
  }

  /// \brief Function to create a Saver to save the dynamic data processed by the dataset pipeline.
  /// \note Usage restrictions:
  ///     1. Supported dataset formats: 'mindrecord' only.
  ///     2. To save the samples in order, set dataset's shuffle to false and num_files to 1.
  ///     3. Before calling the function, do not use batch operation, repeat operation or data augmentation operations
  ///        with random attribute in map operation.
  ///     4. Mindrecord does not support bool, uint64, multi-dimensional uint8(drop dimension) nor
  ///        multi-dimensional string.
  /// \param[in] dataset_path Path to dataset file.
  /// \param[in] num_files Number of dataset files (default=1).
  /// \param[in] dataset_type Dataset format (default="mindrecord").
  /// \return Returns true if no error encountered else false.
  /// \par Example
  /// \code
  ///      /* Create a dataset and save its data into MindRecord */
  ///      std::string folder_path = "/path/to/cifar_dataset";
  ///      std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<SequentialSampler>(0, 10));
  ///      std::string save_file = "Cifar10Data.mindrecord";
  ///      bool rc = ds->Save(save_file);
  /// \endcode
  bool Save(const std::string &dataset_path, int32_t num_files = 1, const std::string &dataset_type = "mindrecord") {
    return SaveCharIF(StringToChar(dataset_path), num_files, StringToChar(dataset_type));
  }

  /// \brief Function to create a BatchDataset.
  /// \note Combines batch_size number of consecutive rows into batches.
  /// \param[in] batch_size The number of rows each batch is created with.
  /// \param[in] drop_remainder Determines whether or not to drop the last possibly incomplete
  ///     batch. If true, and if there are less than batch_size rows
  ///     available to make the last batch, then those rows will
  ///     be dropped and not propagated to the next node.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Create a dataset where every 100 rows is combined into a batch */
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true);
  ///      ds = ds->Batch(100, true);
  /// \endcode
  std::shared_ptr<BatchDataset> Batch(int32_t batch_size, bool drop_remainder = false);

  /// \brief Function to create a BucketBatchByLengthDataset.
  /// \note Bucket elements according to their lengths. Each bucket will be padded and batched when
  ///    they are full.
  /// \param[in] column_names Columns passed to element_length_function.
  /// \param[in] bucket_boundaries A list consisting of the upper boundaries of the buckets.
  ///    Must be strictly increasing. If there are n boundaries, n+1 buckets are created: One bucket for
  ///    [0, bucket_boundaries[0]), one bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
  ///    0<i<n, and one bucket for [bucket_boundaries[n-1], inf).
  /// \param[in] bucket_batch_sizes A list consisting of the batch sizes for each bucket.
  ///    Must contain elements equal to the size of bucket_boundaries + 1.
  /// \param[in] element_length_function A function pointer that takes in MSTensorVec and outputs a MSTensorVec.
  ///    The output must contain a single tensor containing a single int32_t. If no value is provided,
  ///    then size of column_names must be 1, and the size of the first dimension of that column will be taken
  ///    as the length (default=nullptr).
  /// \param[in] pad_info Represents how to batch each column. The key corresponds to the column name, the value must
  ///    be a tuple of 2 elements.  The first element corresponds to the shape to pad to, and the second element
  ///    corresponds to the value to pad with. If a column is not specified, then that column will be padded to the
  ///    longest in the current batch, and 0 will be used as the padding value. Any unspecified dimensions will be
  ///    padded to the longest in the current batch, unless if pad_to_bucket_boundary is true. If no padding is
  ///    wanted, set pad_info to empty map (default=empty map).
  /// \param[in] pad_to_bucket_boundary If true, will pad each unspecified dimension in pad_info to the
  ///    bucket_boundary minus 1. If there are any elements that fall into the last bucket,
  ///    an error will occur (default=false).
  /// \param[in] drop_remainder If true, will drop the last batch for each bucket if it is not a full batch
  ///    (default=false).
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Bucket elements according to their lengths */
  ///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      ds = ds->BucketBatchByLength({"image"}, {1, 2, 3}, {4, 5, 6, 7});
  /// \endcode
  std::shared_ptr<BucketBatchByLengthDataset> BucketBatchByLength(
    const std::vector<std::string> &column_names, const std::vector<int32_t> &bucket_boundaries,
    const std::vector<int32_t> &bucket_batch_sizes,
    const std::function<MSTensorVec(MSTensorVec)> &element_length_function = nullptr,
    const std::map<std::string, std::pair<std::vector<int64_t>, MSTensor>> &pad_info = {},
    bool pad_to_bucket_boundary = false, bool drop_remainder = false) {
    return std::make_shared<BucketBatchByLengthDataset>(
      shared_from_this(), VectorStringToChar(column_names), bucket_boundaries, bucket_batch_sizes,
      element_length_function, MapStringToChar(pad_info), pad_to_bucket_boundary, drop_remainder);
  }

  /// \brief Function to create a SentencePieceVocab from source dataset.
  /// \note Build a SentencePieceVocab from a dataset.
  /// \param[in] col_names Column names to get words from. It can be a vector of column names.
  /// \param[in] vocab_size Vocabulary size.
  /// \param[in] character_coverage Percentage of characters covered by the model, must be between
  ///     0.98 and 1.0 Good defaults are: 0.9995 for languages with rich character sets like
  ///     Japanese or Chinese character sets, and 1.0 for other languages with small character sets.
  /// \param[in] model_type Model type. Choose from unigram (default), bpe, char, or word.
  ///     The input sentence must be pretokenized when using word type.
  /// \param[in] params A vector contains more option parameters of sentencepiece library.
  /// \return Shared pointer to the SentencePieceVocab.
  /// \par Example
  /// \code
  ///      /* Build a SentencePieceVocab from TextFile dataset */
  ///      std::string vocab_file = "/path/to/txtfile";
  ///      std::shared_ptr<Dataset> ds_vocab = TextFile({vocab_file}, 0, ShuffleMode::kFalse);
  ///      std::shared_ptr<SentencePieceVocab> vocab =
  ///          ds_vocab->BuildSentencePieceVocab({}, 5000, 0.9995, SentencePieceModel::kUnigram, {});
  /// \endcode
  std::shared_ptr<SentencePieceVocab> BuildSentencePieceVocab(
    const std::vector<std::string> &col_names, int32_t vocab_size, float character_coverage,
    SentencePieceModel model_type, const std::unordered_map<std::string, std::string> &params) {
    return BuildSentencePieceVocabCharIF(VectorStringToChar(col_names), vocab_size, character_coverage, model_type,
                                         UnorderedMapStringToChar(params));
  }

  /// \brief Function to create a Vocab from source dataset.
  /// \note Build a vocab from a dataset. This would collect all the unique words in a dataset and return a vocab
  ///    which contains top_k most frequent words (if top_k is specified).
  /// \param[in] columns Column names to get words from. It can be a vector of column names.
  /// \param[in] freq_range A tuple of integers (min_frequency, max_frequency). Words within the frequency
  ///    range would be kept. 0 <= min_frequency <= max_frequency <= total_words. min_frequency/max_frequency
  ///    can be set to default, which corresponds to 0/total_words separately.
  /// \param[in] top_k Number of words to be built into vocab. top_k most frequent words are
  ///    taken. The top_k is taken after freq_range. If not enough top_k, all words will be taken.
  /// \param[in] special_tokens A list of strings, each one is a special token.
  /// \param[in] special_first Whether special_tokens will be prepended/appended to vocab, If special_tokens
  ///    is specified and special_first is set to default, special_tokens will be prepended.
  /// \return Shared pointer to the Vocab.
  /// \par Example
  /// \code
  ///      /* Build a Vocab from TextFile dataset */
  ///      std::string vocab_file = "/path/to/txtfile";
  ///      std::shared_ptr<Dataset> ds = TextFile({vocab_file}, 0, ShuffleMode::kFalse);
  ///      std::shared_ptr<Vocab> vocab = ds->BuildVocab();
  /// \endcode
  std::shared_ptr<Vocab> BuildVocab(const std::vector<std::string> &columns = {},
                                    const std::pair<int64_t, int64_t> &freq_range = {0, kDeMaxFreq},
                                    int64_t top_k = kDeMaxTopk, const std::vector<std::string> &special_tokens = {},
                                    bool special_first = true) {
    return BuildVocabCharIF(VectorStringToChar(columns), freq_range, top_k, VectorStringToChar(special_tokens),
                            special_first);
  }

  /// \brief Function to create a ConcatDataset.
  /// \note Concat the datasets in the input.
  /// \param[in] datasets List of shared pointers to the dataset that should be concatenated together.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Create a dataset by concatenating dataset_1 and dataset_2 with "+" operator */
  ///      std::shared_ptr<Dataset> dataset = dataset_1 + dataset_2;
  ///      /* Create a dataset by concatenating dataset_1 and dataset_2 with concat operation */
  ///      std::shared_ptr<Dataset> dataset = dataset_1->Concat({dataset_2});
  /// \endcode
  std::shared_ptr<ConcatDataset> Concat(const std::vector<std::shared_ptr<Dataset>> &datasets) {
    std::vector<std::shared_ptr<Dataset>> all_datasets{shared_from_this()};
    all_datasets.insert(std::end(all_datasets), std::begin(datasets), std::end(datasets));
    return std::make_shared<ConcatDataset>(all_datasets);
  }

  /// \brief Function to filter dataset by predicate.
  /// \note If input_columns is not provided or empty, all columns will be used.
  /// \param[in] predicate Function callable which returns a boolean value. If false then filter the element.
  /// \param[in] input_columns List of names of the input columns to filter.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Define a predicate function */
  ///      MSTensorVec Predicate1(MSTensorVec in) {
  ///        // Return true if input is equal to 3
  ///        uint64_t input_value;
  ///        TensorRow input = VecToRow(in);
  ///        (void)input.at(0)->GetItemAt(&input_value, {0});
  ///        bool result = (input_value == 3);
  ///        // Convert from boolean to TensorRow
  ///        TensorRow output;
  ///        std::shared_ptr<Tensor> out;
  ///        (void)Tensor::CreateEmpty(mindspore::dataset::TensorShape({}),
  ///                                  mindspore::dataset::DataType(mindspore::dataset::DataType::Type::DE_BOOL), &out);
  ///        (void)out->SetItemAt({}, result);
  ///        output.push_back(out);
  ///        return RowToVec(output);
  ///      }
  ///
  ///      /* Apply predicate function for datase */
  ///      std::shared_ptr<Dataset> ds = ds->Filter(Predicate1, {"label"});
  /// \endcode
  std::shared_ptr<FilterDataset> Filter(const std::function<MSTensorVec(MSTensorVec)> &predicate,
                                        const std::vector<std::string> &input_columns = {}) {
    return std::make_shared<FilterDataset>(shared_from_this(), predicate, VectorStringToChar(input_columns));
  }

  /// \brief Function to create a MapDataset.
  /// \note Applies each operation in operations to this dataset.
  /// \param[in] operations Vector of raw pointers to TensorTransform objects to be applied on the dataset. Operations
  ///     are applied in the order they appear in this list.
  /// \param[in] input_columns Vector of the names of the columns that will be passed to the first
  ///     operation as input. The size of this list must match the number of
  ///     input columns expected by the first operation. The default input_columns
  ///     is the first column.
  /// \param[in] output_columns Vector of names assigned to the columns outputted by the last operation.
  ///     This parameter is mandatory if len(input_columns) != len(output_columns).
  ///     The size of this list must match the number of output columns of the
  ///     last operation. The default output_columns will have the same
  ///     name as the input columns, i.e., the columns will be replaced.
  /// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
  /// \param[in] callbacks List of Dataset callbacks to be called.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///     // Create objects for the tensor ops
  ///     std::shared_ptr<TensorTransform> decode_op = std::make_shared<vision::Decode>(true);
  ///     std::shared_ptr<TensorTransform> random_color_op = std::make_shared<vision::RandomColor>(0.0, 0.0);
  ///
  ///     /* 1) Simple map example */
  ///     // Apply decode_op on column "image". This column will be replaced by the outputted
  ///     // column of decode_op.
  ///     dataset = dataset->Map({decode_op}, {"image"});
  ///
  ///     // Decode and rename column "image" to "decoded_image".
  ///     dataset = dataset->Map({decode_op}, {"image"}, {"decoded_image"});
  ///
  ///    /* 2) Map example with more than one operation */
  ///    // Create a dataset where the images are decoded, then randomly color jittered.
  ///    // decode_op takes column "image" as input and outputs one column. The column
  ///    // outputted by decode_op is passed as input to random_jitter_op.
  ///    // random_jitter_op will output one column. Column "image" will be replaced by
  ///    // the column outputted by random_jitter_op (the very last operation). All other
  ///    // columns are unchanged.
  ///    dataset = dataset->Map({decode_op, random_jitter_op}, {"image"})
  /// \endcode
  std::shared_ptr<MapDataset> Map(const std::vector<TensorTransform *> &operations,
                                  const std::vector<std::string> &input_columns = {},
                                  const std::vector<std::string> &output_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  const std::vector<std::shared_ptr<DSCallback>> &callbacks = {}) {
    std::vector<std::shared_ptr<TensorOperation>> transform_ops;
    (void)std::transform(
      operations.begin(), operations.end(), std::back_inserter(transform_ops),
      [](TensorTransform *op) -> std::shared_ptr<TensorOperation> { return op != nullptr ? op->Parse() : nullptr; });
    return std::make_shared<MapDataset>(shared_from_this(), transform_ops, VectorStringToChar(input_columns),
                                        VectorStringToChar(output_columns), cache, callbacks);
  }

  /// \brief Function to create a MapDataset.
  /// \note Applies each operation in operations to this dataset.
  /// \param[in] operations Vector of shared pointers to TensorTransform objects to be applied on the dataset.
  ///     Operations are applied in the order they appear in this list.
  /// \param[in] input_columns Vector of the names of the columns that will be passed to the first
  ///     operation as input. The size of this list must match the number of
  ///     input columns expected by the first operation. The default input_columns
  ///     is the first column.
  /// \param[in] output_columns Vector of names assigned to the columns outputted by the last operation.
  ///     This parameter is mandatory if len(input_columns) != len(output_columns).
  ///     The size of this list must match the number of output columns of the
  ///     last operation. The default output_columns will have the same
  ///     name as the input columns, i.e., the columns will be replaced.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] callbacks List of Dataset callbacks to be called.
  /// \return Shared pointer to the current Dataset.
  std::shared_ptr<MapDataset> Map(const std::vector<std::shared_ptr<TensorTransform>> &operations,
                                  const std::vector<std::string> &input_columns = {},
                                  const std::vector<std::string> &output_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  const std::vector<std::shared_ptr<DSCallback>> &callbacks = {}) {
    std::vector<std::shared_ptr<TensorOperation>> transform_ops;
    (void)std::transform(operations.begin(), operations.end(), std::back_inserter(transform_ops),
                         [](const std::shared_ptr<TensorTransform> &op) -> std::shared_ptr<TensorOperation> {
                           return op != nullptr ? op->Parse() : nullptr;
                         });
    return std::make_shared<MapDataset>(shared_from_this(), transform_ops, VectorStringToChar(input_columns),
                                        VectorStringToChar(output_columns), cache, callbacks);
  }

  /// \brief Function to create a MapDataset.
  /// \note Applies each operation in operations to this dataset.
  /// \param[in] operations Vector of TensorTransform objects to be applied on the dataset. Operations are applied in
  ///     the order they appear in this list.
  /// \param[in] input_columns Vector of the names of the columns that will be passed to the first
  ///     operation as input. The size of this list must match the number of
  ///     input columns expected by the first operation. The default input_columns
  ///     is the first column.
  /// \param[in] output_columns Vector of names assigned to the columns outputted by the last operation.
  ///     This parameter is mandatory if len(input_columns) != len(output_columns).
  ///     The size of this list must match the number of output columns of the
  ///     last operation. The default output_columns will have the same
  ///     name as the input columns, i.e., the columns will be replaced.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] callbacks List of Dataset callbacks to be called.
  /// \return Shared pointer to the current Dataset.
  std::shared_ptr<MapDataset> Map(const std::vector<std::reference_wrapper<TensorTransform>> &operations,
                                  const std::vector<std::string> &input_columns = {},
                                  const std::vector<std::string> &output_columns = {},
                                  const std::shared_ptr<DatasetCache> &cache = nullptr,
                                  const std::vector<std::shared_ptr<DSCallback>> &callbacks = {}) {
    std::vector<std::shared_ptr<TensorOperation>> transform_ops;
    (void)std::transform(operations.begin(), operations.end(), std::back_inserter(transform_ops),
                         [](TensorTransform &op) -> std::shared_ptr<TensorOperation> { return op.Parse(); });
    return std::make_shared<MapDataset>(shared_from_this(), transform_ops, VectorStringToChar(input_columns),
                                        VectorStringToChar(output_columns), cache, callbacks);
  }

  /// \brief Function to create a Project Dataset.
  /// \note Applies project to the dataset.
  /// \param[in] columns The name of columns to project.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Reorder the original column names in dataset */
  ///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      ds = ds->Project({"label", "image"});
  /// \endcode
  std::shared_ptr<ProjectDataset> Project(const std::vector<std::string> &columns) {
    return std::make_shared<ProjectDataset>(shared_from_this(), VectorStringToChar(columns));
  }

  /// \brief Function to create a Rename Dataset.
  /// \note Renames the columns in the input dataset.
  /// \param[in] input_columns List of the input columns to rename.
  /// \param[in] output_columns List of the output columns.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Rename the original column names in dataset */
  ///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      ds = ds->Rename({"image", "label"}, {"image_output", "label_output"});
  /// \endcode
  std::shared_ptr<RenameDataset> Rename(const std::vector<std::string> &input_columns,
                                        const std::vector<std::string> &output_columns) {
    return std::make_shared<RenameDataset>(shared_from_this(), VectorStringToChar(input_columns),
                                           VectorStringToChar(output_columns));
  }
  /// \brief Function to create a RepeatDataset.
  /// \note Repeats this dataset count times. Repeat indefinitely if count is -1.
  /// \param[in] count Number of times the dataset should be repeated.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Create a dataset where the dataset is repeated for 50 epochs */
  ///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      ds = ds->Repeat(50);
  /// \endcode
  std::shared_ptr<RepeatDataset> Repeat(int32_t count = -1) {
    return std::make_shared<RepeatDataset>(shared_from_this(), count);
  }

  /// \brief Function to create a Shuffle Dataset.
  /// \note Randomly shuffles the rows of this dataset.
  /// \param[in] buffer_size The size of the buffer (must be larger than 1) for shuffling
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Create a shuffled dataset using a shuffle buffer of size 4 */
  ///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      ds = ds->Shuffle(4);
  /// \endcode
  std::shared_ptr<ShuffleDataset> Shuffle(int32_t buffer_size) {
    return std::make_shared<ShuffleDataset>(shared_from_this(), buffer_size);
  }

  /// \brief Function to create a SkipDataset.
  /// \note Skips count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be skipped.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Create a dataset which skips first 3 elements from data */
  ///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      ds = ds->Skip(3);
  /// \endcode
  std::shared_ptr<SkipDataset> Skip(int32_t count) { return std::make_shared<SkipDataset>(shared_from_this(), count); }

  /// \brief Function to create a TakeDataset.
  /// \note Takes count elements in this dataset.
  /// \param[in] count Number of elements the dataset to be taken.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Create a dataset where the dataset includes 50 elements. */
  ///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      ds = ds->Take(50);
  /// \endcode
  std::shared_ptr<TakeDataset> Take(int32_t count = -1) {
    return std::make_shared<TakeDataset>(shared_from_this(), count);
  }

  /// \brief Function to create a Zip Dataset.
  /// \note Applies zip to the dataset.
  /// \param[in] datasets A list of shared pointers to the datasets that we want to zip.
  /// \return Shared pointer to the current Dataset.
  /// \par Example
  /// \code
  ///      /* Create a dataset which is the combination of dataset and dataset_1 */
  ///      std::shared_ptr<Dataset> ds1 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  ///      std::shared_ptr<Dataset> ds2 = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
  ///      std::shared_ptr<Dataset> ds = ds->Zip({ds1, ds2});
  /// \endcode
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
  std::shared_ptr<Iterator> CreateIteratorCharIF(int32_t num_epochs);

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

/// \class SchemaObj
/// \brief A schema object to set column type, data type and data shape.
class DATASET_API SchemaObj {
 public:
  /// \brief Constructor
  explicit SchemaObj(const std::string &schema_file = "") : SchemaObj(StringToChar(schema_file)) {}

  /// \brief Destructor
  ~SchemaObj() = default;

  /// \brief SchemaObj Init function.
  /// \return bool true if schema initialization is successful.
  Status Init();

  /// \brief Add new column to the schema with unknown shape of rank 1.
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(mindspore::DataType).
  /// \return Status code.
  Status add_column(const std::string &name, mindspore::DataType ms_type) {
    return add_column_char(StringToChar(name), ms_type);
  }

  /// \brief Add new column to the schema with unknown shape of rank 1.
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(std::string).
  /// \param[in] shape Shape of the column.
  /// \return Status code.
  Status add_column(const std::string &name, const std::string &ms_type) {
    return add_column_char(StringToChar(name), StringToChar(ms_type));
  }

  /// \brief Add new column to the schema.
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(mindspore::DataType).
  /// \param[in] shape Shape of the column.
  /// \return Status code.
  Status add_column(const std::string &name, mindspore::DataType ms_type, const std::vector<int32_t> &shape) {
    return add_column_char(StringToChar(name), ms_type, shape);
  }

  /// \brief Add new column to the schema.
  /// \param[in] name Name of the column.
  /// \param[in] ms_type Data type of the column(std::string).
  /// \param[in] shape Shape of the column.
  /// \return Status code.
  Status add_column(const std::string &name, const std::string &ms_type, const std::vector<int32_t> &shape) {
    return add_column_char(StringToChar(name), StringToChar(ms_type), shape);
  }

  /// \brief Get a JSON string of the schema.
  /// \return JSON string of the schema.
  std::string to_json() { return CharToString(to_json_char()); }

  /// \brief Serialize schema config to JSON.
  /// \return Status code.
  Status schema_to_json(nlohmann::json *out_json);

  /// \brief Get a JSON string of the schema.
  /// \return JSON string of the schema.
  std::string to_string() { return to_json(); }

  /// \brief Set a new value to dataset_type.
  /// \param[in] dataset_type Data type of the schema.
  void set_dataset_type(const std::string &dataset_type);

  /// \brief Set a new value to num_rows.
  /// \param[in] dataset_type Number of rows of the schema.
  void set_num_rows(int32_t num_rows);

  /// \brief Get the current num_rows
  /// \return Number of rows.
  int32_t get_num_rows() const;

  /// \brief Get schema file from JSON file.
  /// \param[in] json_obj parsed JSON object.
  /// \return Status code.
  Status from_json(nlohmann::json json_obj);

  /// \brief Get schema file from JSON file.
  /// \param[in] json_string Name of JSON file to be parsed.
  /// \return Status code.
  Status FromJSONString(const std::string &json_string) { return FromJSONStringCharIF(StringToChar(json_string)); }

  /// \brief Parse and add column information.
  /// \param[in] json_string Name of JSON string for column dataset attribute information, decoded from schema file.
  /// \return Status code.
  Status ParseColumnString(const std::string &json_string) {
    return ParseColumnStringCharIF(StringToChar(json_string));
  }

 private:
  /// \brief Parse the columns and add them to columns.
  /// \param[in] columns Dataset attribution information, decoded from schema file.
  ///    Support both nlohmann::json::value_t::array and nlohmann::json::value_t::onject.
  /// \return Status code.
  Status parse_column(nlohmann::json columns);

  // Char constructor of SchemaObj
  explicit SchemaObj(const std::vector<char> &schema_file);

  // Char interface of add_column
  Status add_column_char(const std::vector<char> &name, mindspore::DataType ms_type);

  Status add_column_char(const std::vector<char> &name, const std::vector<char> &ms_type);

  Status add_column_char(const std::vector<char> &name, mindspore::DataType ms_type, const std::vector<int32_t> &shape);

  Status add_column_char(const std::vector<char> &name, const std::vector<char> &ms_type,
                         const std::vector<int32_t> &shape);

  // Char interface of to_json
  std::vector<char> to_json_char();

  // Char interface of FromJSONString
  Status FromJSONStringCharIF(const std::vector<char> &json_string);

  // Char interface of ParseColumnString
  Status ParseColumnStringCharIF(const std::vector<char> &json_string);

  struct Data;
  std::shared_ptr<Data> data_;
};

/// \class BatchDataset
/// \brief The result of applying Batch operation to the input dataset.
class DATASET_API BatchDataset : public Dataset {
 public:
  /// \brief Constructor of BatchDataset.
  /// \note Combines batch_size number of consecutive rows into batches.
  /// \param[in] input The dataset which need to apply batch operation.
  /// \param[in] batch_size The number of rows each batch is created with.
  /// \param[in] drop_remainder Determines whether or not to drop the last possibly incomplete
  ///     batch. If true, and if there are less than batch_size rows
  ///     available to make the last batch, then those rows will
  ///     be dropped and not propagated to the next node.
  BatchDataset(const std::shared_ptr<Dataset> &input, int32_t batch_size, bool drop_remainder = false);

  /// \brief Destructor of BatchDataset.
  ~BatchDataset() override = default;
};

/// \class BucketBatchByLengthDataset
/// \brief The result of applying BucketBatchByLength operation to the input dataset.
class DATASET_API BucketBatchByLengthDataset : public Dataset {
 public:
  /// \brief Constructor of BucketBatchByLengthDataset.
  /// \note Bucket elements according to their lengths. Each bucket will be padded and batched when
  ///    they are full.
  /// \param[in] input The dataset which need to apply bucket batch by length operation.
  /// \param[in] column_names Columns passed to element_length_function.
  /// \param[in] bucket_boundaries A list consisting of the upper boundaries of the buckets.
  ///    Must be strictly increasing. If there are n boundaries, n+1 buckets are created: One bucket for
  ///    [0, bucket_boundaries[0]), one bucket for [bucket_boundaries[i], bucket_boundaries[i+1]) for each
  ///    0<i<n, and one bucket for [bucket_boundaries[n-1], inf).
  /// \param[in] bucket_batch_sizes A list consisting of the batch sizes for each bucket.
  ///    Must contain elements equal to the size of bucket_boundaries + 1.
  /// \param[in] element_length_function A function pointer that takes in MSTensorVec and outputs a MSTensorVec.
  ///    The output must contain a single tensor containing a single int32_t. If no value is provided,
  ///    then size of column_names must be 1, and the size of the first dimension of that column will be taken
  ///    as the length (default=nullptr).
  /// \param[in] pad_info Represents how to batch each column. The key corresponds to the column name, the value must
  ///    be a tuple of 2 elements.  The first element corresponds to the shape to pad to, and the second element
  ///    corresponds to the value to pad with. If a column is not specified, then that column will be padded to the
  ///    longest in the current batch, and 0 will be used as the padding value. Any unspecified dimensions will be
  ///    padded to the longest in the current batch, unless if pad_to_bucket_boundary is true. If no padding is
  ///    wanted, set pad_info to empty map (default=empty map).
  /// \param[in] pad_to_bucket_boundary If true, will pad each unspecified dimension in pad_info to the
  ///    bucket_boundary minus 1. If there are any elements that fall into the last bucket,
  ///    an error will occur (default=false).
  /// \param[in] drop_remainder If true, will drop the last batch for each bucket if it is not a full batch
  ///    (default=false).
  BucketBatchByLengthDataset(
    const std::shared_ptr<Dataset> &input, const std::vector<std::vector<char>> &column_names,
    const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
    const std::function<MSTensorVec(MSTensorVec)> &element_length_function = nullptr,
    const std::map<std::vector<char>, std::pair<std::vector<int64_t>, MSTensor>> &pad_info = {},
    bool pad_to_bucket_boundary = false, bool drop_remainder = false);

  /// \brief Destructor of BucketBatchByLengthDataset.
  ~BucketBatchByLengthDataset() override = default;
};

/// \class ConcatDataset
/// \brief The result of applying Concat operation to the input Dataset.
class DATASET_API ConcatDataset : public Dataset {
 public:
  /// \brief Constructor of ConcatDataset.
  /// \note Concat the datasets in the input.
  /// \param[in] input List of shared pointers to the dataset that should be concatenated together.
  explicit ConcatDataset(const std::vector<std::shared_ptr<Dataset>> &datasets);

  /// \brief Destructor of ConcatDataset.
  ~ConcatDataset() override = default;
};

/// \class FilterDataset
/// \brief The result of applying filter predicate to the input Dataset.
class DATASET_API FilterDataset : public Dataset {
 public:
  /// \brief Constructor of FilterDataset.
  /// \note If input_columns is not provided or empty, all columns will be used.
  /// \param[in] input The dataset which need to apply filter operation.
  /// \param[in] predicate Function callable which returns a boolean value. If false then filter the element.
  /// \param[in] input_columns List of names of the input columns to filter.
  FilterDataset(const std::shared_ptr<Dataset> &input, const std::function<MSTensorVec(MSTensorVec)> &predicate,
                const std::vector<std::vector<char>> &input_columns);

  /// \brief Destructor of FilterDataset.
  ~FilterDataset() override = default;
};

/// \class MapDataset
/// \brief The result of applying the Map operation to the input Dataset.
class DATASET_API MapDataset : public Dataset {
 public:
  /// \brief Constructor of MapDataset.
  /// \note Applies each operation in operations to this dataset.
  /// \param[in] input The dataset which need to apply map operation.
  /// \param[in] operations Vector of raw pointers to TensorTransform objects to be applied on the dataset. Operations
  ///     are applied in the order they appear in this list.
  /// \param[in] input_columns Vector of the names of the columns that will be passed to the first
  ///     operation as input. The size of this list must match the number of
  ///     input columns expected by the first operation. The default input_columns
  ///     is the first column.
  /// \param[in] output_columns Vector of names assigned to the columns outputted by the last operation.
  ///     This parameter is mandatory if len(input_columns) != len(output_columns).
  ///     The size of this list must match the number of output columns of the
  ///     last operation. The default output_columns will have the same
  ///     name as the input columns, i.e., the columns will be replaced.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] callbacks List of Dataset callbacks to be called.
  MapDataset(const std::shared_ptr<Dataset> &input, const std::vector<std::shared_ptr<TensorOperation>> &operations,
             const std::vector<std::vector<char>> &input_columns, const std::vector<std::vector<char>> &output_columns,
             const std::shared_ptr<DatasetCache> &cache, const std::vector<std::shared_ptr<DSCallback>> &callbacks);

  /// \brief Destructor of MapDataset.
  ~MapDataset() override = default;
};

/// \class ProjectDataset
/// \brief The result of applying the Project operation to the input Dataset.
class DATASET_API ProjectDataset : public Dataset {
 public:
  /// \brief Constructor of ProjectDataset.
  /// \note Applies project to the dataset.
  /// \param[in] input The dataset which need to apply project operation.
  /// \param[in] columns The name of columns to project.
  ProjectDataset(const std::shared_ptr<Dataset> &input, const std::vector<std::vector<char>> &columns);

  /// \brief Destructor of ProjectDataset.
  ~ProjectDataset() override = default;
};

/// \class RenameDataset
/// \brief The result of applying the Rename operation to the input Dataset.
class DATASET_API RenameDataset : public Dataset {
 public:
  /// \brief Constructor of RenameDataset.
  /// \note Renames the columns in the input dataset.
  /// \param[in] input The dataset which need to apply rename operation.
  /// \param[in] input_columns List of the input columns to rename.
  /// \param[in] output_columns List of the output columns.
  RenameDataset(const std::shared_ptr<Dataset> &input, const std::vector<std::vector<char>> &input_columns,
                const std::vector<std::vector<char>> &output_columns);

  /// \brief Destructor of RenameDataset.
  ~RenameDataset() override = default;
};

/// \class RepeatDataset
/// \brief The result of applying the Repeat operation to the input Dataset.
class DATASET_API RepeatDataset : public Dataset {
 public:
  /// \brief Constructor of RepeatDataset.
  /// \note Repeats this dataset count times. Repeat indefinitely if count is -1.
  /// \param[in] input The dataset which need to apply repeat operation.
  /// \param[in] count Number of times the dataset should be repeated.
  RepeatDataset(const std::shared_ptr<Dataset> &input, int32_t count);

  /// \brief Destructor of RepeatDataset.
  ~RepeatDataset() override = default;
};

/// \class ShuffleDataset
/// \brief The result of applying the Shuffle operation to the input Dataset.
class DATASET_API ShuffleDataset : public Dataset {
 public:
  /// \brief Constructor of ShuffleDataset.
  /// \note Randomly shuffles the rows of this dataset.
  /// \param[in] input The dataset which need to apply shuffle operation.
  /// \param[in] buffer_size The size of the buffer (must be larger than 1) for shuffling
  ShuffleDataset(const std::shared_ptr<Dataset> &input, int32_t buffer_size);

  /// \brief Destructor of ShuffleDataset.
  ~ShuffleDataset() override = default;
};

/// \class SkipDataset
/// \brief The result of applying the Skip operation to the input Dataset.
class DATASET_API SkipDataset : public Dataset {
 public:
  /// \brief Constructor of SkipDataset.
  /// \note Skips count elements in this dataset.
  /// \param[in] input The dataset which need to apply skip operation.
  /// \param[in] count Number of elements the dataset to be skipped.
  SkipDataset(const std::shared_ptr<Dataset> &input, int32_t count);

  /// \brief Destructor of SkipDataset.
  ~SkipDataset() override = default;
};

/// \class TakeDataset
/// \brief The result of applying the Take operation to the input Dataset.
class DATASET_API TakeDataset : public Dataset {
 public:
  /// \brief Constructor of TakeDataset.
  /// \note Takes count elements in this dataset.
  /// \param[in] input The dataset which need to apply take operation.
  /// \param[in] count Number of elements the dataset to be taken.
  TakeDataset(const std::shared_ptr<Dataset> &input, int32_t count);

  /// \brief Destructor of TakeDataset.
  ~TakeDataset() override = default;
};

/// \class ZipDataset
/// \brief The result of applying the Zip operation to the input Dataset.
class DATASET_API ZipDataset : public Dataset {
 public:
  /// \brief Constructor of ZipDataset.
  /// \note Applies zip to the dataset.
  /// \param[in] inputs A list of shared pointers to the datasets that we want to zip.
  explicit ZipDataset(const std::vector<std::shared_ptr<Dataset>> &datasets);

  /// \brief Destructor of ZipDataset.
  ~ZipDataset() override = default;
};

/// \brief Function to create a SchemaObj.
/// \param[in] schema_file Path of schema file.
/// \note The reason for using this API is that std::string will be constrained by the
///    compiler option '_GLIBCXX_USE_CXX11_ABI' while char is free of this restriction.
///    Check API `mindspore::dataset::Schema` and find more usage.
/// \return Shared pointer to the current schema.
std::shared_ptr<SchemaObj> DATASET_API SchemaCharIF(const std::vector<char> &schema_file);

/// \brief Function to create a SchemaObj.
/// \param[in] schema_file Path of schema file (default = "", which means do not set the path).
/// \return Shared pointer to the current schema.
/// \par Example
/// \code
///      /* Define a schema to make RandomDataset generate specific data. */
///      std::shared_ptr<SchemaObj> schema = Schema();
///      schema->add_column("image", mindspore::DataType::kNumberTypeUInt8, {2});
///      schema->add_column("label", mindspore::DataType::kNumberTypeUInt8, {1});
///      std::shared_ptr<Dataset> ds = RandomData(50, schema);
/// \endcode
inline std::shared_ptr<SchemaObj> DATASET_API Schema(const std::string &schema_file = "") {
  return SchemaCharIF(StringToChar(schema_file));
}

/// \class AGNewsDataset
/// \brief A source dataset that reads and parses AG News datasets.
class DATASET_API AGNewsDataset : public Dataset {
 public:
  /// \brief Constructor of AGNewsDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data list csv file to be read, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  AGNewsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of AGNewsDataset.
  ~AGNewsDataset() override = default;
};

/// \brief Function to create a AGNewsDataset.
/// \note The generated dataset has three columns ['index', 'title', 'description'].
///     The index range is [1, 4].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train" or "test" (default = "all").
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0 means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch
/// (Default=ShuffleMode::kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use.(default=nullptr which means no cache is used).
/// \return Shared pointer to the AGNewsDataset.
inline std::shared_ptr<AGNewsDataset> DATASET_API AGNews(const std::string &dataset_dir,
                                                         const std::string &usage = "all", int64_t num_samples = 0,
                                                         ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                         int32_t num_shards = 1, int32_t shard_id = 0,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AGNewsDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                         num_shards, shard_id, cache);
}

/// \class AlbumDataset
/// \brief A source dataset for reading and parsing Album dataset.
class DATASET_API AlbumDataset : public Dataset {
 public:
  /// \brief Constructor of AlbumDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] data_schema Path to dataset schema file.
  /// \param[in] column_names Column names used to specify columns to load, if empty, will read all columns
  ///     (default = {}).
  /// \param[in] decode The option to decode the images in dataset (default = false).
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
               const std::vector<std::vector<char>> &column_names, bool decode, const std::shared_ptr<Sampler> &sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of AlbumDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] data_schema Path to dataset schema file.
  /// \param[in] column_names Column names used to specify columns to load.
  /// \param[in] decode The option to decode the images in dataset.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
               const std::vector<std::vector<char>> &column_names, bool decode, const Sampler *sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of AlbumDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] data_schema Path to dataset schema file.
  /// \param[in] column_names Column names used to specify columns to load.
  /// \param[in] decode The option to decode the images in dataset.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
               const std::vector<std::vector<char>> &column_names, bool decode,
               const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of AlbumDataset.
  ~AlbumDataset() override = default;
};

/// \brief Function to create an AlbumDataset.
/// \note The generated dataset is specified through setting a schema.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] data_schema Path to dataset schema file.
/// \param[in] column_names Column names used to specify columns to load, if empty, will read all columns
///     (default = {}).
/// \param[in] decode The option to decode the images in dataset (default = false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the AlbumDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/album_dataset_directory";
///      std::string schema_file = "/path/to/album_schema_file";
///      std::vector<std::string> column_names = {"image", "label", "id"};
///      std::shared_ptr<Dataset> ds = Album(folder_path, schema_file, column_names);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: As we defined before, each data dictionary owns keys "image", "label" and "id" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<AlbumDataset> DATASET_API
Album(const std::string &dataset_dir, const std::string &data_schema, const std::vector<std::string> &column_names = {},
      bool decode = false, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
      const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AlbumDataset>(StringToChar(dataset_dir), StringToChar(data_schema),
                                        VectorStringToChar(column_names), decode, sampler, cache);
}

/// \brief Function to create an AlbumDataset.
/// \note The generated dataset is specified through setting a schema.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] data_schema Path to dataset schema file.
/// \param[in] column_names Column names used to specify columns to load.
/// \param[in] decode The option to decode the images in dataset.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the AlbumDataset
inline std::shared_ptr<AlbumDataset> DATASET_API Album(const std::string &dataset_dir, const std::string &data_schema,
                                                       const std::vector<std::string> &column_names, bool decode,
                                                       const Sampler *sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AlbumDataset>(StringToChar(dataset_dir), StringToChar(data_schema),
                                        VectorStringToChar(column_names), decode, sampler, cache);
}

/// \brief Function to create an AlbumDataset.
/// \note The generated dataset is specified through setting a schema.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] data_schema Path to dataset schema file.
/// \param[in] column_names Column names used to specify columns to load.
/// \param[in] decode The option to decode the images in dataset.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the AlbumDataset.
inline std::shared_ptr<AlbumDataset> DATASET_API Album(const std::string &dataset_dir, const std::string &data_schema,
                                                       const std::vector<std::string> &column_names, bool decode,
                                                       const std::reference_wrapper<Sampler> &sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AlbumDataset>(StringToChar(dataset_dir), StringToChar(data_schema),
                                        VectorStringToChar(column_names), decode, sampler, cache);
}

/// \class AmazonReviewDataset
/// \brief A source dataset for reading and parsing Amazon Review Polarity and Amazon Review Full datasets.
class DATASET_API AmazonReviewDataset : public Dataset {
 public:
  /// \brief Constructor of AmazonReviewDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of AmazonReview, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  AmazonReviewDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                      ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                      const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of AmazonReviewDataset.
  ~AmazonReviewDataset() override = default;
};

/// \brief Function to create a AmazonReviewDataset.
/// \note This dataset includes polarity and full, which can be read according to your own needs.
///     The generated dataset has three columns ["label","title","content"]. Their types are all string.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of AmazonReview, can be "train", "test" or "all" (default="all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, which means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the AmazonReviewDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/amazon_review_dataset_directory";
///      std::shared_ptr<Dataset> ds = AmazonReview(folder_path, "test", 0, ShuffleMode::kGlobal);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In AmazonReview dataset, each data dictionary owns keys "label", "title", "content" */
///      auto title = row["title"];
/// \endcode
inline std::shared_ptr<AmazonReviewDataset> DATASET_API
AmazonReview(const std::string &dataset_dir, const std::string &usage = "all", int64_t num_samples = 0,
             ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<AmazonReviewDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                               num_shards, shard_id, cache);
}

/// \class Caltech256Dataset
/// \brief A source dataset for reading and parsing Caltech256 dataset.
class DATASET_API Caltech256Dataset : public Dataset {
 public:
  /// \brief Constructor of Caltech256Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Caltech256Dataset(const std::vector<char> &dataset_dir, bool decode, const std::shared_ptr<Sampler> &sampler,
                    const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Caltech256Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Caltech256Dataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
                    const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Caltech256Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Caltech256Dataset(const std::vector<char> &dataset_dir, bool decode, const std::reference_wrapper<Sampler> &sampler,
                    const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of Caltech256Dataset.
  ~Caltech256Dataset() override = default;
};

/// \brief Function to create a Caltech256Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the Caltech256Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_path = "/path/to/caltech256_dataset_directory";
///      std::shared_ptr<Dataset> ds = Caltech256(dataset_path, true, std::make_shared<RandomSampler>(false, 10));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In Caltech256 dataset, each data dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<Caltech256Dataset> DATASET_API
Caltech256(const std::string &dataset_dir, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
           bool decode = false, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Caltech256Dataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \brief Function to create a Caltech256Dataset
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the Caltech256Dataset.
inline std::shared_ptr<Caltech256Dataset> DATASET_API Caltech256(const std::string &dataset_dir, const Sampler *sampler,
                                                                 bool decode = false,
                                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Caltech256Dataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \brief Function to create a Caltech256Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the Caltech256Dataset.
inline std::shared_ptr<Caltech256Dataset> DATASET_API Caltech256(const std::string &dataset_dir,
                                                                 const std::reference_wrapper<Sampler> &sampler,
                                                                 bool decode = false,
                                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Caltech256Dataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \class CelebADataset
/// \brief A source dataset for reading and parsing CelebA dataset.
class DATASET_API CelebADataset : public Dataset {
 public:
  /// \brief Constructor of CelebADataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage One of "all", "train", "valid" or "test" (default = "all").
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///      given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
  /// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
  CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                const std::shared_ptr<Sampler> &sampler, bool decode, const std::set<std::vector<char>> &extensions,
                const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of CelebADataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage One of "all", "train", "valid" or "test".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                bool decode, const std::set<std::vector<char>> &extensions, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of CelebADataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage One of "all", "train", "valid" or "test".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                const std::reference_wrapper<Sampler> &sampler, bool decode,
                const std::set<std::vector<char>> &extensions, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of CelebADataset.
  ~CelebADataset() override = default;
};

/// \brief Function to create a CelebADataset.
/// \note The generated dataset has two columns ['image', 'attr'].
///      The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train", "valid" or "test" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///      given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the CelebADataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/celeba_dataset_directory";
///      std::shared_ptr<Dataset> ds = CelebA(folder_path, "all", std::make_shared<SequentialSampler>(0, 5));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In CelebA dataset, each data dictionary owns keys "image" and "attr" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<CelebADataset> DATASET_API
CelebA(const std::string &dataset_dir, const std::string &usage = "all",
       const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(), bool decode = false,
       const std::set<std::string> &extensions = {}, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CelebADataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, decode,
                                         SetStringToChar(extensions), cache);
}

/// \brief Function to create a CelebADataset.
/// \note The generated dataset has two columns ['image', 'attr'].
///      The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train", "valid" or "test".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the CelebADataset.
inline std::shared_ptr<CelebADataset> DATASET_API CelebA(const std::string &dataset_dir, const std::string &usage,
                                                         const Sampler *sampler, bool decode = false,
                                                         const std::set<std::string> &extensions = {},
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CelebADataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, decode,
                                         SetStringToChar(extensions), cache);
}

/// \brief Function to create a CelebADataset.
/// \note The generated dataset has two columns ['image', 'attr'].
///      The type of the image tensor is uint8. The attr tensor is uint32 and one hot type.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train", "valid" or "test".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] extensions Set of file extensions to be included in the dataset (default={}).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the CelebADataset.
inline std::shared_ptr<CelebADataset> DATASET_API CelebA(const std::string &dataset_dir, const std::string &usage,
                                                         const std::reference_wrapper<Sampler> &sampler,
                                                         bool decode = false,
                                                         const std::set<std::string> &extensions = {},
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CelebADataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, decode,
                                         SetStringToChar(extensions), cache);
}

/// \class Cifar10Dataset
/// \brief A source dataset for reading and parsing Cifar10 dataset.
class DATASET_API Cifar10Dataset : public Dataset {
 public:
  /// \brief Constructor of Cifar10Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of CIFAR10, can be "train", "test" or "all" (default = "all").
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                 const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Cifar10Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of CIFAR10, can be "train", "test" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                 const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Cifar10Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of CIFAR10, can be "train", "test" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                 const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of Cifar10Dataset.
  ~Cifar10Dataset() override = default;
};

/// \brief Function to create a Cifar10 Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of CIFAR10, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the Cifar10Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/cifar10_dataset_directory";
///      std::shared_ptr<Dataset> ds = Cifar10(folder_path, "all", std::make_shared<RandomSampler>(false, 10));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In CIFAR10 dataset, each data dictionary owns keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<Cifar10Dataset> DATASET_API
Cifar10(const std::string &dataset_dir, const std::string &usage = "all",
        const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
        const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar10 Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of CIFAR10, can be "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the Cifar10Dataset.
inline std::shared_ptr<Cifar10Dataset> DATASET_API Cifar10(const std::string &dataset_dir, const std::string &usage,
                                                           const Sampler *sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar10 Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of CIFAR10, can be "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the Cifar10Dataset.
inline std::shared_ptr<Cifar10Dataset> DATASET_API Cifar10(const std::string &dataset_dir, const std::string &usage,
                                                           const std::reference_wrapper<Sampler> &sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class Cifar100Dataset
/// \brief A source dataset for reading and parsing Cifar100 dataset.
class DATASET_API Cifar100Dataset : public Dataset {
 public:
  /// \brief Constructor of Cifar100Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of CIFAR100, can be "train", "test" or "all" (default = "all").
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                  const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Cifar100Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of CIFAR100, can be "train", "test" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Cifar100Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of CIFAR100, can be "train", "test" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                  const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of Cifar100Dataset.
  ~Cifar100Dataset() override = default;
};

/// \brief Function to create a Cifar100 Dataset.
/// \note The generated dataset has three columns ["image", "coarse_label", "fine_label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of CIFAR100, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the Cifar100Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/cifar100_dataset_directory";
///      std::shared_ptr<Dataset> ds = Cifar100(folder_path, "all", std::make_shared<RandomSampler>());
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In CIFAR100 dataset, each dictionary has 3 keys: "image", "fine_label" and "coarse_label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<Cifar100Dataset> DATASET_API
Cifar100(const std::string &dataset_dir, const std::string &usage = "all",
         const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar100Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar100 Dataset.
/// \note The generated dataset has three columns ["image", "coarse_label", "fine_label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of CIFAR100, can be "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the Cifar100Dataset.
inline std::shared_ptr<Cifar100Dataset> DATASET_API Cifar100(const std::string &dataset_dir, const std::string &usage,
                                                             const Sampler *sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar100Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a Cifar100 Dataset.
/// \note The generated dataset has three columns ["image", "coarse_label", "fine_label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of CIFAR100, can be "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the Cifar100Dataset.
inline std::shared_ptr<Cifar100Dataset> DATASET_API Cifar100(const std::string &dataset_dir, const std::string &usage,
                                                             const std::reference_wrapper<Sampler> &sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Cifar100Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class CityscapesDataset
/// \brief A source dataset for reading and parsing Cityscapes dataset.
class DATASET_API CityscapesDataset : public Dataset {
 public:
  /// \brief Constructor of CityscapesDataset.
  /// \param[in] dataset_dir The dataset dir to be read.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all" if
  ///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
  /// \param[in] quality_mode The quality mode of processed image. Acceptable quality_modes include
  ///     "fine" or "coarse".
  /// \param[in] task The type of task which is used to select output data. Acceptable tasks include
  ///     "instance", "semantic", "polygon" or "color".
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  CityscapesDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                    const std::vector<char> &quality_mode, const std::vector<char> &task, bool decode,
                    const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of CityscapesDataset.
  /// \param[in] dataset_dir The dataset dir to be read.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all" if
  ///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
  /// \param[in] quality_mode The quality mode of processed image. Acceptable quality_modes include
  ///     "fine" or "coarse".
  /// \param[in] task The type of task which is used to select output data. Acceptable tasks include
  ///     "instance", "semantic", "polygon" or "color".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  CityscapesDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                    const std::vector<char> &quality_mode, const std::vector<char> &task, bool decode,
                    const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of CityscapesDataset.
  /// \param[in] dataset_dir The dataset dir to be read.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all" if
  ///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
  /// \param[in] quality_mode The quality mode of processed image. Acceptable quality_modes include
  ///     "fine" or "coarse".
  /// \param[in] task The type of task which is used to select output data. Acceptable tasks include
  ///     "instance", "semantic", "polygon" or "color".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  CityscapesDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                    const std::vector<char> &quality_mode, const std::vector<char> &task, bool decode,
                    const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of CityscapesDataset.
  ~CityscapesDataset() override = default;
};

/// \brief Function to create a CityscapesDataset.
/// \note The generated dataset has two columns ["image", "task"].
/// \param[in] dataset_dir The dataset dir to be read.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all" if
///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
/// \param[in] quality_mode The quality mode of processed image. Acceptable quality_modes include
///     "fine" or "coarse".
/// \param[in] task The type of task which is used to select output data. Acceptable tasks include
///     "instance", "semantic", "polygon" or "color".
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current CityscapesDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/cityscapes_dataset_directory";
///      std::shared_ptr<Dataset> ds = Cityscapes(dataset_path, "train", "fine", "color");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In Cityscapes dataset, each data dictionary owns keys "image" and "task" */
///      auto task = row["task"];
/// \endcode
inline std::shared_ptr<CityscapesDataset> DATASET_API Cityscapes(
  const std::string &dataset_dir, const std::string &usage, const std::string &quality_mode, const std::string &task,
  bool decode = false, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
  const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CityscapesDataset>(StringToChar(dataset_dir), StringToChar(usage), StringToChar(quality_mode),
                                             StringToChar(task), decode, sampler, cache);
}

/// \brief Function to create a CityscapesDataset.
/// \note The generated dataset has two columns ["image", "task"].
/// \param[in] dataset_dir The dataset dir to be read.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all" if
///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
/// \param[in] quality_mode The quality mode of processed image. Acceptable quality_modes include
///     "fine" or "coarse".
/// \param[in] task The type of task which is used to select output data. Acceptable tasks include
///     "instance", "semantic", "polygon" or "color".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current CityscapesDataset.
inline std::shared_ptr<CityscapesDataset> DATASET_API Cityscapes(
  const std::string &dataset_dir, const std::string &usage, const std::string &quality_mode, const std::string &task,
  bool decode, const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CityscapesDataset>(StringToChar(dataset_dir), StringToChar(usage), StringToChar(quality_mode),
                                             StringToChar(task), decode, sampler, cache);
}

/// \brief Function to create a CityscapesDataset.
/// \note The generated dataset has two columns ["image", "task"].
/// \param[in] dataset_dir The dataset dir to be read.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all" if
///     quality_mode is "fine" otherwise "train", "train_extra", "val" or "all".
/// \param[in] quality_mode The quality mode of processed image. Acceptable quality_modes include
///     "fine" or "coarse".
/// \param[in] task The type of task which is used to select output data. Acceptable tasks include
///     "instance", "semantic", "polygon" or "color".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current CityscapesDataset.
inline std::shared_ptr<CityscapesDataset> DATASET_API Cityscapes(
  const std::string &dataset_dir, const std::string &usage, const std::string &quality_mode, const std::string &task,
  bool decode, const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CityscapesDataset>(StringToChar(dataset_dir), StringToChar(usage), StringToChar(quality_mode),
                                             StringToChar(task), decode, sampler, cache);
}

/// \class CLUEDataset
/// \brief A source dataset for reading and parsing CLUE dataset.
class DATASET_API CLUEDataset : public Dataset {
 public:
  /// \brief Constructor of CLUEDataset.
  /// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
  ///     will be sorted in a lexicographical order.
  /// \param[in] task The kind of task, one of "AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC" and "CSL" (default="AFQMC").
  /// \param[in] usage Part of dataset of CLUE, can be "train", "test" or "eval" data (default="train").
  /// \param[in] num_samples The number of samples to be included in the dataset
  ///     (Default = 0 means all samples).
  /// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
  ///     Can be any of:
  ///     ShuffleMode::kFalse - No shuffling is performed.
  ///     ShuffleMode::kFiles - Shuffle files only.
  ///     ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified (Default = 0).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  CLUEDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &task,
              const std::vector<char> &usage, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
              int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of CLUEDataset.
  ~CLUEDataset() override = default;
};

/// \brief Function to create a CLUEDataset.
/// \note The generated dataset has a variable number of columns depending on the task and usage.
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] task The kind of task, one of "AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC" and "CSL" (default="AFQMC").
/// \param[in] usage Part of dataset of CLUE, can be "train", "test" or "eval" data (default="train").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0 means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the CLUEDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string train_file = "/path/to/clue_dataset_file";
///      std::shared_ptr<Dataset> ds = CLUE({train_file}, "AFQMC", "train", 0, ShuffleMode::kFalse);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      auto text = row["sentence1"];
/// \endcode
inline std::shared_ptr<CLUEDataset> DATASET_API CLUE(const std::vector<std::string> &dataset_files,
                                                     const std::string &task = "AFQMC",
                                                     const std::string &usage = "train", int64_t num_samples = 0,
                                                     ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                                     int32_t shard_id = 0,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CLUEDataset>(VectorStringToChar(dataset_files), StringToChar(task), StringToChar(usage),
                                       num_samples, shuffle, num_shards, shard_id, cache);
}

/// \class CMUArcticDataset
/// \brief A source dataset for reading and parsing CMUArctic dataset.
class DATASET_API CMUArcticDataset : public Dataset {
 public:
  /// \brief Constructor of CMUArcticDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Part of dataset of CMUArctic, can be "aew", "ahw", "aup", "awb", "axb", "bdl", "clb", "eey",
  ///     "fem", "gka", "jmk", "ksp", "ljm", "lnh", "rms", "rxr", "slp" or "slt".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  CMUArcticDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                   const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of CMUArcticDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Part of dataset of CMUArctic, can be "aew", "ahw", "aup", "awb", "axb", "bdl", "clb", "eey",
  ///     "fem", "gka", "jmk", "ksp", "ljm", "lnh", "rms", "rxr", "slp" or "slt".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  CMUArcticDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name, const Sampler *sampler,
                   const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of CMUArcticDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Part of dataset of CMUArctic, can be "aew", "ahw", "aup", "awb", "axb", "bdl", "clb", "eey",
  ///     "fem", "gka", "jmk", "ksp", "ljm", "lnh", "rms", "rxr", "slp" or "slt".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  CMUArcticDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                   const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of CMUArcticDataset.
  ~CMUArcticDataset() override = default;
};

/// \brief Function to create a CMUArcticDataset.
/// \note The generated dataset has four columns ["waveform", "sample_rate", "transcript", "utterance_id"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Part of dataset of CMUArctic, can be "aew", "ahw", "aup", "awb", "axb", "bdl", "clb", "eey",
///     "fem", "gka", "jmk", "ksp", "ljm", "lnh", "rms", "rxr", "slp" or "slt" (default = "aew").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///      given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the CMUArcticDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/cmu_arctic_dataset_directory";
///      std::shared_ptr<Dataset> ds =
///          CMUArcticDataset(folder_path, name = "aew", std::make_shared<RandomSampler>(false, 10));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In CMUArctic dataset, each data dictionary has keys "waveform", "sample_rate", "transcript"
///         and "utterance_id" */
///      auto waveform = row["waveform"];
/// \endcode
inline std::shared_ptr<CMUArcticDataset> DATASET_API
CMUArctic(const std::string &dataset_dir, const std::string &name = "aew",
          const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CMUArcticDataset>(StringToChar(dataset_dir), StringToChar(name), sampler, cache);
}

/// \brief Function to create a CMUArcticDataset.
/// \note The generated dataset has four columns ["waveform", "sample_rate", "transcript", "utterance_id"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Part of dataset of CMUArctic, can be "aew", "ahw", "aup", "awb", "axb", "bdl", "clb", "eey",
///     "fem", "gka", "jmk", "ksp", "ljm", "lnh", "rms", "rxr", "slp" or "slt".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the CMUArcticDataset.
inline std::shared_ptr<CMUArcticDataset> DATASET_API CMUArctic(const std::string &dataset_dir, const std::string &name,
                                                               const Sampler *sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CMUArcticDataset>(StringToChar(dataset_dir), StringToChar(name), sampler, cache);
}

/// \brief Function to create a CMUArcticDataset.
/// \note The generated dataset has four columns ["waveform", "sample_rate", "transcript", "utterance_id"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Part of dataset of CMUArctic, can be "aew", "ahw", "aup", "awb", "axb", "bdl", "clb", "eey",
///     "fem", "gka", "jmk", "ksp", "ljm", "lnh", "rms", "rxr", "slp" or "slt".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the CMUArcticDataset.
inline std::shared_ptr<CMUArcticDataset> DATASET_API CMUArctic(const std::string &dataset_dir, const std::string &name,
                                                               const std::reference_wrapper<Sampler> sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CMUArcticDataset>(StringToChar(dataset_dir), StringToChar(name), sampler, cache);
}

/// \class CocoDataset
/// \brief A source dataset for reading and parsing Coco dataset.
class DATASET_API CocoDataset : public Dataset {
 public:
  /// \brief Constructor of CocoDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] annotation_file Path to the annotation json.
  /// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] extra_metadata Flag to add extra meta-data to row. (default=false).
  CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
              const std::vector<char> &task, const bool &decode, const std::shared_ptr<Sampler> &sampler,
              const std::shared_ptr<DatasetCache> &cache, const bool &extra_metadata);

  /// \brief Constructor of CocoDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] annotation_file Path to the annotation json.
  /// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] extra_metadata Flag to add extra meta-data to row. (default=false).
  CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
              const std::vector<char> &task, const bool &decode, const Sampler *sampler,
              const std::shared_ptr<DatasetCache> &cache, const bool &extra_metadata);

  /// \brief Constructor of CocoDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] annotation_file Path to the annotation json.
  /// \param[in] task Set the task type of reading coco data, now support 'Detection'/'Stuff'/'Panoptic'/'Keypoint'.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] extra_metadata Flag to add extra meta-data to row. (default=false).
  CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
              const std::vector<char> &task, const bool &decode, const std::reference_wrapper<Sampler> &sampler,
              const std::shared_ptr<DatasetCache> &cache, const bool &extra_metadata);

  /// \brief Constructor of CocoDataset.
  ~CocoDataset() override = default;
};

/// \brief Function to create a CocoDataset.
/// \note The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                  ['iscrowd', dtype=uint32]].
///     - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///     - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                 ['num_keypoints', dtype=uint32]].
///     - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                 ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
///     - task='Captioning', column: [['image', dtype=uint8], ['captions', dtype=string]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] annotation_file Path to the annotation json.
/// \param[in] task Set the task type of reading coco data. Supported task types are "Detection", "Stuff", "Panoptic",
///     "Keypoint" and "Captioning".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \param[in] extra_metadata Flag to add extra meta-data to row. (default=false).
/// \return Shared pointer to the CocoDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/coco_dataset_directory";
///      std::string annotation_file = "/path/to/annotation_file";
///      std::shared_ptr<Dataset> ds = Coco(folder_path, annotation_file);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In COCO dataset, each dictionary has keys "image" and "annotation" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<CocoDataset> DATASET_API
Coco(const std::string &dataset_dir, const std::string &annotation_file, const std::string &task = "Detection",
     const bool &decode = false, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
     const std::shared_ptr<DatasetCache> &cache = nullptr, const bool &extra_metadata = false) {
  return std::make_shared<CocoDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), StringToChar(task),
                                       decode, sampler, cache, extra_metadata);
}

/// \brief Function to create a CocoDataset.
/// \note The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                  ['iscrowd', dtype=uint32]].
///     - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///     - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                 ['num_keypoints', dtype=uint32]].
///     - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                 ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
///     - task='Captioning', column: [['image', dtype=uint8], ['captions', dtype=string]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] annotation_file Path to the annotation json.
/// \param[in] task Set the task type of reading coco data. Supported task types are "Detection", "Stuff", "Panoptic",
///     "Keypoint" and "Captioning".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \param[in] extra_metadata Flag to add extra meta-data to row. (default=false).
/// \return Shared pointer to the CocoDataset.
inline std::shared_ptr<CocoDataset> DATASET_API Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                                     const std::string &task, const bool &decode,
                                                     const Sampler *sampler,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr,
                                                     const bool &extra_metadata = false) {
  return std::make_shared<CocoDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), StringToChar(task),
                                       decode, sampler, cache, extra_metadata);
}

/// \brief Function to create a CocoDataset.
/// \note The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                  ['iscrowd', dtype=uint32]].
///     - task='Stuff', column: [['image', dtype=uint8], ['segmentation',dtype=float32], ['iscrowd', dtype=uint32]].
///     - task='Keypoint', column: [['image', dtype=uint8], ['keypoints', dtype=float32],
///                                 ['num_keypoints', dtype=uint32]].
///     - task='Panoptic', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['category_id', dtype=uint32],
///                                 ['iscrowd', dtype=uint32], ['area', dtype=uitn32]].
///     - task='Captioning', column: [['image', dtype=uint8], ['captions', dtype=string]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] annotation_file Path to the annotation json.
/// \param[in] task Set the task type of reading coco data. Supported task types are "Detection", "Stuff", "Panoptic",
///     "Keypoint" and "Captioning".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \param[in] extra_metadata Flag to add extra meta-data to row. (default=false).
/// \return Shared pointer to the CocoDataset.
inline std::shared_ptr<CocoDataset> DATASET_API Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                                     const std::string &task, const bool &decode,
                                                     const std::reference_wrapper<Sampler> &sampler,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr,
                                                     const bool &extra_metadata = false) {
  return std::make_shared<CocoDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), StringToChar(task),
                                       decode, sampler, cache, extra_metadata);
}

/// \class CoNLL2000Dataset
/// \brief A source dataset for reading and parsing CoNLL2000Dataset.
class DATASET_API CoNLL2000Dataset : public Dataset {
 public:
  /// \brief Constructor of CoNLL2000Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data list txt file to be read, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  CoNLL2000Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                   ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                   const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of CoNLL2000Dataset.
  ~CoNLL2000Dataset() override = default;
};

/// \brief Function to create a CoNLL2000Dataset.
/// \note The generated dataset has three column ['word', 'pos_tag', 'chunk_tag'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of CoNLL2000, can be "train", "test" or "all" (default="all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the CoNLL2000Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_dir = "/path/to/conll2000_dataset_directory";
///      std::shared_ptr<Dataset> ds = CoNLL2000(dataset_dir, "all", 0, ShuffleMode::kGlobal);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In CoNLL2000 dataset, each dictionary has keys "word", "pos_tag", "chunk_tag" */
///      auto word = row["word"];
/// \endcode
inline std::shared_ptr<CoNLL2000Dataset> DATASET_API CoNLL2000(const std::string &dataset_dir,
                                                               const std::string &usage = "all",
                                                               int64_t num_samples = 0,
                                                               ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                               int32_t num_shards = 1, int32_t shard_id = 0,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CoNLL2000Dataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                            num_shards, shard_id, cache);
}

/// \class CSVDataset
/// \brief A source dataset that reads and parses comma-separated values (CSV) datasets.
class DATASET_API CSVDataset : public Dataset {
 public:
  /// \brief Constructor of CSVDataset.
  /// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
  ///    will be sorted in a lexicographical order.
  /// \param[in] field_delim A char that indicates the delimiter to separate fields (default=',').
  /// \param[in] column_defaults List of default values for the CSV field (default={}). Each item in the list is
  ///    either a valid type (float, int, or string). If this is not provided, treats all columns as string type.
  /// \param[in] column_names List of column names of the dataset (default={}). If this is not provided, infers the
  ///    column_names from the first row of CSV file.
  /// \param[in] num_samples The number of samples to be included in the dataset.
  ///    (Default = 0 means all samples).
  /// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///    specified only when num_shards is also specified (Default = 0).
  /// \param[in] cache Tensor cache to use.(default=nullptr which means no cache is used).
  CSVDataset(const std::vector<std::vector<char>> &dataset_files, char field_delim,
             const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
             const std::vector<std::vector<char>> &column_names, int64_t num_samples, ShuffleMode shuffle,
             int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of CSVDataset.
  ~CSVDataset() override = default;
};

/// \brief Function to create a CSVDataset.
/// \note The generated dataset has a variable number of columns.
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///    will be sorted in a lexicographical order.
/// \param[in] field_delim A char that indicates the delimiter to separate fields (default=',').
/// \param[in] column_defaults List of default values for the CSV field (default={}). Each item in the list is
///    either a valid type (float, int, or string). If this is not provided, treats all columns as string type.
/// \param[in] column_names List of column names of the dataset (default={}). If this is not provided, infers the
///    column_names from the first row of CSV file.
/// \param[in] num_samples The number of samples to be included in the dataset.
///    (Default = 0 means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1)
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///    specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use.(default=nullptr which means no cache is used).
/// \return Shared pointer to the CSVDataset
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string train_file = "/path/to/csv_file";
///      std::vector<std::string> column_names = {"col1", "col2", "col3", "col4"};
///      std::shared_ptr<Dataset> ds = CSV({train_file}, ',', {}, column_names, 0, ShuffleMode::kFalse);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: As we defined before, the dataset has column "col1", "col2", "col3" and "col4" */
///      auto col1 = row["col1"];
/// \endcode
inline std::shared_ptr<CSVDataset> DATASET_API CSV(const std::vector<std::string> &dataset_files,
                                                   char field_delim = ',',
                                                   const std::vector<std::shared_ptr<CsvBase>> &column_defaults = {},
                                                   const std::vector<std::string> &column_names = {},
                                                   int64_t num_samples = 0, ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                   int32_t num_shards = 1, int32_t shard_id = 0,
                                                   const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<CSVDataset>(VectorStringToChar(dataset_files), field_delim, column_defaults,
                                      VectorStringToChar(column_names), num_samples, shuffle, num_shards, shard_id,
                                      cache);
}

/// \class DBpediaDataset
/// \brief A source dataset for reading and parsing DBpedia dataset.
class DATASET_API DBpediaDataset : public Dataset {
 public:
  /// \brief Constructor of DBpediaDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of DBpedia, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  DBpediaDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of DBpediaDataset.
  ~DBpediaDataset() override = default;
};

/// \brief Function to create a DBpediaDataset.
/// \note The generated dataset has three columns ["class", "title", "content"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of DBpedia, can be "train", "test" or "all" (default = "all").
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the DBpediaDataset
inline std::shared_ptr<DBpediaDataset> DATASET_API DBpedia(const std::string &dataset_dir,
                                                           const std::string &usage = "all", int64_t num_samples = 0,
                                                           ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                           int32_t num_shards = 1, int32_t shard_id = 0,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<DBpediaDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                          num_shards, shard_id, cache);
}

/// \class DIV2KDataset
/// \brief A source dataset for reading and parsing DIV2K dataset.
class DATASET_API DIV2KDataset : public Dataset {
 public:
  /// \brief Constructor of DIV2KDataset.
  /// \param[in] dataset_dir The dataset dir to be read.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "valid" or "all".
  /// \param[in] downgrade The mode of downgrade. Acceptable downgrades include "bicubic", "unknown", "mild",
  ///     "difficult" or "wild".
  /// \param[in] scale The scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  DIV2KDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const std::vector<char> &downgrade,
               int32_t scale, bool decode, const std::shared_ptr<Sampler> &sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of DIV2KDataset.
  /// \param[in] dataset_dir The dataset dir to be read.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "valid" or "all".
  /// \param[in] downgrade The mode of downgrade. Acceptable downgrades include "bicubic", "unknown", "mild",
  ///     "difficult" or "wild".
  /// \param[in] scale The scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  DIV2KDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const std::vector<char> &downgrade,
               int32_t scale, bool decode, const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of DIV2KDataset.
  /// \param[in] dataset_dir The dataset dir to be read.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "valid" or "all".
  /// \param[in] downgrade The mode of downgrade. Acceptable downgrades include "bicubic", "unknown", "mild",
  ///     "difficult" or "wild".
  /// \param[in] scale The scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  DIV2KDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const std::vector<char> &downgrade,
               int32_t scale, bool decode, const std::reference_wrapper<Sampler> &sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of DIV2KDataset.
  ~DIV2KDataset() override = default;
};

/// \brief Function to create a DIV2KDataset.
/// \note The generated dataset has two columns ["hr_image", "lr_image"].
/// \param[in] dataset_dir The dataset dir to be read.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "valid" or "all".
/// \param[in] downgrade The mode of downgrade. Acceptable downgrades include "bicubic", "unknown", "mild", "difficult"
///     or "wild".
/// \param[in] scale The scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current DIV2KDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_path = "/path/to/div2k_dataset_directory";
///      std::shared_ptr<Dataset> ds = DIV2K(dataset_path, "train", "bicubic", 2);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In DIV2K dataset, each dictionary has keys "hr_image" and "lr_image" */
///      auto hr_image = row["hr_image"];
/// \endcode
inline std::shared_ptr<DIV2KDataset> DATASET_API
DIV2K(const std::string &dataset_dir, const std::string &usage, const std::string &downgrade, int32_t scale,
      bool decode = false, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
      const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<DIV2KDataset>(StringToChar(dataset_dir), StringToChar(usage), StringToChar(downgrade), scale,
                                        decode, sampler, cache);
}

/// \brief Function to create a DIV2KDataset.
/// \note The generated dataset has two columns ["hr_image", "lr_image"].
/// \param[in] dataset_dir The dataset dir to be read.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "valid" or "all".
/// \param[in] downgrade The mode of downgrade. Acceptable downgrades include "bicubic", "unknown", "mild", "difficult"
///     or "wild".
/// \param[in] scale The scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current DIV2KDataset.
inline std::shared_ptr<DIV2KDataset> DATASET_API DIV2K(const std::string &dataset_dir, const std::string &usage,
                                                       const std::string &downgrade, int32_t scale, bool decode,
                                                       const Sampler *sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<DIV2KDataset>(StringToChar(dataset_dir), StringToChar(usage), StringToChar(downgrade), scale,
                                        decode, sampler, cache);
}

/// \brief Function to create a DIV2KDataset.
/// \note The generated dataset has two columns ["hr_image", "lr_image"].
/// \param[in] dataset_dir The dataset dir to be read.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "valid" or "all".
/// \param[in] downgrade The mode of downgrade. Acceptable downgrades include "bicubic", "unknown", "mild", "difficult"
///     or "wild".
/// \param[in] scale The scale of downgrade. Acceptable scales include 2, 3, 4 or 8.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current DIV2KDataset.
inline std::shared_ptr<DIV2KDataset> DATASET_API DIV2K(const std::string &dataset_dir, const std::string &usage,
                                                       const std::string &downgrade, int32_t scale, bool decode,
                                                       const std::reference_wrapper<Sampler> &sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<DIV2KDataset>(StringToChar(dataset_dir), StringToChar(usage), StringToChar(downgrade), scale,
                                        decode, sampler, cache);
}

/// \class EMnistDataset
/// \brief A source dataset for reading and parsing EMnist dataset.
class DATASET_API EMnistDataset : public Dataset {
 public:
  /// \brief Constructor of EMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Name of splits for EMNIST, can be "byclass", "bymerge", "balanced", "letters", "digits"
  ///     or "mnist".
  /// \param[in] usage Part of dataset of EMNIST, can be "train", "test" or "all".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  EMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name, const std::vector<char> &usage,
                const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of EMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Name of splits for EMNIST, can be "byclass", "bymerge", "balanced", "letters", "digits"
  ///     or "mnist".
  /// \param[in] usage Part of dataset of EMNIST, can be "train", "test" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  EMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name, const std::vector<char> &usage,
                const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of EMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Name of splits for EMNIST, can be "byclass", "bymerge", "balanced", "letters", "digits"
  ///     or "mnist".
  /// \param[in] usage Part of dataset of EMNIST, can be "train", "test" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  EMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name, const std::vector<char> &usage,
                const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of EMnistDataset.
  ~EMnistDataset() override = default;
};

/// \brief Function to create a EMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Name of splits for EMNIST, can be "byclass", "bymerge", "balanced", "letters", "digits" or "mnist".
/// \param[in] usage Usage of EMNIST, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not.
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current EMnistDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/emnist_dataset_directory";
///      std::shared_ptr<Dataset> ds = EMnist(folder_path, "mnist", "train", std::make_shared<RandomSampler>(false, 5));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In EMNIST dataset dataset, each dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<EMnistDataset> DATASET_API
EMnist(const std::string &dataset_dir, const std::string &name, const std::string &usage = "all",
       const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<EMnistDataset>(StringToChar(dataset_dir), StringToChar(name), StringToChar(usage), sampler,
                                         cache);
}

/// \brief Function to create a EMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] name Name of splits for EMNIST, can be "byclass", "bymerge", "balanced", "letters", "digits" or "mnist".
/// \param[in] usage Usage of EMNIST, can be "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current EMnistDataset.
inline std::shared_ptr<EMnistDataset> DATASET_API EMnist(const std::string &dataset_dir, const std::string &usage,
                                                         const std::string &name, const Sampler *sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<EMnistDataset>(StringToChar(dataset_dir), StringToChar(name), StringToChar(usage), sampler,
                                         cache);
}

/// \brief Function to create a EMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Name of splits for EMNIST, can be "byclass", "bymerge", "balanced", "letters", "digits" or "mnist".
/// \param[in] usage Usage of EMNIST, can be "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current EMnistDataset.
inline std::shared_ptr<EMnistDataset> DATASET_API EMnist(const std::string &dataset_dir, const std::string &name,
                                                         const std::string &usage,
                                                         const std::reference_wrapper<Sampler> &sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<EMnistDataset>(StringToChar(dataset_dir), StringToChar(name), StringToChar(usage), sampler,
                                         cache);
}

/// \class EnWik9Dataset
/// \brief A source dataset for reading and parsing EnWik9 dataset.
class DATASET_API EnWik9Dataset : public Dataset {
 public:
  /// \brief Function to create a EnWik9Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  EnWik9Dataset(const std::vector<char> &dataset_dir, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of EnWik9Dataset.
  ~EnWik9Dataset() override = default;
};

/// \brief Function to create a EnWik9Dataset.
/// \note The generated dataset has one column ['text'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode.kFalse - No shuffling is performed.
///     ShuffleMode.kFiles - Shuffle files only.
///     ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the EnWik9Dataset.
inline std::shared_ptr<EnWik9Dataset> DATASET_API EnWik9(const std::string &dataset_dir, int64_t num_samples = 0,
                                                         ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                         int32_t num_shards = 1, int32_t shard_id = 0,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<EnWik9Dataset>(StringToChar(dataset_dir), num_samples, shuffle, num_shards, shard_id, cache);
}

/// \class FakeImageDataset
/// \brief A source dataset for generating fake images.
class DATASET_API FakeImageDataset : public Dataset {
 public:
  /// \brief Constructor of FakeImageDataset.
  /// \param[in] num_images The number of images to generate, which must be positive.
  /// \param[in] image_size Size of the images, which must be a vector of three positive values.
  /// \param[in] num_classes The number of classes of the images, which must be positive.
  /// \param[in] base_seed The base seed to generate the images.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///    given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  FakeImageDataset(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes, int32_t base_seed,
                   const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of FakeImageDataset.
  /// \param[in] num_images The number of images to generate, which must be positive.
  /// \param[in] image_size Size of the images, which must be a vector of three positive values.
  /// \param[in] num_classes The number of classes of the images, which must be positive.
  /// \param[in] base_seed The base seed to generate the images.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  FakeImageDataset(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes, int32_t base_seed,
                   const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of FakeImageDataset.
  /// \param[in] num_images The number of images to generate, which must be positive.
  /// \param[in] image_size Size of the images, which must be a vector of three positive values.
  /// \param[in] num_classes The number of classes of the images, which must be positive.
  /// \param[in] base_seed The base seed to generate the images.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  FakeImageDataset(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes, int32_t base_seed,
                   const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of FakeImageDataset.
  ~FakeImageDataset() override = default;
};

/// \brief Function to create a FakeImageDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] num_images The number of images to generate, which must be positive (default = 1000).
/// \param[in] image_size Size of the images, which must be a vector of three positive values
///    (default = {224, 224, 3}).
/// \param[in] num_classes The number of classes of the images, which must be positive (default = 10).
/// \param[in] base_seed The base seed to generate the images (default = 0).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///    given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current FakeDataset.
inline std::shared_ptr<FakeImageDataset> DATASET_API
FakeImage(int32_t num_images = 1000, const std::vector<int32_t> &image_size = {224, 224, 3}, int32_t num_classes = 10,
          int32_t base_seed = 0, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FakeImageDataset>(num_images, image_size, num_classes, base_seed, sampler, cache);
}

/// \brief Function to create a FakeImageDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] num_images The number of images to generate, which must be positive.
/// \param[in] image_size Size of the images, which must be a vector of three positive values.
/// \param[in] num_classes The number of classes of the images, which must be positive.
/// \param[in] base_seed The base seed to generate the images.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current FakeImageDataset.
inline std::shared_ptr<FakeImageDataset> DATASET_API FakeImage(int32_t num_images,
                                                               const std::vector<int32_t> &image_size,
                                                               int32_t num_classes, int32_t base_seed,
                                                               const Sampler *sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FakeImageDataset>(num_images, image_size, num_classes, base_seed, sampler, cache);
}

/// \brief Function to create a FakeImageDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] num_images The number of images to generate, which must be positive.
/// \param[in] image_size Size of the images, which must be a vector of three positive values.
/// \param[in] num_classes The number of classes of the images, which must be positive.
/// \param[in] base_seed The base seed to generate the images.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current FakeImageDataset.
inline std::shared_ptr<FakeImageDataset> DATASET_API FakeImage(int32_t num_images,
                                                               const std::vector<int32_t> &image_size,
                                                               int32_t num_classes, int32_t base_seed,
                                                               const std::reference_wrapper<Sampler> &sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FakeImageDataset>(num_images, image_size, num_classes, base_seed, sampler, cache);
}

/// \class FashionMnistDataset
/// \brief A source dataset that reads and parses FASHION-MNIST dataset.
class DATASET_API FashionMnistDataset : public Dataset {
 public:
  /// \brief Constructor of FashionMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of FASHION-MNIST, can be "train", "test" or "all".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  FashionMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                      const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of FashionMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of FASHION-MNIST, can be "train", "test" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  FashionMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                      const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of FashionMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of FASHION-MNIST, can be "train", "test" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  FashionMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                      const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of FashionMnistDataset.
  ~FashionMnistDataset() override = default;
};

/// \brief Function to create a FashionMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of FASHION-MNIST, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the FashionMnistDataset.
inline std::shared_ptr<FashionMnistDataset> DATASET_API
FashionMnist(const std::string &dataset_dir, const std::string &usage = "all",
             const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FashionMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a FashionMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of FASHION-MNIST, can be "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the FashionMnistDataset.
inline std::shared_ptr<FashionMnistDataset> DATASET_API
FashionMnist(const std::string &dataset_dir, const std::string &usage, const Sampler *sampler,
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FashionMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a FashionMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of FASHION-MNIST, can be "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the FashionMnistDataset.
inline std::shared_ptr<FashionMnistDataset> DATASET_API
FashionMnist(const std::string &dataset_dir, const std::string &usage, const std::reference_wrapper<Sampler> &sampler,
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FashionMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class FlickrDataset
/// \brief A source dataset for reading and parsing Flickr dataset.
class DATASET_API FlickrDataset : public Dataset {
 public:
  /// \brief Constructor of FlickrDataset.
  /// \param[in] dataset_dir The dataset dir to be read
  /// \param[in] annotation_file The annotation file to be read
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  FlickrDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file, bool decode,
                const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of FlickrDataset.
  /// \param[in] dataset_dir The dataset dir to be read
  /// \param[in] annotation_file The annotation file to be read
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  FlickrDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file, bool decode,
                const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of FlickrDataset.
  /// \param[in] dataset_dir The dataset dir to be read
  /// \param[in] annotation_file The annotation file to be read
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
  FlickrDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file, bool decode,
                const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of FlickrDataset.
  ~FlickrDataset() override = default;
};

/// \brief Function to create a FlickrDataset
/// \note The generated dataset has two columns ["image", "annotation"]
/// \param[in] dataset_dir The dataset dir to be read
/// \param[in] annotation_file The annotation file to be read
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current FlickrDataset
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_path = "/path/to/flickr30k_dataset_directory";
///      std::string file_path = "/path/to/token_file";
///      std::shared_ptr<Dataset> ds = Flickr(dataset_path, file_path);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In FLICKR dataset, each dictionary has keys "image" and "annotation" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<FlickrDataset> DATASET_API
Flickr(const std::string &dataset_dir, const std::string &annotation_file, bool decode = false,
       const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FlickrDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), decode, sampler,
                                         cache);
}

/// \brief Function to create a FlickrDataset
/// \note The generated dataset has two columns ["image", "annotation"]
/// \param[in] dataset_dir The dataset dir to be read
/// \param[in] annotation_file The annotation file to be read
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current FlickrDataset
inline std::shared_ptr<FlickrDataset> DATASET_API Flickr(const std::string &dataset_dir,
                                                         const std::string &annotation_file, bool decode,
                                                         const Sampler *sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FlickrDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), decode, sampler,
                                         cache);
}

/// \brief Function to create a FlickrDataset
/// \note The generated dataset has two columns ["image", "annotation"]
/// \param[in] dataset_dir The dataset dir to be read
/// \param[in] annotation_file The annotation file to be read
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current FlickrDataset
inline std::shared_ptr<FlickrDataset> DATASET_API Flickr(const std::string &dataset_dir,
                                                         const std::string &annotation_file, bool decode,
                                                         const std::reference_wrapper<Sampler> &sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<FlickrDataset>(StringToChar(dataset_dir), StringToChar(annotation_file), decode, sampler,
                                         cache);
}

/// \class Food101Dataset
/// \brief A source dataset for reading and parsing Food101 dataset.
class DATASET_API Food101Dataset : public Dataset {
 public:
  /// \brief Constructor of Food101Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Food101Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                 const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Food101Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Food101Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                 const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Food101Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Food101Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                 const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of Food101Dataset.
  ~Food101Dataset() override = default;
};

/// \brief Function to create a Food101Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all". Default: "all".
/// \param[in] decode Decode the images after reading. Default: false.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset. Default: RandomSampler().
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the Food101Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_path = "/path/to/Food101_dataset_directory";
///      std::shared_ptr<Dataset> ds = Food101(dataset_path);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In Food101 dataset, each data dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<Food101Dataset> DATASET_API
Food101(const std::string &dataset_dir, const std::string &usage = "all", bool decode = false,
        const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
        const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Food101Dataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a Food101Dataset
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all" .
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the Food101Dataset.
inline std::shared_ptr<Food101Dataset> DATASET_API Food101(const std::string &dataset_dir, const std::string &usage,
                                                           bool decode, const Sampler *sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Food101Dataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a Food101Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the Food101Dataset.
inline std::shared_ptr<Food101Dataset> DATASET_API Food101(const std::string &dataset_dir, const std::string &usage,
                                                           bool decode, const std::reference_wrapper<Sampler> &sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Food101Dataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \class GTZANDataset
/// \brief A source dataset for reading and parsing GTZAN dataset.
class DATASET_API GTZANDataset : public Dataset {
 public:
  /// \brief Constructor of GTZANDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of GTZAN, can be "train", "valid", "test", or "all" (default = "all").
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  GTZANDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
               const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of GTZANDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of GTZAN, can be "train", "valid", "test", or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  GTZANDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of GTZANDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of GTZAN, can be "train", "valid", "test", or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  GTZANDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
               const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of GTZANDataset.
  ~GTZANDataset() override = default;
};

/// \brief Function to create a GTZANDataset.
/// \note The generated dataset has three columns ["waveform", "sample_rate", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of GTZAN, can be "train", "valid", "test", or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the GTZANDataset.
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/gtzan_dataset_directory";
///      std::shared_ptr<Dataset> ds =
///          GTZANDataset(folder_path, usage = "all", std::make_shared<RandomSampler>(false, 10));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In GTZAN dataset, each data dictionary has keys "waveform", "sample_rate" and "label" */
///      auto waveform = row["waveform"];
/// \endcode
inline std::shared_ptr<GTZANDataset> DATASET_API
GTZAN(const std::string &dataset_dir, const std::string &usage = "all",
      const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
      const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<GTZANDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a GTZANDataset.
/// \note The generated dataset has three columns ["waveform", "sample_rate", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of GTZAN, can be "train", "valid", "test", or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the GTZANDataset.
inline std::shared_ptr<GTZANDataset> DATASET_API GTZAN(const std::string &dataset_dir, const std::string &usage,
                                                       const Sampler *sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<GTZANDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a GTZANDataset.
/// \note The generated dataset has three columns ["waveform", "sample_rate", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of GTZAN, can be "train", "valid", "test", or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the GTZANDataset.
inline std::shared_ptr<GTZANDataset> DATASET_API GTZAN(const std::string &dataset_dir, const std::string &usage,
                                                       const std::reference_wrapper<Sampler> sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<GTZANDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class ImageFolderDataset
/// \brief A source dataset that reads images from a tree of directories.
class DATASET_API ImageFolderDataset : public Dataset {
 public:
  /// \brief Constructor of ImageFolderDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode A flag to decode in ImageFolder.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] extensions File extensions to be read.
  /// \param[in] class_indexing a class name to label map.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode, const std::shared_ptr<Sampler> &sampler,
                     const std::set<std::vector<char>> &extensions,
                     const std::map<std::vector<char>, int32_t> &class_indexing,
                     const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of ImageFolderDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode A flag to decode in ImageFolder.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] extensions File extensions to be read.
  /// \param[in] class_indexing a class name to label map.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
                     const std::set<std::vector<char>> &extensions,
                     const std::map<std::vector<char>, int32_t> &class_indexing,
                     const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of ImageFolderDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode A flag to decode in ImageFolder.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] extensions File extensions to be read.
  /// \param[in] class_indexing a class name to label map.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode, const std::reference_wrapper<Sampler> &sampler,
                     const std::set<std::vector<char>> &extensions,
                     const std::map<std::vector<char>, int32_t> &class_indexing,
                     const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of ImageFolderDataset.
  ~ImageFolderDataset() override = default;
};

/// \brief Function to create an ImageFolderDataset.
/// \note A source dataset that reads images from a tree of directories.
///     All images within one folder have the same label.
///     The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode A flag to decode in ImageFolder.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] extensions File extensions to be read.
/// \param[in] class_indexing a class name to label map.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the ImageFolderDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_path = "/path/to/image_directory";
///      std::shared_ptr<Dataset> ds = ImageFolder(dataset_path, true, std::make_shared<RandomSampler>(false, 10));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In ImageFolder dataset, each data dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<ImageFolderDataset> DATASET_API
ImageFolder(const std::string &dataset_dir, bool decode = false,
            const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
            const std::set<std::string> &extensions = {}, const std::map<std::string, int32_t> &class_indexing = {},
            const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ImageFolderDataset>(StringToChar(dataset_dir), decode, sampler, SetStringToChar(extensions),
                                              MapStringToChar(class_indexing), cache);
}

/// \brief Function to create an ImageFolderDataset
/// \note A source dataset that reads images from a tree of directories.
///     All images within one folder have the same label.
///     The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode A flag to decode in ImageFolder.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] extensions File extensions to be read.
/// \param[in] class_indexing a class name to label map.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the ImageFolderDataset.
inline std::shared_ptr<ImageFolderDataset> DATASET_API ImageFolder(
  const std::string &dataset_dir, bool decode, const Sampler *sampler, const std::set<std::string> &extensions = {},
  const std::map<std::string, int32_t> &class_indexing = {}, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ImageFolderDataset>(StringToChar(dataset_dir), decode, sampler, SetStringToChar(extensions),
                                              MapStringToChar(class_indexing), cache);
}

/// \brief Function to create an ImageFolderDataset.
/// \note A source dataset that reads images from a tree of directories.
///     All images within one folder have the same label.
///     The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode A flag to decode in ImageFolder.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] extensions File extensions to be read.
/// \param[in] class_indexing a class name to label map.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the ImageFolderDataset.
inline std::shared_ptr<ImageFolderDataset> DATASET_API
ImageFolder(const std::string &dataset_dir, bool decode, const std::reference_wrapper<Sampler> &sampler,
            const std::set<std::string> &extensions = {}, const std::map<std::string, int32_t> &class_indexing = {},
            const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ImageFolderDataset>(StringToChar(dataset_dir), decode, sampler, SetStringToChar(extensions),
                                              MapStringToChar(class_indexing), cache);
}

/// \class IMDBDataset
/// \brief A source dataset for reading and parsing IMDB dataset.
class DATASET_API IMDBDataset : public Dataset {
 public:
  /// \brief Constructor of IMDBDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  IMDBDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
              const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of IMDBDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  IMDBDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
              const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of IMDBDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  IMDBDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
              const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of IMDBDataset.
  ~IMDBDataset() override = default;
};

/// \brief A source dataset for reading and parsing IMDB dataset.
/// \note The generated dataset has two columns ["text", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all"
///     (Default="all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the IMDBDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_path = "/path/to/imdb_dataset_directory";
///      std::shared_ptr<Dataset> ds = IMDB(dataset_path, "all");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In IMDB dataset, each data dictionary has keys "text" and "label" */
///      auto text = row["text"];
/// \endcode
inline std::shared_ptr<IMDBDataset> DATASET_API
IMDB(const std::string &dataset_dir, const std::string &usage = "all",
     const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<IMDBDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief A source dataset for reading and parsing IMDB dataset.
/// \note The generated dataset has two columns ["text", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the IMDBDataset.
inline std::shared_ptr<IMDBDataset> DATASET_API IMDB(const std::string &dataset_dir, const std::string &usage,
                                                     const Sampler *sampler,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<IMDBDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief A source dataset for reading and parsing IMDB dataset.
/// \note The generated dataset has two columns ["text", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the IMDBDataset.
inline std::shared_ptr<IMDBDataset> DATASET_API IMDB(const std::string &dataset_dir, const std::string &usage,
                                                     const std::reference_wrapper<Sampler> &sampler,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<IMDBDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class IWSLT2016Dataset.
/// \brief A source dataset for reading and parsing IWSLT2016 dataset.
class DATASET_API IWSLT2016Dataset : public Dataset {
 public:
  /// \brief Constructor of IWSLT2016Dataset.
  /// \note The generated dataset has two columns ["text", "translation"].
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of IWSLT2016, can be "train", "valid", "test" or "all".
  /// \param[in] language_pair List containing src and tgt language.
  /// \param[in] valid_set A string to identify validation set.
  /// \param[in] test_set A string to identify test set.
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///    specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  IWSLT2016Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                   const std::vector<std::vector<char>> &language_pair, const std::vector<char> &valid_set,
                   const std::vector<char> &test_set, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                   int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of IWSLT2016Dataset.
  ~IWSLT2016Dataset() override = default;
};

/// \brief Function to create a IWSLT2016Dataset.
/// \note The generated dataset has two columns ["text", "translation"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of IWSLT2016, can be "train", "valid", "test" or "all" (default = "all").
/// \param[in] language_pair List containing src and tgt language (Default = {"de", "en"}).
/// \param[in] valid_set A string to identify validation set (Default = "tst2013").
/// \param[in] test_set A string to identify test set (Default = "tst2014").
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the IWSLT2016Dataset.
inline std::shared_ptr<IWSLT2016Dataset> DATASET_API
IWSLT2016(const std::string &dataset_dir, const std::string &usage = "all",
          const std::vector<std::string> &language_pair = {"de", "en"}, const std::string &valid_set = "tst2013",
          const std::string &test_set = "tst2014", int64_t num_samples = 0, ShuffleMode shuffle = ShuffleMode::kGlobal,
          int32_t num_shards = 1, int32_t shard_id = 0, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<IWSLT2016Dataset>(StringToChar(dataset_dir), StringToChar(usage),
                                            VectorStringToChar(language_pair), StringToChar(valid_set),
                                            StringToChar(test_set), num_samples, shuffle, num_shards, shard_id, cache);
}

/// \class IWSLT2017Dataset.
/// \brief A source dataset for reading and parsing IWSLT2017 dataset.
class DATASET_API IWSLT2017Dataset : public Dataset {
 public:
  /// \brief Constructor of IWSLT2017Dataset.
  /// \note The generated dataset has two columns ["text", "translation"].
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of IWSLT2017, can be "train", "valid", "test" or "all".
  /// \param[in] language_pair List containing src and tgt language.
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode::kFalse - No shuffling is performed.
  ///     ShuffleMode::kFiles - Shuffle files only.
  ///     ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  /// \return Shared pointer to the IWSLT2017Dataset.
  IWSLT2017Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                   const std::vector<std::vector<char>> &language_pair, int64_t num_samples, ShuffleMode shuffle,
                   int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of IWSLT2017Dataset.
  ~IWSLT2017Dataset() override = default;
};

/// \brief Function to create a IWSLT2017Dataset.
/// \note The generated dataset has two columns ["text", "translation"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of IWSLT2017, can be "train", "valid", "test" or "all" (default = "all").
/// \param[in] language_pair List containing src and tgt language (Default = {"de", "en"}).
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the IWSLT2017Dataset.
inline std::shared_ptr<IWSLT2017Dataset> DATASET_API
IWSLT2017(const std::string &dataset_dir, const std::string &usage = "all",
          const std::vector<std::string> &language_pair = {"de", "en"}, int64_t num_samples = 0,
          ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<IWSLT2017Dataset>(StringToChar(dataset_dir), StringToChar(usage),
                                            VectorStringToChar(language_pair), num_samples, shuffle, num_shards,
                                            shard_id, cache);
}

/// \class KITTIDataset
/// \brief A source dataset that reads KITTI images and labels.
class DATASET_API KITTIDataset : public Dataset {
 public:
  /// \brief Constructor of KITTIDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data file to read.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  KITTIDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
               const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of KITTIDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data file to read.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  KITTIDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
               const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of KITTIDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data file to read.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  KITTIDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
               const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of KITTIDataset.
  ~KITTIDataset() override = default;
};

/// \brief Function to create a KITTIDataset.
/// \note When usage is 'train', the generated dataset has multi-columns, 'image', 'label', 'truncated',
///     'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y'; When usage is 'test',
///     the generated dataset has one column 'image'.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of data file to read (default = "train").
/// \param[in] decode Decode the images after reading (default = false).
/// \param[in] sampler Sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/kitti_dataset_directory";
///      std::shared_ptr<Dataset> ds = KITTI(folder_path);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In KITTI dataset, each dictionary has key "image" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<KITTIDataset> DATASET_API
KITTI(const std::string &dataset_dir, const std::string &usage = "train", bool decode = false,
      const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
      const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<KITTIDataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a KITTIDataset.
/// \note When usage is 'train', the generated dataset has multi-columns, 'image', 'label', 'truncated',
///     'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y'; When usage is 'test',
///     the generated dataset has one column 'image'.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of data file to read.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<KITTIDataset> DATASET_API KITTI(const std::string &dataset_dir, const std::string &usage,
                                                       bool decode, const Sampler *sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<KITTIDataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a KITTIDataset.
/// \note When usage is 'train', the generated dataset has multi-columns, 'image', 'label', 'truncated',
///     'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y'; When usage is 'test',
///     the generated dataset has one column 'image'.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of data file to read.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<KITTIDataset> DATASET_API KITTI(const std::string &dataset_dir, const std::string &usage,
                                                       bool decode, const std::reference_wrapper<Sampler> &sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<KITTIDataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \class KMnistDataset.
/// \brief A source dataset for reading and parsing KMnist dataset.
class DATASET_API KMnistDataset : public Dataset {
 public:
  /// \brief Function to create a KMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage of KMNIST, can be "train", "test" or "all".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  /// \return Shared pointer to the current KMnistDataset.
  KMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Function to create a KMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage of Kmnist, can be "train", "test" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  /// \return Shared pointer to the current KMnistDataset.
  KMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                const std::shared_ptr<DatasetCache> &cache);

  /// \brief Function to create a KMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage of Kmnist, can be "train", "test" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  /// \return Shared pointer to the current KMnistDataset.
  KMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of KMnistDataset.
  ~KMnistDataset() override = default;
};

/// \brief Function to create a KMnistDataset.
/// \note The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset
/// \param[in] usage of KMNIST, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current KMnistDataset.
inline std::shared_ptr<KMnistDataset> DATASET_API
KMnist(const std::string &dataset_dir, const std::string &usage = "all",
       const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<KMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a KMnistDataset.
/// \note The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage of Kmnist, can be "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current KMnistDataset.
inline std::shared_ptr<KMnistDataset> DATASET_API KMnist(const std::string &dataset_dir, const std::string &usage,
                                                         const Sampler *sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<KMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a KMnistDataset.
/// \note The generated dataset has two columns ["image", "label"]
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage of Kmnist, can be "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current KMnistDataset.
inline std::shared_ptr<KMnistDataset> DATASET_API KMnist(const std::string &dataset_dir, const std::string &usage,
                                                         const std::reference_wrapper<Sampler> &sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<KMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class LFWDataset
/// \brief A source dataset for reading and parsing LFW dataset.
class DATASET_API LFWDataset : public Dataset {
 public:
  /// \brief Constructor of LFWDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] task Set the task type of reading LFW, support "people" and "pairs".
  /// \param[in] usage The image split to use, support "10fold", "train", "test" and "all".
  /// \param[in] image_set Image set of image funneling to use, support "original", "funneled" and "deepfunneled".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LFWDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task, const std::vector<char> &usage,
             const std::vector<char> &image_set, bool decode, const std::shared_ptr<Sampler> &sampler,
             const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LFWDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] task Set the task type of reading LFW, support "people" and "pairs".
  /// \param[in] usage The image split to use, support "10fold", "train", "test" and "all".
  /// \param[in] image_set Image set of image funneling to use, support "original", "funneled" and "deepfunneled".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LFWDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task, const std::vector<char> &usage,
             const std::vector<char> &image_set, bool decode, const Sampler *sampler,
             const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LFWDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] task Set the task type of reading LFW, support "people" and "pairs".
  /// \param[in] usage The image split to use, support "10fold", "train", "test" and "all".
  /// \param[in] image_set Image set of image funneling to use, support "original", "funneled" and "deepfunneled".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LFWDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task, const std::vector<char> &usage,
             const std::vector<char> &image_set, bool decode, const std::reference_wrapper<Sampler> &sampler,
             const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of LFWDataset.
  ~LFWDataset() override = default;
};

/// \brief Function to create a LFWDataset.
/// \note When usage is 'people', the generated dataset has two columns ["image", "label"];
///     When task is 'pairs', the generated dataset has three columns ["image1", "image2", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] task Set the task type of reading LFW, support "people" and "pairs" (default = "people").
/// \param[in] usage The image split to use, support "10fold", "train", "test" and "all" (default = "all",
///     will read samples including train and test).
/// \param[in] image_set Image set of image funneling to use, support "original", "funneled" and "deepfunneled"
///     (default = "funneled").
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the LFWDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/lfw_dataset_directory";
///      std::shared_ptr<Dataset> ds = LFW(folder_path);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In LFW dataset, each data dictionary owns keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<LFWDataset> DATASET_API
LFW(const std::string &dataset_dir, const std::string &task = "people", const std::string &usage = "all",
    const std::string &image_set = "funneled", bool decode = false,
    const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
    const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LFWDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      StringToChar(image_set), decode, sampler, cache);
}

/// \brief Function to create a LFWDataset.
/// \note When usage is 'people', the generated dataset has two columns ["image", "label"];
///     When task is 'pairs', the generated dataset has three columns ["image1", "image2", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] task Set the task type of reading LFW, support "people" and "pairs".
/// \param[in] usage The image split to use, support "10fold", "train", "test" and "all".
/// \param[in] image_set Image set of image funneling to use, support "original", "funneled" and "deepfunneled".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the LFWDataset.
inline std::shared_ptr<LFWDataset> DATASET_API LFW(const std::string &dataset_dir, const std::string &task,
                                                   const std::string &usage, const std::string &image_set, bool decode,
                                                   const Sampler *sampler,
                                                   const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LFWDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      StringToChar(image_set), decode, sampler, cache);
}

/// \brief Function to create a LFWDataset.
/// \note When usage is 'people', the generated dataset has two columns ["image", "label"];
///     When task is 'pairs', the generated dataset has three columns ["image1", "image2", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] task Set the task type of reading LFW, support "people" and "pairs".
/// \param[in] usage The image split to use, support "10fold", "train", "test" and "all".
/// \param[in] image_set Image set of image funneling to use, support "original", "funneled" and "deepfunneled".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the LFWDataset.
inline std::shared_ptr<LFWDataset> DATASET_API LFW(const std::string &dataset_dir, const std::string &task,
                                                   const std::string &usage, const std::string &image_set, bool decode,
                                                   const std::reference_wrapper<Sampler> &sampler,
                                                   const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LFWDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      StringToChar(image_set), decode, sampler, cache);
}

/// \class LibriTTSDataset
/// \brief A source dataset for reading and parsing LibriTTSDataset dataset.
class DATASET_API LibriTTSDataset : public Dataset {
 public:
  /// \brief Constructor of LibriTTSDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of LibriTTS, can be "dev-clean", "dev-other", "test-clean",
  ///     "test-other", "train-clean-100", "train-clean-360", "train-other-500" or "all".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LibriTTSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                  const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LibriTTSDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of LibriTTS, can be "dev-clean", "dev-other", "test-clean",
  ///     "test-other", "train-clean-100", "train-clean-360", "train-other-500" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LibriTTSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LibriTTSDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of LibriTTS, can be "dev-clean", "dev-other", "test-clean",
  ///     "test-other", "train-clean-100", "train-clean-360", "train-other-500" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LibriTTSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                  const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of LibriTTSDataset.
  ~LibriTTSDataset() override = default;
};

/// \brief Function to create a LibriTTSDataset.
/// \note The generated dataset has seven columns ['waveform', 'sample_rate', 'original_text', 'normalized_text',
///     'speaker_id', 'chapter_id', 'utterance_id'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of LibriTTS, can be "dev-clean", "dev-other", "test-clean", "test-other",
///     "train-clean-100", "train-clean-360", "train-other-500", or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the LibriTTSDataset.
/// \par Example
/// \code
///      /* Define dataset path and LibriTTS object */
///      std::string folder_path = "/path/to/libri_tts_dataset_directory";
///      std::shared_ptr<Dataset> ds = LibriTTS(folder_path);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In LibriTTS dataset, each data dictionary has seven columns ["waveform", "sample_rate",
///         "original_text", "normalized_text", "speaker_id", "chapter_id", "utterance_id"].*/
///      auto waveform = row["waveform"];
/// \endcode
inline std::shared_ptr<LibriTTSDataset> DATASET_API
LibriTTS(const std::string &dataset_dir, const std::string &usage = "all",
         const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LibriTTSDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a LibriTTSDataset.
/// \note The generated dataset has seven columns ['waveform', 'sample_rate', 'original_text', 'normalized_text',
///     'speaker_id', 'chapter_id', 'utterance_id'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of LibriTTS, can be "dev-clean", "dev-other", "test-clean", "test-other",
///     "train-clean-100", "train-clean-360", "train-other-500", or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the LibriTTSDataset.
inline std::shared_ptr<LibriTTSDataset> DATASET_API LibriTTS(const std::string &dataset_dir, const std::string &usage,
                                                             const Sampler *sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LibriTTSDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a LibriTTSDataset.
/// \note The generated dataset has seven columns ['waveform', 'sample_rate', 'original_text', 'normalized_text',
///     'speaker_id', 'chapter_id', 'utterance_id'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of LibriTTS, can be "dev-clean", "dev-other", "test-clean", "test-other",
///     "train-clean-100", "train-clean-360", "train-other-500", or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the LibriTTSDataset.
inline std::shared_ptr<LibriTTSDataset> DATASET_API LibriTTS(const std::string &dataset_dir, const std::string &usage,
                                                             const std::reference_wrapper<Sampler> &sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LibriTTSDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class LJSpeechDataset
/// \brief A source dataset for reading and parsing LJSpeech dataset.
class DATASET_API LJSpeechDataset : public Dataset {
 public:
  /// \brief Constructor of LJSpeechDataset.
  /// \param[in] dataset_file The dataset file to be read.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  LJSpeechDataset(const std::vector<char> &dataset_dir, const std::shared_ptr<Sampler> &sampler,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LJSpeechDataset.
  /// \param[in] dataset_file The dataset file to be read.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LJSpeechDataset(const std::vector<char> &dataset_dir, const std::reference_wrapper<Sampler> &sampler,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LJSpeechDataset.
  /// \param[in] dataset_file The dataset file to be read.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LJSpeechDataset(const std::vector<char> &dataset_dir, const Sampler *sampler,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of LJSpeechDataset.
  ~LJSpeechDataset() override = default;
};

/// \brief Function to create a LJSpeech Dataset.
/// \note The generated dataset has four columns ["waveform", "sample_rate", "transcription",
///     "normalized_transcription"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use. (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<LJSpeechDataset> DATASET_API
LJSpeech(const std::string &dataset_dir, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LJSpeechDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \brief Function to create a LJSpeech Dataset.
/// \note The generated dataset has four columns ["waveform", "sample_rate", "transcription",
///     "normalized_transcription"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<LJSpeechDataset> DATASET_API LJSpeech(const std::string &dataset_dir, Sampler *sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LJSpeechDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \brief Function to create a LJSpeech Dataset.
/// \note The generated dataset has four columns ["waveform", "sample_rate", "transcription",
///     "normalized_transcription"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<LJSpeechDataset> DATASET_API LJSpeech(const std::string &dataset_dir,
                                                             const std::reference_wrapper<Sampler> &sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LJSpeechDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \class LSUNDataset
/// \brief A source dataset for reading LSUN datast.
class DATASET_API LSUNDataset : public Dataset {
 public:
  /// \brief Constructor of LSUNDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
  /// \param[in] classes Classes list to load.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LSUNDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
              const std::vector<std::vector<char>> &classes, bool decode, const std::shared_ptr<Sampler> &sampler,
              const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LSUNDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
  /// \param[in] classes Classes list to load.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LSUNDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
              const std::vector<std::vector<char>> &classes, bool decode, const Sampler *sampler,
              const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of LSUNDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
  /// \param[in] classes Classes list to load.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  LSUNDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
              const std::vector<std::vector<char>> &classes, bool decode, const std::reference_wrapper<Sampler> sampler,
              const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of LSUNDataset.
  ~LSUNDataset() override = default;
};

/// \brief Function to create a LSUNDataset.
/// \note The generated dataset has two columns "image" and "label".
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all` (Default=`all`).
/// \param[in] classes Classes list to load, such as 'bedroom', 'classroom' (Default={}, means load all classes).
/// \param[in] decode Decode the images after reading (Default=false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (Default = RandomSampler()).
/// \param[in] cache Tensor cache to use (Default=nullptr, which means no cache is used).
/// \return Shared pointer to the current LSUNDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/lsun_dataset_directory";
///      std::shared_ptr<Dataset> ds = LSUN(folder_path, "all");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In LSUNDataset, each data dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<LSUNDataset> DATASET_API
LSUN(const std::string &dataset_dir, const std::string &usage = "all", const std::vector<std::string> &classes = {},
     bool decode = false, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LSUNDataset>(StringToChar(dataset_dir), StringToChar(usage), VectorStringToChar(classes),
                                       decode, sampler, cache);
}

/// \brief Function to create a LSUNDataset.
/// \note The generated dataset has two columns "image" and "label".
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
/// \param[in] classes Classes list to load.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (Default=nullptr, which means no cache is used).
/// \return Shared pointer to the current LSUNDataset.
inline std::shared_ptr<LSUNDataset> DATASET_API LSUN(const std::string &dataset_dir, const std::string &usage,
                                                     const std::vector<std::string> &classes, bool decode,
                                                     const Sampler *sampler,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LSUNDataset>(StringToChar(dataset_dir), StringToChar(usage), VectorStringToChar(classes),
                                       decode, sampler, cache);
}

/// \brief Function to create a LSUNDataset.
/// \note The generated dataset has two columns "image" and "label".
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Dataset splits of LSUN, can be `train`, `valid`, `test` or `all`.
/// \param[in] classes Classes list to load.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (Default=nullptr, which means no cache is used).
/// \return Shared pointer to the current LSUNDataset.
inline std::shared_ptr<LSUNDataset> DATASET_API LSUN(const std::string &dataset_dir, const std::string &usage,
                                                     const std::vector<std::string> &classes, bool decode,
                                                     const std::reference_wrapper<Sampler> sampler,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<LSUNDataset>(StringToChar(dataset_dir), StringToChar(usage), VectorStringToChar(classes),
                                       decode, sampler, cache);
}

/// \class ManifestDataset
/// \brief A source dataset for reading and parsing Manifest dataset.
class DATASET_API ManifestDataset : public Dataset {
 public:
  /// \brief Constructor of ManifestDataset.
  /// \param[in] dataset_file The dataset file to be read.
  /// \param[in] usage Part of dataset of ManifestDataset, can be "train", "eval" or "inference" data (default="train").
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
  ///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                  const std::shared_ptr<Sampler> &sampler, const std::map<std::vector<char>, int32_t> &class_indexing,
                  bool decode, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of ManifestDataset.
  /// \param[in] dataset_file The dataset file to be read.
  /// \param[in] usage Part of dataset of ManifestDataset, can be "train", "eval" or "inference" data.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
  ///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage, const Sampler *sampler,
                  const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of ManifestDataset.
  /// \param[in] dataset_file The dataset file to be read.
  /// \param[in] usage Part of dataset of ManifestDataset, can be "train", "eval" or "inference" data.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
  ///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
  /// \param[in] decode Decode the images after reading (default=false).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                  const std::reference_wrapper<Sampler> &sampler,
                  const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of ManifestDataset.
  ~ManifestDataset() override = default;
};

/// \brief Function to create a ManifestDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_file The dataset file to be read.
/// \param[in] usage Part of dataset of ManifestDataset, can be "train", "eval" or "inference" data (default="train").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the ManifestDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string file_path = "/path/to/manifest_file";
///      std::shared_ptr<Dataset> ds = Manifest(file_path);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In Manifest dataset, each data dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<ManifestDataset> DATASET_API
Manifest(const std::string &dataset_file, const std::string &usage = "train",
         const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
         const std::map<std::string, int32_t> &class_indexing = {}, bool decode = false,
         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ManifestDataset>(StringToChar(dataset_file), StringToChar(usage), sampler,
                                           MapStringToChar(class_indexing), decode, cache);
}

/// \brief Function to create a ManifestDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_file The dataset file to be read.
/// \param[in] usage Part of dataset of ManifestDataset, can be "train", "eval" or "inference" data.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the ManifestDataset.
inline std::shared_ptr<ManifestDataset> DATASET_API Manifest(const std::string &dataset_file, const std::string &usage,
                                                             const Sampler *sampler,
                                                             const std::map<std::string, int32_t> &class_indexing = {},
                                                             bool decode = false,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ManifestDataset>(StringToChar(dataset_file), StringToChar(usage), sampler,
                                           MapStringToChar(class_indexing), decode, cache);
}

/// \brief Function to create a ManifestDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_file The dataset file to be read.
/// \param[in] usage Part of dataset of ManifestDataset, can be "train", "eval" or "inference" data.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] class_indexing A str-to-int mapping from label name to index (default={}, the folder
///     names will be sorted alphabetically and each class will be given a unique index starting from 0).
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the ManifestDataset.
inline std::shared_ptr<ManifestDataset> DATASET_API Manifest(const std::string &dataset_file, const std::string &usage,
                                                             const std::reference_wrapper<Sampler> &sampler,
                                                             const std::map<std::string, int32_t> &class_indexing = {},
                                                             bool decode = false,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<ManifestDataset>(StringToChar(dataset_file), StringToChar(usage), sampler,
                                           MapStringToChar(class_indexing), decode, cache);
}

/// \class MindDataDataset
/// \brief A source dataset for reading and parsing MindRecord dataset.
class DATASET_API MindDataDataset : public Dataset {
 public:
  /// \brief Constructor of MindDataDataset.
  /// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
  ///     in the same path will be found and loaded automatically.
  /// \param[in] columns_list List of columns to be read (default={}).
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()),
  ///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
  /// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
  /// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
  /// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  ///    ShuffleMode::kInfile - Shuffle samples in file.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MindDataDataset(const std::vector<char> &dataset_file, const std::vector<std::vector<char>> &columns_list,
                  const std::shared_ptr<Sampler> &sampler, const nlohmann::json *padded_sample, int64_t num_padded,
                  ShuffleMode shuffle_mode = ShuffleMode::kGlobal,
                  const std::shared_ptr<DatasetCache> &cache = nullptr);

  /// \brief Constructor of MindDataDataset.
  /// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
  ///     in the same path will be found and loaded automatically.
  /// \param[in] columns_list List of columns to be read.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  ///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
  /// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
  /// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
  /// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  ///    ShuffleMode::kInfile - Shuffle samples in file.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MindDataDataset(const std::vector<char> &dataset_file, const std::vector<std::vector<char>> &columns_list,
                  const Sampler *sampler, const nlohmann::json *padded_sample, int64_t num_padded,
                  ShuffleMode shuffle_mode = ShuffleMode::kGlobal,
                  const std::shared_ptr<DatasetCache> &cache = nullptr);

  /// \brief Constructor of MindDataDataset.
  /// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
  ///     in the same path will be found and loaded automatically.
  /// \param[in] columns_list List of columns to be read.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  ///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
  /// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
  /// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
  /// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  ///    ShuffleMode::kInfile - Shuffle samples in file.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MindDataDataset(const std::vector<char> &dataset_file, const std::vector<std::vector<char>> &columns_list,
                  const std::reference_wrapper<Sampler> &sampler, const nlohmann::json *padded_sample,
                  int64_t num_padded, ShuffleMode shuffle_mode = ShuffleMode::kGlobal,
                  const std::shared_ptr<DatasetCache> &cache = nullptr);

  /// \brief Constructor of MindDataDataset.
  /// \param[in] dataset_files List of dataset files to be read directly.
  /// \param[in] columns_list List of columns to be read.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  ///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
  /// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
  /// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
  /// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  ///    ShuffleMode::kInfile - Shuffle data within each file.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                  const std::vector<std::vector<char>> &columns_list, const std::shared_ptr<Sampler> &sampler,
                  const nlohmann::json *padded_sample, int64_t num_padded,
                  ShuffleMode shuffle_mode = ShuffleMode::kGlobal,
                  const std::shared_ptr<DatasetCache> &cache = nullptr);

  /// \brief Constructor of MindDataDataset.
  /// \param[in] dataset_files List of dataset files to be read directly.
  /// \param[in] columns_list List of columns to be read.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  ///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
  /// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
  /// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
  /// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  ///    ShuffleMode::kInfile - Shuffle data within each file.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                  const std::vector<std::vector<char>> &columns_list, const Sampler *sampler,
                  const nlohmann::json *padded_sample, int64_t num_padded,
                  ShuffleMode shuffle_mode = ShuffleMode::kGlobal,
                  const std::shared_ptr<DatasetCache> &cache = nullptr);

  /// \brief Constructor of MindDataDataset.
  /// \param[in] dataset_files List of dataset files to be read directly.
  /// \param[in] columns_list List of columns to be read.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  ///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
  /// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
  /// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
  /// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
  ///    Can be any of:
  ///    ShuffleMode::kFalse - No shuffling is performed.
  ///    ShuffleMode::kFiles - Shuffle files only.
  ///    ShuffleMode::kGlobal - Shuffle both the files and samples.
  ///    ShuffleMode::kInfile - Shuffle samples in file.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                  const std::vector<std::vector<char>> &columns_list, const std::reference_wrapper<Sampler> &sampler,
                  const nlohmann::json *padded_sample, int64_t num_padded,
                  ShuffleMode shuffle_mode = ShuffleMode::kGlobal,
                  const std::shared_ptr<DatasetCache> &cache = nullptr);

  /// \brief Destructor of MindDataDataset.
  ~MindDataDataset() override = default;
};

/// \brief Function to create a MindDataDataset.
/// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
///     in the same path will be found and loaded automatically.
/// \param[in] columns_list List of columns to be read (default={}).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()),
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
///    ShuffleMode::kInfile - Shuffle samples in file.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current MindDataDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string file_path = "/path/to/mindrecord_file";
///      std::vector<std::string> column_names = {"data", "file_name", "label"};
///      std::shared_ptr<Dataset> ds = MindData(file_path, column_names);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: As we defined before, each data dictionary owns keys "data", "file_name" and "label" */
///      auto data = row["data"];
/// \endcode
inline std::shared_ptr<MindDataDataset> DATASET_API
MindData(const std::string &dataset_file, const std::vector<std::string> &columns_list = {},
         const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
         nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0,
         ShuffleMode shuffle_mode = ShuffleMode::kGlobal, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MindDataDataset>(StringToChar(dataset_file), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded, shuffle_mode, cache);
}

/// \brief Function to create a MindDataDataset.
/// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
///     in the same path will be found and loaded automatically.
/// \param[in] columns_list List of columns to be read.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
///    ShuffleMode::kInfile - Shuffle samples in file.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MindDataDataset.
inline std::shared_ptr<MindDataDataset> DATASET_API
MindData(const std::string &dataset_file, const std::vector<std::string> &columns_list, const Sampler *sampler,
         nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0,
         ShuffleMode shuffle_mode = ShuffleMode::kGlobal, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MindDataDataset>(StringToChar(dataset_file), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded, shuffle_mode, cache);
}
/// \brief Function to create a MindDataDataset.
/// \param[in] dataset_file File name of one component of a mindrecord source. Other files with identical source
///     in the same path will be found and loaded automatically.
/// \param[in] columns_list List of columns to be read.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
///    ShuffleMode::kInfile - Shuffle samples in file.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MindDataDataset.
inline std::shared_ptr<MindDataDataset> DATASET_API MindData(
  const std::string &dataset_file, const std::vector<std::string> &columns_list,
  const std::reference_wrapper<Sampler> &sampler, nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0,
  ShuffleMode shuffle_mode = ShuffleMode::kGlobal, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MindDataDataset>(StringToChar(dataset_file), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded, shuffle_mode, cache);
}

/// \brief Function to create a MindDataDataset.
/// \param[in] dataset_files List of dataset files to be read directly.
/// \param[in] columns_list List of columns to be read (default={}).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()),
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
///    ShuffleMode::kInfile - Shuffle samples in file.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MindDataDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string file_path1 = "/path/to/mindrecord_file1";
///      std::string file_path2 = "/path/to/mindrecord_file2";
///      std::vector<std::string> file_list = {file_path1, file_path2};
///      std::vector<std::string> column_names = {"data", "file_name", "label"};
///      std::shared_ptr<Dataset> ds = MindData(file_list, column_names);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: As we defined before, each data dictionary owns keys "data", "file_name" and "label" */
///      auto data = row["data"];
/// \endcode
inline std::shared_ptr<MindDataDataset> DATASET_API
MindData(const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list = {},
         const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
         nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0,
         ShuffleMode shuffle_mode = ShuffleMode::kGlobal, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MindDataDataset>(VectorStringToChar(dataset_files), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded, shuffle_mode, cache);
}

/// \brief Function to create a MindDataDataset.
/// \param[in] dataset_files List of dataset files to be read directly.
/// \param[in] columns_list List of columns to be read.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
///    ShuffleMode::kInfile - Shuffle data within each file.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MindDataDataset.
inline std::shared_ptr<MindDataDataset> DATASET_API
MindData(const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list,
         const Sampler *sampler, nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0,
         ShuffleMode shuffle_mode = ShuffleMode::kGlobal, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MindDataDataset>(VectorStringToChar(dataset_files), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded, shuffle_mode, cache);
}

/// \brief Function to create a MindDataDataset.
/// \param[in] dataset_files List of dataset files to be read directly.
/// \param[in] columns_list List of columns to be read.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
///     supported sampler list: SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler, DistributedSampler.
/// \param[in] padded_sample Samples will be appended to dataset, where keys are the same as column_list.
/// \param[in] num_padded Number of padding samples. Dataset size plus num_padded should be divisible by num_shards.
/// \param[in] shuffle_mode The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///    Can be any of:
///    ShuffleMode::kFalse - No shuffling is performed.
///    ShuffleMode::kFiles - Shuffle files only.
///    ShuffleMode::kGlobal - Shuffle both the files and samples.
///    ShuffleMode::kInfile - Shuffle samples in file.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MindDataDataset.
inline std::shared_ptr<MindDataDataset> DATASET_API MindData(
  const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list,
  const std::reference_wrapper<Sampler> &sampler, nlohmann::json *padded_sample = nullptr, int64_t num_padded = 0,
  ShuffleMode shuffle_mode = ShuffleMode::kGlobal, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MindDataDataset>(VectorStringToChar(dataset_files), VectorStringToChar(columns_list), sampler,
                                           padded_sample, num_padded, shuffle_mode, cache);
}

/// \class MnistDataset
/// \brief A source dataset for reading and parsing MNIST dataset.
class DATASET_API MnistDataset : public Dataset {
 public:
  /// \brief Constructor of MnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of MNIST, can be "train", "test" or "all" (default = "all").
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
               const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of MnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of MNIST, can be "train", "test" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of MnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of MNIST, can be "train", "test" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
               const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of MnistDataset.
  ~MnistDataset() override = default;
};

/// \brief Function to create a MnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of MNIST, can be "train", "test" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MnistDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/mnist_dataset_directory";
///      std::shared_ptr<Dataset> ds = Mnist(folder_path, "all", std::make_shared<RandomSampler>(false, 20));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In MNIST dataset, each dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<MnistDataset> DATASET_API
Mnist(const std::string &dataset_dir, const std::string &usage = "all",
      const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
      const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a MnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of MNIST, can be "train", "test" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MnistDataset.
inline std::shared_ptr<MnistDataset> DATASET_API Mnist(const std::string &dataset_dir, const std::string &usage,
                                                       const Sampler *sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a MnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of MNIST, can be "train", "test" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the MnistDataset.
inline std::shared_ptr<MnistDataset> DATASET_API Mnist(const std::string &dataset_dir, const std::string &usage,
                                                       const std::reference_wrapper<Sampler> &sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<MnistDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class Multi30kDataset
/// \brief A source dataset that reads and parses Multi30k dataset.
class DATASET_API Multi30kDataset : public Dataset {
 public:
  /// \brief Constructor of Multi30kDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of MULTI30K, can be "train", "test", "valid" or "all".
  /// \param[in] language_pair List containing text and translation language.
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode::kFalse - No shuffling is performed.
  ///     ShuffleMode::kFiles - Shuffle files only.
  ///     ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  Multi30kDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                  const std::vector<std::vector<char>> &language_pair, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of Multi30kDataset.
  ~Multi30kDataset() override = default;
};

/// \brief Function to create a Multi30kDataset.
/// \note The generated dataset has two columns ["text", "translation"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of MULTI30K, can be "train", "test", "valid" or "all" (default = "all").
/// \param[in] language_pair List containing text and translation language (default = {"en", "de"}).
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the Multi30kDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_dir = "/path/to/multi30k_dataset_directory";
///      std::shared_ptr<Dataset> ds = Multi30k(dataset_dir, "all");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In Multi30kdataset, each dictionary has keys "text" and "translation" */
///      auto text = row["text"];
/// \endcode
inline std::shared_ptr<Multi30kDataset> DATASET_API
Multi30k(const std::string &dataset_dir, const std::string &usage = "all",
         const std::vector<std::string> &language_pair = {"en", "de"}, int64_t num_samples = 0,
         ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Multi30kDataset>(StringToChar(dataset_dir), StringToChar(usage),
                                           VectorStringToChar(language_pair), num_samples, shuffle, num_shards,
                                           shard_id, cache);
}

/// \class OmniglotDataset
/// \brief A source dataset for reading and parsing Omniglot dataset.
class DATASET_API OmniglotDataset : public Dataset {
 public:
  /// \brief Constructor of OmniglotDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] background A flag to use background dataset or evaluation dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  OmniglotDataset(const std::vector<char> &dataset_dir, bool background, bool decode,
                  const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of OmniglotDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] background A flag to use background dataset or evaluation dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  OmniglotDataset(const std::vector<char> &dataset_dir, bool background, bool decode, const Sampler *sampler,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of OmniglotDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] background A flag to use background dataset or evaluation dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  OmniglotDataset(const std::vector<char> &dataset_dir, bool background, bool decode,
                  const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// Destructor of OmniglotDataset.
  ~OmniglotDataset() override = default;
};

/// \brief Function to create an OmniglotDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] background A flag to use background dataset or evaluation dataset (Default=true).
/// \param[in] decode Decode the images after reading (Default=false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current OmniglotDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/omniglot_dataset_directory";
///      std::shared_ptr<Dataset> ds = Omniglot(folder_path, true, false, std::make_shared<RandomSampler>(false, 5));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In Omniglot dataset, each dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<OmniglotDataset> DATASET_API
Omniglot(const std::string &dataset_dir, bool background = true, bool decode = false,
         const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<OmniglotDataset>(StringToChar(dataset_dir), background, decode, sampler, cache);
}

/// \brief Function to create an OmniglotDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] background A flag to use background dataset or evaluation dataset.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current OmniglotDataset.
inline std::shared_ptr<OmniglotDataset> DATASET_API Omniglot(const std::string &dataset_dir, bool background,
                                                             bool decode, const Sampler *sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<OmniglotDataset>(StringToChar(dataset_dir), background, decode, sampler, cache);
}

/// \brief Function to create an OmniglotDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] background A flag to use background dataset or evaluation dataset.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current OmniglotDataset.
inline std::shared_ptr<OmniglotDataset> DATASET_API Omniglot(const std::string &dataset_dir, bool background,
                                                             bool decode,
                                                             const std::reference_wrapper<Sampler> &sampler,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<OmniglotDataset>(StringToChar(dataset_dir), background, decode, sampler, cache);
}

/// \class PennTreebankDataset
/// \brief A source dataset for reading and parsing PennTreebank dataset.
class DATASET_API PennTreebankDataset : public Dataset {
 public:
  /// \brief Constructor of PennTreebank Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data list txt file to be read, can be "train", "test", 'valid' or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  PennTreebankDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                      ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                      const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of PennTreebankDataset.
  ~PennTreebankDataset() override = default;
};

/// \brief Function to create a PennTreebank Dataset.
/// \note The generated dataset has one column ['text'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train" , 'valid' or "test" (default = "all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode.kFalse - No shuffling is performed.
///     ShuffleMode.kFiles - Shuffle files only.
///     ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] usage One of "all", "train", "valid" or "test" (default = "all").
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the TextFileDataset.
inline std::shared_ptr<PennTreebankDataset> DATASET_API
PennTreebank(const std::string &dataset_dir, const std::string &usage = "all", int64_t num_samples = 0,
             ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<PennTreebankDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                               num_shards, shard_id, cache);
}

/// \class PhotoTourDataset
/// \brief A source dataset for reading and parsing PhotoTour dataset.
class DATASET_API PhotoTourDataset : public Dataset {
 public:
  /// \brief Constructor of PhotoTourDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Name of the dataset to load, should be one of 'notredame', 'yosemite', 'liberty',
  ///     'notredame_harris', 'yosemite_harris' or 'liberty_harris'.
  /// \param[in] usage Part of dataset of PhotoTour, can be `train` or `test`.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  PhotoTourDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name, const std::vector<char> &usage,
                   const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of PhotoTourDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Name of the dataset to load, should be one of 'notredame', 'yosemite', 'liberty',
  ///     'notredame_harris', 'yosemite_harris' or 'liberty_harris'.
  /// \param[in] usage Part of dataset of PhotoTour, can be `train` or `test`.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  PhotoTourDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name, const std::vector<char> &usage,
                   const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of PhotoTourDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] name Name of the dataset to load, should be one of 'notredame', 'yosemite', 'liberty',
  ///     'notredame_harris', 'yosemite_harris' or 'liberty_harris'.
  /// \param[in] usage Part of dataset of PhotoTour, can be `train` or `test`.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  PhotoTourDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name, const std::vector<char> &usage,
                   const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of PhotoTourDataset.
  ~PhotoTourDataset() override = default;
};

/// \brief Function to create a PhotoTourDataset.
/// \note If usage is 'train', the generated dataset has one column ["image"], else
///     three columns ["image1", "image2", "matches"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Name of the dataset to load, should be one of 'notredame', 'yosemite', 'liberty',
///     'notredame_harris', 'yosemite_harris' or 'liberty_harris'.
/// \param[in] usage Part of dataset of PhotoTour, can be `train` or `test` (default="train").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples
///     from the dataset. If sampler is not given, a `RandomSampler` will
///     be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current PhotoTourDataset.
inline std::shared_ptr<PhotoTourDataset> DATASET_API
PhotoTour(const std::string &dataset_dir, const std::string &name, const std::string &usage = "train",
          const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<PhotoTourDataset>(StringToChar(dataset_dir), StringToChar(name), StringToChar(usage), sampler,
                                            cache);
}

/// \brief Function to create a PhotoTourDataset.
/// \note If usage is 'train', the generated dataset has one column ["image"], else
///     three columns ["image1", "image2", "matches"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Name of the dataset to load, should be one of 'notredame', 'yosemite', 'liberty',
///     'notredame_harris', 'yosemite_harris' or 'liberty_harris'.
/// \param[in] usage Part of dataset of PhotoTour, can be `train` or `test`.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current PhotoTourDataset.
inline std::shared_ptr<PhotoTourDataset> DATASET_API PhotoTour(const std::string &dataset_dir, const std::string &name,
                                                               const std::string &usage, const Sampler *sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<PhotoTourDataset>(StringToChar(dataset_dir), StringToChar(name), StringToChar(usage), sampler,
                                            cache);
}

/// \brief Function to create a PhotoTourDataset.
/// \note If usage is 'train', the generated dataset has one column ["image"], else
///     three columns ["image1", "image2", "matches"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] name Name of the dataset to load, should be one of 'notredame', 'yosemite', 'liberty',
///     'notredame_harris', 'yosemite_harris' or 'liberty_harris'.
/// \param[in] usage Part of dataset of PhotoTour, can be `train` or `test`.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current PhotoTourDataset.
inline std::shared_ptr<PhotoTourDataset> DATASET_API PhotoTour(const std::string &dataset_dir, const std::string &name,
                                                               const std::string &usage,
                                                               const std::reference_wrapper<Sampler> &sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<PhotoTourDataset>(StringToChar(dataset_dir), StringToChar(name), StringToChar(usage), sampler,
                                            cache);
}

/// \class Places365Dataset
/// \brief A source dataset that reads and parses Places365 dataset.
class DATASET_API Places365Dataset : public Dataset {
 public:
  /// \brief Constructor of Places365Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Dataset splits of Places365, can be `train-standard`, `train-challenge` or `val`.
  /// \param[in] small Use the small images instead of the high resolution ones.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  Places365Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool small, bool decode,
                   const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Places365Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Dataset splits of Places365, can be `train-standard`, `train-challenge` or `val`.
  /// \param[in] small Use the small images instead of the high resolution ones.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Places365Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool small, bool decode,
                   const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of Places365Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Dataset splits of Places365, can be `train-standard`, `train-challenge` or `val`.
  /// \param[in] small Use the small images instead of the high resolution ones.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  Places365Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool small, bool decode,
                   const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of Places365Dataset.
  ~Places365Dataset() override = default;
};

/// \brief Function to create a Places365Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Dataset splits of Places365, can be `train-standard`, `train-challenge`
///     or `val` (default="train-standard").
/// \param[in] small Use the small images instead of the high resolution ones (default=false).
/// \param[in] decode Decode the images after reading (default=true).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples
///     from the dataset. If sampler is not given, a `RandomSampler` will
///     be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Places365Dataset.
inline std::shared_ptr<Places365Dataset> DATASET_API
Places365(const std::string &dataset_dir, const std::string &usage = "train-standard", const bool small = false,
          const bool decode = true, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Places365Dataset>(StringToChar(dataset_dir), StringToChar(usage), small, decode, sampler,
                                            cache);
}

/// \brief Function to create a Places365Dataset
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Dataset splits of Places365, can be `train-standard`, `train-challenge` or `val`.
/// \param[in] small Use the small images instead of the high resolution ones.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Places365Dataset.
inline std::shared_ptr<Places365Dataset> DATASET_API Places365(const std::string &dataset_dir, const std::string &usage,
                                                               const bool small, const bool decode,
                                                               const Sampler *sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Places365Dataset>(StringToChar(dataset_dir), StringToChar(usage), small, decode, sampler,
                                            cache);
}

/// \brief Function to create a Places365Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Dataset splits of Places365, can be `train-standard`, `train-challenge` or `val`.
/// \param[in] small Use the small images instead of the high resolution ones.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr which means no cache is used).
/// \return Shared pointer to the current Places365Dataset.
inline std::shared_ptr<Places365Dataset> DATASET_API Places365(const std::string &dataset_dir, const std::string &usage,
                                                               const bool small, const bool decode,
                                                               const std::reference_wrapper<Sampler> &sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<Places365Dataset>(StringToChar(dataset_dir), StringToChar(usage), small, decode, sampler,
                                            cache);
}

/// \class QMnistDataset
/// \brief A source dataset that reads and parses QMNIST dataset.
class DATASET_API QMnistDataset : public Dataset {
 public:
  /// \brief Constructor of QMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of QMNIST, can be "train", "test", "test10k", "test50k", "nist" or "all".
  /// \param[in] compat Whether the label for each example is class number (compat=true)
  ///     or the full QMNIST information (compat=false).
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  QMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool compat,
                const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of QMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of QMNIST, can be "train", "test", "test10k", "test50k", "nist" or "all".
  /// \param[in] compat Whether the label for each example is class number (compat=true)
  ///     or the full QMNIST information (compat=false).
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  QMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool compat,
                const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of QMnistDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of QMNIST, can be "train", "test", "test10k", "test50k", "nist" or "all".
  /// \param[in] compat Whether the label for each example is class number (compat=true)
  ///     or the full QMNIST information (compat=false).
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  QMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool compat,
                const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of QMnistDataset.
  ~QMnistDataset() override = default;
};

/// \brief Function to create a QMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of QMNIST, can be "train", "test", "test10k", "test50k", "nist" or "all" (default = "all").
/// \param[in] compat Whether the label for each example is class number or the full QMNIST information
///     (default = true).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the QMnistDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/qmnist_dataset_directory";
///      std::shared_ptr<Dataset> ds = QMnist(folder_path, "train", true, std::make_shared<RandomSampler>(false, 5));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In QMNIST dataset, each dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<QMnistDataset> DATASET_API
QMnist(const std::string &dataset_dir, const std::string &usage = "all", bool compat = true,
       const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<QMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), compat, sampler, cache);
}

/// \brief Function to create a QMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of QMNIST, can be "train", "test", "test10k", "test50k", "nist" or "all".
/// \param[in] compat Whether the label for each example is class number or the full QMNIST information.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the QMnistDataset.
inline std::shared_ptr<QMnistDataset> DATASET_API QMnist(const std::string &dataset_dir, const std::string &usage,
                                                         bool compat, const Sampler *sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<QMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), compat, sampler, cache);
}

/// \brief Function to create a QMnistDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of QMNIST, can be "train", "test", "test10k", "test50k", "nist" or "all".
/// \param[in] compat Whether the label for each example is class number or the full QMNIST information.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the QMnistDataset.
inline std::shared_ptr<QMnistDataset> DATASET_API QMnist(const std::string &dataset_dir, const std::string &usage,
                                                         bool compat, const std::reference_wrapper<Sampler> &sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<QMnistDataset>(StringToChar(dataset_dir), StringToChar(usage), compat, sampler, cache);
}

/// \brief Function to create a ConcatDataset.
/// \note Reload "+" operator to concat two datasets.
/// \param[in] datasets1 Shared pointer to the first dataset to be concatenated.
/// \param[in] datasets2 Shared pointer to the second dataset to be concatenated.
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<ConcatDataset> DATASET_API operator+(const std::shared_ptr<Dataset> &datasets1,
                                                            const std::shared_ptr<Dataset> &datasets2) {
  return std::make_shared<ConcatDataset>(std::vector({datasets1, datasets2}));
}

/// \class RandomDataDataset
/// \brief A source dataset that generates random data.
class DATASET_API RandomDataDataset : public Dataset {
 public:
  /// \brief Constructor of RandomDataDataset.
  /// \param[in] total_rows Number of rows for the dataset to generate (default=0, number of rows is random).
  /// \param[in] schema SchemaObj to set column type, data type and data shape.
  /// \param[in] columns_list List of columns to be read (default={}, read all columns).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  RandomDataDataset(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema,
                    const std::vector<std::vector<char>> &columns_list, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of RandomDataDataset.
  /// \param[in] total_rows Number of rows for the dataset to generate (default=0, number of rows is random).
  /// \param[in] schema_path Path of schema file.
  /// \param[in] columns_list List of columns to be read (default={}, read all columns).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  RandomDataDataset(const int32_t &total_rows, const std::vector<char> &schema_path,
                    const std::vector<std::vector<char>> &columns_list, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of RandomDataDataset.
  ~RandomDataDataset() override = default;
};

/// \brief Function to create a RandomDataset.
/// \param[in] total_rows Number of rows for the dataset to generate (default=0, number of rows is random).
/// \param[in] schema SchemaObj to set column type, data type and data shape.
/// \param[in] columns_list List of columns to be read (default={}, read all columns).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the RandomDataset.
/// \par Example
/// \code
///      /* Define MindData objects */
///      std::shared_ptr<SchemaObj> schema = Schema();
///      schema->add_column("column1", mindspore::DataType::kNumberTypeUInt8, {2});
///      schema->add_column("column2", mindspore::DataType::kNumberTypeUInt8, {1});
///      std::shared_ptr<Dataset> ds = RandomData(50, schema);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: As we defined the schema before, each data dictionary owns keys "column1" and "column2" */
///      auto column1 = row["column1"];
/// \endcode
template <typename T = std::shared_ptr<SchemaObj>>
std::shared_ptr<RandomDataDataset> DATASET_API RandomData(const int32_t &total_rows = 0, const T &schema = nullptr,
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

/// \class RenderedSST2Dataset
/// \brief A source dataset for reading and parsing RenderedSST2 dataset.
class DATASET_API RenderedSST2Dataset : public Dataset {
 public:
  /// \brief Constructor of RenderedSST2Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  RenderedSST2Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                      const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of RenderedSST2Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  RenderedSST2Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                      const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of RenderedSST2Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  RenderedSST2Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                      const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of RenderedSST2Dataset.
  ~RenderedSST2Dataset() override = default;
};

/// \brief Function to create a RenderedSST2Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all". Default: "all".
/// \param[in] decode Decode the images after reading. Default: false.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset. Default: RandomSampler().
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the RenderedSST2Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string dataset_path = "/path/to/RenderedSST2_dataset_directory";
///      std::shared_ptr<Dataset> ds = RenderedSST2(dataset_path);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In RenderedSST2 dataset, each data dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<RenderedSST2Dataset> DATASET_API
RenderedSST2(const std::string &dataset_dir, const std::string &usage = "all", bool decode = false,
             const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<RenderedSST2Dataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a RenderedSST2Dataset
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the RenderedSST2Dataset.
inline std::shared_ptr<RenderedSST2Dataset> DATASET_API
RenderedSST2(const std::string &dataset_dir, const std::string &usage, bool decode, const Sampler *sampler,
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<RenderedSST2Dataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a RenderedSST2Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of dataset. Acceptable usages include "train", "test", "val" or "all".
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the RenderedSST2Dataset.
inline std::shared_ptr<RenderedSST2Dataset> DATASET_API
RenderedSST2(const std::string &dataset_dir, const std::string &usage, bool decode,
             const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<RenderedSST2Dataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \class SBUDataset
/// \brief A source dataset that reads and parses SBU dataset.
class DATASET_API SBUDataset : public Dataset {
 public:
  /// \brief Constructor of SBUDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  SBUDataset(const std::vector<char> &dataset_dir, bool decode, const std::shared_ptr<Sampler> &sampler,
             const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SBUDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SBUDataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
             const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SBUDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SBUDataset(const std::vector<char> &dataset_dir, bool decode, const std::reference_wrapper<Sampler> &sampler,
             const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of SBUDataset.
  ~SBUDataset() override = default;
};

/// \brief Function to create a SBUDataset.
/// \note The generated dataset has two columns ["image", "caption"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode Decode the images after reading (default=false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current SBUDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/sbu_dataset_directory";
///      std::shared_ptr<Dataset> ds = SBU(folder_path, true, std::make_shared<RandomSampler>(false, 5));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In SBU dataset, each dictionary has keys "image" and "caption" */
///      auto caption = row["caption"];
/// \endcode
inline std::shared_ptr<SBUDataset> DATASET_API
SBU(const std::string &dataset_dir, bool decode = false,
    const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
    const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SBUDataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \brief Function to create a SBUDataset.
/// \note The generated dataset has two columns ["image", "caption"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current SBUDataset.
inline std::shared_ptr<SBUDataset> DATASET_API SBU(const std::string &dataset_dir, bool decode, const Sampler *sampler,
                                                   const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SBUDataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \brief Function to create a SBUDataset.
/// \note The generated dataset has two columns ["image", "caption"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current SBUDataset.
inline std::shared_ptr<SBUDataset> DATASET_API SBU(const std::string &dataset_dir, bool decode,
                                                   const std::reference_wrapper<Sampler> &sampler,
                                                   const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SBUDataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \class SemeionDataset
/// \brief A source dataset for reading and parsing Semeion dataset.
class DATASET_API SemeionDataset : public Dataset {
 public:
  /// \brief Constructor of SemeionDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SemeionDataset(const std::vector<char> &dataset_dir, const ::std::shared_ptr<Sampler> &sampler,
                 const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SemeionDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SemeionDataset(const std::vector<char> &dataset_dir, const Sampler *sampler,
                 const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SemeionDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SemeionDataset(const std::vector<char> &dataset_dir, const ::std::reference_wrapper<Sampler> &samlper,
                 const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of SemeionDataset.
  ~SemeionDataset() override = default;
};

/// \brief Function to create a Semeion Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SemeionDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/semeion_dataset_directory";
///      std::shared_ptr<Dataset> ds = SEMEION(folder_path, std::make_shared<SequentialSampler>(0, 6));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In SEMEION dataset, each dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<SemeionDataset> DATASET_API
Semeion(const std::string &dataset_dir, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
        const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SemeionDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \brief Function to create a Semeion Dataset
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SemeionDataset.
inline std::shared_ptr<SemeionDataset> DATASET_API Semeion(const std::string &dataset_dir,
                                                           const std::reference_wrapper<Sampler> &sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SemeionDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \brief Function to create a Semeion Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SemeionDataset.
inline std::shared_ptr<SemeionDataset> DATASET_API Semeion(const std::string &dataset_dir, Sampler *sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SemeionDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \class SogouNewsDataset
/// \brief A source dataset for reading and parsing Sogou News dataset.
class DATASET_API SogouNewsDataset : public Dataset {
 public:
  /// \brief Constructor of SogouNewsDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of SogouNews, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  SogouNewsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                   ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                   const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of SogouNewsDataset.
  ~SogouNewsDataset() override = default;
};

/// \brief Function to create a SogouNewsDataset.
/// \note This dataset includes polarity and full, which can be read according to your own needs.
/// \note The generated dataset has three columns ["index", "title" , "content"]. Their types are all string.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of SogouNews, can be "train", "test" or "all" data (default="all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SogouNewsDataset.
inline std::shared_ptr<SogouNewsDataset> DATASET_API SogouNews(const std::string &dataset_dir,
                                                               const std::string &usage = "all",
                                                               int64_t num_samples = 0,
                                                               ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                               int32_t num_shards = 1, int32_t shard_id = 0,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SogouNewsDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                            num_shards, shard_id, cache);
}

/// \class SpeechCommandsDataset.
/// \brief A source dataset that reads and parses SpeechCommands dataset.
class DATASET_API SpeechCommandsDataset : public Dataset {
 public:
  /// \brief Constructor of SpeechCommandsDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of SpeechCommands, can be "train", "test", "valid" or "all".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SpeechCommandsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                        const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SpeechCommandsDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of SpeechCommands, can be "train", "test", "valid" or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SpeechCommandsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                        const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SpeechCommandsDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of SpeechCommands, can be "train", "test", "valid" or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SpeechCommandsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                        const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of SpeechCommandsDataset.
  ~SpeechCommandsDataset() override = default;
};

/// \brief Function to create a SpeechCommands Dataset.
/// \note The generated dataset has five columns ["waveform", "sample_rate", "label", "speaker_id", "utterance_number"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of SpeechCommands, can be "train", "test", "valid" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SpeechCommandsDataset.
inline std::shared_ptr<SpeechCommandsDataset> DATASET_API
SpeechCommands(const std::string &dataset_dir, const std::string &usage = "all",
               const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SpeechCommandsDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a SpeechCommands Dataset.
/// \note The generated dataset has five columns ["waveform", "sample_rate", "label", "speaker_id", "utterance_number"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of SpeechCommands, can be "train", "test", "valid" or "all".
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SpeechCommandsDataset.
inline std::shared_ptr<SpeechCommandsDataset> DATASET_API
SpeechCommands(const std::string &dataset_dir, const std::string &usage, const Sampler *sampler,
               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SpeechCommandsDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a SpeechCommands Dataset.
/// \note The generated dataset has five columns ["waveform", "sample_rate", "label", "speaker_id", "utterance_number"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of SpeechCommands, can be "train", "test", "valid" or "all".
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SpeechCommandsDataset.
inline std::shared_ptr<SpeechCommandsDataset> DATASET_API
SpeechCommands(const std::string &dataset_dir, const std::string &usage, const std::reference_wrapper<Sampler> &sampler,
               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SpeechCommandsDataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class SQuADDataset
/// \brief A source dataset that reads and parses SQuAD dataset.
class DATASET_API SQuADDataset : public Dataset {
 public:
  /// \brief Constructor of SQuADDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of SQuAD, can be "train", "dev" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode::kFalse - No shuffling is performed.
  ///     ShuffleMode::kFiles - Shuffle files only.
  ///     ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  SQuADDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
               ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of SQuADUDataset.
  ~SQuADDataset() override = default;
};

/// \brief Function to create a SQuADDataset.
/// \note The generated dataset has four columns ["context", "question", "text", "answer_start"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of SQuAD, can be "train", "dev" or "all" (Default="all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default=0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default=1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default=0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the SQuADDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/squad_dataset_directory";
///      std::shared_ptr<Dataset> ds = SQuAD(folder_path, "train", 0, ShuffleMode::kFalse,
///                                          std::make_shared<SequentialSampler>(0, 6));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In SQuAD dataset, each dictionary has keys "context", "question", "text", "answer_start" */
///      auto context = row["context"];
/// \endcode
inline std::shared_ptr<SQuADDataset> DATASET_API SQuAD(const std::string &dataset_dir, const std::string &usage = "all",
                                                       int64_t num_samples = 0,
                                                       ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                       int32_t num_shards = 1, int32_t shard_id = 0,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SQuADDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                        num_shards, shard_id, cache);
}

/// \class SST2Dataset
/// \brief A source dataset for reading and parsing SST2 dataset.
class DATASET_API SST2Dataset : public Dataset {
 public:
  /// \brief Constructor of SST2Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of SST2, can be "train", "test" or "dev".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kGlobal - Shuffle the samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  SST2Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
              ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of SST2.
  ~SST2Dataset() override = default;
};

/// \brief Function to create a SST2Dataset.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of SST2, can be "train", "test" or "dev". Default: "train".
/// \param[in] num_samples The number of samples to be included in the dataset.
///     Default: 0, means all samples.
/// \param[in] shuffle The mode for shuffling data every epoch. Default: ShuffleMode::kGlobal.
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. Default: 1.
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified. Default: 0.
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the SST2Dataset
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/sst2_dataset_directory";
///      std::shared_ptr<Dataset> ds = SST2(folder_path, "train");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
/// \endcode
inline std::shared_ptr<SST2Dataset> DATASET_API SST2(const std::string &dataset_dir, const std::string &usage = "train",
                                                     int64_t num_samples = 0,
                                                     ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                                     int32_t shard_id = 0,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SST2Dataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle, num_shards,
                                       shard_id, cache);
}

/// \class STL10Dataset
/// \brief A source dataset that reads and parses STL10 dataset.
class DATASET_API STL10Dataset : public Dataset {
 public:
  /// \brief Constructor of STL10Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of STL10, can be "train", "test", "unlabeled", "train+unlabeled"
  ///     or "all".
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  STL10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
               const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of STL10Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of STL10, can be "train", "test", "unlabeled", "train+unlabeled"
  ///     or "all".
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  STL10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of STL10Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of STL10, can be "train", "test", "unlabeled", "train+unlabeled"
  ///     or "all".
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  STL10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
               const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of STL10Dataset.
  ~STL10Dataset() override = default;
};

/// \brief Function to create a STL10 Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of STL10, can be "train", "test", "unlabeled", "train+unlabeled" or "all" (default = "all").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use. (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<STL10Dataset> DATASET_API
STL10(const std::string &dataset_dir, const std::string &usage = "all",
      const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
      const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<STL10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a STL10 Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of STL10, can be "train", "test", "unlabeled" or "train+unlabeled" or "all"
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<STL10Dataset> DATASET_API STL10(const std::string &dataset_dir, const std::string &usage,
                                                       const Sampler *sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<STL10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \brief Function to create a STL10 Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of STL10, can be "train", "test", "unlabeled", "train+unlabeled" or "all"
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<STL10Dataset> DATASET_API STL10(const std::string &dataset_dir, const std::string &usage,
                                                       const std::reference_wrapper<Sampler> &sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<STL10Dataset>(StringToChar(dataset_dir), StringToChar(usage), sampler, cache);
}

/// \class SUN397Dataset.
/// \brief A source dataset that reads and parses SUN397 dataset.
class DATASET_API SUN397Dataset : public Dataset {
 public:
  /// \brief Constructor of SUN397Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  SUN397Dataset(const std::vector<char> &dataset_dir, bool decode, const std::shared_ptr<Sampler> &sampler,
                const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SUN397Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SUN397Dataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
                const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of SUN397Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  SUN397Dataset(const std::vector<char> &dataset_dir, bool decode, const std::reference_wrapper<Sampler> &sampler,
                const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of SUN397Dataset.
  ~SUN397Dataset() override = default;
};

/// \brief Function to create a SUN397Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode Decode the images after reading. Default: true.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples
///     be used to randomly iterate the entire dataset. Default: RandomSampler().
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the current SUN397Dataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/sun397_dataset_directory";
///      std::shared_ptr<Dataset> ds = SUN397(folder_path, false, std::make_shared<RandomSampler>(false, 5));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In SUN397 dataset dataset, each dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<SUN397Dataset> DATASET_API
SUN397(const std::string &dataset_dir, bool decode = true,
       const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SUN397Dataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \brief Function to create a SUN397Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the current SUN397Dataset.
inline std::shared_ptr<SUN397Dataset> DATASET_API SUN397(const std::string &dataset_dir, bool decode,
                                                         const Sampler *sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SUN397Dataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \brief Function to create a SUN397Dataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use. Default: nullptr, which means no cache is used.
/// \return Shared pointer to the current SUN397Dataset.
inline std::shared_ptr<SUN397Dataset> DATASET_API SUN397(const std::string &dataset_dir, bool decode,
                                                         const std::reference_wrapper<Sampler> &sampler,
                                                         const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<SUN397Dataset>(StringToChar(dataset_dir), decode, sampler, cache);
}

/// \class TedliumDataset
/// \brief A source dataset for reading and parsing tedlium dataset.
class DATASET_API TedliumDataset : public Dataset {
 public:
  /// \brief Constructor of TedliumDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] release Release of the dataset, can be "release1", "release2", "release3".
  /// \param[in] usage Part of dataset of TEDLIUM, for release3, only can be "all", for release1 and release2,
  ///     can be "train", "test" or "all".
  /// \param[in] extensions The extensions of audio file. Only support ".sph" now.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  TedliumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &release, const std::vector<char> &usage,
                 const std::vector<char> &extensions, const std::shared_ptr<Sampler> &sampler,
                 const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of TedliumDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] release Release of the dataset, can be "release1", "release2", "release3".
  /// \param[in] usage Part of dataset of TEDLIUM, for release3, only can be "all", for release1 and release2,
  ///     can be "train", "test" or "all".
  /// \param[in] extensions The extensions of audio file. Only support ".sph" now.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  TedliumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &release, const std::vector<char> &usage,
                 const std::vector<char> &extensions, const Sampler *sampler,
                 const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of TedliumDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] release Release of the dataset, can be "release1", "release2", "release3".
  /// \param[in] usage Part of dataset of TEDLIUM, for release3, only can be "all", for release1 and release2,
  ///     can be "train", "test" or "all".
  /// \param[in] extensions The extensions of audio file. Only support ".sph" now.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  TedliumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &release, const std::vector<char> &usage,
                 const std::vector<char> &extensions, const std::reference_wrapper<Sampler> &samlper,
                 const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of TedliumDataset.
  ~TedliumDataset() override = default;
};

/// \brief Function to create a TedliumDataset.
/// \note The generated dataset has six columns ["waveform", "sample_rate", "transcript", "talk_id", "speaker_id",
///     "identifier"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] release Release of the dataset, can be "release1", "release2", "release3".
/// \param[in] usage Part of dataset of TEDLIUM, for release3, only can be "all", for release1 and release2,
///     can be "train", "test" or "all" (default = "all").
/// \param[in] extensions The extensions of audio file. Only support ".sph" now (default = ".sph").
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the TedliumDataset.
inline std::shared_ptr<TedliumDataset> DATASET_API Tedlium(
  const std::string &dataset_dir, const std::string &release, const std::string &usage = "all",
  const std::string &extensions = ".sph", const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
  const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<TedliumDataset>(StringToChar(dataset_dir), StringToChar(release), StringToChar(usage),
                                          StringToChar(extensions), sampler, cache);
}

/// \brief Function to create a TedliumDataset.
/// \note The generated dataset has six columns ["waveform", "sample_rate","transcript", "talk_id", "speaker_id",
///     "identifier"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] release Release of the dataset, can be "release1", "release2", "release3".
/// \param[in] usage Part of dataset of TEDLIUM, for release3, only can be "all", for release1 and release2,
///     can be "train", "test" or "all".
/// \param[in] extensions The extensions of audio file. Only support ".sph" now.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the TedliumDataset.
inline std::shared_ptr<TedliumDataset> DATASET_API Tedlium(const std::string &dataset_dir, const std::string &release,
                                                           const std::string &usage, const std::string &extensions,
                                                           const std::reference_wrapper<Sampler> &sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<TedliumDataset>(StringToChar(dataset_dir), StringToChar(release), StringToChar(usage),
                                          StringToChar(extensions), sampler, cache);
}

/// \brief Function to create a TedliumDataset.
/// \note The generated dataset has six columns ["waveform", "sample_rate","transcript", "talk_id", "speaker_id",
///     "identifier"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] release Release of the dataset, can be "release1", "release2", "release3".
/// \param[in] usage Part of dataset of TEDLIUM, for release3, only can be "all", for release1 and release2,
///     can be "train", "test" or "all".
/// \param[in] extensions The extensions of audio file. Only support ".sph" now.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the TedliumDataset.
inline std::shared_ptr<TedliumDataset> DATASET_API Tedlium(const std::string &dataset_dir, const std::string &release,
                                                           const std::string &usage, const std::string &extensions,
                                                           Sampler *sampler,
                                                           const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<TedliumDataset>(StringToChar(dataset_dir), StringToChar(release), StringToChar(usage),
                                          StringToChar(extensions), sampler, cache);
}

/// \class TextFileDataset
/// \brief A source dataset that reads and parses datasets stored on disk in text format.
class DATASET_API TextFileDataset : public Dataset {
 public:
  /// \brief Constructor of TextFileDataset.
  /// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
  ///     will be sorted in a lexicographical order.
  /// \param[in] num_samples The number of samples to be included in the dataset
  ///     (Default = 0 means all samples).
  /// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified (Default = 0).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  TextFileDataset(const std::vector<std::vector<char>> &dataset_files, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of TextFileDataset.
  ~TextFileDataset() override = default;
};

/// \brief Function to create a TextFileDataset.
/// \note The generated dataset has one column ['text'].
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0 means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode.kFalse - No shuffling is performed.
///     ShuffleMode.kFiles - Shuffle files only.
///     ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the TextFileDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string file_path = "/path/to/text_file_dataset_file";
///      std::shared_ptr<Dataset> ds = TextFile({file_path}, 2);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In TextFile dataset, each dictionary has key "text" */
///      auto text = row["text"];
/// \endcode
inline std::shared_ptr<TextFileDataset> DATASET_API TextFile(const std::vector<std::string> &dataset_files,
                                                             int64_t num_samples = 0,
                                                             ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                             int32_t num_shards = 1, int32_t shard_id = 0,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<TextFileDataset>(VectorStringToChar(dataset_files), num_samples, shuffle, num_shards,
                                           shard_id, cache);
}

/// \class TFRecordDataset
/// \brief A source dataset for reading and parsing datasets stored on disk in TFData format.
class DATASET_API TFRecordDataset : public Dataset {
 public:
  /// \brief Constructor of TFRecordDataset.
  /// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
  ///     will be sorted in a lexicographical order.
  /// \param[in] schema Path to schema file.
  /// \param[in] columns_list List of columns to be read. (Default = {}, read all columns).
  /// \param[in] num_samples The number of samples to be included in the dataset.
  ///     (Default = 0 means all samples).
  ///     Processing priority for `num_samples` is as the following:
  ///     1. If `num_samples` is greater than 0, read `num_samples` rows.
  ///     2. Otherwise, if numRows (parsed from `schema` ) is greater than 0, read numRows rows.
  ///     3. Otherwise, read the full dataset.
  ///     `num_samples` or numRows (parsed from `schema` ) will be interpreted as number of rows per shard.
  ///     It is highly recommended to provide `num_samples` or numRows (parsed from `schema` )
  ///     when `compression_type` is "GZIP" or "ZLIB" to avoid performance degradation due to multiple
  ///     decompressions of the same file to obtain the file size.
  /// \param[in] shuffle The mode for shuffling data every epoch. (Default = ShuffleMode::kGlobal).
  ///     Can be any of:
  ///     ShuffleMode::kFalse - No shuffling is performed.
  ///     ShuffleMode::kFiles - Shuffle files only.
  ///     ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1).
  /// \param[in] shard_id The shard ID within num_shards. This argument should be specified only
  ///     when num_shards is also specified. (Default = 0).
  /// \param[in] shard_equal_rows Get equal rows for all shards.
  ///     (Default = false, number of rows of each shard may be not equal).
  ///     When `compression_type` is "GZIP" or "ZLIB", and `num_samples` or numRows (parsed from `schema` ) is
  ///     provided, shard_equal_rows` will be implied as true.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] compression_type Compression type to use.
  ///     (Default = "", which means no compression is used).
  ///     Can be any of:
  ///     "" - No compression is used.
  ///     "GZIP" - GZIP compression is used.
  ///     "ZLIB" - ZLIB compression is used.
  TFRecordDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &schema,
                  const std::vector<std::vector<char>> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                  const std::shared_ptr<DatasetCache> &cache, const std::vector<char> &compression_type);

  /// \brief Constructor of TFRecordDataset.
  /// \note Parameter 'schema' is shared pointer to Schema object
  /// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
  ///     will be sorted in a lexicographical order.
  /// \param[in] schema SchemaObj to set column type, data type and data shape.
  /// \param[in] columns_list List of columns to be read (Default = {}, read all columns).
  /// \param[in] num_samples The number of samples to be included in the dataset
  ///     (Default = 0 means all samples).
  ///     Processing priority for `num_samples` is as the following:
  ///     1. If `num_samples` is greater than 0, read `num_samples` rows.
  ///     2. Otherwise, if numRows (parsed from `schema` ) is greater than 0, read numRows rows.
  ///     3. Otherwise, read the full dataset.
  ///     `num_samples` or numRows (parsed from `schema` ) will be interpreted as number of rows per shard.
  ///     It is highly recommended to provide `num_samples` or numRows (parsed from `schema` )
  ///     when `compression_type` is "GZIP" or "ZLIB" to avoid performance degradation due to multiple
  ///     decompressions of the same file to obtain the file size.
  /// \param[in] shuffle The mode for shuffling data every epoch. (Default = ShuffleMode::kGlobal).
  ///     Can be any of:
  ///     ShuffleMode::kFalse - No shuffling is performed.
  ///     ShuffleMode::kFiles - Shuffle files only.
  ///     ShuffleMode::kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1).
  /// \param[in] shard_id The shard ID within num_shards. This argument should be specified only
  ///     when num_shards is also specified. (Default = 0).
  /// \param[in] shard_equal_rows Get equal rows for all shards.
  ///     (Default = false, number of rows of each shard may be not equal).
  ///     When `compression_type` is "GZIP" or "ZLIB", and `num_samples` or numRows (parsed from `schema` ) is
  ///     provided, shard_equal_rows` will be implied as true.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] compression_type Compression type to use.
  ///     (Default = "", which means no compression is used).
  ///     Can be any of:
  ///     "" - No compression is used.
  ///     "GZIP" - GZIP compression is used.
  ///     "ZLIB" - ZLIB compression is used.
  TFRecordDataset(const std::vector<std::vector<char>> &dataset_files, const std::shared_ptr<SchemaObj> &schema,
                  const std::vector<std::vector<char>> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                  int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                  const std::shared_ptr<DatasetCache> &cache, const std::vector<char> &compression_type);

  /// \brief Destructor of TFRecordDataset.
  ~TFRecordDataset() override = default;
};

/// \brief Function to create a TFRecordDataset.
/// \param[in] dataset_files List of files to be read to search for a pattern of files. The list
///     will be sorted in a lexicographical order.
/// \param[in] schema SchemaObj or string to schema path. (Default = nullptr, which means that the
///     meta data from the TFData file is considered the schema).
/// \param[in] columns_list List of columns to be read (Default = {}, read all columns).
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0 means all samples).
///     Processing priority for `num_samples` is as the following:
///     1. If `num_samples` is greater than 0, read `num_samples` rows.
///     2. Otherwise, if numRows (parsed from `schema` ) is greater than 0, read numRows rows.
///     3. Otherwise, read the full dataset.
///     `num_samples` or numRows (parsed from `schema` ) will be interpreted as number of rows per shard.
///     It is highly recommended to provide `num_samples` or numRows (parsed from `schema` )
///     when `compression_type` is "GZIP" or "ZLIB" to avoid performance degradation due to multiple
///     decompressions of the same file to obtain the file size.
/// \param[in] shuffle The mode for shuffling data every epoch. (Default = ShuffleMode::kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into. (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be specified only
///     when num_shards is also specified. (Default = 0).
/// \param[in] shard_equal_rows Get equal rows for all shards.
///     (Default = false, number of rows of each shard may be not equal).
///     When `compression_type` is "GZIP" or "ZLIB", and `num_samples` or numRows (parsed from `schema` ) is
///     provided, shard_equal_rows` will be implied as true.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \param[in] compression_type Compression type to use.
///     (Default = "", which means no compression is used).
///     Can be any of:
///     "" - No compression is used.
///     "GZIP" - GZIP compression is used.
///     "ZLIB" - ZLIB compression is used.
/// \return Shared pointer to the TFRecordDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string file_path = "/path/to/tfrecord_file";
///      std::string schema_path = "/path/to/schema_file";
///      std::shared_ptr<Dataset> ds = TFRecord({file_path}, schema_path, {"image"});
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: The columns of generated dataset depend on the source TFRecord files. */
///      auto image = row["image"];
/// \endcode
template <typename T = std::shared_ptr<SchemaObj>>
std::shared_ptr<TFRecordDataset> DATASET_API
TFRecord(const std::vector<std::string> &dataset_files, const T &schema = nullptr,
         const std::vector<std::string> &columns_list = {}, int64_t num_samples = 0,
         ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
         bool shard_equal_rows = false, const std::shared_ptr<DatasetCache> &cache = nullptr,
         const std::string &compression_type = "") {
  std::shared_ptr<TFRecordDataset> ds;
  if constexpr (std::is_same<T, std::nullptr_t>::value || std::is_same<T, std::shared_ptr<SchemaObj>>::value) {
    std::shared_ptr<SchemaObj> schema_obj = schema;
    ds = std::make_shared<TFRecordDataset>(VectorStringToChar(dataset_files), std::move(schema_obj),
                                           VectorStringToChar(columns_list), num_samples, shuffle, num_shards, shard_id,
                                           shard_equal_rows, cache, StringToChar(compression_type));
  } else {
    ds = std::make_shared<TFRecordDataset>(VectorStringToChar(dataset_files), StringToChar(schema),
                                           VectorStringToChar(columns_list), num_samples, shuffle, num_shards, shard_id,
                                           shard_equal_rows, cache, StringToChar(compression_type));
  }
  return ds;
}

/// \class UDPOSDataset
/// \brief A source dataset for reading and parsing UDPOS dataset.
class DATASET_API UDPOSDataset : public Dataset {
 public:
  /// \brief Constructor of UDPOS Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data list txt file to be read, can be "train", "test", 'valid' or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  UDPOSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
               ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of UDPOSDataset.
  ~UDPOSDataset() override = default;
};

/// \brief Function to create a UDPOSDataset.
/// \note The generated dataset has three column ['word','universal','stanford'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of UDPOS, can be "train", "test", "valid" or "all" (default="all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch. (Default=ShuffleMode.kGlobal)
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the UDPOSDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/udpos_dataset_directory";
///      std::shared_ptr<Dataset> ds = UDPOS(dataset_dir, "test", 0, ShuffleMode::kGlobal);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In UDPOS dataset, each dictionary has keys "word", "universal", "stanford" */
///      auto word = row["word"];
/// \endcode
inline std::shared_ptr<UDPOSDataset> DATASET_API UDPOS(const std::string &dataset_dir, const std::string &usage = "all",
                                                       int64_t num_samples = 0,
                                                       ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                       int32_t num_shards = 1, int32_t shard_id = 0,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<UDPOSDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                        num_shards, shard_id, cache);
}

/// \class USPSDataset
/// \brief A source dataset that reads and parses USPS datasets.
class DATASET_API USPSDataset : public Dataset {
 public:
  /// \brief Constructor of USPSDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Usage of USPS, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified (Default = 0).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  USPSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
              ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of USPSDataset.
  ~USPSDataset() override = default;
};

/// \brief Function to create a USPSDataset.
/// \note The generated dataset has two columns ["image", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Usage of USPS, can be "train", "test" or "all" (Default = "all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0 means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode.kFalse - No shuffling is performed.
///     ShuffleMode.kFiles - Shuffle files only.
///     ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \return Shared pointer to the current USPSDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/usps_dataset_directory";
///      std::shared_ptr<Dataset> ds = USPS(folder_path, "train");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In USPS dataset, each dictionary has keys "image" and "label" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<USPSDataset> DATASET_API USPS(const std::string &dataset_dir, const std::string &usage = "all",
                                                     int64_t num_samples = 0,
                                                     ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1,
                                                     int32_t shard_id = 0,
                                                     const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<USPSDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle, num_shards,
                                       shard_id, cache);
}

/// \class VOCDataset
/// \brief A source dataset for reading and parsing VOC dataset.
class DATASET_API VOCDataset : public Dataset {
 public:
  /// \brief Constructor of VOCDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection".
  /// \param[in] usage The type of data list text file to be read (default = "train").
  /// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] extra_metadata Flag to add extra meta-data to row (default=false).
  VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task, const std::vector<char> &usage,
             const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
             const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache, bool extra_metadata);

  /// \brief Constructor of VOCDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection".
  /// \param[in] usage The type of data list text file to be read.
  /// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] extra_metadata Flag to add extra meta-data to row (default=false).
  VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task, const std::vector<char> &usage,
             const std::map<std::vector<char>, int32_t> &class_indexing, bool decode, const Sampler *sampler,
             const std::shared_ptr<DatasetCache> &cache, bool extra_metadata);

  /// \brief Constructor of VOCDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection".
  /// \param[in] usage The type of data list text file to be read.
  /// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task.
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
  /// \param[in] extra_metadata Flag to add extra meta-data to row (default=false).
  VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task, const std::vector<char> &usage,
             const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
             const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache,
             bool extra_metadata);

  /// \brief Destructor of VOCDataset.
  ~VOCDataset() override = default;
};

/// \brief Function to create a VOCDataset.
/// \note The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                  ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///     - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection".
/// \param[in] usage The type of data list text file to be read (default = "train").
/// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \param[in] extra_metadata Flag to add extra meta-data to row (default=false).
/// \return Shared pointer to the VOCDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/voc_dataset_directory";
///      std::shared_ptr<Dataset> ds = VOC(folder_path, "Detection", "train", {}, false,
///                                        std::make_shared<SequentialSampler>(0, 6));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In VOC dataset, if task='Segmentation', each dictionary has keys "image" and "target" */
///      /* Note: In VOC dataset, if task='Detection', each dictionary has keys "image" and "annotation" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<VOCDataset> DATASET_API
VOC(const std::string &dataset_dir, const std::string &task = "Segmentation", const std::string &usage = "train",
    const std::map<std::string, int32_t> &class_indexing = {}, bool decode = false,
    const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
    const std::shared_ptr<DatasetCache> &cache = nullptr, bool extra_metadata = false) {
  return std::make_shared<VOCDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      MapStringToChar(class_indexing), decode, sampler, cache, extra_metadata);
}

/// \brief Function to create a VOCDataset.
/// \note The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                  ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///     - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection".
/// \param[in] usage The type of data list text file to be read.
/// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \param[in] extra_metadata Flag to add extra meta-data to row (default=false).
/// \return Shared pointer to the VOCDataset.
inline std::shared_ptr<VOCDataset> DATASET_API VOC(const std::string &dataset_dir, const std::string &task,
                                                   const std::string &usage,
                                                   const std::map<std::string, int32_t> &class_indexing, bool decode,
                                                   const Sampler *sampler,
                                                   const std::shared_ptr<DatasetCache> &cache = nullptr,
                                                   bool extra_metadata = false) {
  return std::make_shared<VOCDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      MapStringToChar(class_indexing), decode, sampler, cache, extra_metadata);
}

/// \brief Function to create a VOCDataset.
/// \note The generated dataset has multi-columns :
///     - task='Detection', column: [['image', dtype=uint8], ['bbox', dtype=float32], ['label', dtype=uint32],
///                                  ['difficult', dtype=uint32], ['truncate', dtype=uint32]].
///     - task='Segmentation', column: [['image', dtype=uint8], ['target',dtype=uint8]].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] task Set the task type of reading voc data, now only support "Segmentation" or "Detection".
/// \param[in] usage The type of data list text file to be read.
/// \param[in] class_indexing A str-to-int mapping from label name to index, only valid in "Detection" task.
/// \param[in] decode Decode the images after reading.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr which means no cache is used).
/// \param[in] extra_metadata Flag to add extra meta-data to row (default=false).
/// \return Shared pointer to the VOCDataset.
inline std::shared_ptr<VOCDataset> DATASET_API VOC(const std::string &dataset_dir, const std::string &task,
                                                   const std::string &usage,
                                                   const std::map<std::string, int32_t> &class_indexing, bool decode,
                                                   const std::reference_wrapper<Sampler> &sampler,
                                                   const std::shared_ptr<DatasetCache> &cache = nullptr,
                                                   bool extra_metadata = false) {
  return std::make_shared<VOCDataset>(StringToChar(dataset_dir), StringToChar(task), StringToChar(usage),
                                      MapStringToChar(class_indexing), decode, sampler, cache, extra_metadata);
}

/// \class WIDERFaceDataset
/// \brief A source dataset for reading and parsing WIDERFace dataset.
class DATASET_API WIDERFaceDataset : public Dataset {
 public:
  /// \brief Constructor of WIDERFaceDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data to be read, can be "train", "test", "valid" or "all". "all" will read samples
  ///     from "train" and "valid".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  WIDERFaceDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                   const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of WIDERFaceDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data to be read, can be "train", "test", "valid" or "all". "all" will read samples
  ///     from "train" and "valid".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  WIDERFaceDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                   const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of WIDERFaceDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data to be read, can be "train", "test", "valid" or "all". "all" will read samples
  ///     from "train" and "valid".
  /// \param[in] decode Decode the images after reading.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  WIDERFaceDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                   const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of WIDERFaceDataset.
  ~WIDERFaceDataset() override = default;
};

/// \brief Function to create a WIDERFace Dataset.
/// \note When usage is "train", "valid" or "all", the generated dataset has eight columns ["image", "bbox", "blur",
///     "expression", "illumination", "occlusion", "pose", "invalid"]. When usage is "test", it only has one column
///     ["image"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of data to be read, can be "train", "test", "valid" or "all" (default="all"). "all" will
///     read samples from "train" and "valid".
/// \param[in] decode The option to decode the images in dataset (default = false).
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///     given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the WIDERFaceDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/wider_face_dataset_directory";
///      std::shared_ptr<Dataset> ds = WIDERFace(folder_path, "train", std::make_shared<SequentialSampler>(0, 2));
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In WIDERFace dataset, if task='test', each dictionary has key "image" */
///      /* Note: In WIDERFace dataset, if task='all', 'train' or 'valid', each dictionary has keys "image", "bbox",
///      "blur", "expression", "illumination", "occlusion", "pose", "invalid" */
///      auto image = row["image"];
/// \endcode
inline std::shared_ptr<WIDERFaceDataset> DATASET_API
WIDERFace(const std::string &dataset_dir, const std::string &usage = "all", bool decode = false,
          const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
          const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<WIDERFaceDataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a WIDERFace Dataset.
/// \note When usage is "train", "valid" or "all", the generated dataset has eight columns ["image", "bbox", "blur",
///     "expression", "illumination", "occlusion", "pose", "invalid"]. When usage is "test", it only has one column
///     ["image"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of data to be read, can be "train", "test", "valid" or "all". "all" will read samples
///     from "train" and "valid".
/// \param[in] decode The option to decode the images in dataset.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the WIDERFaceDataset.
inline std::shared_ptr<WIDERFaceDataset> DATASET_API WIDERFace(const std::string &dataset_dir, const std::string &usage,
                                                               bool decode, const Sampler *sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<WIDERFaceDataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \brief Function to create a WIDERFace Dataset.
/// \note When usage is "train", "valid" or "all", the generated dataset has eight columns ["image", "bbox", "blur",
///     "expression", "illumination", "occlusion", "pose", "invalid"]. When usage is "test", it only has one column
///     ["image"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage The type of data to be read, can be "train", "test", "valid" or "all". "all" will read samples
///     from "train" and "valid".
/// \param[in] decode The option to decode the images in dataset.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the WIDERFaceDataset.
inline std::shared_ptr<WIDERFaceDataset> DATASET_API WIDERFace(const std::string &dataset_dir, const std::string &usage,
                                                               bool decode,
                                                               const std::reference_wrapper<Sampler> &sampler,
                                                               const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<WIDERFaceDataset>(StringToChar(dataset_dir), StringToChar(usage), decode, sampler, cache);
}

/// \class WikiTextDataset
/// \brief A source dataset for reading and parsing WikiTextDataset dataset.
class DATASET_API WikiTextDataset : public Dataset {
 public:
  /// \brief Constructor of WikiTextDataset Dataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage The type of data list txt file to be read, can be "train", "test", 'valid' or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  WikiTextDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                  ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                  const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of WikiTextDataset.
  ~WikiTextDataset() override = default;
};

/// \brief Function to create a WikiText Dataset.
/// \note The generated dataset has one column ['text'].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage One of "all", "train" , 'valid' or "test" (default = "all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode.kFalse - No shuffling is performed.
///     ShuffleMode.kFiles - Shuffle files only.
///     ShuffleMode.kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the WikiTextDataset.
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/wiki_dataset_directory";
///      std::shared_ptr<Dataset> ds = WikiText(folder_path, "all");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: In WikiText dataset, each dictionary has key "text" */
///      auto text = row["text"];
/// \endcode
inline std::shared_ptr<WikiTextDataset> DATASET_API WikiText(const std::string &dataset_dir,
                                                             const std::string &usage = "all", int64_t num_samples = 0,
                                                             ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                             int32_t num_shards = 1, int32_t shard_id = 0,
                                                             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<WikiTextDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                           num_shards, shard_id, cache);
}

/// \class YahooAnswersDataset
/// \brief A source dataset for reading and parsing YahooAnswers dataset.
class DATASET_API YahooAnswersDataset : public Dataset {
 public:
  /// \brief Constructor of YahooAnswersDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of YahooAnswers, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  YahooAnswersDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                      ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                      const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of YahooAnswers.
  ~YahooAnswersDataset() override = default;
};

/// \brief Function to create a YahooAnswersDataset.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of YahooAnswers, can be "train", "test" or "all" (default = "all").
/// \param[in] num_samples The number of samples to be included in the dataset.
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode::kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the YahooAnswersDataset
/// \par Example
/// \code
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/yahoo_answers_dataset_directory";
///      std::shared_ptr<Dataset> ds = YahooAnswers(folder_path, "train");
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
///      std::unordered_map<std::string, mindspore::MSTensor> row;
///      iter->GetNextRow(&row);
///
///      /* Note: As we defined before, each data dictionary owns keys "class", "title", "content", "answer" */
///      auto title = row["title"];
/// \endcode
inline std::shared_ptr<YahooAnswersDataset> DATASET_API
YahooAnswers(const std::string &dataset_dir, const std::string &usage = "all", int64_t num_samples = 0,
             ShuffleMode shuffle = ShuffleMode::kGlobal, int32_t num_shards = 1, int32_t shard_id = 0,
             const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<YahooAnswersDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                               num_shards, shard_id, cache);
}

/// \class YelpReviewDataset
/// \brief A source dataset for reading and parsing Yelp Review dataset.
class DATASET_API YelpReviewDataset : public Dataset {
 public:
  /// \brief Constructor of YelpReviewDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] usage Part of dataset of YelpReview, can be "train", "test" or "all".
  /// \param[in] num_samples The number of samples to be included in the dataset.
  /// \param[in] shuffle The mode for shuffling data every epoch.
  ///     Can be any of:
  ///     ShuffleMode.kFalse - No shuffling is performed.
  ///     ShuffleMode.kFiles - Shuffle files only.
  ///     ShuffleMode.kGlobal - Shuffle both the files and samples.
  /// \param[in] num_shards Number of shards that the dataset should be divided into.
  /// \param[in] shard_id The shard ID within num_shards. This argument should be
  ///     specified only when num_shards is also specified.
  /// \param[in] cache Tensor cache to use.
  YelpReviewDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                    ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                    const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of YelpReviewDataset.
  ~YelpReviewDataset() override = default;
};

/// \brief Function to create a YelpReviewDataset.
/// \note This dataset includes polarity and full, which can be read according to your own needs.
/// \note The generated dataset has two columns ["label", "text"]. Their types are all string.
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] usage Part of dataset of YelpReview, can be "train", "test" or "all" (default="all").
/// \param[in] num_samples The number of samples to be included in the dataset
///     (Default = 0, means all samples).
/// \param[in] shuffle The mode for shuffling data every epoch (Default=ShuffleMode.kGlobal).
///     Can be any of:
///     ShuffleMode::kFalse - No shuffling is performed.
///     ShuffleMode::kFiles - Shuffle files only.
///     ShuffleMode::kGlobal - Shuffle both the files and samples.
/// \param[in] num_shards Number of shards that the dataset should be divided into (Default = 1).
/// \param[in] shard_id The shard ID within num_shards. This argument should be
///     specified only when num_shards is also specified (Default = 0).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the YelpReviewDataset.
inline std::shared_ptr<YelpReviewDataset> DATASET_API YelpReview(const std::string &dataset_dir,
                                                                 const std::string &usage = "all",
                                                                 int64_t num_samples = 0,
                                                                 ShuffleMode shuffle = ShuffleMode::kGlobal,
                                                                 int32_t num_shards = 1, int32_t shard_id = 0,
                                                                 const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<YelpReviewDataset>(StringToChar(dataset_dir), StringToChar(usage), num_samples, shuffle,
                                             num_shards, shard_id, cache);
}

/// \class YesNoDataset.
/// \brief A source dataset for reading and parsing YesNo dataset.
class DATASET_API YesNoDataset : public Dataset {
 public:
  /// \brief Constructor of YesNoDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
  ///     given, a `RandomSampler` will be used to randomly iterate the entire dataset.
  /// \param[in] cache Tensor cache to use.
  YesNoDataset(const std::vector<char> &dataset_dir, const std::shared_ptr<Sampler> &sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of YesNoDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  YesNoDataset(const std::vector<char> &dataset_dir, const Sampler *sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Constructor of YesNoDataset.
  /// \param[in] dataset_dir Path to the root directory that contains the dataset.
  /// \param[in] sampler Sampler object used to choose samples from the dataset.
  /// \param[in] cache Tensor cache to use.
  YesNoDataset(const std::vector<char> &dataset_dir, const std::reference_wrapper<Sampler> &sampler,
               const std::shared_ptr<DatasetCache> &cache);

  /// \brief Destructor of YesNoDataset.
  ~YesNoDataset() override = default;
};

/// \brief Function to create a YesNo Dataset.
/// \note The generated dataset has three columns ["waveform", "sample_rate", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Shared pointer to a sampler object used to choose samples from the dataset. If sampler is not
///    given, a `RandomSampler` will be used to randomly iterate the entire dataset (default = RandomSampler()).
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<YesNoDataset> DATASET_API
YesNo(const std::string &dataset_dir, const std::shared_ptr<Sampler> &sampler = std::make_shared<RandomSampler>(),
      const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<YesNoDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \brief Function to create a YesNo Dataset.
/// \note The generated dataset has three columns ["waveform", "sample_rate", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Raw pointer to a sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<YesNoDataset> DATASET_API YesNo(const std::string &dataset_dir, Sampler *sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<YesNoDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \brief Function to create a YesNo Dataset.
/// \note The generated dataset has three columns ["waveform", "sample_rate", "label"].
/// \param[in] dataset_dir Path to the root directory that contains the dataset.
/// \param[in] sampler Sampler object used to choose samples from the dataset.
/// \param[in] cache Tensor cache to use (default=nullptr, which means no cache is used).
/// \return Shared pointer to the current Dataset.
inline std::shared_ptr<YesNoDataset> DATASET_API YesNo(const std::string &dataset_dir,
                                                       const std::reference_wrapper<Sampler> &sampler,
                                                       const std::shared_ptr<DatasetCache> &cache = nullptr) {
  return std::make_shared<YesNoDataset>(StringToChar(dataset_dir), sampler, cache);
}

/// \brief Function to create a cache to be attached to a dataset.
/// \note The reason for providing this API is that std::string will be constrained by the
///    compiler option '_GLIBCXX_USE_CXX11_ABI' while char is free of this restriction.
///    Check API `mindspore::dataset::CreateDatasetCache` and find more usage.
/// \param[in] id A user assigned session id for the current pipeline.
/// \param[in] mem_sz Size of the memory set aside for the row caching (default=0 which means unlimited,
///     note that it might bring in the risk of running out of memory on the machine).
/// \param[in] spill Spill to disk if out of memory.
/// \param[in] hostname optional host name (default=std::nullopt, means to use "127.0.0.1").
/// \param[in] port optional port (default=std::nullopt, means to use 50052).
/// \param[in] num_connections optional number of connections (default=std::nullopt, means to use 12).
/// \param[in] prefetch_sz optional prefetch size (default=std::nullopt, means to use 20).
/// \return Shared pointer to DatasetCache. If error, nullptr is returned.
std::shared_ptr<DatasetCache> DATASET_API CreateDatasetCacheCharIF(
  session_id_type id, uint64_t mem_sz, bool spill, const std::optional<std::vector<char>> &hostname = std::nullopt,
  const std::optional<int32_t> &port = std::nullopt, const std::optional<int32_t> &num_connections = std::nullopt,
  const std::optional<int32_t> &prefetch_sz = std::nullopt);

/// \brief Function the create a cache to be attached to a dataset.
/// \param[in] id A user assigned session id for the current pipeline.
/// \param[in] mem_sz Size of the memory set aside for the row caching (default=0 which means unlimited,
///     note that it might bring in the risk of running out of memory on the machine).
/// \param[in] spill Spill to disk if out of memory.
/// \param[in] hostname optional host name (default=std::nullopt, means to use "127.0.0.1").
/// \param[in] port optional port (default=std::nullopt, means to use 50052).
/// \param[in] num_connections optional number of connections (default=std::nullopt, means to use 12).
/// \param[in] prefetch_sz optional prefetch size (default=std::nullopt, means to use 20).
/// \return Shared pointer to DatasetCache. If error, nullptr is returned.
/// \par Example
/// \code
///      /* Define a Cache object */
///      std::shared_ptr<DatasetCache> cache = CreateDatasetCache(233, 0, false, "127.0.0.1", 50053, 1, 1);
///
///      /* Define dataset path and MindData object */
///      std::string folder_path = "/path/to/image_directory";
///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, nullptr, {}, {}, cache);
///
///      /* Create iterator to read dataset */
///      std::shared_ptr<Iterator> iter = ds->CreateIterator();
/// \endcode
inline std::shared_ptr<DatasetCache> DATASET_API CreateDatasetCache(
  session_id_type id, uint64_t mem_sz, bool spill, const std::optional<std::string> &hostname = std::nullopt,
  const std::optional<int32_t> &port = std::nullopt, const std::optional<int32_t> &num_connections = std::nullopt,
  const std::optional<int32_t> &prefetch_sz = std::nullopt) {
  std::optional<std::vector<char>> hostname_c = std::nullopt;
  if (hostname != std::nullopt) {
    hostname_c = std::vector<char>(hostname->begin(), hostname->end());
  }
  return CreateDatasetCacheCharIF(id, mem_sz, spill, hostname_c, port, num_connections, prefetch_sz);
}

/// \brief Function to create a ZipDataset.
/// \note Applies zip to the dataset.
/// \param[in] datasets List of shared pointers to the datasets that we want to zip.
/// \return Shared pointer to the ZipDataset.
inline std::shared_ptr<ZipDataset> DATASET_API Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  return std::make_shared<ZipDataset>(datasets);
}
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_DATASETS_H_
