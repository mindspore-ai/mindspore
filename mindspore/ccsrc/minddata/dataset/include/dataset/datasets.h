/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "nlohmann/json_fwd.hpp"

#include "utils/dual_abi_helper.h"
#include "include/dataset/iterator.h"
#include "include/dataset/samplers.h"
#include "include/dataset/transforms.h"
#include "minddata/dataset/util/visible.h"

namespace mindspore::dataset {
class CsvBase;
class DatasetCache;
class DatasetNode;
class DSCallback;
class Iterator;
class PullBasedIterator;
class SamplerObj;
class SchemaObj;
class Tensor;
class TensorOperation;
class TensorShape;
class TreeAdapter;
class TreeAdapterLite;
class TreeGetters;

// Dataset classes (in alphabetical order)
class BatchDataset;
class BucketBatchByLengthDataset;
class ConcatDataset;
class CSVDataset;
class FilterDataset;
class MapDataset;
class ProjectDataset;
class RenameDataset;
class RepeatDataset;
class ShuffleDataset;
class SkipDataset;
class TakeDataset;
class TransferDataset;
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
  ///      std::string folder_path = "/path/to/image_folder";
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, false, std::make_shared<SequentialSampler>(0, 10));
  ///      std::string save_file = "ImageData.mindrecord";
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
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
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
    (void)all_datasets.insert(all_datasets.end(), datasets.begin(), datasets.end());
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
  ///     std::shared_ptr<TensorTransform> resize_op = std::make_shared<vision::Resize>({224, 224});
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
  ///    // outputted by decode_op is passed as input to resize_op.
  ///    // resize_op will output one column. Column "image" will be replaced by
  ///    // the column outputted by resize_op (the very last operation). All other
  ///    // columns are unchanged.
  ///    dataset = dataset->Map({decode_op, resize_op}, {"image"})
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
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
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
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
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
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
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
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
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
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
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
  ///      std::shared_ptr<Dataset> ds = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
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
  ///      std::shared_ptr<Dataset> ds2 = ImageFolder(folder_path, true, std::make_shared<RandomSampler>(false, 10));
  ///      ds2 = ds2->Rename({"image", "label"}, {"image_2", "label_2"});
  ///      std::shared_ptr<Dataset> ds = ds1->Zip({ds2});
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
}  // namespace mindspore::dataset
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_DATASETS_H_
