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

#include "minddata/dataset/include/datasets.h"
#include <algorithm>
#include <fstream>
#include <unordered_set>
#include <utility>
#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/include/transforms.h"
// Source dataset headers (in alphabetical order)
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/engine/datasetops/source/album_op.h"
#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/clue_op.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/csv_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#include "minddata/dataset/engine/datasetops/source/mindrecord_op.h"
#include "minddata/dataset/engine/ir/cache/dataset_cache_impl.h"
#endif
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#endif
// Dataset operator headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/datasetops/zip_op.h"

// Sampler headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

// IR non-leaf nodes
#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#include "minddata/dataset/engine/ir/datasetops/concat_node.h"
#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/project_node.h"
#include "minddata/dataset/engine/ir/datasetops/rename_node.h"
#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/take_node.h"
#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/bucket_batch_by_length_node.h"
#include "minddata/dataset/engine/ir/datasetops/build_sentence_piece_vocab_node.h"
#include "minddata/dataset/engine/ir/datasetops/build_vocab_node.h"
#endif

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/services.h"

// IR leaf nodes
#include "minddata/dataset/engine/ir/datasetops/source/album_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"

// IR leaf nodes disabled for android
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"
#endif

namespace mindspore {
namespace dataset {
namespace api {

// Function to create the iterator, which will build and launch the execution tree.
std::shared_ptr<Iterator> Dataset::CreateIterator(std::vector<std::string> columns) {
  std::shared_ptr<Iterator> iter;
  try {
    auto ds = shared_from_this();

    // The specified columns will be selected from the dataset and passed down the pipeline
    // in the order specified, other columns will be discarded.
    if (!columns.empty()) {
      ds = ds->Project(columns);
    }

    iter = std::make_shared<Iterator>();
    Status rc = iter->BuildAndLaunchTree(ds);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "CreateIterator failed." << rc;
      return nullptr;
    }

    return iter;
  } catch (const std::exception &err) {
    MS_LOG(ERROR) << "CreateIterator: Iterator exception caught: " << err.what();
    return nullptr;
  }

  return iter;
}

// Function to return a transferred Node that transfers data through a device.
bool Dataset::DeviceQueue(bool send_epoch_end) {
  Status rc;

  // Build and launch tree
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Failed to init runtime context. Error status: " << rc;
    return false;
  }

  // Get a uuid for queue name
  std::string queue_name = Services::GetUniqueID();

  // TODO(CRC):
  // Get device type from ms context
  std::string device_type = "CPU";

  // Get device ID from children
  int32_t device_id = 0;
  rc = TransferNode::get_distribution(shared_from_this(), &device_id);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Failed to get shard id. Error status: " << rc;
    return false;
  }

  // Add TransferNode IR on top of dataset d
  auto ds = std::make_shared<TransferNode>(shared_from_this(), queue_name, device_id, device_type, send_epoch_end);

  // Get ToDevice consumer
  auto consumer = std::make_unique<ToDevice>(device_type, send_epoch_end, -1);
  ToDevice *consumer_ = consumer.get();
  rc = consumer->Init(ds);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "ToDevice: Failed to init. Error status: " << rc;
    return false;
  }
  runtime_context->AssignConsumer(std::move(consumer));

  // Send data to device
  rc = consumer_->Send();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "ToDevice: Failed to send data to device. Error status: " << rc;
    return false;
  }

  return true;
}

#ifndef ENABLE_ANDROID
// Function to create the saver, which will build and launch the execution tree and save data
bool Dataset::Save(std::string dataset_path, int32_t num_files, std::string dataset_type) {
  Status rc;
  // Build and launch tree
  auto ds = shared_from_this();
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "CreateSaver failed." << rc;
    return false;
  }

  // Get SaveToDisk consumer
  auto consumer = std::make_unique<SaveToDisk>(dataset_path, num_files, dataset_type);
  rc = consumer->ValidateParams();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "CreateSaver failed." << rc;
    return false;
  }
  SaveToDisk *consumer_ = consumer.get();
  rc = consumer->Init(ds);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "CreateSaver failed." << rc;
    return false;
  }
  runtime_context->AssignConsumer(std::move(consumer));

  // Save data into file
  rc = consumer_->Save();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Saver: Failed to save data into file. Error status: " << rc;
    return false;
  }

  // Shut down the data pipeline
  rc = runtime_context->Terminate();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Saver: Failed to shut down pipeline. Error status: " << rc;
    return false;
  }

  return true;
}
#endif

// Constructor
Dataset::Dataset() {
  // Fetch some default value from config manager
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  num_workers_ = cfg->num_parallel_workers();
  rows_per_buffer_ = cfg->rows_per_buffer();
  connector_que_size_ = cfg->op_connector_size();
  worker_connector_size_ = cfg->worker_connector_size();
  tree_getters_ = std::make_shared<TreeGetters>();
}

int64_t Dataset::GetDatasetSize() {
  int64_t dataset_size;
  auto ds = shared_from_this();
  Status rc;
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetDatasetSize: Initializing RuntimeContext failed.";
    return -1;
  }
  if (!tree_getters_->isInitialized()) {
    rc = tree_getters_->Init(ds);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "GetDatasetSize: Initializing TreeGetters failed.";
      return -1;
    }
  }
  rc = tree_getters_->GetDatasetSize(&dataset_size);
  return rc.IsError() ? -1 : dataset_size;
}

std::vector<DataType> Dataset::GetOutputTypes() {
  std::vector<DataType> types;
  Status rc;
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetOutputTypes: Initializing RuntimeContext failed.";
    types.clear();
    return types;
  }
  if (!tree_getters_->isInitialized()) {
    rc = tree_getters_->Init(shared_from_this());
    if (rc.IsError()) {
      MS_LOG(ERROR) << "GetOutputTypes: Initializing TreeGetters failed.";
      types.clear();
      return types;
    }
  }
  rc = tree_getters_->GetOutputTypes(&types);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetOutputTypes: Get Output Types failed.";
    types.clear();
    return types;
  }
  return types;
}

std::vector<TensorShape> Dataset::GetOutputShapes() {
  std::vector<TensorShape> shapes;
  Status rc;
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetOutputShapes: Initializing RuntimeContext failed.";
    shapes.clear();
    return shapes;
  }
  if (!tree_getters_->isInitialized()) {
    rc = tree_getters_->Init(shared_from_this());
    if (rc.IsError()) {
      MS_LOG(ERROR) << "GetOutputShapes: Initializing TreeGetters failed.";
      shapes.clear();
      return shapes;
    }
  }
  rc = tree_getters_->GetOutputShapes(&shapes);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetOutputShapes: Get Output Shapes failed.";
    shapes.clear();
    return shapes;
  }
  return shapes;
}

int64_t Dataset::GetNumClasses() {
  int64_t num_classes;
  auto ds = shared_from_this();
  Status rc;
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetNumClasses: Initializing RuntimeContext failed.";
    return -1;
  }
  if (!tree_getters_->isInitialized()) {
    rc = tree_getters_->Init(ds);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "GetNumClasses: Initializing TreeGetters failed.";
      return -1;
    }
  }
  rc = tree_getters_->GetNumClasses(&num_classes);
  return rc.IsError() ? -1 : num_classes;
}

// Constructor to initialize the cache
Dataset::Dataset(const std::shared_ptr<DatasetCache> &dataset_cache) : Dataset() { cache_ = dataset_cache; }

/// \brief Function to create a SchemaObj
/// \param[in] schema_file Path of schema file
/// \return Shared pointer to the current schema
std::shared_ptr<SchemaObj> Schema(const std::string &schema_file) {
  auto schema = std::make_shared<SchemaObj>(schema_file);

  return schema->init() ? schema : nullptr;
}

// FUNCTIONS TO CREATE DATASETS FOR LEAF-NODE DATASETS
// (In alphabetical order)

// Function to create a AlbumNode.
std::shared_ptr<AlbumNode> Album(const std::string &dataset_dir, const std::string &data_schema,
                                 const std::vector<std::string> &column_names, bool decode,
                                 const std::shared_ptr<SamplerObj> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<AlbumNode>(dataset_dir, data_schema, column_names, decode, sampler, cache);

  return ds;
}

// Function to create a CelebANode.
std::shared_ptr<CelebANode> CelebA(const std::string &dataset_dir, const std::string &usage,
                                   const std::shared_ptr<SamplerObj> &sampler, bool decode,
                                   const std::set<std::string> &extensions,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CelebANode>(dataset_dir, usage, sampler, decode, extensions, cache);

  return ds;
}

// Function to create a Cifar10Node.
std::shared_ptr<Cifar10Node> Cifar10(const std::string &dataset_dir, const std::string &usage,
                                     const std::shared_ptr<SamplerObj> &sampler,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<Cifar10Node>(dataset_dir, usage, sampler, cache);

  return ds;
}

// Function to create a Cifar100Node.
std::shared_ptr<Cifar100Node> Cifar100(const std::string &dataset_dir, const std::string &usage,
                                       const std::shared_ptr<SamplerObj> &sampler,
                                       const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<Cifar100Node>(dataset_dir, usage, sampler, cache);

  return ds;
}

// Function to create a CLUENode.
std::shared_ptr<CLUENode> CLUE(const std::vector<std::string> &clue_files, const std::string &task,
                               const std::string &usage, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                               int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CLUENode>(clue_files, task, usage, num_samples, shuffle, num_shards, shard_id, cache);

  return ds;
}

// Function to create a CocoNode.
std::shared_ptr<CocoNode> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                               const std::string &task, const bool &decode, const std::shared_ptr<SamplerObj> &sampler,
                               const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CocoNode>(dataset_dir, annotation_file, task, decode, sampler, cache);

  return ds;
}

// Function to create a CSVNode.
std::shared_ptr<CSVNode> CSV(const std::vector<std::string> &dataset_files, char field_delim,
                             const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                             const std::vector<std::string> &column_names, int64_t num_samples, ShuffleMode shuffle,
                             int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CSVNode>(dataset_files, field_delim, column_defaults, column_names, num_samples, shuffle,
                                      num_shards, shard_id, cache);

  return ds;
}

// Function to create a ImageFolderNode.
std::shared_ptr<ImageFolderNode> ImageFolder(const std::string &dataset_dir, bool decode,
                                             const std::shared_ptr<SamplerObj> &sampler,
                                             const std::set<std::string> &extensions,
                                             const std::map<std::string, int32_t> &class_indexing,
                                             const std::shared_ptr<DatasetCache> &cache) {
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderNode.
  auto ds =
    std::make_shared<ImageFolderNode>(dataset_dir, decode, sampler, recursive, extensions, class_indexing, cache);

  return ds;
}

#ifndef ENABLE_ANDROID
// Function to create a ManifestNode.
std::shared_ptr<ManifestNode> Manifest(const std::string &dataset_file, const std::string &usage,
                                       const std::shared_ptr<SamplerObj> &sampler,
                                       const std::map<std::string, int32_t> &class_indexing, bool decode,
                                       const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<ManifestNode>(dataset_file, usage, sampler, class_indexing, decode, cache);

  return ds;
}

// Function to create a MindDataNode.
std::shared_ptr<MindDataNode> MindData(const std::string &dataset_file, const std::vector<std::string> &columns_list,
                                       const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                       int64_t num_padded) {
  auto ds = std::make_shared<MindDataNode>(dataset_file, columns_list, sampler, padded_sample, num_padded);

  return ds;
}

// Function to create a MindDataNode.
std::shared_ptr<MindDataNode> MindData(const std::vector<std::string> &dataset_files,
                                       const std::vector<std::string> &columns_list,
                                       const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                       int64_t num_padded) {
  auto ds = std::make_shared<MindDataNode>(dataset_files, columns_list, sampler, padded_sample, num_padded);

  return ds;
}
#endif

// Function to create a MnistNode.
std::shared_ptr<MnistNode> Mnist(const std::string &dataset_dir, const std::string &usage,
                                 const std::shared_ptr<SamplerObj> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<MnistNode>(dataset_dir, usage, sampler, cache);

  return ds;
}

// Function to overload "+" operator to concat two datasets
std::shared_ptr<ConcatNode> operator+(const std::shared_ptr<Dataset> &datasets1,
                                      const std::shared_ptr<Dataset> &datasets2) {
  std::shared_ptr<ConcatNode> ds = std::make_shared<ConcatNode>(std::vector({datasets2, datasets1}));

  return ds;
}

// Function to create a TextFileNode.
std::shared_ptr<TextFileNode> TextFile(const std::vector<std::string> &dataset_files, int64_t num_samples,
                                       ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                       const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<TextFileNode>(dataset_files, num_samples, shuffle, num_shards, shard_id, cache);

  return ds;
}

#ifndef ENABLE_ANDROID
// Function to create a VOCNode.
std::shared_ptr<VOCNode> VOC(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                             const std::map<std::string, int32_t> &class_indexing, bool decode,
                             const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<VOCNode>(dataset_dir, task, usage, class_indexing, decode, sampler, cache);

  return ds;
}
#endif

// Function to create a ZipNode.
std::shared_ptr<ZipNode> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  auto ds = std::make_shared<ZipNode>(datasets);

  return ds;
}

// FUNCTIONS TO CREATE DATASETS FOR DATASET OPS
// (In alphabetical order)

// Function to create a Batch dataset
std::shared_ptr<BatchNode> Dataset::Batch(int32_t batch_size, bool drop_remainder) {
  // Default values
  std::vector<std::string> cols_to_map = {};
  std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map;
  bool pad = false;
  auto ds = std::make_shared<BatchNode>(shared_from_this(), batch_size, drop_remainder, pad, cols_to_map, pad_map);

  return ds;
}

#ifndef ENABLE_ANDROID
// Function to create a BucketBatchByLength dataset
std::shared_ptr<BucketBatchByLengthNode> Dataset::BucketBatchByLength(
  const std::vector<std::string> &column_names, const std::vector<int32_t> &bucket_boundaries,
  const std::vector<int32_t> &bucket_batch_sizes, std::function<TensorRow(TensorRow)> element_length_function,
  const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info, bool pad_to_bucket_boundary,
  bool drop_remainder) {
  auto ds = std::make_shared<BucketBatchByLengthNode>(shared_from_this(), column_names, bucket_boundaries,
                                                      bucket_batch_sizes, element_length_function, pad_info,
                                                      pad_to_bucket_boundary, drop_remainder);

  return ds;
}

// Function to create a SentencePieceVocab from dataset
std::shared_ptr<SentencePieceVocab> Dataset::BuildSentencePieceVocab(
  const std::vector<std::string> &col_names, uint32_t vocab_size, float character_coverage,
  SentencePieceModel model_type, const std::unordered_map<std::string, std::string> &params) {
  auto vocab = std::make_shared<SentencePieceVocab>();
  auto ds = std::make_shared<BuildSentenceVocabNode>(shared_from_this(), vocab, col_names, vocab_size,
                                                     character_coverage, model_type, params);

  // Validate input params
  if (!ds->ValidateParams()) {
    return nullptr;
  }

  // Run tree here to start building vocab
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  if (iter == nullptr) {
    MS_LOG(ERROR) << "Fail to run iterator in BuildSentencePieceVocab.";
    return nullptr;
  }

  // Finish building vocab by triggering GetNextRow
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  if (!iter->GetNextRow(&row)) {
    return nullptr;
  }

  return vocab;
}

// Function to create a Vocab from dataset
std::shared_ptr<Vocab> Dataset::BuildVocab(const std::vector<std::string> &columns,
                                           const std::pair<int64_t, int64_t> &freq_range, int64_t top_k,
                                           const std::vector<std::string> &special_tokens, bool special_first) {
  auto vocab = std::make_shared<Vocab>();
  auto ds = std::make_shared<BuildVocabNode>(shared_from_this(), vocab, columns, freq_range, top_k, special_tokens,
                                             special_first);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  // Run tree here to starting building vocab
  std::shared_ptr<Iterator> iter = ds->CreateIterator();
  if (iter == nullptr) {
    MS_LOG(ERROR) << "Fail to run iterator in BuildVocab.";
    return nullptr;
  }

  // Finish building vocab by triggering GetNextRow
  std::unordered_map<std::string, std::shared_ptr<Tensor>> row;
  if (!iter->GetNextRow(&row)) {
    return nullptr;
  }

  return vocab;
}
#endif

// Function to create a Concat dataset
std::shared_ptr<ConcatNode> Dataset::Concat(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  auto ds = std::make_shared<ConcatNode>(datasets);
  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create a Map dataset.
std::shared_ptr<MapNode> Dataset::Map(std::vector<std::shared_ptr<TensorOperation>> operations,
                                      std::vector<std::string> input_columns, std::vector<std::string> output_columns,
                                      const std::vector<std::string> &project_columns,
                                      const std::shared_ptr<DatasetCache> &cache) {
  auto ds =
    std::make_shared<MapNode>(shared_from_this(), operations, input_columns, output_columns, project_columns, cache);

  return ds;
}

// Function to create a ProjectNode.
std::shared_ptr<ProjectNode> Dataset::Project(const std::vector<std::string> &columns) {
  auto ds = std::make_shared<ProjectNode>(shared_from_this(), columns);

  return ds;
}

// Function to create a RenameNode.
std::shared_ptr<RenameNode> Dataset::Rename(const std::vector<std::string> &input_columns,
                                            const std::vector<std::string> &output_columns) {
  auto ds = std::make_shared<RenameNode>(shared_from_this(), input_columns, output_columns);

  return ds;
}

// Function to create Repeat dataset.
std::shared_ptr<Dataset> Dataset::Repeat(int32_t count) {
  // Workaround for repeat == 1, do not inject repeat.
  if (count == 1) {
    return shared_from_this();
  }

  auto ds = std::make_shared<RepeatNode>(shared_from_this(), count);

  return ds;
}

// Function to create a ShuffleOp
std::shared_ptr<ShuffleNode> Dataset::Shuffle(int32_t buffer_size) {
  // Pass in reshuffle_each_epoch with true
  auto ds = std::make_shared<ShuffleNode>(shared_from_this(), buffer_size, true);

  return ds;
}

// Function to create a SkipNode.
std::shared_ptr<SkipNode> Dataset::Skip(int32_t count) {
  auto ds = std::make_shared<SkipNode>(shared_from_this(), count);

  return ds;
}

// Function to create a TakeNode.
std::shared_ptr<Dataset> Dataset::Take(int32_t count) {
  // If count is greater than the number of element in dataset or equal to -1,
  // all the element in dataset will be taken
  if (count == -1) {
    return shared_from_this();
  }

  auto ds = std::make_shared<TakeNode>(shared_from_this(), count);

  return ds;
}

// Function to create a Zip dataset
std::shared_ptr<ZipNode> Dataset::Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  // Default values
  auto ds = std::make_shared<ZipNode>(datasets);
  ds->children.push_back(shared_from_this());

  return ds;
}

Status Dataset::AddCacheOp(std::vector<std::shared_ptr<DatasetOp>> *node_ops) {
  if (cache_ != nullptr) {
    RETURN_IF_NOT_OK(cache_->Build());
    std::shared_ptr<DatasetOp> cache_op;
    RETURN_IF_NOT_OK(cache_->CreateCacheOp(num_workers_, &cache_op));
    node_ops->push_back(cache_op);
  }
  return Status::OK();
}

int64_t Dataset::GetBatchSize() {
  int64_t batch_size;
  auto ds = shared_from_this();
  Status rc;
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetBatchSize: Initializing RuntimeContext failed.";
    return -1;
  }
  if (!tree_getters_->isInitialized()) {
    rc = tree_getters_->Init(ds);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "GetBatchSize: Initializing TreeGetters failed.";
      return -1;
    }
  }
  rc = tree_getters_->GetBatchSize(&batch_size);
  return rc.IsError() ? -1 : batch_size;
}

int64_t Dataset::GetRepeatCount() {
  int64_t repeat_count;
  auto ds = shared_from_this();
  Status rc;
  std::unique_ptr<RuntimeContext> runtime_context = std::make_unique<RuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "GetRepeatCount: Initializing RuntimeContext failed.";
    return -1;
  }
  if (!tree_getters_->isInitialized()) {
    rc = tree_getters_->Init(ds);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "GetRepeatCount: Initializing TreeGetters failed.";
      return -1;
    }
  }
  rc = tree_getters_->GetRepeatCount(&repeat_count);
  return rc.IsError() ? 0 : repeat_count;
}

SchemaObj::SchemaObj(const std::string &schema_file) : schema_file_(schema_file), num_rows_(0), dataset_type_("") {}

// SchemaObj init function
bool SchemaObj::init() {
  if (schema_file_ != "") {
    Path schema_file(schema_file_);
    if (!schema_file.Exists()) {
      MS_LOG(ERROR) << "The file " << schema_file << " does not exist or permission denied!";
      return false;
    }

    nlohmann::json js;
    try {
      std::ifstream in(schema_file_);
      in >> js;
      if (js.find("columns") == js.end()) {
        MS_LOG(ERROR) << "\"columns\" node is required in the schema json file.";
        return false;
      }
    } catch (const std::exception &err) {
      MS_LOG(ERROR) << "Schema file failed to load";
      return false;
    }
    return from_json(js);
  }
  return true;
}

// Function to add a column to schema with a mstype de_type
bool SchemaObj::add_column(std::string name, TypeId de_type, std::vector<int32_t> shape) {
  nlohmann::json new_column;
  new_column["name"] = name;
  // if de_type is mstype
  DataType data_type = dataset::MSTypeToDEType(de_type);
  new_column["type"] = data_type.ToString();
  if (shape.size() > 0) {
    new_column["shape"] = shape;
    new_column["rank"] = shape.size();
  } else {
    new_column["rank"] = 1;
  }
  columns_.push_back(new_column);
  return true;
}

// Function to add a column to schema with a string de_type
bool SchemaObj::add_column(std::string name, std::string de_type, std::vector<int32_t> shape) {
  nlohmann::json new_column;
  new_column["name"] = name;
  DataType data_type(de_type);
  new_column["type"] = data_type.ToString();
  if (shape.size() > 0) {
    new_column["shape"] = shape;
    new_column["rank"] = shape.size();
  } else {
    new_column["rank"] = 1;
  }
  columns_.push_back(new_column);
  return true;
}

std::string SchemaObj::to_json() {
  nlohmann::json json_file;
  json_file["columns"] = columns_;
  if (dataset_type_ != "") {
    json_file["datasetType"] = dataset_type_;
  }

  if (num_rows_ > 0) {
    json_file["numRows"] = num_rows_;
  }

  return json_file.dump(2);
}

bool SchemaObj::parse_column(nlohmann::json columns) {
  std::string name, de_type;
  std::vector<int32_t> shape;

  columns_.clear();
  if (columns.type() == nlohmann::json::value_t::array) {
    // reference to python list
    for (auto column : columns) {
      auto key_name = column.find("name");
      if (key_name == column.end()) {
        MS_LOG(ERROR) << "Column's name is missing";
        return false;
      }
      name = *key_name;

      auto key_type = column.find("type");
      if (key_type == column.end()) {
        MS_LOG(ERROR) << "Column's type is missing";
        return false;
      }
      de_type = *key_type;

      shape.clear();
      auto key_shape = column.find("shape");
      if (key_shape != column.end()) {
        shape.insert(shape.end(), (*key_shape).begin(), (*key_shape).end());
      }
      if (!add_column(name, de_type, shape)) {
        return false;
      }
    }
  } else if (columns.type() == nlohmann::json::value_t::object) {
    for (const auto &it_child : columns.items()) {
      name = it_child.key();
      auto key_type = it_child.value().find("type");
      if (key_type == it_child.value().end()) {
        MS_LOG(ERROR) << "Column's type is missing";
        return false;
      }
      de_type = *key_type;

      shape.clear();
      auto key_shape = it_child.value().find("shape");
      if (key_shape != it_child.value().end()) {
        shape.insert(shape.end(), (*key_shape).begin(), (*key_shape).end());
      }

      if (!add_column(name, de_type, shape)) {
        return false;
      }
    }
  } else {
    MS_LOG(ERROR) << "columns must be dict or list, columns contain name, type, shape(optional).";
    return false;
  }
  return true;
}

bool SchemaObj::from_json(nlohmann::json json_obj) {
  for (const auto &it_child : json_obj.items()) {
    if (it_child.key() == "datasetType") {
      dataset_type_ = it_child.value();
    } else if (it_child.key() == "numRows") {
      num_rows_ = it_child.value();
    } else if (it_child.key() == "columns") {
      if (!parse_column(it_child.value())) {
        MS_LOG(ERROR) << "parse columns failed";
        return false;
      }
    } else {
      MS_LOG(ERROR) << "Unknown field " << it_child.key();
      return false;
    }
  }
  if (columns_.empty()) {
    MS_LOG(ERROR) << "Columns are missing.";
    return false;
  }
  if (num_rows_ <= 0) {
    MS_LOG(ERROR) << "numRows must be greater than 0";
    return false;
  }

  return true;
}

// OTHER FUNCTIONS

// Helper function to compute a default shuffle size
Status ComputeShuffleSize(int64_t num_files, int64_t num_devices, int64_t num_rows, int64_t total_rows,
                          int64_t *shuffle_size) {
  const int64_t average_files_multiplier = 4;
  const int64_t shuffle_max = 10000;
  int64_t avg_rows_per_file = 0;

  // Adjust the num rows per shard if sharding was given
  if (num_devices > 0) {
    if (num_rows % num_devices == 0) {
      num_rows = num_rows / num_devices;
    } else {
      num_rows = (num_rows / num_devices) + 1;
    }
  }

  // Cap based on total rows directive.  Some ops do not have this and give value of 0.
  if (total_rows > 0) {
    num_rows = std::min(num_rows, total_rows);
  }

  // get the average per file
  CHECK_FAIL_RETURN_UNEXPECTED(num_files != 0, "The size of dataset_files must greater than 0.");
  avg_rows_per_file = num_rows / num_files;

  *shuffle_size = std::max(avg_rows_per_file * average_files_multiplier, shuffle_max);
  return Status::OK();
}

// Helper function to inject a shuffle operator over top of current operator being built
Status AddShuffleOp(int64_t num_files, int64_t num_devices, int64_t num_rows, int64_t total_rows,
                    int32_t connector_que_size, int32_t rows_per_buffer, std::shared_ptr<DatasetOp> *shuffle_op) {
  std::shared_ptr<ShuffleOp> new_shuffle_op = nullptr;
  int64_t shuffle_size = 0;
  RETURN_EMPTY_IF_ERROR(ComputeShuffleSize(num_files, num_devices, num_rows, total_rows, &shuffle_size));
  MS_LOG(INFO) << "Dataset::AddShuffleOp - num_rows: " << num_rows << ", shuffle_size: " << shuffle_size;
  // Add the shuffle op
  *shuffle_op = std::make_shared<ShuffleOp>(shuffle_size, GetSeed(), connector_que_size, true, rows_per_buffer);
  return Status::OK();
}

// Helper function to validate dataset directory parameter
Status ValidateDatasetDirParam(const std::string &dataset_name, std::string dataset_dir) {
  if (dataset_dir.empty()) {
    std::string err_msg = dataset_name + ": dataset_dir is not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  Path dir(dataset_dir);
  if (!dir.IsDirectory()) {
    std::string err_msg = dataset_name + ": dataset_dir: [" + dataset_dir + "] is an invalid directory path.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (access(dataset_dir.c_str(), R_OK) == -1) {
    std::string err_msg = dataset_name + ": No access to specified dataset path: " + dataset_dir;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Helper function to validate dataset files parameter
Status ValidateDatasetFilesParam(const std::string &dataset_name, const std::vector<std::string> &dataset_files) {
  if (dataset_files.empty()) {
    std::string err_msg = dataset_name + ": dataset_files is not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  for (auto f : dataset_files) {
    Path dataset_file(f);
    if (!dataset_file.Exists()) {
      std::string err_msg = dataset_name + ": dataset file: [" + f + "] is invalid or does not exist.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (access(dataset_file.toString().c_str(), R_OK) == -1) {
      std::string err_msg = dataset_name + ": No access to specified dataset file: " + f;
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

// Helper function to validate dataset num_shards and shard_id parameters
Status ValidateDatasetShardParams(const std::string &dataset_name, int32_t num_shards, int32_t shard_id) {
  if (num_shards <= 0) {
    std::string err_msg = dataset_name + ": Invalid num_shards: " + std::to_string(num_shards);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (shard_id < 0 || shard_id >= num_shards) {
    // num_shards;
    std::string err_msg = dataset_name + ": Invalid input, shard_id: " + std::to_string(shard_id) +
                          ", num_shards: " + std::to_string(num_shards);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Helper function to validate dataset sampler parameter
Status ValidateDatasetSampler(const std::string &dataset_name, const std::shared_ptr<SamplerObj> &sampler) {
  if (sampler == nullptr) {
    std::string err_msg = dataset_name + ": Sampler is not constructed correctly, sampler: nullptr";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status ValidateStringValue(const std::string &dataset_name, const std::string &str,
                           const std::unordered_set<std::string> &valid_strings) {
  if (valid_strings.find(str) == valid_strings.end()) {
    std::string mode;
    mode = std::accumulate(valid_strings.begin(), valid_strings.end(), mode,
                           [](std::string a, std::string b) { return std::move(a) + " " + std::move(b); });
    std::string err_msg = dataset_name + ": " + str + " does not match any mode in [" + mode + " ]";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Helper function to validate dataset input/output column parameter
Status ValidateDatasetColumnParam(const std::string &dataset_name, const std::string &column_param,
                                  const std::vector<std::string> &columns) {
  if (columns.empty()) {
    std::string err_msg = dataset_name + ":" + column_param + " should not be empty string";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (uint32_t i = 0; i < columns.size(); ++i) {
    if (columns[i].empty()) {
      std::string err_msg = dataset_name + ":" + column_param + "[" + std::to_string(i) + "] must not be empty";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  std::set<std::string> columns_set(columns.begin(), columns.end());
  if (columns_set.size() != columns.size()) {
    std::string err_msg = dataset_name + ":" + column_param + ": Every column name should not be same with others";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

#ifndef ENABLE_ANDROID

std::shared_ptr<DatasetCache> CreateDatasetCache(session_id_type id, uint64_t mem_sz, bool spill,
                                                 std::optional<std::string> hostname, std::optional<int32_t> port,
                                                 std::optional<int32_t> num_connections,
                                                 std::optional<int32_t> prefetch_sz) {
  auto cache = std::make_shared<DatasetCacheImpl>(id, mem_sz, spill, hostname, port, num_connections, prefetch_sz);
  return cache->ValidateParams() ? cache : nullptr;
}
#endif

std::shared_ptr<SamplerObj> SelectSampler(int64_t num_samples, bool shuffle, int32_t num_shards, int32_t shard_id) {
  if (shuffle) {
    if (num_shards > 1) {
      // If shuffle enabled, sharding enabled, use distributed random sampler
      return DistributedSampler(num_shards, shard_id, shuffle, num_samples);
    }
    // If shuffle enabled, sharding disabled, use random sampler
    return RandomSampler(num_samples >= 0, num_samples);
  }
  if (num_shards > 1) {
    // If shuffle disabled, sharding enabled, use distributed sequential sampler
    return DistributedSampler(num_shards, shard_id, shuffle, num_samples);
  }
  // If shuffle disabled, sharding disabled, use sequential sampler
  return SequentialSampler(0, num_samples);
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
