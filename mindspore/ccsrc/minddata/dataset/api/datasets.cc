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

#include "minddata/dataset/engine/runtime_context.h"
#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"

#include "minddata/dataset/core/client.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"

#include "minddata/dataset/kernels/c_func_op.h"
#include "minddata/dataset/kernels/tensor_op.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/cache/dataset_cache_impl.h"
#endif

#ifndef ENABLE_ANDROID
#include "minddata/dataset/text/sentence_piece_vocab.h"
#include "minddata/dataset/text/vocab.h"
#endif

// Sampler headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"

#include "minddata/dataset/engine/ir/datasetops/dataset_node.h"

// IR non-leaf nodes
#include "minddata/dataset/engine/ir/datasetops/batch_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/bucket_batch_by_length_node.h"
#include "minddata/dataset/engine/ir/datasetops/build_sentence_piece_vocab_node.h"
#include "minddata/dataset/engine/ir/datasetops/build_vocab_node.h"
#include "minddata/dataset/engine/ir/datasetops/concat_node.h"
#include "minddata/dataset/engine/ir/datasetops/filter_node.h"
#endif

#include "minddata/dataset/engine/ir/datasetops/map_node.h"
#include "minddata/dataset/engine/ir/datasetops/project_node.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/rename_node.h"
#endif

#include "minddata/dataset/engine/ir/datasetops/repeat_node.h"
#include "minddata/dataset/engine/ir/datasetops/shuffle_node.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/skip_node.h"
#include "minddata/dataset/engine/ir/datasetops/take_node.h"
#include "minddata/dataset/engine/ir/datasetops/transfer_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"
#endif

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/services.h"

// IR leaf nodes
#include "minddata/dataset/engine/ir/datasetops/source/album_node.h"

// IR leaf nodes disabled for android
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"
#endif

namespace mindspore {
namespace dataset {

// Function to create the iterator, which will build and launch the execution tree.
std::shared_ptr<Iterator> Dataset::CreateIterator(std::vector<std::string> columns, int32_t num_epochs) {
  std::shared_ptr<Iterator> iter;
  try {
    auto ds = shared_from_this();

    // The specified columns will be selected from the dataset and passed down the pipeline
    // in the order specified, other columns will be discarded.
    if (!columns.empty()) {
      ds = ds->Project(columns);
    }

    iter = std::make_shared<Iterator>();
    Status rc = iter->BuildAndLaunchTree(ds, num_epochs);
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

#ifndef ENABLE_ANDROID
// Function to return a transferred Node that transfers data through a device.
bool Dataset::DeviceQueue(std::string queue_name, std::string device_type, int32_t num_epochs, bool send_epoch_end,
                          int32_t total_batches, bool create_data_info_queue) {
  Status rc;

  // Build and launch tree
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Failed to init runtime context. Error status: " << rc;
    return false;
  }

  // Add TransferNode IR on top of dataset
  auto ds = std::make_shared<TransferNode>(shared_from_this()->IRNode(), queue_name, device_type, send_epoch_end,
                                           total_batches, create_data_info_queue);

  // Get ToDevice consumer
  auto consumer = std::make_unique<ToDevice>(num_epochs);
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

// Function to create the saver, which will build and launch the execution tree and save data
bool Dataset::Save(std::string dataset_path, int32_t num_files, std::string dataset_type) {
  Status rc;
  // Build and launch tree
  auto ds = shared_from_this();
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
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
  rc = consumer->Init(ds->IRNode());
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
Dataset::Dataset() { tree_getters_ = std::make_shared<TreeGetters>(); }

int64_t Dataset::GetDatasetSize(bool estimate) {
  int64_t dataset_size;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), -1);
  std::shared_ptr<DatasetSizeGetter> size_getter = std::make_shared<DatasetSizeGetter>();
  RETURN_SECOND_IF_ERROR(size_getter->Init(this->IRNode()), -1);
  RETURN_SECOND_IF_ERROR(size_getter->GetDatasetSize(&dataset_size, estimate), -1);
  return dataset_size;
}

std::vector<DataType> Dataset::GetOutputTypes() {
  std::vector<DataType> types;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->GetOutputTypes(&types), {});
  return types;
}

std::vector<TensorShape> Dataset::GetOutputShapes() {
  std::vector<TensorShape> shapes;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->GetOutputShapes(&shapes), {});
  return shapes;
}

int64_t Dataset::GetNumClasses() {
  int64_t num_classes;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), -1);
  RETURN_SECOND_IF_ERROR(tree_getters_->Init(this->IRNode()), -1);
  RETURN_SECOND_IF_ERROR(tree_getters_->GetNumClasses(&num_classes), -1);
  return num_classes;
}

std::vector<std::string> Dataset::GetColumnNames() {
  std::vector<std::string> col_names;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->GetColumnNames(&col_names), {});
  return col_names;
}

std::vector<std::pair<std::string, std::vector<int32_t>>> Dataset::GetClassIndexing() {
  std::vector<std::pair<std::string, std::vector<int32_t>>> output_class_indexing;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(tree_getters_->GetClassIndexing(&output_class_indexing), {});
  return output_class_indexing;
}

/// \brief Function to create a SchemaObj
/// \param[in] schema_file Path of schema file
/// \return Shared pointer to the current schema
std::shared_ptr<SchemaObj> Schema(const std::string &schema_file) {
  auto schema = std::make_shared<SchemaObj>(schema_file);

  return schema->Init() ? schema : nullptr;
}

// FUNCTIONS TO CREATE DATASETS FOR LEAF CLASSES
// (In alphabetical order)

// Function to create a AlbumDataset.
std::shared_ptr<AlbumDataset> Album(const std::string &dataset_dir, const std::string &data_schema,
                                    const std::vector<std::string> &column_names, bool decode,
                                    const std::shared_ptr<SamplerObj> &sampler,
                                    const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<AlbumDataset>(dataset_dir, data_schema, column_names, decode, sampler, cache);

  return ds;
}

#ifndef ENABLE_ANDROID
// Function to create a CelebADataset.
std::shared_ptr<CelebADataset> CelebA(const std::string &dataset_dir, const std::string &usage,
                                      const std::shared_ptr<SamplerObj> &sampler, bool decode,
                                      const std::set<std::string> &extensions,
                                      const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CelebADataset>(dataset_dir, usage, sampler, decode, extensions, cache);

  return ds;
}

// Function to create a Cifar10Dataset.
std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir, const std::string &usage,
                                        const std::shared_ptr<SamplerObj> &sampler,
                                        const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<Cifar10Dataset>(dataset_dir, usage, sampler, cache);

  return ds;
}

// Function to create a Cifar100Dataset.
std::shared_ptr<Cifar100Dataset> Cifar100(const std::string &dataset_dir, const std::string &usage,
                                          const std::shared_ptr<SamplerObj> &sampler,
                                          const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<Cifar100Dataset>(dataset_dir, usage, sampler, cache);

  return ds;
}

// Function to create a CLUEDataset.
std::shared_ptr<CLUEDataset> CLUE(const std::vector<std::string> &clue_files, const std::string &task,
                                  const std::string &usage, int64_t num_samples, ShuffleMode shuffle,
                                  int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CLUEDataset>(clue_files, task, usage, num_samples, shuffle, num_shards, shard_id, cache);

  return ds;
}

// Function to create a CocoDataset.
std::shared_ptr<CocoDataset> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                  const std::string &task, const bool &decode,
                                  const std::shared_ptr<SamplerObj> &sampler,
                                  const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CocoDataset>(dataset_dir, annotation_file, task, decode, sampler, cache);

  return ds;
}

// Function to create a CSVDataset.
std::shared_ptr<CSVDataset> CSV(const std::vector<std::string> &dataset_files, char field_delim,
                                const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                                const std::vector<std::string> &column_names, int64_t num_samples, ShuffleMode shuffle,
                                int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CSVDataset>(dataset_files, field_delim, column_defaults, column_names, num_samples,
                                         shuffle, num_shards, shard_id, cache);

  return ds;
}

// Function to create a ImageFolderDataset.
std::shared_ptr<ImageFolderDataset> ImageFolder(const std::string &dataset_dir, bool decode,
                                                const std::shared_ptr<SamplerObj> &sampler,
                                                const std::set<std::string> &extensions,
                                                const std::map<std::string, int32_t> &class_indexing,
                                                const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<ImageFolderDataset>(dataset_dir, decode, sampler, extensions, class_indexing, cache);

  return ds;
}

// Function to create a ManifestDataset.
std::shared_ptr<ManifestDataset> Manifest(const std::string &dataset_file, const std::string &usage,
                                          const std::shared_ptr<SamplerObj> &sampler,
                                          const std::map<std::string, int32_t> &class_indexing, bool decode,
                                          const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<ManifestDataset>(dataset_file, usage, sampler, class_indexing, decode, cache);

  return ds;
}

// Function to create a MindDataDataset.
std::shared_ptr<MindDataDataset> MindData(const std::string &dataset_file, const std::vector<std::string> &columns_list,
                                          const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                          int64_t num_padded) {
  auto ds = std::make_shared<MindDataDataset>(dataset_file, columns_list, sampler, padded_sample, num_padded);

  return ds;
}

// Function to create a MindDataDataset.
std::shared_ptr<MindDataDataset> MindData(const std::vector<std::string> &dataset_files,
                                          const std::vector<std::string> &columns_list,
                                          const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                          int64_t num_padded) {
  auto ds = std::make_shared<MindDataDataset>(dataset_files, columns_list, sampler, padded_sample, num_padded);

  return ds;
}

// Function to create a MnistDataset.
std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir, const std::string &usage,
                                    const std::shared_ptr<SamplerObj> &sampler,
                                    const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<MnistDataset>(dataset_dir, usage, sampler, cache);

  return ds;
}
// Function to overload "+" operator to concat two datasets
std::shared_ptr<ConcatDataset> operator+(const std::shared_ptr<Dataset> &datasets1,
                                         const std::shared_ptr<Dataset> &datasets2) {
  return std::make_shared<ConcatDataset>(std::vector({datasets1, datasets2}));
}

// Function to create a TextFileDataset.
std::shared_ptr<TextFileDataset> TextFile(const std::vector<std::string> &dataset_files, int64_t num_samples,
                                          ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                          const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<TextFileDataset>(dataset_files, num_samples, shuffle, num_shards, shard_id, cache);

  return ds;
}

// Function to create a VOCDataset.
std::shared_ptr<VOCDataset> VOC(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                                const std::map<std::string, int32_t> &class_indexing, bool decode,
                                const std::shared_ptr<SamplerObj> &sampler,
                                const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<VOCDataset>(dataset_dir, task, usage, class_indexing, decode, sampler, cache);

  return ds;
}

// Function to create a ZipDatset.
std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  auto ds = std::make_shared<ZipDataset>(datasets);
  return ds;
}
#endif
// FUNCTIONS TO CREATE DATASETS FOR DATASET OPS
// (In alphabetical order)

// Function to create a Batch dataset
BatchDataset::BatchDataset(std::shared_ptr<Dataset> input, int32_t batch_size, bool drop_remainder) {
  // Default values
  auto ds = std::make_shared<BatchNode>(input->IRNode(), batch_size, drop_remainder);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#ifndef ENABLE_ANDROID
// Function to create a BucketBatchByLength dataset
BucketBatchByLengthDataset::BucketBatchByLengthDataset(
  std::shared_ptr<Dataset> input, const std::vector<std::string> &column_names,
  const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
  std::function<TensorRow(TensorRow)> element_length_function,
  const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info, bool pad_to_bucket_boundary,
  bool drop_remainder) {
  std::shared_ptr<TensorOp> c_func = nullptr;
  if (element_length_function != nullptr) {
    c_func = std::make_shared<CFuncOp>(element_length_function);
  }
  auto ds =
    std::make_shared<BucketBatchByLengthNode>(input->IRNode(), column_names, bucket_boundaries, bucket_batch_sizes,
                                              c_func, pad_info, pad_to_bucket_boundary, drop_remainder);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ConcatDataset::ConcatDataset(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  std::vector<std::shared_ptr<DatasetNode>> all_datasets;
  (void)std::transform(datasets.begin(), datasets.end(), std::back_inserter(all_datasets),
                       [](std::shared_ptr<Dataset> dataset) -> std::shared_ptr<DatasetNode> {
                         return (dataset != nullptr) ? dataset->IRNode() : nullptr;
                       });

  auto ds = std::make_shared<ConcatNode>(all_datasets);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FilterDataset::FilterDataset(std::shared_ptr<Dataset> input, std::function<TensorRow(TensorRow)> predicate,
                             const std::vector<std::string> &input_columns) {
  std::shared_ptr<TensorOp> c_func = nullptr;
  if (predicate) c_func = std::make_shared<CFuncOp>(predicate);
  auto ds = std::make_shared<FilterNode>(input->IRNode(), c_func, input_columns);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#endif

MapDataset::MapDataset(std::shared_ptr<Dataset> input, std::vector<std::shared_ptr<TensorOperation>> operations,
                       const std::vector<std::string> &input_columns, const std::vector<std::string> &output_columns,
                       const std::vector<std::string> &project_columns, const std::shared_ptr<DatasetCache> &cache,
                       std::vector<std::shared_ptr<DSCallback>> callbacks) {
  auto ds = std::make_shared<MapNode>(input->IRNode(), operations, input_columns, output_columns, project_columns,
                                      cache, callbacks);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ProjectDataset::ProjectDataset(std::shared_ptr<Dataset> input, const std::vector<std::string> &columns) {
  auto ds = std::make_shared<ProjectNode>(input->IRNode(), columns);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#ifndef ENABLE_ANDROID
RenameDataset::RenameDataset(std::shared_ptr<Dataset> input, const std::vector<std::string> &input_columns,
                             const std::vector<std::string> &output_columns) {
  auto ds = std::make_shared<RenameNode>(input->IRNode(), input_columns, output_columns);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#endif

RepeatDataset::RepeatDataset(std::shared_ptr<Dataset> input, int32_t count) {
  auto ds = std::make_shared<RepeatNode>(input->IRNode(), count);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ShuffleDataset::ShuffleDataset(std::shared_ptr<Dataset> input, int32_t buffer_size) {
  // Pass in reshuffle_each_epoch with true
  auto ds = std::make_shared<ShuffleNode>(input->IRNode(), buffer_size, true);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#ifndef ENABLE_ANDROID
SkipDataset::SkipDataset(std::shared_ptr<Dataset> input, int32_t count) {
  auto ds = std::make_shared<SkipNode>(input->IRNode(), count);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TakeDataset::TakeDataset(std::shared_ptr<Dataset> input, int32_t count) {
  auto ds = std::make_shared<TakeNode>(input->IRNode(), count);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ZipDataset::ZipDataset(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  std::vector<std::shared_ptr<DatasetNode>> all_datasets;
  (void)std::transform(
    datasets.begin(), datasets.end(), std::back_inserter(all_datasets),
    [](std::shared_ptr<Dataset> dataset) -> std::shared_ptr<DatasetNode> { return dataset->IRNode(); });

  auto ds = std::make_shared<ZipNode>(all_datasets);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#endif
int64_t Dataset::GetBatchSize() {
  int64_t batch_size = -1;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), -1);
  RETURN_SECOND_IF_ERROR(tree_getters_->Init(this->IRNode()), -1);
  RETURN_SECOND_IF_ERROR(tree_getters_->GetBatchSize(&batch_size), -1);
  return batch_size;
}

int64_t Dataset::GetRepeatCount() {
  int64_t repeat_count = 0;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), -1);
  RETURN_SECOND_IF_ERROR(tree_getters_->Init(this->IRNode()), 0);
  RETURN_SECOND_IF_ERROR(tree_getters_->GetRepeatCount(&repeat_count), 0);
  return repeat_count;
}

std::shared_ptr<Dataset> Dataset::SetNumWorkers(int32_t num_workers) {
  if (ir_node_ == nullptr || ir_node_->SetNumWorkers(num_workers) == nullptr) {
    return nullptr;
  }
  return shared_from_this();
}

#ifndef ENABLE_ANDROID
std::shared_ptr<SentencePieceVocab> Dataset::BuildSentencePieceVocab(
  const std::vector<std::string> &col_names, int32_t vocab_size, float character_coverage,
  SentencePieceModel model_type, const std::unordered_map<std::string, std::string> &params) {
  auto vocab = std::make_shared<SentencePieceVocab>();
  auto ds = std::make_shared<BuildSentenceVocabNode>(IRNode(), vocab, col_names, vocab_size, character_coverage,
                                                     model_type, params);

  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  Status rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "BuildSentencePieceVocab: Failed to init runtime context. Error status: " << rc;
    return nullptr;
  }

  auto consumer = std::make_unique<BuildVocabConsumer>();
  BuildVocabConsumer *bv_consumer = consumer.get();
  rc = consumer->Init(ds);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "BuildSentencePieceVocab: Failed to init consumer. Error status: " << rc;
    return nullptr;
  }
  runtime_context->AssignConsumer(std::move(consumer));

  // Run tree here to starting building SentencePieceVocab
  rc = bv_consumer->Start();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "BuildSentencePieceVocab: Failed to start consumer. Error status: " << rc;
    return nullptr;
  }
  return vocab;
}

std::shared_ptr<Vocab> Dataset::BuildVocab(const std::vector<std::string> &columns,
                                           const std::pair<int64_t, int64_t> &freq_range, int64_t top_k,
                                           const std::vector<std::string> &special_tokens, bool special_first) {
  auto vocab = std::make_shared<Vocab>();
  auto ds =
    std::make_shared<BuildVocabNode>(IRNode(), vocab, columns, freq_range, top_k, special_tokens, special_first);

  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  Status rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "BuildVocab: Failed to init runtime context. Error status: " << rc;
    return nullptr;
  }

  auto consumer = std::make_unique<BuildVocabConsumer>();
  BuildVocabConsumer *bv_consumer = consumer.get();
  rc = consumer->Init(ds);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "BuildVocab: Failed to init consumer. Error status: " << rc;
    return nullptr;
  }
  runtime_context->AssignConsumer(std::move(consumer));

  // Run tree here to starting building vocab
  rc = bv_consumer->Start();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "BuildVocab: Failed to start consumer. Error status: " << rc;
    return nullptr;
  }
  return vocab;
}
#endif

std::shared_ptr<BatchDataset> Dataset::Batch(int32_t batch_size, bool drop_remainder) {
  return std::make_shared<BatchDataset>(shared_from_this(), batch_size, drop_remainder);
}

SchemaObj::SchemaObj(const std::string &schema_file) : schema_file_(schema_file), num_rows_(0), dataset_type_("") {}

// SchemaObj Init function
Status SchemaObj::Init() {
  if (!schema_file_.empty()) {
    Path schema_file(schema_file_);
    CHECK_FAIL_RETURN_UNEXPECTED(schema_file.Exists(),
                                 "The file " + schema_file_ + " does not exist or permission denied!");

    nlohmann::json js;
    try {
      std::ifstream in(schema_file_);
      in >> js;
      CHECK_FAIL_RETURN_UNEXPECTED(js.find("columns") != js.end(),
                                   "\"columns\" node is required in the schema json file.");
    } catch (const std::exception &err) {
      std::string err_msg = "Schema file failed to load: ";
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    return from_json(js);
  }
  return Status::OK();
}

// Function to add a column to schema with a mstype de_type and known shape
Status SchemaObj::add_column(const std::string &name, TypeId de_type, const std::vector<int32_t> &shape) {
  DataType data_type = dataset::MSTypeToDEType(de_type);
  return add_column(name, data_type.ToString(), shape);
}

// Function to add a column to schema with a string de_type and known shape
Status SchemaObj::add_column(const std::string &name, const std::string &de_type, const std::vector<int32_t> &shape) {
  DataType data_type(de_type);
  CHECK_FAIL_RETURN_UNEXPECTED(data_type != DataType::DE_UNKNOWN, "Type is unknown.");

  nlohmann::json new_column;
  new_column["name"] = name;
  new_column["type"] = data_type.ToString();
  new_column["shape"] = shape;
  new_column["rank"] = shape.size();

  columns_.push_back(new_column);
  return Status::OK();
}

// Function to add a column to schema with a mstype de_type and without shape
Status SchemaObj::add_column(const std::string &name, TypeId de_type) {
  DataType data_type = dataset::MSTypeToDEType(de_type);
  return add_column(name, data_type.ToString());
}

// Function to add a column to schema with a string de_type and without shape
Status SchemaObj::add_column(const std::string &name, const std::string &de_type) {
  DataType data_type(de_type);
  CHECK_FAIL_RETURN_UNEXPECTED(data_type != DataType::DE_UNKNOWN, "Type is unknown.");

  nlohmann::json new_column;
  new_column["name"] = name;
  new_column["type"] = data_type.ToString();
  new_column["rank"] = 1;

  columns_.push_back(new_column);
  return Status::OK();
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

Status SchemaObj::parse_column(nlohmann::json columns) {
  std::string name, de_type;
  std::vector<int32_t> shape;

  columns_.clear();
  if (columns.type() == nlohmann::json::value_t::array) {
    // reference to python list
    for (auto column : columns) {
      auto key_name = column.find("name");
      if (key_name == column.end()) {
        RETURN_STATUS_SYNTAX_ERROR("Column's name is missing");
      }
      name = *key_name;

      auto key_type = column.find("type");
      if (key_type == column.end()) {
        RETURN_STATUS_SYNTAX_ERROR("Column's type is missing");
      }
      de_type = *key_type;

      shape.clear();
      auto key_shape = column.find("shape");
      if (key_shape != column.end()) {
        shape.insert(shape.end(), (*key_shape).begin(), (*key_shape).end());
      }
      RETURN_IF_NOT_OK(add_column(name, de_type, shape));
    }
  } else if (columns.type() == nlohmann::json::value_t::object) {
    for (const auto &it_child : columns.items()) {
      name = it_child.key();
      auto key_type = it_child.value().find("type");
      if (key_type == it_child.value().end()) {
        RETURN_STATUS_SYNTAX_ERROR("Column's type is missing");
      }
      de_type = *key_type;

      shape.clear();
      auto key_shape = it_child.value().find("shape");
      if (key_shape != it_child.value().end()) {
        shape.insert(shape.end(), (*key_shape).begin(), (*key_shape).end());
      }

      RETURN_IF_NOT_OK(add_column(name, de_type, shape));
    }
  } else {
    RETURN_STATUS_SYNTAX_ERROR("columns must be dict or list, columns contain name, type, shape(optional).");
  }
  return Status::OK();
}

Status SchemaObj::from_json(nlohmann::json json_obj) {
  for (const auto &it_child : json_obj.items()) {
    if (it_child.key() == "datasetType") {
      dataset_type_ = it_child.value();
    } else if (it_child.key() == "numRows") {
      num_rows_ = it_child.value();
    } else if (it_child.key() == "columns") {
      RETURN_IF_NOT_OK(parse_column(it_child.value()));
    } else {
      RETURN_STATUS_SYNTAX_ERROR("Unknown field " + it_child.key());
    }
  }
  if (columns_.empty()) {
    RETURN_STATUS_SYNTAX_ERROR("Columns are missing.");
  }
  if (num_rows_ < 0) {
    RETURN_STATUS_SYNTAX_ERROR("numRows must be greater than or equal to 0");
  }

  return Status::OK();
}

Status SchemaObj::FromJSONString(const std::string &json_string) {
  try {
    nlohmann::json js = nlohmann::json::parse(json_string);
    CHECK_FAIL_RETURN_UNEXPECTED(js.find("columns") != js.end(),
                                 "\"columns\" node is required in the schema json JSON.");
    RETURN_IF_NOT_OK(from_json(js));
  } catch (const std::exception &err) {
    std::string err_msg = "FromJSONString: JSON string failed to parse: ";
    err_msg += err.what();
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status SchemaObj::ParseColumnString(const std::string &json_string) {
  try {
    nlohmann::json js = nlohmann::json::parse(json_string);
    RETURN_IF_NOT_OK(parse_column(js));
  } catch (const std::exception &err) {
    std::string err_msg = "ParseColumnString: JSON string failed to parse: ";
    err_msg += err.what();
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// OTHER FUNCTIONS

#ifndef ENABLE_ANDROID

std::shared_ptr<DatasetCache> CreateDatasetCache(session_id_type id, uint64_t mem_sz, bool spill,
                                                 std::optional<std::string> hostname, std::optional<int32_t> port,
                                                 std::optional<int32_t> num_connections,
                                                 std::optional<int32_t> prefetch_sz) {
  auto cache = std::make_shared<DatasetCacheImpl>(id, mem_sz, spill, hostname, port, num_connections, prefetch_sz);
  return cache;
}
#endif

AlbumDataset::AlbumDataset(const std::string &dataset_dir, const std::string &data_schema,
                           const std::vector<std::string> &column_names, bool decode,
                           const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<AlbumNode>(dataset_dir, data_schema, column_names, decode, sampler, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#ifndef ENABLE_ANDROID
CelebADataset::CelebADataset(const std::string &dataset_dir, const std::string &usage,
                             const std::shared_ptr<SamplerObj> &sampler, bool decode,
                             const std::set<std::string> &extensions, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CelebANode>(dataset_dir, usage, sampler, decode, extensions, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
Cifar10Dataset::Cifar10Dataset(const std::string &dataset_dir, const std::string &usage,
                               const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<Cifar10Node>(dataset_dir, usage, sampler, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
Cifar100Dataset::Cifar100Dataset(const std::string &dataset_dir, const std::string &usage,
                                 const std::shared_ptr<SamplerObj> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<Cifar100Node>(dataset_dir, usage, sampler, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
CLUEDataset::CLUEDataset(const std::vector<std::string> &dataset_files, const std::string &task,
                         const std::string &usage, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                         int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CLUENode>(dataset_files, task, usage, num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
CocoDataset::CocoDataset(const std::string &dataset_dir, const std::string &annotation_file, const std::string &task,
                         const bool &decode, const std::shared_ptr<SamplerObj> &sampler,
                         const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CocoNode>(dataset_dir, annotation_file, task, decode, sampler, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
CSVDataset::CSVDataset(const std::vector<std::string> &dataset_files, char field_delim,
                       const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                       const std::vector<std::string> &column_names, int64_t num_samples, ShuffleMode shuffle,
                       int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CSVNode>(dataset_files, field_delim, column_defaults, column_names, num_samples, shuffle,
                                      num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
ImageFolderDataset::ImageFolderDataset(const std::string &dataset_dir, bool decode,
                                       const std::shared_ptr<SamplerObj> &sampler,
                                       const std::set<std::string> &extensions,
                                       const std::map<std::string, int32_t> &class_indexing,
                                       const std::shared_ptr<DatasetCache> &cache) {
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderDataset.
  auto ds =
    std::make_shared<ImageFolderNode>(dataset_dir, decode, sampler, recursive, extensions, class_indexing, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ManifestDataset::ManifestDataset(const std::string &dataset_file, const std::string &usage,
                                 const std::shared_ptr<SamplerObj> &sampler,
                                 const std::map<std::string, int32_t> &class_indexing, bool decode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<ManifestNode>(dataset_file, usage, sampler, class_indexing, decode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MindDataDataset::MindDataDataset(const std::string &dataset_file, const std::vector<std::string> &columns_list,
                                 const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                 int64_t num_padded) {
  auto ds = std::make_shared<MindDataNode>(dataset_file, columns_list, sampler, padded_sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MindDataDataset::MindDataDataset(const std::vector<std::string> &dataset_files,
                                 const std::vector<std::string> &columns_list,
                                 const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                 int64_t num_padded) {
  auto ds = std::make_shared<MindDataNode>(dataset_files, columns_list, sampler, padded_sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

MnistDataset::MnistDataset(const std::string &dataset_dir, const std::string &usage,
                           const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<MnistNode>(dataset_dir, usage, sampler, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
TextFileDataset::TextFileDataset(const std::vector<std::string> &dataset_files, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<TextFileNode>(dataset_files, num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

VOCDataset::VOCDataset(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                       const std::map<std::string, int32_t> &class_indexing, bool decode,
                       const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<VOCNode>(dataset_dir, task, usage, class_indexing, decode, sampler, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

RandomDataDataset::RandomDataDataset(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema,
                                     const std::vector<std::string> &columns_list,
                                     std::shared_ptr<DatasetCache> cache) {
  auto ds = std::make_shared<RandomNode>(total_rows, std::move(schema), std::move(columns_list), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
RandomDataDataset::RandomDataDataset(const int32_t &total_rows, std::string schema_path,
                                     const std::vector<std::string> &columns_list,
                                     std::shared_ptr<DatasetCache> cache) {
  auto ds = std::make_shared<RandomNode>(total_rows, std::move(schema_path), std::move(columns_list), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TFRecordDataset::TFRecordDataset(const std::vector<std::string> &dataset_files, std::string schema,
                                 const std::vector<std::string> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                                 int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                                 std::shared_ptr<DatasetCache> cache) {
  auto ds = std::make_shared<TFRecordNode>(dataset_files, schema, columns_list, num_samples, shuffle, num_shards,
                                           shard_id, shard_equal_rows, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
TFRecordDataset::TFRecordDataset(const std::vector<std::string> &dataset_files, std::shared_ptr<SchemaObj> schema,
                                 const std::vector<std::string> &columns_list, int64_t num_samples, ShuffleMode shuffle,
                                 int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                                 std::shared_ptr<DatasetCache> cache) {
  auto ds = std::make_shared<TFRecordNode>(dataset_files, schema, columns_list, num_samples, shuffle, num_shards,
                                           shard_id, shard_equal_rows, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#endif
}  // namespace dataset
}  // namespace mindspore
