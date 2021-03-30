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

#include "minddata/dataset/include/datasets.h"
#include <algorithm>
#include <fstream>
#include <unordered_set>
#include <utility>
#include <nlohmann/json.hpp>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/engine/runtime_context.h"
#include "minddata/dataset/include/iterator.h"
#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/core/client.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/consumers/pull_based_tree_consumer.h"

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
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"

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
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"

// IR leaf nodes disabled for android
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"
#endif

namespace mindspore {
namespace dataset {

// convert MSTensorVec to DE TensorRow, return empty if fails
TensorRow VecToRow(const MSTensorVec &v) {
  TensorRow row;
  row.reserve(v.size());
  for (const MSTensor &t : v) {
    std::shared_ptr<Tensor> rt;
    Status rc = Tensor::CreateFromMSTensor(t, &rt);
    if (rc.IsError()) {
      MS_LOG_ERROR << "Convert from MSTensor to DETensor failed:" << rc.ToString() << ".";
      return {};
    }
    row.emplace_back(rt);
  }
  return row;
}

// convert DE TensorRow to MSTensorVec, won't fail
MSTensorVec RowToVec(const TensorRow &v) {
  MSTensorVec rv;
  rv.reserve(v.size());
  std::transform(v.begin(), v.end(), std::back_inserter(rv), [](std::shared_ptr<Tensor> t) -> MSTensor {
    return mindspore::MSTensor(std::make_shared<DETensor>(t));
  });
  return rv;
}

// Convert a std::function<TensorRow(TensorRow)> to std::function<MSTensorVec(MSTensor)> with this helper
TensorRow FuncPtrConverter(std::function<MSTensorVec(MSTensorVec)> func, TensorRow in_row) {
  return VecToRow(func(RowToVec(in_row)));
}

// Function to create the iterator, which will build and launch the execution tree.
std::shared_ptr<Iterator> Dataset::CreateIteratorCharIF(std::vector<std::vector<char>> columns, int32_t num_epochs) {
  std::shared_ptr<Iterator> iter;
  try {
    auto ds = shared_from_this();

    // The specified columns will be selected from the dataset and passed down the pipeline
    // in the order specified, other columns will be discarded.
    if (!VectorCharToString(columns).empty()) {
      ds = ds->Project(VectorCharToString(columns));
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

// Function to create the iterator, which will build and launch the execution tree.
std::shared_ptr<PullIterator> Dataset::CreatePullBasedIterator(std::vector<std::vector<char>> columns) {
  // The specified columns will be selected from the dataset and passed down the pipeline
  // in the order specified, other columns will be discarded.
  // This code is not in a try/catch block because there is no execution tree class that will be created.
  auto ds = shared_from_this();
  if (!VectorCharToString(columns).empty()) {
    ds = ds->Project(VectorCharToString(columns));
  }

  std::shared_ptr<PullIterator> iter = std::make_shared<PullIterator>();
  Status rc = iter->BuildAndLaunchTree(ds);
  if (rc.IsError()) MS_LOG(ERROR) << "CreateIterator: Iterator exception caught: " << rc;
  RETURN_SECOND_IF_ERROR(rc, nullptr);
  return iter;
}

#ifndef ENABLE_ANDROID
// Function to return a transferred Node that transfers data through a device.
bool Dataset::DeviceQueueCharIF(const std::vector<char> &queue_name, const std::vector<char> &device_type,
                                int32_t device_id, int32_t num_epochs, bool send_epoch_end, int32_t total_batches,
                                bool create_data_info_queue) {
  Status rc;

  // Build and launch tree
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  rc = runtime_context->Init();
  if (rc.IsError()) {
    MS_LOG(ERROR) << "Failed to init runtime context. Error status: " << rc;
    return false;
  }

  // Add TransferNode IR on top of dataset
  auto ds =
    std::make_shared<TransferNode>(shared_from_this()->IRNode(), CharToString(queue_name), CharToString(device_type),
                                   device_id, send_epoch_end, total_batches, create_data_info_queue);

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
bool Dataset::SaveCharIF(const std::vector<char> &dataset_path, int32_t num_files,
                         const std::vector<char> &dataset_type) {
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
  auto consumer = std::make_unique<SaveToDisk>(CharToString(dataset_path), num_files, CharToString(dataset_type));
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
  DatasetSizeGetter *consumer = size_getter.get();
  runtime_context->AssignConsumer(size_getter);
  RETURN_SECOND_IF_ERROR(consumer->Init(this->IRNode()), -1);
  RETURN_SECOND_IF_ERROR(consumer->GetDatasetSize(&dataset_size, estimate), -1);
  return dataset_size;
}

std::vector<mindspore::DataType> Dataset::GetOutputTypes() {
  std::vector<DataType> types;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  TreeGetters *consumer = tree_getters_.get();
  runtime_context->AssignConsumer(tree_getters_);
  RETURN_SECOND_IF_ERROR(consumer->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(consumer->GetOutputTypes(&types), {});
  std::vector<mindspore::DataType> ret_types;
  std::transform(
    types.begin(), types.end(), std::back_inserter(ret_types),
    [](const DataType &d) -> mindspore::DataType { return static_cast<mindspore::DataType>(DETypeToMSType(d)); });
  return ret_types;
}

std::vector<std::vector<int64_t>> Dataset::GetOutputShapes() {
  std::vector<TensorShape> shapes;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  TreeGetters *consumer = tree_getters_.get();
  runtime_context->AssignConsumer(tree_getters_);
  RETURN_SECOND_IF_ERROR(consumer->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(consumer->GetOutputShapes(&shapes), {});
  std::vector<std::vector<int64_t>> ret_shapes;
  std::transform(shapes.begin(), shapes.end(), std::back_inserter(ret_shapes),
                 [](const TensorShape &s) -> std::vector<int64_t> { return s.AsVector(); });
  return ret_shapes;
}

int64_t Dataset::GetNumClasses() {
  int64_t num_classes;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), -1);
  TreeGetters *consumer = tree_getters_.get();
  runtime_context->AssignConsumer(tree_getters_);
  RETURN_SECOND_IF_ERROR(consumer->Init(this->IRNode()), -1);
  RETURN_SECOND_IF_ERROR(consumer->GetNumClasses(&num_classes), -1);
  return num_classes;
}

std::vector<std::vector<char>> Dataset::GetColumnNamesCharIF() {
  std::vector<std::string> col_names;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  TreeGetters *consumer = tree_getters_.get();
  runtime_context->AssignConsumer(tree_getters_);
  RETURN_SECOND_IF_ERROR(consumer->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(consumer->GetColumnNames(&col_names), {});
  return VectorStringToChar(col_names);
}

std::vector<std::pair<std::vector<char>, std::vector<int32_t>>> Dataset::GetClassIndexingCharIF() {
  std::vector<std::pair<std::string, std::vector<int32_t>>> output_class_indexing;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), {});
  TreeGetters *consumer = tree_getters_.get();
  runtime_context->AssignConsumer(tree_getters_);
  RETURN_SECOND_IF_ERROR(consumer->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(consumer->GetClassIndexing(&output_class_indexing), {});
  return ClassIndexStringToChar(output_class_indexing);
}

/// \brief Function to create a SchemaObj
/// \param[in] schema_file Path of schema file
/// \return Shared pointer to the current schema
std::shared_ptr<SchemaObj> SchemaCharIF(const std::vector<char> &schema_file) {
  auto schema = std::make_shared<SchemaObj>(CharToString(schema_file));
  return schema->Init() ? schema : nullptr;
}

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
  std::shared_ptr<Dataset> input, const std::vector<std::vector<char>> &column_names,
  const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
  std::function<MSTensorVec(MSTensorVec)> element_length_function,
  const std::map<std::vector<char>, std::pair<std::vector<int64_t>, MSTensor>> &pad_info, bool pad_to_bucket_boundary,
  bool drop_remainder) {
  std::shared_ptr<TensorOp> c_func = nullptr;
  if (element_length_function != nullptr) {
    c_func = std::make_shared<CFuncOp>(std::bind(FuncPtrConverter, element_length_function, std::placeholders::_1));
  }

  std::map<std::vector<char>, std::pair<TensorShape, std::shared_ptr<Tensor>>> map;
  for (auto const &p : pad_info) {
    const MSTensor &t = p.second.second;
    std::shared_ptr<Tensor> rt;
    Status rc = Tensor::CreateFromMemory(TensorShape(t.Shape()), MSTypeToDEType(static_cast<TypeId>(t.DataType())),
                                         (const uchar *)(t.Data().get()), t.DataSize(), &rt);
    if (rc.IsError()) {
      MS_LOG_ERROR << "Fail to create DETensor from MSTensor for pad_info: " << rc.ToString() << ".";
      map.clear();
      break;
    }
    map.insert({p.first, {TensorShape(p.second.first), rt}});
  }

  auto ds = std::make_shared<BucketBatchByLengthNode>(input->IRNode(), VectorCharToString(column_names),
                                                      bucket_boundaries, bucket_batch_sizes, c_func,
                                                      PadInfoCharToString(map), pad_to_bucket_boundary, drop_remainder);

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

FilterDataset::FilterDataset(std::shared_ptr<Dataset> input, std::function<MSTensorVec(MSTensorVec)> predicate,
                             const std::vector<std::vector<char>> &input_columns) {
  std::shared_ptr<TensorOp> c_func = nullptr;
  if (predicate) c_func = std::make_shared<CFuncOp>(std::bind(FuncPtrConverter, predicate, std::placeholders::_1));
  auto ds = std::make_shared<FilterNode>(input->IRNode(), c_func, VectorCharToString(input_columns));

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#endif

MapDataset::MapDataset(std::shared_ptr<Dataset> input, std::vector<std::shared_ptr<TensorOperation>> operations,
                       const std::vector<std::vector<char>> &input_columns,
                       const std::vector<std::vector<char>> &output_columns,
                       const std::vector<std::vector<char>> &project_columns,
                       const std::shared_ptr<DatasetCache> &cache, std::vector<std::shared_ptr<DSCallback>> callbacks) {
  auto ds = std::make_shared<MapNode>(input->IRNode(), operations, VectorCharToString(input_columns),
                                      VectorCharToString(output_columns), VectorCharToString(project_columns), cache,
                                      callbacks);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ProjectDataset::ProjectDataset(std::shared_ptr<Dataset> input, const std::vector<std::vector<char>> &columns) {
  auto ds = std::make_shared<ProjectNode>(input->IRNode(), VectorCharToString(columns));

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#ifndef ENABLE_ANDROID
RenameDataset::RenameDataset(std::shared_ptr<Dataset> input, const std::vector<std::vector<char>> &input_columns,
                             const std::vector<std::vector<char>> &output_columns) {
  auto ds = std::make_shared<RenameNode>(input->IRNode(), VectorCharToString(input_columns),
                                         VectorCharToString(output_columns));

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
std::shared_ptr<SentencePieceVocab> Dataset::BuildSentencePieceVocabCharIF(
  const std::vector<std::vector<char>> &col_names, int32_t vocab_size, float character_coverage,
  SentencePieceModel model_type, const std::map<std::vector<char>, std::vector<char>> &params) {
  auto vocab = std::make_shared<SentencePieceVocab>();
  auto ds = std::make_shared<BuildSentenceVocabNode>(IRNode(), vocab, VectorCharToString(col_names), vocab_size,
                                                     character_coverage, model_type, UnorderedMapCharToString(params));

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

std::shared_ptr<Vocab> Dataset::BuildVocabCharIF(const std::vector<std::vector<char>> &columns,
                                                 const std::pair<int64_t, int64_t> &freq_range, int64_t top_k,
                                                 const std::vector<std::vector<char>> &special_tokens,
                                                 bool special_first) {
  auto vocab = std::make_shared<Vocab>();
  auto ds = std::make_shared<BuildVocabNode>(IRNode(), vocab, VectorCharToString(columns), freq_range, top_k,
                                             VectorCharToString(special_tokens), special_first);

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

struct SchemaObj::Data {
  int32_t num_rows_;
  std::string dataset_type_;
  std::string schema_file_;
  nlohmann::json columns_;
};

SchemaObj::SchemaObj(const std::vector<char> &schema_file) : data_(std::make_shared<Data>()) {
  data_->schema_file_ = CharToString(schema_file);
  data_->dataset_type_ = "";
  data_->num_rows_ = 0;
}

// SchemaObj Init function
Status SchemaObj::Init() {
  if (!data_->schema_file_.empty()) {
    Path schema_file(data_->schema_file_);
    CHECK_FAIL_RETURN_UNEXPECTED(schema_file.Exists(),
                                 "The file " + data_->schema_file_ + " does not exist or permission denied!");

    nlohmann::json js;
    try {
      std::ifstream in(data_->schema_file_);
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
Status SchemaObj::add_column_char(const std::vector<char> &name, mindspore::DataType de_type,
                                  const std::vector<int32_t> &shape) {
  DataType data_type = dataset::MSTypeToDEType(static_cast<TypeId>(de_type));
  return add_column_char(name, StringToChar(data_type.ToString()), shape);
}

// Function to add a column to schema with a string de_type and known shape
Status SchemaObj::add_column_char(const std::vector<char> &name, const std::vector<char> &de_type,
                                  const std::vector<int32_t> &shape) {
  DataType data_type(CharToString(de_type));
  CHECK_FAIL_RETURN_UNEXPECTED(data_type != DataType::DE_UNKNOWN, "Type is unknown.");

  nlohmann::json new_column;
  new_column["name"] = CharToString(name);
  new_column["type"] = data_type.ToString();
  new_column["shape"] = shape;
  new_column["rank"] = shape.size();

  data_->columns_.push_back(new_column);
  return Status::OK();
}

// Function to add a column to schema with a mstype de_type and without shape
Status SchemaObj::add_column_char(const std::vector<char> &name, mindspore::DataType de_type) {
  DataType data_type = dataset::MSTypeToDEType(static_cast<TypeId>(de_type));
  return add_column_char(name, StringToChar(data_type.ToString()));
}

// Function to add a column to schema with a string de_type and without shape
Status SchemaObj::add_column_char(const std::vector<char> &name, const std::vector<char> &de_type) {
  DataType data_type(CharToString(de_type));
  CHECK_FAIL_RETURN_UNEXPECTED(data_type != DataType::DE_UNKNOWN, "Type is unknown.");

  nlohmann::json new_column;
  new_column["name"] = CharToString(name);
  new_column["type"] = data_type.ToString();
  new_column["rank"] = 1;

  data_->columns_.push_back(new_column);
  return Status::OK();
}

const std::vector<char> SchemaObj::to_json_char() {
  nlohmann::json json_file;
  json_file["columns"] = data_->columns_;
  std::string str_dataset_type_(data_->dataset_type_);
  if (str_dataset_type_ != "") {
    json_file["datasetType"] = str_dataset_type_;
  }

  if (data_->num_rows_ > 0) {
    json_file["numRows"] = data_->num_rows_;
  }

  return StringToChar(json_file.dump(2));
}

void SchemaObj::set_dataset_type(std::string dataset_type) { data_->dataset_type_ = dataset_type.data(); }

void SchemaObj::set_num_rows(int32_t num_rows) { data_->num_rows_ = num_rows; }

int32_t SchemaObj::get_num_rows() const { return data_->num_rows_; }

Status SchemaObj::parse_column(nlohmann::json columns) {
  std::string name, de_type;
  std::vector<int32_t> shape;

  data_->columns_.clear();
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
      std::string str_dataset_type_ = it_child.value();
      data_->dataset_type_ = str_dataset_type_.data();
    } else if (it_child.key() == "numRows") {
      data_->num_rows_ = it_child.value();
    } else if (it_child.key() == "columns") {
      RETURN_IF_NOT_OK(parse_column(it_child.value()));
    } else {
      RETURN_STATUS_SYNTAX_ERROR("Unknown field " + it_child.key());
    }
  }
  if (data_->columns_.empty()) {
    RETURN_STATUS_SYNTAX_ERROR("Columns are missing.");
  }
  if (data_->num_rows_ < 0) {
    RETURN_STATUS_SYNTAX_ERROR("numRows must be greater than or equal to 0");
  }

  return Status::OK();
}

Status SchemaObj::FromJSONStringCharIF(const std::vector<char> &json_string) {
  try {
    nlohmann::json js = nlohmann::json::parse(CharToString(json_string));
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

Status SchemaObj::ParseColumnStringCharIF(const std::vector<char> &json_string) {
  try {
    nlohmann::json js = nlohmann::json::parse(CharToString(json_string));
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

std::shared_ptr<DatasetCache> CreateDatasetCacheCharIF(session_id_type id, uint64_t mem_sz, bool spill,
                                                       std::optional<std::vector<char>> hostname,
                                                       std::optional<int32_t> port,
                                                       std::optional<int32_t> num_connections,
                                                       std::optional<int32_t> prefetch_sz) {
  auto cache = std::make_shared<DatasetCacheImpl>(id, mem_sz, spill, hostname, port, num_connections, prefetch_sz);
  return cache;
}
#endif

AlbumDataset::AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
                           const std::vector<std::vector<char>> &column_names, bool decode,
                           const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<AlbumNode>(CharToString(dataset_dir), CharToString(data_schema),
                                        VectorCharToString(column_names), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

AlbumDataset::AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
                           const std::vector<std::vector<char>> &column_names, bool decode, const Sampler *sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<AlbumNode>(CharToString(dataset_dir), CharToString(data_schema),
                                        VectorCharToString(column_names), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
AlbumDataset::AlbumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &data_schema,
                           const std::vector<std::vector<char>> &column_names, bool decode,
                           const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<AlbumNode>(CharToString(dataset_dir), CharToString(data_schema),
                                        VectorCharToString(column_names), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#ifndef ENABLE_ANDROID
CelebADataset::CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                             const std::shared_ptr<Sampler> &sampler, bool decode,
                             const std::set<std::vector<char>> &extensions,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CelebANode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, decode,
                                         SetCharToString(extensions), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
CelebADataset::CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                             const Sampler *sampler, bool decode, const std::set<std::vector<char>> &extensions,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CelebANode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, decode,
                                         SetCharToString(extensions), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
CelebADataset::CelebADataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                             const std::reference_wrapper<Sampler> sampler, bool decode,
                             const std::set<std::vector<char>> &extensions,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<CelebANode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, decode,
                                         SetCharToString(extensions), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Cifar10Dataset::Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                               const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<Cifar10Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
Cifar10Dataset::Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                               const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<Cifar10Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
Cifar10Dataset::Cifar10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                               const std::reference_wrapper<Sampler> sampler,
                               const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<Cifar10Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Cifar100Dataset::Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<Cifar100Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
Cifar100Dataset::Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<Cifar100Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
Cifar100Dataset::Cifar100Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 const std::reference_wrapper<Sampler> sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<Cifar100Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CLUEDataset::CLUEDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &task,
                         const std::vector<char> &usage, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                         int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CLUENode>(VectorCharToString(dataset_files), CharToString(task), CharToString(usage),
                                       num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CocoDataset::CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                         const std::vector<char> &task, const bool &decode, const std::shared_ptr<Sampler> &sampler,
                         const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CocoNode>(CharToString(dataset_dir), CharToString(annotation_file), CharToString(task),
                                       decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
CocoDataset::CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                         const std::vector<char> &task, const bool &decode, const Sampler *sampler,
                         const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CocoNode>(CharToString(dataset_dir), CharToString(annotation_file), CharToString(task),
                                       decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
CocoDataset::CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                         const std::vector<char> &task, const bool &decode,
                         const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<CocoNode>(CharToString(dataset_dir), CharToString(annotation_file), CharToString(task),
                                       decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CSVDataset::CSVDataset(const std::vector<std::vector<char>> &dataset_files, char field_delim,
                       const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                       const std::vector<std::vector<char>> &column_names, int64_t num_samples, ShuffleMode shuffle,
                       int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds =
    std::make_shared<CSVNode>(VectorCharToString(dataset_files), field_delim, column_defaults,
                              VectorCharToString(column_names), num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ImageFolderDataset::ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode,
                                       const std::shared_ptr<Sampler> &sampler,
                                       const std::set<std::vector<char>> &extensions,
                                       const std::map<std::vector<char>, int32_t> &class_indexing,
                                       const std::shared_ptr<DatasetCache> &cache) {
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<ImageFolderNode>(CharToString(dataset_dir), decode, sampler_obj, recursive,
                                              SetCharToString(extensions), MapCharToString(class_indexing), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ImageFolderDataset::ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
                                       const std::set<std::vector<char>> &extensions,
                                       const std::map<std::vector<char>, int32_t> &class_indexing,
                                       const std::shared_ptr<DatasetCache> &cache) {
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<ImageFolderNode>(CharToString(dataset_dir), decode, sampler_obj, recursive,
                                              SetCharToString(extensions), MapCharToString(class_indexing), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ImageFolderDataset::ImageFolderDataset(const std::vector<char> &dataset_dir, bool decode,
                                       const std::reference_wrapper<Sampler> sampler,
                                       const std::set<std::vector<char>> &extensions,
                                       const std::map<std::vector<char>, int32_t> &class_indexing,
                                       const std::shared_ptr<DatasetCache> &cache) {
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderDataset.
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<ImageFolderNode>(CharToString(dataset_dir), decode, sampler_obj, recursive,
                                              SetCharToString(extensions), MapCharToString(class_indexing), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

ManifestDataset::ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                                 const std::shared_ptr<Sampler> &sampler,
                                 const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<ManifestNode>(CharToString(dataset_file), CharToString(usage), sampler_obj,
                                           MapCharToString(class_indexing), decode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
ManifestDataset::ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                                 const Sampler *sampler, const std::map<std::vector<char>, int32_t> &class_indexing,
                                 bool decode, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<ManifestNode>(CharToString(dataset_file), CharToString(usage), sampler_obj,
                                           MapCharToString(class_indexing), decode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
ManifestDataset::ManifestDataset(const std::vector<char> &dataset_file, const std::vector<char> &usage,
                                 const std::reference_wrapper<Sampler> sampler,
                                 const std::map<std::vector<char>, int32_t> &class_indexing, bool decode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<ManifestNode>(CharToString(dataset_file), CharToString(usage), sampler_obj,
                                           MapCharToString(class_indexing), decode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

MindDataDataset::MindDataDataset(const std::vector<char> &dataset_file,
                                 const std::vector<std::vector<char>> &columns_list,
                                 const std::shared_ptr<Sampler> &sampler, const nlohmann::json *padded_sample,
                                 int64_t num_padded) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }
  auto ds = std::make_shared<MindDataNode>(CharToString(dataset_file), VectorCharToString(columns_list), sampler_obj,
                                           sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MindDataDataset::MindDataDataset(const std::vector<char> &dataset_file,
                                 const std::vector<std::vector<char>> &columns_list, const Sampler *sampler,
                                 const nlohmann::json *padded_sample, int64_t num_padded) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }
  auto ds = std::make_shared<MindDataNode>(CharToString(dataset_file), VectorCharToString(columns_list), sampler_obj,
                                           sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MindDataDataset::MindDataDataset(const std::vector<char> &dataset_file,
                                 const std::vector<std::vector<char>> &columns_list,
                                 const std::reference_wrapper<Sampler> sampler, const nlohmann::json *padded_sample,
                                 int64_t num_padded) {
  auto sampler_obj = sampler.get().Parse();
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }

  auto ds = std::make_shared<MindDataNode>(CharToString(dataset_file), VectorCharToString(columns_list), sampler_obj,
                                           sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MindDataDataset::MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                                 const std::vector<std::vector<char>> &columns_list,
                                 const std::shared_ptr<Sampler> &sampler, const nlohmann::json *padded_sample,
                                 int64_t num_padded) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }

  auto ds = std::make_shared<MindDataNode>(VectorCharToString(dataset_files), VectorCharToString(columns_list),
                                           sampler_obj, sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MindDataDataset::MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                                 const std::vector<std::vector<char>> &columns_list, const Sampler *sampler,
                                 const nlohmann::json *padded_sample, int64_t num_padded) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }

  auto ds = std::make_shared<MindDataNode>(VectorCharToString(dataset_files), VectorCharToString(columns_list),
                                           sampler_obj, sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MindDataDataset::MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                                 const std::vector<std::vector<char>> &columns_list,
                                 const std::reference_wrapper<Sampler> sampler, const nlohmann::json *padded_sample,
                                 int64_t num_padded) {
  auto sampler_obj = sampler.get().Parse();
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }
  auto ds = std::make_shared<MindDataNode>(VectorCharToString(dataset_files), VectorCharToString(columns_list),
                                           sampler_obj, sample, num_padded);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#endif

MnistDataset::MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<MnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MnistDataset::MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<MnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
MnistDataset::MnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<MnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#ifndef ENABLE_ANDROID
TextFileDataset::TextFileDataset(const std::vector<std::vector<char>> &dataset_files, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<TextFileNode>(VectorCharToString(dataset_files), num_samples, shuffle, num_shards,
                                           shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

VOCDataset::VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                       bool decode, const std::shared_ptr<Sampler> &sampler,
                       const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<VOCNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      MapCharToString(class_indexing), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
VOCDataset::VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                       bool decode, const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<VOCNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      MapCharToString(class_indexing), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
VOCDataset::VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                       bool decode, const std::reference_wrapper<Sampler> sampler,
                       const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<VOCNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      MapCharToString(class_indexing), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}  // namespace dataset

RandomDataDataset::RandomDataDataset(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema,
                                     const std::vector<std::vector<char>> &columns_list,
                                     std::shared_ptr<DatasetCache> cache) {
  auto ds = std::make_shared<RandomNode>(total_rows, std::move(schema), VectorCharToString(columns_list), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
RandomDataDataset::RandomDataDataset(const int32_t &total_rows, const std::vector<char> &schema_path,
                                     const std::vector<std::vector<char>> &columns_list,
                                     std::shared_ptr<DatasetCache> cache) {
  auto ds =
    std::make_shared<RandomNode>(total_rows, CharToString(schema_path), VectorCharToString(columns_list), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TFRecordDataset::TFRecordDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &schema,
                                 const std::vector<std::vector<char>> &columns_list, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                                 std::shared_ptr<DatasetCache> cache) {
  auto ds = std::make_shared<TFRecordNode>(VectorCharToString(dataset_files), CharToString(schema),
                                           VectorCharToString(columns_list), num_samples, shuffle, num_shards, shard_id,
                                           shard_equal_rows, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
TFRecordDataset::TFRecordDataset(const std::vector<std::vector<char>> &dataset_files, std::shared_ptr<SchemaObj> schema,
                                 const std::vector<std::vector<char>> &columns_list, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                                 std::shared_ptr<DatasetCache> cache) {
  // std::cout << "SchemaObj.to_string2 " << schema->to_json() << std::endl;
  auto ds = std::make_shared<TFRecordNode>(VectorCharToString(dataset_files), schema, VectorCharToString(columns_list),
                                           num_samples, shuffle, num_shards, shard_id, shard_equal_rows, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#endif
}  // namespace dataset
}  // namespace mindspore
