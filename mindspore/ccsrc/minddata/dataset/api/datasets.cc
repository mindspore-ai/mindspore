/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/include/dataset/datasets.h"

#include <algorithm>
#include <fstream>
#include <unordered_set>
#include <utility>

#include <nlohmann/json.hpp>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/type_id.h"
#include "minddata/dataset/engine/consumers/pull_based_tree_consumer.h"
#include "minddata/dataset/engine/consumers/tree_consumer.h"
#include "minddata/dataset/engine/runtime_context.h"
#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/iterator.h"
#include "minddata/dataset/include/dataset/samplers.h"
#include "minddata/dataset/kernels/c_func_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/path.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/status.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/cache/dataset_cache_impl.h"
#include "minddata/dataset/include/dataset/text.h"
#endif

// Sampler headers (in alphabetical order)
#include "minddata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"

// IR dataset node
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
#include "minddata/dataset/engine/ir/datasetops/data_queue_node.h"
#include "minddata/dataset/engine/ir/datasetops/zip_node.h"
#endif

// IR leaf nodes
#include "minddata/dataset/engine/ir/datasetops/source/ag_news_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/album_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/amazon_review_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/caltech256_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/celeba_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cifar100_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cityscapes_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/clue_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/cmu_arctic_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/conll2000_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/csv_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/dbpedia_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/div2k_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/emnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/en_wik9_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/fake_image_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/fashion_mnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/flickr_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/gtzan_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/imdb_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/iwslt2016_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/iwslt2017_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/kitti_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/kmnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/lfw_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/libri_tts_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/lj_speech_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/lsun_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/minddata_node.h"
#endif
#include "minddata/dataset/engine/ir/datasetops/source/mnist_node.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/ir/datasetops/source/multi30k_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/omniglot_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/penn_treebank_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/photo_tour_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/places365_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/qmnist_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/random_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sbu_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/semeion_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/sogou_news_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/speech_commands_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/squad_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/stl10_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tedlium_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/text_file_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/tf_record_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/udpos_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/usps_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/wider_face_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/wiki_text_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yahoo_answers_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yelp_review_node.h"
#include "minddata/dataset/engine/ir/datasetops/source/yes_no_node.h"
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
  std::transform(v.begin(), v.end(), std::back_inserter(rv), [](const std::shared_ptr<Tensor> &t) -> MSTensor {
    return mindspore::MSTensor(std::make_shared<DETensor>(t));
  });
  return rv;
}

// Convert a std::function<TensorRow(TensorRow)> to std::function<MSTensorVec(MSTensor)> with this helper
TensorRow FuncPtrConverter(const std::function<MSTensorVec(MSTensorVec)> &func, const TensorRow &in_row) {
  return VecToRow(func(RowToVec(in_row)));
}

// Function to create the iterator, which will build and launch the execution tree.
std::shared_ptr<Iterator> Dataset::CreateIteratorCharIF(int32_t num_epochs) {
  std::shared_ptr<Iterator> iter;
  try {
    auto ds = shared_from_this();

    iter = std::make_shared<Iterator>();
    Status rc = iter->BuildAndLaunchTree(ds, num_epochs);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "CreateIterator failed." << rc;
      return nullptr;
    }
  } catch (const std::exception &err) {
    MS_LOG(ERROR) << "CreateIterator: Iterator exception caught: " << err.what();
    return nullptr;
  }

  return iter;
}

// Function to create the iterator, which will build and launch the execution tree.
std::shared_ptr<PullIterator> Dataset::CreatePullBasedIterator() {
  auto ds = shared_from_this();
  std::shared_ptr<PullIterator> iter = std::make_shared<PullIterator>();
  Status rc = iter->BuildAndLaunchTree(ds, 1);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "CreateIterator: Iterator exception caught: " << rc;
  }
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

  // Add DataQueueNode IR on top of dataset
  auto ds =
    std::make_shared<DataQueueNode>(shared_from_this()->IRNode(), CharToString(queue_name), CharToString(device_type),
                                    device_id, send_epoch_end, total_batches, create_data_info_queue);

  // Get ToDevice consumer
  auto consumer = std::make_unique<ToDevice>(num_epochs);
  ToDevice *consumer_ptr = consumer.get();
  if (consumer_ptr == nullptr) {
    MS_LOG(ERROR) << "ToDevice: Failed to get consumer.";
    return false;
  }
  rc = consumer->Init(ds);
  if (rc.IsError()) {
    MS_LOG(ERROR) << "ToDevice: Failed to init. Error status: " << rc;
    return false;
  }
  runtime_context->AssignConsumer(std::move(consumer));

  // Send data to device
  rc = consumer_ptr->Send();
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
  SaveToDisk *consumer_ptr = consumer.get();
  if (consumer_ptr == nullptr) {
    MS_LOG(ERROR) << "ToDevice: Failed to get consumer.";
    return false;
  }
  rc = consumer->Init(ds->IRNode());
  if (rc.IsError()) {
    MS_LOG(ERROR) << "CreateSaver failed." << rc;
    return false;
  }

  runtime_context->AssignConsumer(std::move(consumer));

  // Save data into file
  rc = consumer_ptr->Save();
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
  int64_t dataset_size = -1;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), -1);
  std::shared_ptr<DatasetSizeGetter> size_getter = std::make_shared<DatasetSizeGetter>();
  DatasetSizeGetter *consumer = size_getter.get();
  if (consumer == nullptr) {
    MS_LOG(ERROR) << "DatasetSizeGetter: Failed to get consumer.";
    return -1;
  }
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
  if (consumer == nullptr) {
    MS_LOG(ERROR) << "TreeGetters: Failed to get consumer.";
    return std::vector<mindspore::DataType>();
  }
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
  if (consumer == nullptr) {
    MS_LOG(ERROR) << "TreeGetters: Failed to get consumer.";
    return std::vector<std::vector<int64_t>>();
  }
  runtime_context->AssignConsumer(tree_getters_);
  RETURN_SECOND_IF_ERROR(consumer->Init(this->IRNode()), {});
  RETURN_SECOND_IF_ERROR(consumer->GetOutputShapes(&shapes), {});
  std::vector<std::vector<int64_t>> ret_shapes;
  std::transform(shapes.begin(), shapes.end(), std::back_inserter(ret_shapes),
                 [](const TensorShape &s) -> std::vector<int64_t> { return s.AsVector(); });
  return ret_shapes;
}

int64_t Dataset::GetNumClasses() {
  int64_t num_classes = -1;
  std::unique_ptr<NativeRuntimeContext> runtime_context = std::make_unique<NativeRuntimeContext>();
  RETURN_SECOND_IF_ERROR(runtime_context->Init(), -1);
  TreeGetters *consumer = tree_getters_.get();
  if (consumer == nullptr) {
    MS_LOG(ERROR) << "TreeGetters: Failed to get consumer.";
    return -1;
  }
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
  if (consumer == nullptr) {
    MS_LOG(ERROR) << "TreeGetters: Failed to get consumer.";
    return std::vector<std::vector<char>>();
  }
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
  if (consumer == nullptr) {
    MS_LOG(ERROR) << "TreeGetters: Failed to get consumer.";
    return std::vector<std::pair<std::vector<char>, std::vector<int32_t>>>();
  }
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
BatchDataset::BatchDataset(const std::shared_ptr<Dataset> &input, int32_t batch_size, bool drop_remainder) {
  // Default values
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<BatchNode>(input->IRNode(), batch_size, drop_remainder);
    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

#ifndef ENABLE_ANDROID
// Function to create a BucketBatchByLength dataset
BucketBatchByLengthDataset::BucketBatchByLengthDataset(
  const std::shared_ptr<Dataset> &input, const std::vector<std::vector<char>> &column_names,
  const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
  const std::function<MSTensorVec(MSTensorVec)> &element_length_function,
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
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<BucketBatchByLengthNode>(
      input->IRNode(), VectorCharToString(column_names), bucket_boundaries, bucket_batch_sizes, c_func,
      PadInfoCharToString(map), pad_to_bucket_boundary, drop_remainder);

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

ConcatDataset::ConcatDataset(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  std::vector<std::shared_ptr<DatasetNode>> all_datasets;
  (void)std::transform(datasets.begin(), datasets.end(), std::back_inserter(all_datasets),
                       [](const std::shared_ptr<Dataset> &dataset) -> std::shared_ptr<DatasetNode> {
                         return (dataset != nullptr) ? dataset->IRNode() : nullptr;
                       });
  auto ds = std::make_shared<ConcatNode>(all_datasets);

  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FilterDataset::FilterDataset(const std::shared_ptr<Dataset> &input,
                             const std::function<MSTensorVec(MSTensorVec)> &predicate,
                             const std::vector<std::vector<char>> &input_columns) {
  std::shared_ptr<TensorOp> c_func = nullptr;
  if (predicate) {
    c_func = std::make_shared<CFuncOp>(std::bind(FuncPtrConverter, predicate, std::placeholders::_1));
  }
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<FilterNode>(input->IRNode(), c_func, VectorCharToString(input_columns));
    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}
#endif

MapDataset::MapDataset(const std::shared_ptr<Dataset> &input,
                       const std::vector<std::shared_ptr<TensorOperation>> &operations,
                       const std::vector<std::vector<char>> &input_columns,
                       const std::vector<std::vector<char>> &output_columns, const std::shared_ptr<DatasetCache> &cache,
                       const std::vector<std::shared_ptr<DSCallback>> &callbacks) {
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<MapNode>(input->IRNode(), operations, VectorCharToString(input_columns),
                                        VectorCharToString(output_columns), cache, callbacks);

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

ProjectDataset::ProjectDataset(const std::shared_ptr<Dataset> &input, const std::vector<std::vector<char>> &columns) {
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<ProjectNode>(input->IRNode(), VectorCharToString(columns));

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

#ifndef ENABLE_ANDROID
RenameDataset::RenameDataset(const std::shared_ptr<Dataset> &input, const std::vector<std::vector<char>> &input_columns,
                             const std::vector<std::vector<char>> &output_columns) {
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<RenameNode>(input->IRNode(), VectorCharToString(input_columns),
                                           VectorCharToString(output_columns));

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}
#endif

RepeatDataset::RepeatDataset(const std::shared_ptr<Dataset> &input, int32_t count) {
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<RepeatNode>(input->IRNode(), count);

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

ShuffleDataset::ShuffleDataset(const std::shared_ptr<Dataset> &input, int32_t buffer_size) {
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    // Pass in reshuffle_each_epoch with true
    auto ds = std::make_shared<ShuffleNode>(input->IRNode(), buffer_size, true);

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

#ifndef ENABLE_ANDROID
SkipDataset::SkipDataset(const std::shared_ptr<Dataset> &input, int32_t count) {
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<SkipNode>(input->IRNode(), count);

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

TakeDataset::TakeDataset(const std::shared_ptr<Dataset> &input, int32_t count) {
  if (input == nullptr) {
    ir_node_ = nullptr;
  } else {
    auto ds = std::make_shared<TakeNode>(input->IRNode(), count);

    ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
  }
}

ZipDataset::ZipDataset(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  std::vector<std::shared_ptr<DatasetNode>> all_datasets;
  (void)std::transform(datasets.begin(), datasets.end(), std::back_inserter(all_datasets),
                       [](const std::shared_ptr<Dataset> &dataset) -> std::shared_ptr<DatasetNode> {
                         return (dataset != nullptr) ? dataset->IRNode() : nullptr;
                       });
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
  if (bv_consumer == nullptr) {
    MS_LOG(ERROR) << "BuildVocabConsumer: Failed to get bv_consumer.";
    return nullptr;
  }
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
  if (bv_consumer == nullptr) {
    MS_LOG(ERROR) << "BuildVocabConsumer: Failed to get bv_consumer.";
    return nullptr;
  }
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
  if (data_ != nullptr && !data_->schema_file_.empty()) {
    std::string real_path;
    RETURN_IF_NOT_OK(Path::RealPath(data_->schema_file_, real_path));
    Path schema_file(real_path);
    CHECK_FAIL_RETURN_UNEXPECTED(schema_file.Exists(),
                                 "The file " + data_->schema_file_ + " does not exist or permission denied!");

    nlohmann::json js;
    try {
      std::ifstream in(real_path);
      in >> js;
      CHECK_FAIL_RETURN_UNEXPECTED(js.find("columns") != js.end(),
                                   "\"columns\" node is required in the schema json file.");
    } catch (const std::exception &err) {
      std::string err_msg = "Schema file failed to load: ";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
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

Status SchemaObj::schema_to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json json_file;
  json_file["columns"] = data_->columns_;
  std::string str_dataset_type_(data_->dataset_type_);
  if (!str_dataset_type_.empty()) {
    json_file["datasetType"] = str_dataset_type_;
  }

  if (data_->num_rows_ > 0) {
    json_file["numRows"] = data_->num_rows_;
  }
  *out_json = json_file;
  return Status::OK();
}

std::vector<char> SchemaObj::to_json_char() {
  nlohmann::json json_file;
  this->schema_to_json(&json_file);
  return StringToChar(json_file.dump(2));
}

void SchemaObj::set_dataset_type(const std::string &dataset_type) { data_->dataset_type_ = dataset_type; }

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
        LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Column's name is missing");
      }
      name = *key_name;

      auto key_type = column.find("type");
      if (key_type == column.end()) {
        LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Column's type is missing");
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
        LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Column's type is missing");
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
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR("columns must be dict or list, columns contain name, type, shape(optional).");
  }
  return Status::OK();
}

Status SchemaObj::from_json(nlohmann::json json_obj) {
  for (const auto &it_child : json_obj.items()) {
    if (it_child.key() == "datasetType") {
      std::string str_dataset_type_ = it_child.value();
      data_->dataset_type_ = str_dataset_type_;
    } else if (it_child.key() == "numRows") {
      data_->num_rows_ = it_child.value();
    } else if (it_child.key() == "columns") {
      RETURN_IF_NOT_OK(parse_column(it_child.value()));
    } else {
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Unknown field " + it_child.key());
    }
  }
  if (data_->columns_.empty()) {
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Columns are missing.");
  }
  if (data_->num_rows_ < 0) {
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR("numRows must be greater than or equal to 0");
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
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
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
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// OTHER FUNCTIONS

#ifndef ENABLE_ANDROID

std::shared_ptr<DatasetCache> CreateDatasetCacheCharIF(session_id_type id, uint64_t mem_sz, bool spill,
                                                       const std::optional<std::vector<char>> &hostname,
                                                       const std::optional<int32_t> &port,
                                                       const std::optional<int32_t> &num_connections,
                                                       const std::optional<int32_t> &prefetch_sz) {
  auto cache = std::make_shared<DatasetCacheImpl>(id, mem_sz, spill, hostname, port, num_connections, prefetch_sz);
  return cache;
}

AGNewsDataset::AGNewsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                             ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<AGNewsNode>(CharToString(dataset_dir), num_samples, shuffle, CharToString(usage),
                                         num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
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
                           const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<AlbumNode>(CharToString(dataset_dir), CharToString(data_schema),
                                        VectorCharToString(column_names), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#ifndef ENABLE_ANDROID
AmazonReviewDataset::AmazonReviewDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                         int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                         const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<AmazonReviewNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                               num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Caltech256Dataset::Caltech256Dataset(const std::vector<char> &dataset_dir, bool decode,
                                     const std::shared_ptr<Sampler> &sampler,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<Caltech256Node>(CharToString(dataset_dir), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Caltech256Dataset::Caltech256Dataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<Caltech256Node>(CharToString(dataset_dir), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Caltech256Dataset::Caltech256Dataset(const std::vector<char> &dataset_dir, bool decode,
                                     const std::reference_wrapper<Sampler> &sampler,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<Caltech256Node>(CharToString(dataset_dir), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

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
                             const std::reference_wrapper<Sampler> &sampler, bool decode,
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
                               const std::reference_wrapper<Sampler> &sampler,
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
                                 const std::reference_wrapper<Sampler> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<Cifar100Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CityscapesDataset::CityscapesDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                     const std::vector<char> &quality_mode, const std::vector<char> &task, bool decode,
                                     const std::shared_ptr<Sampler> &sampler,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CityscapesNode>(CharToString(dataset_dir), CharToString(usage), CharToString(quality_mode),
                                             CharToString(task), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CityscapesDataset::CityscapesDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                     const std::vector<char> &quality_mode, const std::vector<char> &task, bool decode,
                                     const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CityscapesNode>(CharToString(dataset_dir), CharToString(usage), CharToString(quality_mode),
                                             CharToString(task), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CityscapesDataset::CityscapesDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                     const std::vector<char> &quality_mode, const std::vector<char> &task, bool decode,
                                     const std::reference_wrapper<Sampler> &sampler,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<CityscapesNode>(CharToString(dataset_dir), CharToString(usage), CharToString(quality_mode),
                                             CharToString(task), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CLUEDataset::CLUEDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &task,
                         const std::vector<char> &usage, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                         int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CLUENode>(VectorCharToString(dataset_files), CharToString(task), CharToString(usage),
                                       num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CMUArcticDataset::CMUArcticDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                                   const std::shared_ptr<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CMUArcticNode>(CharToString(dataset_dir), CharToString(name), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CMUArcticDataset::CMUArcticDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                                   const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CMUArcticNode>(CharToString(dataset_dir), CharToString(name), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CMUArcticDataset::CMUArcticDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                                   const std::reference_wrapper<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<CMUArcticNode>(CharToString(dataset_dir), CharToString(name), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CocoDataset::CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                         const std::vector<char> &task, const bool &decode, const std::shared_ptr<Sampler> &sampler,
                         const std::shared_ptr<DatasetCache> &cache, const bool &extra_metadata) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CocoNode>(CharToString(dataset_dir), CharToString(annotation_file), CharToString(task),
                                       decode, sampler_obj, cache, extra_metadata);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CocoDataset::CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                         const std::vector<char> &task, const bool &decode, const Sampler *sampler,
                         const std::shared_ptr<DatasetCache> &cache, const bool &extra_metadata) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<CocoNode>(CharToString(dataset_dir), CharToString(annotation_file), CharToString(task),
                                       decode, sampler_obj, cache, extra_metadata);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CocoDataset::CocoDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                         const std::vector<char> &task, const bool &decode,
                         const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache,
                         const bool &extra_metadata) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<CocoNode>(CharToString(dataset_dir), CharToString(annotation_file), CharToString(task),
                                       decode, sampler_obj, cache, extra_metadata);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

CoNLL2000Dataset::CoNLL2000Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                   int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<CoNLL2000Node>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                            num_shards, shard_id, cache);
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

DBpediaDataset::DBpediaDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                               int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                               const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<DBpediaNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                          num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DBpediaNode>(ds);
}

DIV2KDataset::DIV2KDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::vector<char> &downgrade, int32_t scale, bool decode,
                           const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<DIV2KNode>(CharToString(dataset_dir), CharToString(usage), CharToString(downgrade), scale,
                                        decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

DIV2KDataset::DIV2KDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::vector<char> &downgrade, int32_t scale, bool decode, const Sampler *sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<DIV2KNode>(CharToString(dataset_dir), CharToString(usage), CharToString(downgrade), scale,
                                        decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

DIV2KDataset::DIV2KDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::vector<char> &downgrade, int32_t scale, bool decode,
                           const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<DIV2KNode>(CharToString(dataset_dir), CharToString(usage), CharToString(downgrade), scale,
                                        decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

EMnistDataset::EMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                             const std::vector<char> &usage, const std::shared_ptr<Sampler> &sampler,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<EMnistNode>(CharToString(dataset_dir), CharToString(name), CharToString(usage),
                                         sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

EMnistDataset::EMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                             const std::vector<char> &usage, const Sampler *sampler,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<EMnistNode>(CharToString(dataset_dir), CharToString(name), CharToString(usage),
                                         sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

EMnistDataset::EMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                             const std::vector<char> &usage, const std::reference_wrapper<Sampler> &sampler,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<EMnistNode>(CharToString(dataset_dir), CharToString(name), CharToString(usage),
                                         sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

EnWik9Dataset::EnWik9Dataset(const std::vector<char> &dataset_dir, int64_t num_samples, ShuffleMode shuffle,
                             int32_t num_shards, int32_t shard_id, const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<EnWik9Node>(CharToString(dataset_dir), num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FakeImageDataset::FakeImageDataset(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes,
                                   int32_t base_seed, const std::shared_ptr<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<FakeImageNode>(num_images, image_size, num_classes, base_seed, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FakeImageDataset::FakeImageDataset(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes,
                                   int32_t base_seed, const Sampler *sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<FakeImageNode>(num_images, image_size, num_classes, base_seed, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FakeImageDataset::FakeImageDataset(int32_t num_images, const std::vector<int32_t> &image_size, int32_t num_classes,
                                   int32_t base_seed, const std::reference_wrapper<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<FakeImageNode>(num_images, image_size, num_classes, base_seed, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FashionMnistDataset::FashionMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                         const std::shared_ptr<Sampler> &sampler,
                                         const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<FashionMnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FashionMnistDataset::FashionMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                         const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<FashionMnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FashionMnistDataset::FashionMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                         const std::reference_wrapper<Sampler> &sampler,
                                         const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<FashionMnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FlickrDataset::FlickrDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                             bool decode, const std::shared_ptr<Sampler> &sampler,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds =
    std::make_shared<FlickrNode>(CharToString(dataset_dir), CharToString(annotation_file), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FlickrDataset::FlickrDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                             bool decode, const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds =
    std::make_shared<FlickrNode>(CharToString(dataset_dir), CharToString(annotation_file), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

FlickrDataset::FlickrDataset(const std::vector<char> &dataset_dir, const std::vector<char> &annotation_file,
                             bool decode, const std::reference_wrapper<Sampler> &sampler,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds =
    std::make_shared<FlickrNode>(CharToString(dataset_dir), CharToString(annotation_file), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

GTZANDataset::GTZANDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<GTZANNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

GTZANDataset::GTZANDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<GTZANNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

GTZANDataset::GTZANDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<GTZANNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
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
                                       const std::reference_wrapper<Sampler> &sampler,
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

IMDBDataset::IMDBDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                         const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of IMDBDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<IMDBNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

IMDBDataset::IMDBDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                         const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of IMDBDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<IMDBNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

IMDBDataset::IMDBDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                         const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of IMDBDataset.
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<IMDBNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

IWSLT2016Dataset::IWSLT2016Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                   const std::vector<std::vector<char>> &language_pair,
                                   const std::vector<char> &valid_set, const std::vector<char> &test_set,
                                   int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<IWSLT2016Node>(CharToString(dataset_dir), CharToString(usage),
                                            VectorCharToString(language_pair), CharToString(valid_set),
                                            CharToString(test_set), num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

IWSLT2017Dataset::IWSLT2017Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                   const std::vector<std::vector<char>> &language_pair, int64_t num_samples,
                                   ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto ds =
    std::make_shared<IWSLT2017Node>(CharToString(dataset_dir), CharToString(usage), VectorCharToString(language_pair),
                                    num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

KITTIDataset::KITTIDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                           const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<KITTINode>(CharToString(dataset_dir), CharToString(usage), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

KITTIDataset::KITTIDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                           const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<KITTINode>(CharToString(dataset_dir), CharToString(usage), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

KITTIDataset::KITTIDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                           const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<KITTINode>(CharToString(dataset_dir), CharToString(usage), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

KMnistDataset::KMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                             const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<KMnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

KMnistDataset::KMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                             const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<KMnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

KMnistDataset::KMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                             const std::reference_wrapper<Sampler> &sampler,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<KMnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LFWDataset::LFWDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::vector<char> &image_set, bool decode,
                       const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of LFWDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LFWNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      CharToString(image_set), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LFWDataset::LFWDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::vector<char> &image_set, bool decode,
                       const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of LFWDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LFWNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      CharToString(image_set), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LFWDataset::LFWDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::vector<char> &image_set, bool decode,
                       const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of LFWDataset.
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<LFWNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      CharToString(image_set), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LibriTTSDataset::LibriTTSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LibriTTSNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LibriTTSDataset::LibriTTSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LibriTTSNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LibriTTSDataset::LibriTTSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 const std::reference_wrapper<Sampler> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<LibriTTSNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LJSpeechDataset::LJSpeechDataset(const std::vector<char> &dataset_dir, const std::shared_ptr<Sampler> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LJSpeechNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LJSpeechDataset::LJSpeechDataset(const std::vector<char> &dataset_dir, const Sampler *sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LJSpeechNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LJSpeechDataset::LJSpeechDataset(const std::vector<char> &dataset_dir, const std::reference_wrapper<Sampler> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<LJSpeechNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LSUNDataset::LSUNDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                         const std::vector<std::vector<char>> &classes, bool decode,
                         const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of LSUNDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LSUNNode>(CharToString(dataset_dir), CharToString(usage), VectorCharToString(classes),
                                       decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LSUNDataset::LSUNDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                         const std::vector<std::vector<char>> &classes, bool decode, const Sampler *sampler,
                         const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of LSUNDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<LSUNNode>(CharToString(dataset_dir), CharToString(usage), VectorCharToString(classes),
                                       decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

LSUNDataset::LSUNDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                         const std::vector<std::vector<char>> &classes, bool decode,
                         const std::reference_wrapper<Sampler> sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of LSUNDataset.
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<LSUNNode>(CharToString(dataset_dir), CharToString(usage), VectorCharToString(classes),
                                       decode, sampler_obj, cache);
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
                                 const std::reference_wrapper<Sampler> &sampler,
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
                                 int64_t num_padded, ShuffleMode shuffle_mode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }
  auto ds = std::make_shared<MindDataNode>(CharToString(dataset_file), VectorCharToString(columns_list), sampler_obj,
                                           sample, num_padded, shuffle_mode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

MindDataDataset::MindDataDataset(const std::vector<char> &dataset_file,
                                 const std::vector<std::vector<char>> &columns_list, const Sampler *sampler,
                                 const nlohmann::json *padded_sample, int64_t num_padded, ShuffleMode shuffle_mode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }
  auto ds = std::make_shared<MindDataNode>(CharToString(dataset_file), VectorCharToString(columns_list), sampler_obj,
                                           sample, num_padded, shuffle_mode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

MindDataDataset::MindDataDataset(const std::vector<char> &dataset_file,
                                 const std::vector<std::vector<char>> &columns_list,
                                 const std::reference_wrapper<Sampler> &sampler, const nlohmann::json *padded_sample,
                                 int64_t num_padded, ShuffleMode shuffle_mode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }

  auto ds = std::make_shared<MindDataNode>(CharToString(dataset_file), VectorCharToString(columns_list), sampler_obj,
                                           sample, num_padded, shuffle_mode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

MindDataDataset::MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                                 const std::vector<std::vector<char>> &columns_list,
                                 const std::shared_ptr<Sampler> &sampler, const nlohmann::json *padded_sample,
                                 int64_t num_padded, ShuffleMode shuffle_mode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }

  auto ds = std::make_shared<MindDataNode>(VectorCharToString(dataset_files), VectorCharToString(columns_list),
                                           sampler_obj, sample, num_padded, shuffle_mode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

MindDataDataset::MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                                 const std::vector<std::vector<char>> &columns_list, const Sampler *sampler,
                                 const nlohmann::json *padded_sample, int64_t num_padded, ShuffleMode shuffle_mode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }

  auto ds = std::make_shared<MindDataNode>(VectorCharToString(dataset_files), VectorCharToString(columns_list),
                                           sampler_obj, sample, num_padded, shuffle_mode, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

MindDataDataset::MindDataDataset(const std::vector<std::vector<char>> &dataset_files,
                                 const std::vector<std::vector<char>> &columns_list,
                                 const std::reference_wrapper<Sampler> &sampler, const nlohmann::json *padded_sample,
                                 int64_t num_padded, ShuffleMode shuffle_mode,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  nlohmann::json sample = nullptr;
  if (padded_sample) {
    sample = *padded_sample;
  }
  auto ds = std::make_shared<MindDataNode>(VectorCharToString(dataset_files), VectorCharToString(columns_list),
                                           sampler_obj, sample, num_padded, shuffle_mode, cache);
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
                           const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<MnistNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

#ifndef ENABLE_ANDROID
Multi30kDataset::Multi30kDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 const std::vector<std::vector<char>> &language_pair, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds =
    std::make_shared<Multi30kNode>(CharToString(dataset_dir), CharToString(usage), VectorCharToString(language_pair),
                                   num_samples, shuffle, num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<Multi30kNode>(ds);
}

OmniglotDataset::OmniglotDataset(const std::vector<char> &dataset_dir, bool background, bool decode,
                                 const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of OmniglotDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<OmniglotNode>(CharToString(dataset_dir), background, decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

OmniglotDataset::OmniglotDataset(const std::vector<char> &dataset_dir, bool background, bool decode,
                                 const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of OmniglotDataset.
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<OmniglotNode>(CharToString(dataset_dir), background, decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

OmniglotDataset::OmniglotDataset(const std::vector<char> &dataset_dir, bool background, bool decode,
                                 const std::reference_wrapper<Sampler> &sampler,
                                 const std::shared_ptr<DatasetCache> &cache) {
  // Create logical representation of OmniglotDataset.
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<OmniglotNode>(CharToString(dataset_dir), background, decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

PennTreebankDataset::PennTreebankDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                         int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                         const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<PennTreebankNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                               num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

PhotoTourDataset::PhotoTourDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                                   const std::vector<char> &usage, const std::shared_ptr<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<PhotoTourNode>(CharToString(dataset_dir), CharToString(name), CharToString(usage),
                                            sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

PhotoTourDataset::PhotoTourDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                                   const std::vector<char> &usage, const Sampler *sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<PhotoTourNode>(CharToString(dataset_dir), CharToString(name), CharToString(usage),
                                            sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

PhotoTourDataset::PhotoTourDataset(const std::vector<char> &dataset_dir, const std::vector<char> &name,
                                   const std::vector<char> &usage, const std::reference_wrapper<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<PhotoTourNode>(CharToString(dataset_dir), CharToString(name), CharToString(usage),
                                            sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Places365Dataset::Places365Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                   const bool small, const bool decode, const std::shared_ptr<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds =
    std::make_shared<Places365Node>(CharToString(dataset_dir), CharToString(usage), small, decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Places365Dataset::Places365Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                   const bool small, const bool decode, const Sampler *sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds =
    std::make_shared<Places365Node>(CharToString(dataset_dir), CharToString(usage), small, decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

Places365Dataset::Places365Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                   const bool small, const bool decode, const std::reference_wrapper<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds =
    std::make_shared<Places365Node>(CharToString(dataset_dir), CharToString(usage), small, decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

QMnistDataset::QMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool compat,
                             const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<QMnistNode>(CharToString(dataset_dir), CharToString(usage), compat, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

QMnistDataset::QMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool compat,
                             const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<QMnistNode>(CharToString(dataset_dir), CharToString(usage), compat, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

QMnistDataset::QMnistDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool compat,
                             const std::reference_wrapper<Sampler> &sampler,
                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<QMnistNode>(CharToString(dataset_dir), CharToString(usage), compat, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SemeionDataset::SemeionDataset(const std::vector<char> &dataset_dir, const std::shared_ptr<Sampler> &sampler,
                               const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<SemeionNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SemeionDataset::SemeionDataset(const std::vector<char> &dataset_dir, const Sampler *sampler,
                               const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<SemeionNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SemeionDataset::SemeionDataset(const std::vector<char> &dataset_dir, const std::reference_wrapper<Sampler> &sampler,
                               const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<SemeionNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SQuADDataset::SQuADDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                           ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<SQuADNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                        num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TedliumDataset::TedliumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &release,
                               const std::vector<char> &usage, const std::vector<char> &extensions,
                               const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<TedliumNode>(CharToString(dataset_dir), CharToString(release), CharToString(usage),
                                          CharToString(extensions), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TedliumDataset::TedliumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &release,
                               const std::vector<char> &usage, const std::vector<char> &extensions,
                               const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<TedliumNode>(CharToString(dataset_dir), CharToString(release), CharToString(usage),
                                          CharToString(extensions), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TedliumDataset::TedliumDataset(const std::vector<char> &dataset_dir, const std::vector<char> &release,
                               const std::vector<char> &usage, const std::vector<char> &extensions,
                               const std::reference_wrapper<Sampler> &sampler,
                               const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<TedliumNode>(CharToString(dataset_dir), CharToString(release), CharToString(usage),
                                          CharToString(extensions), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

STL10Dataset::STL10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<STL10Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

STL10Dataset::STL10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, const Sampler *sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<STL10Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

STL10Dataset::STL10Dataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                           const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<STL10Node>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TextFileDataset::TextFileDataset(const std::vector<std::vector<char>> &dataset_files, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<TextFileNode>(VectorCharToString(dataset_files), num_samples, shuffle, num_shards,
                                           shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

USPSDataset::USPSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                         ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                         const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<USPSNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle, num_shards,
                                       shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

VOCDataset::VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                       bool decode, const std::shared_ptr<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache,
                       bool extra_metadata) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<VOCNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      MapCharToString(class_indexing), decode, sampler_obj, cache, extra_metadata);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

VOCDataset::VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                       bool decode, const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache,
                       bool extra_metadata) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<VOCNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      MapCharToString(class_indexing), decode, sampler_obj, cache, extra_metadata);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

VOCDataset::VOCDataset(const std::vector<char> &dataset_dir, const std::vector<char> &task,
                       const std::vector<char> &usage, const std::map<std::vector<char>, int32_t> &class_indexing,
                       bool decode, const std::reference_wrapper<Sampler> &sampler,
                       const std::shared_ptr<DatasetCache> &cache, bool extra_metadata) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<VOCNode>(CharToString(dataset_dir), CharToString(task), CharToString(usage),
                                      MapCharToString(class_indexing), decode, sampler_obj, cache, extra_metadata);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

WikiTextDataset::WikiTextDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                 int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<WikiTextNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                           num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

RandomDataDataset::RandomDataDataset(const int32_t &total_rows, std::shared_ptr<SchemaObj> schema,
                                     const std::vector<std::vector<char>> &columns_list,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<RandomNode>(total_rows, std::move(schema), VectorCharToString(columns_list), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

RandomDataDataset::RandomDataDataset(const int32_t &total_rows, const std::vector<char> &schema_path,
                                     const std::vector<std::vector<char>> &columns_list,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto ds =
    std::make_shared<RandomNode>(total_rows, CharToString(schema_path), VectorCharToString(columns_list), cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SBUDataset::SBUDataset(const std::vector<char> &dataset_dir, bool decode, const std::shared_ptr<Sampler> &sampler,
                       const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<SBUNode>(CharToString(dataset_dir), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SBUDataset::SBUDataset(const std::vector<char> &dataset_dir, bool decode, const Sampler *sampler,
                       const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<SBUNode>(CharToString(dataset_dir), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SBUDataset::SBUDataset(const std::vector<char> &dataset_dir, bool decode,
                       const std::reference_wrapper<Sampler> &sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<SBUNode>(CharToString(dataset_dir), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SogouNewsDataset::SogouNewsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                   int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<SogouNewsNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                            num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SpeechCommandsDataset::SpeechCommandsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                             const std::shared_ptr<Sampler> &sampler,
                                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<SpeechCommandsNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SpeechCommandsDataset::SpeechCommandsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                             const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<SpeechCommandsNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

SpeechCommandsDataset::SpeechCommandsDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                             const std::reference_wrapper<Sampler> &sampler,
                                             const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<SpeechCommandsNode>(CharToString(dataset_dir), CharToString(usage), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TFRecordDataset::TFRecordDataset(const std::vector<std::vector<char>> &dataset_files, const std::vector<char> &schema,
                                 const std::vector<std::vector<char>> &columns_list, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<TFRecordNode>(VectorCharToString(dataset_files), CharToString(schema),
                                           VectorCharToString(columns_list), num_samples, shuffle, num_shards, shard_id,
                                           shard_equal_rows, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

TFRecordDataset::TFRecordDataset(const std::vector<std::vector<char>> &dataset_files,
                                 const std::shared_ptr<SchemaObj> &schema,
                                 const std::vector<std::vector<char>> &columns_list, int64_t num_samples,
                                 ShuffleMode shuffle, int32_t num_shards, int32_t shard_id, bool shard_equal_rows,
                                 const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<TFRecordNode>(VectorCharToString(dataset_files), schema, VectorCharToString(columns_list),
                                           num_samples, shuffle, num_shards, shard_id, shard_equal_rows, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

UDPOSDataset::UDPOSDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, int64_t num_samples,
                           ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<UDPOSNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                        num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<UDPOSNode>(ds);
}

WIDERFaceDataset::WIDERFaceDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                                   const std::shared_ptr<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<WIDERFaceNode>(CharToString(dataset_dir), CharToString(usage), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

WIDERFaceDataset::WIDERFaceDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                                   const Sampler *sampler, const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<WIDERFaceNode>(CharToString(dataset_dir), CharToString(usage), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

WIDERFaceDataset::WIDERFaceDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage, bool decode,
                                   const std::reference_wrapper<Sampler> &sampler,
                                   const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<WIDERFaceNode>(CharToString(dataset_dir), CharToString(usage), decode, sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

YahooAnswersDataset::YahooAnswersDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                         int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                         const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<YahooAnswersNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                               num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<YahooAnswersNode>(ds);
}

YelpReviewDataset::YelpReviewDataset(const std::vector<char> &dataset_dir, const std::vector<char> &usage,
                                     int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id,
                                     const std::shared_ptr<DatasetCache> &cache) {
  auto ds = std::make_shared<YelpReviewNode>(CharToString(dataset_dir), CharToString(usage), num_samples, shuffle,
                                             num_shards, shard_id, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

YesNoDataset::YesNoDataset(const std::vector<char> &dataset_dir, const std::shared_ptr<Sampler> &sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<YesNoNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

YesNoDataset::YesNoDataset(const std::vector<char> &dataset_dir, const Sampler *sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler ? sampler->Parse() : nullptr;
  auto ds = std::make_shared<YesNoNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}

YesNoDataset::YesNoDataset(const std::vector<char> &dataset_dir, const std::reference_wrapper<Sampler> &sampler,
                           const std::shared_ptr<DatasetCache> &cache) {
  auto sampler_obj = sampler.get().Parse();
  auto ds = std::make_shared<YesNoNode>(CharToString(dataset_dir), sampler_obj, cache);
  ir_node_ = std::static_pointer_cast<DatasetNode>(ds);
}
#endif
}  // namespace dataset
}  // namespace mindspore
