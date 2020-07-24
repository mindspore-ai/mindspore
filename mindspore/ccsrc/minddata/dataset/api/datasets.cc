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

#include <fstream>

#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/samplers.h"
#include "minddata/dataset/include/transforms.h"
#include "minddata/dataset/engine/dataset_iterator.h"
// Source dataset headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
// Dataset operator headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/batch_op.h"
#include "minddata/dataset/engine/datasetops/map_op.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/datasetops/project_op.h"
#include "minddata/dataset/engine/datasetops/zip_op.h"
#include "minddata/dataset/engine/datasetops/rename_op.h"
// Sampler headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
namespace api {

#define RETURN_EMPTY_IF_ERROR(_s) \
  do {                            \
    Status __rc = (_s);           \
    if (__rc.IsError()) {         \
      MS_LOG(ERROR) << __rc;      \
      return {};                  \
    }                             \
  } while (false)

// Function to create the iterator, which will build and launch the execution tree.
std::shared_ptr<Iterator> Dataset::CreateIterator() {
  std::shared_ptr<Iterator> iter;
  try {
    iter = std::make_shared<Iterator>();
    Status rc = iter->BuildAndLaunchTree(shared_from_this());
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

// Constructor
Dataset::Dataset() {
  // Fetch some default value from config manager
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  num_workers_ = cfg->num_parallel_workers();
  rows_per_buffer_ = cfg->rows_per_buffer();
  connector_que_size_ = cfg->op_connector_size();
}

// FUNCTIONS TO CREATE DATASETS FOR LEAF-NODE DATASETS
// (In alphabetical order)

// Function to create a Cifar10Dataset.
std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir, int32_t num_samples,
                                        std::shared_ptr<SamplerObj> sampler) {
  auto ds = std::make_shared<Cifar10Dataset>(dataset_dir, num_samples, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a ImageFolderDataset.
std::shared_ptr<ImageFolderDataset> ImageFolder(std::string dataset_dir, bool decode,
                                                std::shared_ptr<SamplerObj> sampler, std::set<std::string> extensions,
                                                std::map<std::string, int32_t> class_indexing) {
  // This arg is exist in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderDataset.
  auto ds = std::make_shared<ImageFolderDataset>(dataset_dir, decode, sampler, recursive, extensions, class_indexing);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a MnistDataset.
std::shared_ptr<MnistDataset> Mnist(std::string dataset_dir, std::shared_ptr<SamplerObj> sampler) {
  auto ds = std::make_shared<MnistDataset>(dataset_dir, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// FUNCTIONS TO CREATE DATASETS FOR DATASET OPS
// (In alphabetical order)

// Function to create a Batch dataset
std::shared_ptr<BatchDataset> Dataset::Batch(int32_t batch_size, bool drop_remainder) {
  // Default values
  std::vector<std::string> cols_to_map = {};
  std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map;
  bool pad = false;
  auto ds = std::make_shared<BatchDataset>(batch_size, drop_remainder, pad, cols_to_map, pad_map);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create a Map dataset.
std::shared_ptr<MapDataset> Dataset::Map(std::vector<std::shared_ptr<TensorOperation>> operations,
                                         std::vector<std::string> input_columns,
                                         std::vector<std::string> output_columns,
                                         const std::vector<std::string> &project_columns) {
  auto ds = std::make_shared<MapDataset>(operations, input_columns, output_columns, project_columns);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create a ProjectDataset.
std::shared_ptr<ProjectDataset> Dataset::Project(const std::vector<std::string> &columns) {
  auto ds = std::make_shared<ProjectDataset>(columns);
  // Call derived class validation method.
  if (!ds->ValidateParams()) {
    return nullptr;
  }

  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create a RenameDataset.
std::shared_ptr<RenameDataset> Dataset::Rename(const std::vector<std::string> &input_columns,
                                               const std::vector<std::string> &output_columns) {
  auto ds = std::make_shared<RenameDataset>(input_columns, output_columns);
  // Call derived class validation method.
  if (!ds->ValidateParams()) {
    return nullptr;
  }

  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create Repeat dataset.
std::shared_ptr<Dataset> Dataset::Repeat(int32_t count) {
  // Workaround for repeat == 1, do not inject repeat.
  if (count == 1) {
    return shared_from_this();
  }

  auto ds = std::make_shared<RepeatDataset>(count);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create a ShuffleOp
std::shared_ptr<ShuffleDataset> Dataset::Shuffle(int32_t shuffle_size) {
  // Pass in reshuffle_each_epoch with true
  auto ds = std::make_shared<ShuffleDataset>(shuffle_size, true);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create a SkipDataset.
std::shared_ptr<SkipDataset> Dataset::Skip(int32_t count) {
  auto ds = std::make_shared<SkipDataset>(count);

  // Call derived class validation method.
  if (!ds->ValidateParams()) {
    return nullptr;
  }

  ds->children.push_back(shared_from_this());

  return ds;
}

// Function to create a Zip dataset
std::shared_ptr<ZipDataset> Dataset::Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  // Default values
  auto ds = std::make_shared<ZipDataset>();

  if (!ds->ValidateParams()) {
    return nullptr;
  }
  for (auto dataset : datasets) {
    ds->children.push_back(dataset);
  }

  return ds;
}

// OTHER FUNCTIONS
// (In alphabetical order)

// Helper function to create default RandomSampler.
std::shared_ptr<SamplerObj> CreateDefaultSampler() {
  const int32_t num_samples = 0;  // 0 means to sample all ids.
  bool replacement = false;
  return std::make_shared<RandomSamplerObj>(replacement, num_samples);
}

/* ####################################### Derived Dataset classes ################################# */

// DERIVED DATASET CLASSES LEAF-NODE DATASETS
// (In alphabetical order)

// Constructor for Cifar10Dataset
Cifar10Dataset::Cifar10Dataset(const std::string &dataset_dir, int32_t num_samples, std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir), num_samples_(num_samples), sampler_(sampler) {}

bool Cifar10Dataset::ValidateParams() {
  if (dataset_dir_.empty()) {
    MS_LOG(ERROR) << "No dataset path is specified.";
    return false;
  }
  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "Number of samples cannot be negative";
    return false;
  }
  return true;
}

// Function to build CifarOp
std::vector<std::shared_ptr<DatasetOp>> Cifar10Dataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // If user does not specify Sampler, create a default sampler based on the shuffle variable.
  if (sampler_ == nullptr) {
    sampler_ = CreateDefaultSampler();
  }

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  node_ops.push_back(std::make_shared<CifarOp>(CifarOp::CifarType::kCifar10, num_workers_, rows_per_buffer_,
                                               dataset_dir_, connector_que_size_, std::move(schema),
                                               std::move(sampler_->Build())));
  return node_ops;
}

ImageFolderDataset::ImageFolderDataset(std::string dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler,
                                       bool recursive, std::set<std::string> extensions,
                                       std::map<std::string, int32_t> class_indexing)
    : dataset_dir_(dataset_dir),
      decode_(decode),
      sampler_(sampler),
      recursive_(recursive),
      class_indexing_(class_indexing),
      exts_(extensions) {}

bool ImageFolderDataset::ValidateParams() {
  if (dataset_dir_.empty()) {
    MS_LOG(ERROR) << "No dataset path is specified.";
    return false;
  }

  return true;
}

std::vector<std::shared_ptr<DatasetOp>> ImageFolderDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // If user does not specify Sampler, create a default sampler, i.e., RandomSampler.
  if (sampler_ == nullptr) {
    sampler_ = CreateDefaultSampler();
  }

  // Do internal Schema generation.
  // This arg is exist in ImageFolderOp, but not externalized (in Python API).
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &scalar)));
  node_ops.push_back(std::make_shared<ImageFolderOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                                     recursive_, decode_, exts_, class_indexing_, std::move(schema),
                                                     std::move(sampler_->Build())));
  return node_ops;
}

MnistDataset::MnistDataset(std::string dataset_dir, std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir), sampler_(sampler) {}

bool MnistDataset::ValidateParams() {
  if (dataset_dir_.empty()) {
    MS_LOG(ERROR) << "No dataset path is specified.";
    return false;
  }

  return true;
}

std::vector<std::shared_ptr<DatasetOp>> MnistDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // If user does not specify Sampler, create a default sampler, i.e., RandomSampler.
  if (sampler_ == nullptr) {
    sampler_ = CreateDefaultSampler();
  }

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  node_ops.push_back(std::make_shared<MnistOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                               std::move(schema), std::move(sampler_->Build())));
  return node_ops;
}

// DERIVED DATASET CLASSES LEAF-NODE DATASETS
// (In alphabetical order)

BatchDataset::BatchDataset(int32_t batch_size, bool drop_remainder, bool pad, std::vector<std::string> cols_to_map,
                           std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map)
    : batch_size_(batch_size),
      drop_remainder_(drop_remainder),
      pad_(pad),
      cols_to_map_(cols_to_map),
      pad_map_(pad_map) {}

std::vector<std::shared_ptr<DatasetOp>> BatchDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

#ifdef ENABLE_PYTHON
  py::function noop;
  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               cols_to_map_, noop, noop, pad_map_));
#else
  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               cols_to_map_, pad_map_));
#endif
  return node_ops;
}

bool BatchDataset::ValidateParams() {
  if (batch_size_ <= 0) {
    MS_LOG(ERROR) << "Batch: Batch size cannot be negative";
    return false;
  }

  return true;
}

MapDataset::MapDataset(std::vector<std::shared_ptr<TensorOperation>> operations, std::vector<std::string> input_columns,
                       std::vector<std::string> output_columns, const std::vector<std::string> &project_columns)
    : operations_(operations),
      input_columns_(input_columns),
      output_columns_(output_columns),
      project_columns_(project_columns) {}

std::vector<std::shared_ptr<DatasetOp>> MapDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Currently default is true, and this is not exposed to user.
  bool perf_mode = true;

  std::vector<std::shared_ptr<TensorOp>> tensor_ops;

  // Build tensorOp from tensorOperation vector
  // This is to ensure each iterator hold its own copy of the tensorOp objects.
  (void)std::transform(
    operations_.begin(), operations_.end(), std::back_inserter(tensor_ops),
    [](std::shared_ptr<TensorOperation> operation) -> std::shared_ptr<TensorOp> { return operation->Build(); });

  // This parameter will be removed with next rebase
  std::vector<std::string> col_orders;
  auto map_op =
    std::make_shared<MapOp>(input_columns_, output_columns_, tensor_ops, num_workers_, connector_que_size_, perf_mode);
  if (!project_columns_.empty()) {
    auto project_op = std::make_shared<ProjectOp>(project_columns_);
    node_ops.push_back(project_op);
  }

  node_ops.push_back(map_op);
  return node_ops;
}

bool MapDataset::ValidateParams() {
  if (operations_.empty()) {
    MS_LOG(ERROR) << "Map: No operation is specified.";
    return false;
  }

  return true;
}

// Function to build ProjectOp
ProjectDataset::ProjectDataset(const std::vector<std::string> &columns) : columns_(columns) {}

bool ProjectDataset::ValidateParams() {
  if (columns_.empty()) {
    MS_LOG(ERROR) << "No columns are specified.";
    return false;
  }
  return true;
}

std::vector<std::shared_ptr<DatasetOp>> ProjectDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ProjectOp>(columns_));
  return node_ops;
}

// Function to build RenameOp
RenameDataset::RenameDataset(const std::vector<std::string> &input_columns,
                             const std::vector<std::string> &output_columns)
    : input_columns_(input_columns), output_columns_(output_columns) {}

bool RenameDataset::ValidateParams() {
  if (input_columns_.empty() || output_columns_.empty()) {
    MS_LOG(ERROR) << "input and output columns must be specified";
    return false;
  }
  if (input_columns_.size() != output_columns_.size()) {
    MS_LOG(ERROR) << "input and output columns must be the same size";
    return false;
  }
  return true;
}

std::vector<std::shared_ptr<DatasetOp>> RenameDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<RenameOp>(input_columns_, output_columns_, connector_que_size_));
  return node_ops;
}

RepeatDataset::RepeatDataset(uint32_t count) : repeat_count_(count) {}

std::vector<std::shared_ptr<DatasetOp>> RepeatDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<RepeatOp>(repeat_count_));
  return node_ops;
}

bool RepeatDataset::ValidateParams() {
  if (repeat_count_ <= 0) {
    MS_LOG(ERROR) << "Repeat: Repeat count cannot be negative";
    return false;
  }

  return true;
}

// Constructor for ShuffleDataset
ShuffleDataset::ShuffleDataset(int32_t shuffle_size, bool reset_every_epoch)
    : shuffle_size_(shuffle_size), shuffle_seed_(GetSeed()), reset_every_epoch_(reset_every_epoch) {}

// Function to build the ShuffleOp
std::vector<std::shared_ptr<DatasetOp>> ShuffleDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ShuffleOp>(shuffle_size_, shuffle_seed_, connector_que_size_, reset_every_epoch_,
                                                 rows_per_buffer_));
  return node_ops;
}

// Function to validate the parameters for ShuffleDataset
bool ShuffleDataset::ValidateParams() {
  if (shuffle_size_ <= 1) {
    MS_LOG(ERROR) << "ShuffleDataset: Invalid input, shuffle_size: " << shuffle_size_;
    return false;
  }

  return true;
}

// Constructor for SkipDataset
SkipDataset::SkipDataset(int32_t count) : skip_count_(count) {}

// Function to build the SkipOp
std::vector<std::shared_ptr<DatasetOp>> SkipDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<SkipOp>(skip_count_, connector_que_size_));
  return node_ops;
}

// Function to validate the parameters for SkipDataset
bool SkipDataset::ValidateParams() {
  if (skip_count_ <= -1) {
    MS_LOG(ERROR) << "Skip: Invalid input, skip_count: " << skip_count_;
    return false;
  }

  return true;
}

// Function to build ZipOp
ZipDataset::ZipDataset() {}

bool ZipDataset::ValidateParams() { return true; }

std::vector<std::shared_ptr<DatasetOp>> ZipDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ZipOp>(rows_per_buffer_, connector_que_size_));
  return node_ops;
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
