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
#include "minddata/dataset/engine/datasetops/source/celeba_op.h"
#include "minddata/dataset/engine/datasetops/source/cifar_op.h"
#include "minddata/dataset/engine/datasetops/source/clue_op.h"
#include "minddata/dataset/engine/datasetops/source/coco_op.h"
#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
// Dataset operator headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/batch_op.h"
#include "minddata/dataset/engine/datasetops/concat_op.h"
#include "minddata/dataset/engine/datasetops/map_op/map_op.h"
#include "minddata/dataset/engine/datasetops/project_op.h"
#include "minddata/dataset/engine/datasetops/rename_op.h"
#include "minddata/dataset/engine/datasetops/repeat_op.h"
#include "minddata/dataset/engine/datasetops/shuffle_op.h"
#include "minddata/dataset/engine/datasetops/skip_op.h"
#include "minddata/dataset/engine/datasetops/take_op.h"
#include "minddata/dataset/engine/datasetops/zip_op.h"

// Sampler headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/path.h"

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

// Constructor
Dataset::Dataset() {
  // Fetch some default value from config manager
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  num_workers_ = cfg->num_parallel_workers();
  rows_per_buffer_ = cfg->rows_per_buffer();
  connector_que_size_ = cfg->op_connector_size();
  worker_connector_size_ = cfg->worker_connector_size();
}

// FUNCTIONS TO CREATE DATASETS FOR LEAF-NODE DATASETS
// (In alphabetical order)

// Function to create a CelebADataset.
std::shared_ptr<CelebADataset> CelebA(const std::string &dataset_dir, const std::string &dataset_type,
                                      const std::shared_ptr<SamplerObj> &sampler, bool decode,
                                      const std::set<std::string> &extensions) {
  auto ds = std::make_shared<CelebADataset>(dataset_dir, dataset_type, sampler, decode, extensions);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a Cifar10Dataset.
std::shared_ptr<Cifar10Dataset> Cifar10(const std::string &dataset_dir, const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<Cifar10Dataset>(dataset_dir, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a Cifar100Dataset.
std::shared_ptr<Cifar100Dataset> Cifar100(const std::string &dataset_dir, const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<Cifar100Dataset>(dataset_dir, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a CLUEDataset.
std::shared_ptr<CLUEDataset> CLUE(const std::vector<std::string> &clue_files, const std::string &task,
                                  const std::string &usage, int64_t num_samples, ShuffleMode shuffle,
                                  int32_t num_shards, int32_t shard_id) {
  auto ds = std::make_shared<CLUEDataset>(clue_files, task, usage, num_samples, shuffle, num_shards, shard_id);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a CocoDataset.
std::shared_ptr<CocoDataset> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                                  const std::string &task, const bool &decode,
                                  const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<CocoDataset>(dataset_dir, annotation_file, task, decode, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a ImageFolderDataset.
std::shared_ptr<ImageFolderDataset> ImageFolder(const std::string &dataset_dir, bool decode,
                                                const std::shared_ptr<SamplerObj> &sampler,
                                                const std::set<std::string> &extensions,
                                                const std::map<std::string, int32_t> &class_indexing) {
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderDataset.
  auto ds = std::make_shared<ImageFolderDataset>(dataset_dir, decode, sampler, recursive, extensions, class_indexing);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a MnistDataset.
std::shared_ptr<MnistDataset> Mnist(const std::string &dataset_dir, const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<MnistDataset>(dataset_dir, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to overload "+" operator to concat two datasets
std::shared_ptr<ConcatDataset> operator+(const std::shared_ptr<Dataset> &datasets1,
                                         const std::shared_ptr<Dataset> &datasets2) {
  std::shared_ptr<ConcatDataset> ds = std::make_shared<ConcatDataset>(std::vector({datasets1, datasets2}));

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a TextFileDataset.
std::shared_ptr<TextFileDataset> TextFile(const std::vector<std::string> &dataset_files, int32_t num_samples,
                                          ShuffleMode shuffle, int32_t num_shards, int32_t shard_id) {
  auto ds = std::make_shared<TextFileDataset>(dataset_files, num_samples, shuffle, num_shards, shard_id);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a VOCDataset.
std::shared_ptr<VOCDataset> VOC(const std::string &dataset_dir, const std::string &task, const std::string &mode,
                                const std::map<std::string, int32_t> &class_indexing, bool decode,
                                const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<VOCDataset>(dataset_dir, task, mode, class_indexing, decode, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a ZipDataset.
std::shared_ptr<ZipDataset> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  auto ds = std::make_shared<ZipDataset>(datasets);

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

// Function to create a Concat dataset
std::shared_ptr<ConcatDataset> Dataset::Concat(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  auto ds = std::make_shared<ConcatDataset>(datasets);
  ds->children.push_back(shared_from_this());

  return ds->ValidateParams() ? ds : nullptr;
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
std::shared_ptr<ShuffleDataset> Dataset::Shuffle(int32_t buffer_size) {
  // Pass in reshuffle_each_epoch with true
  auto ds = std::make_shared<ShuffleDataset>(buffer_size, true);

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

// Function to create a TakeDataset.
std::shared_ptr<Dataset> Dataset::Take(int32_t count) {
  // If count is greater than the number of element in dataset or equal to -1,
  // all the element in dataset will be taken
  if (count == -1) {
    return shared_from_this();
  }

  auto ds = std::make_shared<TakeDataset>(count);

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
  auto ds = std::make_shared<ZipDataset>(datasets);
  ds->children.push_back(shared_from_this());

  return ds->ValidateParams() ? ds : nullptr;
}

// OTHER FUNCTIONS

// Helper function to create default RandomSampler.
std::shared_ptr<SamplerObj> CreateDefaultSampler() {
  const int32_t num_samples = 0;  // 0 means to sample all ids.
  bool replacement = false;
  return std::make_shared<RandomSamplerObj>(replacement, num_samples);
}

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
bool ValidateDatasetDirParam(const std::string &dataset_name, std::string dataset_dir) {
  if (dataset_dir.empty()) {
    MS_LOG(ERROR) << dataset_name << ": dataset_dir is not specified.";
    return false;
  }

  Path dir(dataset_dir);
  if (!dir.IsDirectory()) {
    MS_LOG(ERROR) << dataset_name << ": dataset_dir: [" << dataset_dir << "] is an invalid directory path.";
    return false;
  }

  if (access(dataset_dir.c_str(), R_OK) == -1) {
    MS_LOG(ERROR) << dataset_name << ": No access to specified dataset path: " << dataset_dir;
    return false;
  }

  return true;
}

// Helper function to validate dataset dataset files parameter
bool ValidateDatasetFilesParam(const std::string &dataset_name, const std::vector<std::string> &dataset_files) {
  if (dataset_files.empty()) {
    MS_LOG(ERROR) << dataset_name << ": dataset_files is not specified.";
    return false;
  }

  for (auto f : dataset_files) {
    Path dataset_file(f);
    if (!dataset_file.Exists()) {
      MS_LOG(ERROR) << dataset_name << ": dataset file: [" << f << "] is invalid or does not exist.";
      return false;
    }
  }

  return true;
}

// Helper function to validate dataset num_shards and shard_id parameters
bool ValidateDatasetShardParams(const std::string &dataset_name, int32_t num_shards, int32_t shard_id) {
  if (num_shards <= 0) {
    MS_LOG(ERROR) << dataset_name << ": Invalid num_shards: " << num_shards;
    return false;
  }

  if (shard_id < 0 || shard_id >= num_shards) {
    MS_LOG(ERROR) << dataset_name << ": Invalid input, shard_id: " << shard_id << ", num_shards: " << num_shards;
    return false;
  }

  return true;
}

/* ####################################### Derived Dataset classes ################################# */

// DERIVED DATASET CLASSES LEAF-NODE DATASETS
// (In alphabetical order)

// Constructor for CelebADataset
CelebADataset::CelebADataset(const std::string &dataset_dir, const std::string &dataset_type,
                             const std::shared_ptr<SamplerObj> &sampler, const bool &decode,
                             const std::set<std::string> &extensions)
    : dataset_dir_(dataset_dir),
      dataset_type_(dataset_type),
      sampler_(sampler),
      decode_(decode),
      extensions_(extensions) {}

bool CelebADataset::ValidateParams() {
  if (!ValidateDatasetDirParam("CelebADataset", dataset_dir_)) {
    return false;
  }
  std::set<std::string> dataset_type_list = {"all", "train", "valid", "test"};
  auto iter = dataset_type_list.find(dataset_type_);
  if (iter == dataset_type_list.end()) {
    MS_LOG(ERROR) << "dataset_type should be one of 'all', 'train', 'valid' or 'test'.";
    return false;
  }
  return true;
}

// Function to build CelebADataset
std::vector<std::shared_ptr<DatasetOp>> CelebADataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // If user does not specify Sampler, create a default sampler based on the shuffle variable.
  if (sampler_ == nullptr) {
    sampler_ = CreateDefaultSampler();
  }

  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  // label is like this:0 1 0 0 1......
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("attr", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  node_ops.push_back(std::make_shared<CelebAOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                                decode_, dataset_type_, extensions_, std::move(schema),
                                                std::move(sampler_->Build())));
  return node_ops;
}

// Constructor for Cifar10Dataset
Cifar10Dataset::Cifar10Dataset(const std::string &dataset_dir, std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir), sampler_(sampler) {}

bool Cifar10Dataset::ValidateParams() { return ValidateDatasetDirParam("Cifar10Dataset", dataset_dir_); }

// Function to build CifarOp for Cifar10
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

// Constructor for Cifar100Dataset
Cifar100Dataset::Cifar100Dataset(const std::string &dataset_dir, std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir), sampler_(sampler) {}

bool Cifar100Dataset::ValidateParams() { return ValidateDatasetDirParam("Cifar100Dataset", dataset_dir_); }

// Function to build CifarOp for Cifar100
std::vector<std::shared_ptr<DatasetOp>> Cifar100Dataset::Build() {
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
    schema->AddColumn(ColDescriptor("coarse_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("fine_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  node_ops.push_back(std::make_shared<CifarOp>(CifarOp::CifarType::kCifar100, num_workers_, rows_per_buffer_,
                                               dataset_dir_, connector_que_size_, std::move(schema),
                                               std::move(sampler_->Build())));
  return node_ops;
}

// Constructor for CLUEDataset
CLUEDataset::CLUEDataset(const std::vector<std::string> clue_files, std::string task, std::string usage,
                         int64_t num_samples, ShuffleMode shuffle, int32_t num_shards, int32_t shard_id)
    : dataset_files_(clue_files),
      task_(task),
      usage_(usage),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

bool CLUEDataset::ValidateParams() {
  if (!ValidateDatasetFilesParam("CLUEDataset", dataset_files_)) {
    return false;
  }

  std::vector<std::string> task_list = {"AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC", "CSL"};
  std::vector<std::string> usage_list = {"train", "test", "eval"};

  if (find(task_list.begin(), task_list.end(), task_) == task_list.end()) {
    MS_LOG(ERROR) << "task should be AFQMC, TNEWS, IFLYTEK, CMNLI, WSC or CSL.";
    return false;
  }

  if (find(usage_list.begin(), usage_list.end(), usage_) == usage_list.end()) {
    MS_LOG(ERROR) << "usage should be train, test or eval.";
    return false;
  }

  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "CLUEDataset: Invalid number of samples: " << num_samples_;
    return false;
  }

  if (!ValidateDatasetShardParams("CLUEDataset", num_shards_, shard_id_)) {
    return false;
  }

  return true;
}

// Function to split string based on a character delimiter
std::vector<std::string> CLUEDataset::split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

// Function to build CLUEDataset
std::vector<std::shared_ptr<DatasetOp>> CLUEDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;
  std::map<std::string, std::string> key_map;
  if (task_ == "AFQMC") {
    if (usage_ == "train") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
    } else if (usage_ == "eval") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    }
  } else if (task_ == "CMNLI") {
    if (usage_ == "train") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
    } else if (usage_ == "eval") {
      key_map["sentence1"] = "sentence1";
      key_map["sentence2"] = "sentence2";
      key_map["label"] = "label";
    }
  } else if (task_ == "CSL") {
    if (usage_ == "train") {
      key_map["id"] = "id";
      key_map["abst"] = "abst";
      key_map["keyword"] = "keyword";
      key_map["label"] = "label";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["abst"] = "abst";
      key_map["keyword"] = "keyword";
    } else if (usage_ == "eval") {
      key_map["id"] = "id";
      key_map["abst"] = "abst";
      key_map["keyword"] = "keyword";
      key_map["label"] = "label";
    }
  } else if (task_ == "IFLYTEK") {
    if (usage_ == "train") {
      key_map["label"] = "label";
      key_map["label_des"] = "label_des";
      key_map["sentence"] = "sentence";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence"] = "sentence";
    } else if (usage_ == "eval") {
      key_map["label"] = "label";
      key_map["label_des"] = "label_des";
      key_map["sentence"] = "sentence";
    }
  } else if (task_ == "TNEWS") {
    if (usage_ == "train") {
      key_map["label"] = "label";
      key_map["label_desc"] = "label_desc";
      key_map["sentence"] = "sentence";
      key_map["keywords"] = "keywords";
    } else if (usage_ == "test") {
      key_map["id"] = "id";
      key_map["sentence"] = "sentence";
      key_map["keywords"] = "keywords";
    } else if (usage_ == "eval") {
      key_map["label"] = "label";
      key_map["label_desc"] = "label_desc";
      key_map["sentence"] = "sentence";
      key_map["keywords"] = "keywords";
    }
  } else if (task_ == "WSC") {
    if (usage_ == "train") {
      key_map["span1_index"] = "target/span1_index";
      key_map["span2_index"] = "target/span2_index";
      key_map["span1_text"] = "target/span1_text";
      key_map["span2_text"] = "target/span2_text";
      key_map["idx"] = "idx";
      key_map["label"] = "label";
      key_map["text"] = "text";
    } else if (usage_ == "test") {
      key_map["span1_index"] = "target/span1_index";
      key_map["span2_index"] = "target/span2_index";
      key_map["span1_text"] = "target/span1_text";
      key_map["span2_text"] = "target/span2_text";
      key_map["idx"] = "idx";
      key_map["text"] = "text";
    } else if (usage_ == "eval") {
      key_map["span1_index"] = "target/span1_index";
      key_map["span2_index"] = "target/span2_index";
      key_map["span1_text"] = "target/span1_text";
      key_map["span2_text"] = "target/span2_text";
      key_map["idx"] = "idx";
      key_map["label"] = "label";
      key_map["text"] = "text";
    }
  }

  ColKeyMap ck_map;
  for (auto &p : key_map) {
    ck_map.insert({p.first, split(p.second, '/')});
  }

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);
  std::shared_ptr<ClueOp> clue_op =
    std::make_shared<ClueOp>(num_workers_, rows_per_buffer_, num_samples_, worker_connector_size_, ck_map,
                             dataset_files_, connector_que_size_, shuffle_files, num_shards_, shard_id_);
  RETURN_EMPTY_IF_ERROR(clue_op->Init());
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(ClueOp::CountAllFileRows(dataset_files_, &num_rows));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(dataset_files_.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }

  node_ops.push_back(clue_op);
  return node_ops;
}

// Constructor for CocoDataset
CocoDataset::CocoDataset(const std::string &dataset_dir, const std::string &annotation_file, const std::string &task,
                         const bool &decode, const std::shared_ptr<SamplerObj> &sampler)
    : dataset_dir_(dataset_dir), annotation_file_(annotation_file), task_(task), decode_(decode), sampler_(sampler) {}

bool CocoDataset::ValidateParams() {
  if (!ValidateDatasetDirParam("CocoDataset", dataset_dir_)) {
    return false;
  }
  Path annotation_file(annotation_file_);
  if (!annotation_file.Exists()) {
    MS_LOG(ERROR) << "annotation_file is invalid or not exist";
    return false;
  }
  std::set<std::string> task_list = {"Detection", "Stuff", "Panoptic", "Keypoint"};
  auto task_iter = task_list.find(task_);
  if (task_iter == task_list.end()) {
    MS_LOG(ERROR) << "Invalid task type";
    return false;
  }
  return true;
}

// Function to build CocoDataset
std::vector<std::shared_ptr<DatasetOp>> CocoDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // If user does not specify Sampler, create a default sampler based on the shuffle variable.
  if (sampler_ == nullptr) {
    sampler_ = CreateDefaultSampler();
  }

  CocoOp::TaskType task_type;
  if (task_ == "Detection") {
    task_type = CocoOp::TaskType::Detection;
  } else if (task_ == "Stuff") {
    task_type = CocoOp::TaskType::Stuff;
  } else if (task_ == "Keypoint") {
    task_type = CocoOp::TaskType::Keypoint;
  } else if (task_ == "Panoptic") {
    task_type = CocoOp::TaskType::Panoptic;
  }

  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor(std::string("image"), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  switch (task_type) {
    case CocoOp::TaskType::Detection:
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("bbox"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("category_id"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("iscrowd"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case CocoOp::TaskType::Stuff:
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("segmentation"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("iscrowd"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case CocoOp::TaskType::Keypoint:
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("keypoints"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("num_keypoints"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case CocoOp::TaskType::Panoptic:
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("bbox"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("category_id"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_EMPTY_IF_ERROR(schema->AddColumn(
        ColDescriptor(std::string("iscrowd"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_EMPTY_IF_ERROR(
        schema->AddColumn(ColDescriptor(std::string("area"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    default:
      MS_LOG(ERROR) << "CocoDataset::Build : Invalid task type";
      return {};
  }
  std::shared_ptr<CocoOp> op =
    std::make_shared<CocoOp>(task_type, dataset_dir_, annotation_file_, num_workers_, rows_per_buffer_,
                             connector_que_size_, decode_, std::move(schema), std::move(sampler_->Build()));
  node_ops.push_back(op);
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

bool ImageFolderDataset::ValidateParams() { return ValidateDatasetDirParam("ImageFolderDataset", dataset_dir_); }

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

bool MnistDataset::ValidateParams() { return ValidateDatasetDirParam("MnistDataset", dataset_dir_); }

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

// Constructor for TextFileDataset
TextFileDataset::TextFileDataset(std::vector<std::string> dataset_files, int32_t num_samples, ShuffleMode shuffle,
                                 int32_t num_shards, int32_t shard_id)
    : dataset_files_(dataset_files),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

bool TextFileDataset::ValidateParams() {
  if (!ValidateDatasetFilesParam("TextFileDataset", dataset_files_)) {
    return false;
  }

  if (num_samples_ < 0) {
    MS_LOG(ERROR) << "TextFileDataset: Invalid number of samples: " << num_samples_;
    return false;
  }

  if (!ValidateDatasetShardParams("TextfileDataset", num_shards_, shard_id_)) {
    return false;
  }

  return true;
}

// Function to build TextFileDataset
std::vector<std::shared_ptr<DatasetOp>> TextFileDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("text", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));

  // Create and initalize TextFileOp
  std::shared_ptr<TextFileOp> text_file_op = std::make_shared<TextFileOp>(
    num_workers_, rows_per_buffer_, num_samples_, worker_connector_size_, std::move(schema), dataset_files_,
    connector_que_size_, shuffle_files, num_shards_, shard_id_, std::move(nullptr));
  RETURN_EMPTY_IF_ERROR(text_file_op->Init());

  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(TextFileOp::CountAllFileRows(dataset_files_, &num_rows));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(dataset_files_.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }

  // Add TextFileOp
  node_ops.push_back(text_file_op);
  return node_ops;
}

// Constructor for VOCDataset
VOCDataset::VOCDataset(const std::string &dataset_dir, const std::string &task, const std::string &mode,
                       const std::map<std::string, int32_t> &class_indexing, bool decode,
                       std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir),
      task_(task),
      mode_(mode),
      class_index_(class_indexing),
      decode_(decode),
      sampler_(sampler) {}

bool VOCDataset::ValidateParams() {
  Path dir(dataset_dir_);
  if (!dir.IsDirectory()) {
    MS_LOG(ERROR) << "Invalid dataset path or no dataset path is specified.";
    return false;
  }
  if (task_ == "Segmentation") {
    if (!class_index_.empty()) {
      MS_LOG(ERROR) << "class_indexing is invalid in Segmentation task.";
      return false;
    }
    Path imagesets_file = dir / "ImageSets" / "Segmentation" / mode_ + ".txt";
    if (!imagesets_file.Exists()) {
      MS_LOG(ERROR) << "Invalid mode: " << mode_ << ", file \"" << imagesets_file << "\" is not exists!";
      return false;
    }
  } else if (task_ == "Detection") {
    Path imagesets_file = dir / "ImageSets" / "Main" / mode_ + ".txt";
    if (!imagesets_file.Exists()) {
      MS_LOG(ERROR) << "Invalid mode: " << mode_ << ", file \"" << imagesets_file << "\" is not exists!";
      return false;
    }
  } else {
    MS_LOG(ERROR) << "Invalid task: " << task_;
    return false;
  }
  return true;
}

// Function to build VOCDataset
std::vector<std::shared_ptr<DatasetOp>> VOCDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // If user does not specify Sampler, create a default sampler based on the shuffle variable.
  if (sampler_ == nullptr) {
    sampler_ = CreateDefaultSampler();
  }

  auto schema = std::make_unique<DataSchema>();
  VOCOp::TaskType task_type_;

  if (task_ == "Segmentation") {
    task_type_ = VOCOp::TaskType::Segmentation;
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnTarget), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  } else if (task_ == "Detection") {
    task_type_ = VOCOp::TaskType::Detection;
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnBbox), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnLabel), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnDifficult), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_EMPTY_IF_ERROR(schema->AddColumn(
      ColDescriptor(std::string(kColumnTruncate), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  }

  std::shared_ptr<VOCOp> voc_op;
  voc_op = std::make_shared<VOCOp>(task_type_, mode_, dataset_dir_, class_index_, num_workers_, rows_per_buffer_,
                                   connector_que_size_, decode_, std::move(schema), std::move(sampler_->Build()));
  node_ops.push_back(voc_op);
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

// Function to build ConcatOp
ConcatDataset::ConcatDataset(const std::vector<std::shared_ptr<Dataset>> &datasets) : datasets_(datasets) {
  this->children = datasets_;
}

bool ConcatDataset::ValidateParams() {
  if (datasets_.empty()) {
    MS_LOG(ERROR) << "Concat: concatenated datasets are not specified.";
    return false;
  }
  return true;
}

std::vector<std::shared_ptr<DatasetOp>> ConcatDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ConcatOp>(connector_que_size_));
  return node_ops;
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

  std::vector<std::shared_ptr<TensorOp>> tensor_ops;

  // Build tensorOp from tensorOperation vector
  // This is to ensure each iterator hold its own copy of the tensorOp objects.
  (void)std::transform(
    operations_.begin(), operations_.end(), std::back_inserter(tensor_ops),
    [](std::shared_ptr<TensorOperation> operation) -> std::shared_ptr<TensorOp> { return operation->Build(); });

  // This parameter will be removed with next rebase
  std::vector<std::string> col_orders;
  auto map_op = std::make_shared<MapOp>(input_columns_, output_columns_, tensor_ops, num_workers_, connector_que_size_);
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

RepeatDataset::RepeatDataset(int32_t count) : repeat_count_(count) {}

std::vector<std::shared_ptr<DatasetOp>> RepeatDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<RepeatOp>(repeat_count_));
  return node_ops;
}

bool RepeatDataset::ValidateParams() {
  if (repeat_count_ <= 0 && repeat_count_ != -1) {
    MS_LOG(ERROR) << "Repeat: repeat_count should be either -1 or positive integer, repeat_count_: " << repeat_count_;
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
    MS_LOG(ERROR) << "Skip: skip_count should not be negative, skip_count: " << skip_count_;
    return false;
  }

  return true;
}

// Constructor for TakeDataset
TakeDataset::TakeDataset(int32_t count) : take_count_(count) {}

// Function to build the TakeOp
std::vector<std::shared_ptr<DatasetOp>> TakeDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<TakeOp>(take_count_, connector_que_size_));
  return node_ops;
}

// Function to validate the parameters for TakeDataset
bool TakeDataset::ValidateParams() {
  if (take_count_ < 0 && take_count_ != -1) {
    MS_LOG(ERROR) << "Take: take_count should be either -1 or positive integer, take_count: " << take_count_;
    return false;
  }

  return true;
}

// Function to build ZipOp
ZipDataset::ZipDataset(const std::vector<std::shared_ptr<Dataset>> &datasets) : datasets_(datasets) {
  for (auto dataset : datasets_) {
    this->children.push_back(dataset);
  }
}

bool ZipDataset::ValidateParams() {
  if (datasets_.empty()) {
    MS_LOG(ERROR) << "Zip: dataset to zip are not specified.";
    return false;
  }
  return true;
}

std::vector<std::shared_ptr<DatasetOp>> ZipDataset::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ZipOp>(rows_per_buffer_, connector_que_size_));
  return node_ops;
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
