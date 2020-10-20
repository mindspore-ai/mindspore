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

#include <algorithm>
#include <fstream>
#include <unordered_set>
#include "minddata/dataset/include/datasets.h"
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
#endif
#include "minddata/dataset/engine/datasetops/source/mnist_op.h"
#include "minddata/dataset/engine/datasetops/source/random_data_op.h"
#include "minddata/dataset/engine/datasetops/source/text_file_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/source/tf_reader_op.h"
#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#endif
// Dataset operator headers (in alphabetical order)
#include "minddata/dataset/engine/datasetops/batch_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/datasetops/bucket_batch_by_length_op.h"
#endif
#include "minddata/dataset/engine/datasetops/build_vocab_op.h"
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
#include "minddata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "minddata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/util/path.h"
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
                                 const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<AlbumNode>(dataset_dir, data_schema, column_names, decode, sampler);

  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a CelebANode.
std::shared_ptr<CelebANode> CelebA(const std::string &dataset_dir, const std::string &usage,
                                   const std::shared_ptr<SamplerObj> &sampler, bool decode,
                                   const std::set<std::string> &extensions) {
  auto ds = std::make_shared<CelebANode>(dataset_dir, usage, sampler, decode, extensions);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a Cifar10Node.
std::shared_ptr<Cifar10Node> Cifar10(const std::string &dataset_dir, const std::string &usage,
                                     const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<Cifar10Node>(dataset_dir, usage, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a Cifar100Node.
std::shared_ptr<Cifar100Node> Cifar100(const std::string &dataset_dir, const std::string &usage,
                                       const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<Cifar100Node>(dataset_dir, usage, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a CLUENode.
std::shared_ptr<CLUENode> CLUE(const std::vector<std::string> &clue_files, const std::string &task,
                               const std::string &usage, int64_t num_samples, ShuffleMode shuffle, int32_t num_shards,
                               int32_t shard_id) {
  auto ds = std::make_shared<CLUENode>(clue_files, task, usage, num_samples, shuffle, num_shards, shard_id);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a CocoNode.
std::shared_ptr<CocoNode> Coco(const std::string &dataset_dir, const std::string &annotation_file,
                               const std::string &task, const bool &decode,
                               const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<CocoNode>(dataset_dir, annotation_file, task, decode, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a CSVNode.
std::shared_ptr<CSVNode> CSV(const std::vector<std::string> &dataset_files, char field_delim,
                             const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                             const std::vector<std::string> &column_names, int64_t num_samples, ShuffleMode shuffle,
                             int32_t num_shards, int32_t shard_id) {
  auto ds = std::make_shared<CSVNode>(dataset_files, field_delim, column_defaults, column_names, num_samples, shuffle,
                                      num_shards, shard_id);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a ImageFolderNode.
std::shared_ptr<ImageFolderNode> ImageFolder(const std::string &dataset_dir, bool decode,
                                             const std::shared_ptr<SamplerObj> &sampler,
                                             const std::set<std::string> &extensions,
                                             const std::map<std::string, int32_t> &class_indexing) {
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;

  // Create logical representation of ImageFolderNode.
  auto ds = std::make_shared<ImageFolderNode>(dataset_dir, decode, sampler, recursive, extensions, class_indexing);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

#ifndef ENABLE_ANDROID
// Function to create a ManifestNode.
std::shared_ptr<ManifestNode> Manifest(const std::string &dataset_file, const std::string &usage,
                                       const std::shared_ptr<SamplerObj> &sampler,
                                       const std::map<std::string, int32_t> &class_indexing, bool decode) {
  auto ds = std::make_shared<ManifestNode>(dataset_file, usage, sampler, class_indexing, decode);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}
#endif

// Function to create a MindDataNode.
std::shared_ptr<MindDataNode> MindData(const std::string &dataset_file, const std::vector<std::string> &columns_list,
                                       const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                       int64_t num_padded) {
  auto ds = std::make_shared<MindDataNode>(dataset_file, columns_list, sampler, padded_sample, num_padded);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a MindDataNode.
std::shared_ptr<MindDataNode> MindData(const std::vector<std::string> &dataset_files,
                                       const std::vector<std::string> &columns_list,
                                       const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample,
                                       int64_t num_padded) {
  auto ds = std::make_shared<MindDataNode>(dataset_files, columns_list, sampler, padded_sample, num_padded);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a MnistNode.
std::shared_ptr<MnistNode> Mnist(const std::string &dataset_dir, const std::string &usage,
                                 const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<MnistNode>(dataset_dir, usage, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to overload "+" operator to concat two datasets
std::shared_ptr<ConcatNode> operator+(const std::shared_ptr<Dataset> &datasets1,
                                      const std::shared_ptr<Dataset> &datasets2) {
  std::shared_ptr<ConcatNode> ds = std::make_shared<ConcatNode>(std::vector({datasets2, datasets1}));

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a TextFileNode.
std::shared_ptr<TextFileNode> TextFile(const std::vector<std::string> &dataset_files, int64_t num_samples,
                                       ShuffleMode shuffle, int32_t num_shards, int32_t shard_id) {
  auto ds = std::make_shared<TextFileNode>(dataset_files, num_samples, shuffle, num_shards, shard_id);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}

#ifndef ENABLE_ANDROID
// Function to create a VOCNode.
std::shared_ptr<VOCNode> VOC(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                             const std::map<std::string, int32_t> &class_indexing, bool decode,
                             const std::shared_ptr<SamplerObj> &sampler) {
  auto ds = std::make_shared<VOCNode>(dataset_dir, task, usage, class_indexing, decode, sampler);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
}
#endif

// Function to create a ZipNode.
std::shared_ptr<ZipNode> Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  auto ds = std::make_shared<ZipNode>(datasets);

  // Call derived class validation method.
  return ds->ValidateParams() ? ds : nullptr;
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

  if (!ds->ValidateParams()) {
    return nullptr;
  }

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

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  return ds;
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

  return ds->ValidateParams() ? ds : nullptr;
}

// Function to create a Map dataset.
std::shared_ptr<MapNode> Dataset::Map(std::vector<std::shared_ptr<TensorOperation>> operations,
                                      std::vector<std::string> input_columns, std::vector<std::string> output_columns,
                                      const std::vector<std::string> &project_columns) {
  auto ds = std::make_shared<MapNode>(shared_from_this(), operations, input_columns, output_columns, project_columns);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  return ds;
}

// Function to create a ProjectNode.
std::shared_ptr<ProjectNode> Dataset::Project(const std::vector<std::string> &columns) {
  auto ds = std::make_shared<ProjectNode>(shared_from_this(), columns);
  // Call derived class validation method.
  if (!ds->ValidateParams()) {
    return nullptr;
  }

  return ds;
}

// Function to create a RenameNode.
std::shared_ptr<RenameNode> Dataset::Rename(const std::vector<std::string> &input_columns,
                                            const std::vector<std::string> &output_columns) {
  auto ds = std::make_shared<RenameNode>(shared_from_this(), input_columns, output_columns);
  // Call derived class validation method.
  if (!ds->ValidateParams()) {
    return nullptr;
  }

  return ds;
}

// Function to create Repeat dataset.
std::shared_ptr<Dataset> Dataset::Repeat(int32_t count) {
  // Workaround for repeat == 1, do not inject repeat.
  if (count == 1) {
    return shared_from_this();
  }

  auto ds = std::make_shared<RepeatNode>(shared_from_this(), count);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  return ds;
}

// Function to create a ShuffleOp
std::shared_ptr<ShuffleNode> Dataset::Shuffle(int32_t buffer_size) {
  // Pass in reshuffle_each_epoch with true
  auto ds = std::make_shared<ShuffleNode>(shared_from_this(), buffer_size, true);

  if (!ds->ValidateParams()) {
    return nullptr;
  }

  return ds;
}

// Function to create a SkipNode.
std::shared_ptr<SkipNode> Dataset::Skip(int32_t count) {
  auto ds = std::make_shared<SkipNode>(shared_from_this(), count);

  // Call derived class validation method.
  if (!ds->ValidateParams()) {
    return nullptr;
  }

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

  // Call derived class validation method.
  if (!ds->ValidateParams()) {
    return nullptr;
  }

  return ds;
}

// Function to create a Zip dataset
std::shared_ptr<ZipNode> Dataset::Zip(const std::vector<std::shared_ptr<Dataset>> &datasets) {
  // Default values
  auto ds = std::make_shared<ZipNode>(datasets);
  ds->children.push_back(shared_from_this());

  return ds->ValidateParams() ? ds : nullptr;
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

// Helper function to validate dataset dataset files parameter
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
    MS_LOG(ERROR) << dataset_name << ": Sampler is not constructed correctly, sampler: nullptr";
    std::string err_msg = dataset_name + ": Sampler is not constructed correctly, sampler: nullptr";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

Status ValidateStringValue(const std::string &str, const std::unordered_set<std::string> &valid_strings) {
  if (valid_strings.find(str) == valid_strings.end()) {
    std::string mode;
    mode = std::accumulate(valid_strings.begin(), valid_strings.end(), mode,
                           [](std::string a, std::string b) { return std::move(a) + " " + std::move(b); });
    std::string err_msg = str + " does not match any mode in [" + mode + " ]";
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
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  std::set<std::string> columns_set(columns.begin(), columns.end());
  if (columns_set.size() != columns.size()) {
    // others";
    std::string err_msg = dataset_name + ":" + column_param + ": Every column name should not be same with others";
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

/* ####################################### Derived Dataset classes ################################# */

// DERIVED DATASET CLASSES LEAF-NODE DATASETS
// (In alphabetical order)

// Constructor for AlbumNode
AlbumNode::AlbumNode(const std::string &dataset_dir, const std::string &data_schema,
                     const std::vector<std::string> &column_names, bool decode,
                     const std::shared_ptr<SamplerObj> &sampler)
    : dataset_dir_(dataset_dir),
      schema_path_(data_schema),
      column_names_(column_names),
      decode_(decode),
      sampler_(sampler) {}

Status AlbumNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("AlbumNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("AlbumNode", {schema_path_}));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("AlbumNode", sampler_));

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("AlbumNode", "column_names", column_names_));
  }

  return Status::OK();
}

// Function to build AlbumNode
std::vector<std::shared_ptr<DatasetOp>> AlbumNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->LoadSchemaFile(schema_path_, column_names_));

  // Argument that is not exposed to user in the API.
  std::set<std::string> extensions = {};

  node_ops.push_back(std::make_shared<AlbumOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                               decode_, extensions, std::move(schema), std::move(sampler_->Build())));
  return node_ops;
}

// Constructor for CelebANode
CelebANode::CelebANode(const std::string &dataset_dir, const std::string &usage,
                       const std::shared_ptr<SamplerObj> &sampler, const bool &decode,
                       const std::set<std::string> &extensions)
    : dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler), decode_(decode), extensions_(extensions) {}

Status CelebANode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("CelebANode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("CelebANode", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue(usage_, {"all", "train", "valid", "test"}));

  return Status::OK();
}

// Function to build CelebANode
std::vector<std::shared_ptr<DatasetOp>> CelebANode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  // label is like this:0 1 0 0 1......
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("attr", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  node_ops.push_back(std::make_shared<CelebAOp>(num_workers_, rows_per_buffer_, dataset_dir_, connector_que_size_,
                                                decode_, usage_, extensions_, std::move(schema),
                                                std::move(sampler_->Build())));
  return node_ops;
}

// Constructor for Cifar10Node
Cifar10Node::Cifar10Node(const std::string &dataset_dir, const std::string &usage, std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler) {}

Status Cifar10Node::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("Cifar10Node", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("Cifar10Node", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue(usage_, {"train", "test", "all"}));

  return Status::OK();
}

// Function to build CifarOp for Cifar10
std::vector<std::shared_ptr<DatasetOp>> Cifar10Node::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  node_ops.push_back(std::make_shared<CifarOp>(CifarOp::CifarType::kCifar10, usage_, num_workers_, rows_per_buffer_,
                                               dataset_dir_, connector_que_size_, std::move(schema),
                                               std::move(sampler_->Build())));
  return node_ops;
}

// Constructor for Cifar100Node
Cifar100Node::Cifar100Node(const std::string &dataset_dir, const std::string &usage,
                           std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler) {}

Status Cifar100Node::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("Cifar100Node", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("Cifar100Node", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue(usage_, {"train", "test", "all"}));

  return Status::OK();
}

// Function to build CifarOp for Cifar100
std::vector<std::shared_ptr<DatasetOp>> Cifar100Node::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("coarse_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("fine_label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  node_ops.push_back(std::make_shared<CifarOp>(CifarOp::CifarType::kCifar100, usage_, num_workers_, rows_per_buffer_,
                                               dataset_dir_, connector_que_size_, std::move(schema),
                                               std::move(sampler_->Build())));
  return node_ops;
}

// Constructor for CLUENode
CLUENode::CLUENode(const std::vector<std::string> clue_files, std::string task, std::string usage, int64_t num_samples,
                   ShuffleMode shuffle, int32_t num_shards, int32_t shard_id)
    : dataset_files_(clue_files),
      task_(task),
      usage_(usage),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

Status CLUENode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("CLUENode", dataset_files_));

  std::vector<std::string> task_list = {"AFQMC", "TNEWS", "IFLYTEK", "CMNLI", "WSC", "CSL"};
  std::vector<std::string> usage_list = {"train", "test", "eval"};

  if (find(task_list.begin(), task_list.end(), task_) == task_list.end()) {
    std::string err_msg = "task should be AFQMC, TNEWS, IFLYTEK, CMNLI, WSC or CSL.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (find(usage_list.begin(), usage_list.end(), usage_) == usage_list.end()) {
    std::string err_msg = "usage should be train, test or eval.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (num_samples_ < 0) {
    std::string err_msg = "CLUENode: Invalid number of samples: " + num_samples_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("CLUENode", num_shards_, shard_id_));

  return Status::OK();
}

// Function to split string based on a character delimiter
std::vector<std::string> CLUENode::split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

// Function to build CLUENode
std::vector<std::shared_ptr<DatasetOp>> CLUENode::Build() {
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

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files = dataset_files_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  std::shared_ptr<ClueOp> clue_op =
    std::make_shared<ClueOp>(num_workers_, rows_per_buffer_, num_samples_, worker_connector_size_, ck_map,
                             sorted_dataset_files, connector_que_size_, shuffle_files, num_shards_, shard_id_, nullptr);
  RETURN_EMPTY_IF_ERROR(clue_op->Init());
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(ClueOp::CountAllFileRows(sorted_dataset_files, &num_rows));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }

  node_ops.push_back(clue_op);
  return node_ops;
}

// Constructor for CocoNode
CocoNode::CocoNode(const std::string &dataset_dir, const std::string &annotation_file, const std::string &task,
                   const bool &decode, const std::shared_ptr<SamplerObj> &sampler)
    : dataset_dir_(dataset_dir), annotation_file_(annotation_file), task_(task), decode_(decode), sampler_(sampler) {}

Status CocoNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("CocoNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("CocoNode", sampler_));

  Path annotation_file(annotation_file_);
  if (!annotation_file.Exists()) {
    std::string err_msg = "annotation_file is invalid or not exist";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::set<std::string> task_list = {"Detection", "Stuff", "Panoptic", "Keypoint"};
  auto task_iter = task_list.find(task_);
  if (task_iter == task_list.end()) {
    std::string err_msg = "Invalid task type";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Function to build CocoNode
std::vector<std::shared_ptr<DatasetOp>> CocoNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

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
      MS_LOG(ERROR) << "CocoNode::Build : Invalid task type: " << task_type;
      return {};
  }
  std::shared_ptr<CocoOp> op =
    std::make_shared<CocoOp>(task_type, dataset_dir_, annotation_file_, num_workers_, rows_per_buffer_,
                             connector_que_size_, decode_, std::move(schema), std::move(sampler_->Build()));
  node_ops.push_back(op);
  return node_ops;
}

// Constructor for CSVNode
CSVNode::CSVNode(const std::vector<std::string> &csv_files, char field_delim,
                 const std::vector<std::shared_ptr<CsvBase>> &column_defaults,
                 const std::vector<std::string> &column_names, int64_t num_samples, ShuffleMode shuffle,
                 int32_t num_shards, int32_t shard_id)
    : dataset_files_(csv_files),
      field_delim_(field_delim),
      column_defaults_(column_defaults),
      column_names_(column_names),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

Status CSVNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("CSVNode", dataset_files_));

  if (field_delim_ == '"' || field_delim_ == '\r' || field_delim_ == '\n') {
    std::string err_msg = "CSVNode: The field delimiter should not be \", \\r, \\n";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (num_samples_ < 0) {
    std::string err_msg = "CSVNode: Invalid number of samples: " + num_samples_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("CSVNode", num_shards_, shard_id_));

  if (find(column_defaults_.begin(), column_defaults_.end(), nullptr) != column_defaults_.end()) {
    std::string err_msg = "CSVNode: column_default should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("CSVNode", "column_names", column_names_));
  }

  return Status::OK();
}

// Function to build CSVNode
std::vector<std::shared_ptr<DatasetOp>> CSVNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files = dataset_files_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  std::vector<std::shared_ptr<CsvOp::BaseRecord>> column_default_list;
  for (auto v : column_defaults_) {
    if (v->type == CsvType::INT) {
      column_default_list.push_back(
        std::make_shared<CsvOp::Record<int>>(CsvOp::INT, std::dynamic_pointer_cast<CsvRecord<int>>(v)->value));
    } else if (v->type == CsvType::FLOAT) {
      column_default_list.push_back(
        std::make_shared<CsvOp::Record<float>>(CsvOp::FLOAT, std::dynamic_pointer_cast<CsvRecord<float>>(v)->value));
    } else if (v->type == CsvType::STRING) {
      column_default_list.push_back(std::make_shared<CsvOp::Record<std::string>>(
        CsvOp::STRING, std::dynamic_pointer_cast<CsvRecord<std::string>>(v)->value));
    }
  }

  std::shared_ptr<CsvOp> csv_op = std::make_shared<CsvOp>(
    sorted_dataset_files, field_delim_, column_default_list, column_names_, num_workers_, rows_per_buffer_,
    num_samples_, worker_connector_size_, connector_que_size_, shuffle_files, num_shards_, shard_id_, nullptr);
  RETURN_EMPTY_IF_ERROR(csv_op->Init());
  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(CsvOp::CountAllFileRows(sorted_dataset_files, column_names_.empty(), &num_rows));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }

  node_ops.push_back(csv_op);
  return node_ops;
}

ImageFolderNode::ImageFolderNode(std::string dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler,
                                 bool recursive, std::set<std::string> extensions,
                                 std::map<std::string, int32_t> class_indexing)
    : dataset_dir_(dataset_dir),
      decode_(decode),
      sampler_(sampler),
      recursive_(recursive),
      class_indexing_(class_indexing),
      exts_(extensions) {}

Status ImageFolderNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("ImageFolderNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("ImageFolderNode", sampler_));

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ImageFolderNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

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

#ifndef ENABLE_ANDROID
ManifestNode::ManifestNode(const std::string &dataset_file, const std::string &usage,
                           const std::shared_ptr<SamplerObj> &sampler,
                           const std::map<std::string, int32_t> &class_indexing, bool decode)
    : dataset_file_(dataset_file), usage_(usage), decode_(decode), class_index_(class_indexing), sampler_(sampler) {}

Status ManifestNode::ValidateParams() {
  std::vector<char> forbidden_symbols = {':', '*', '?', '"', '<', '>', '|', '`', '&', '\'', ';'};
  for (char c : dataset_file_) {
    auto p = std::find(forbidden_symbols.begin(), forbidden_symbols.end(), c);
    if (p != forbidden_symbols.end()) {
      std::string err_msg = "filename should not contains :*?\"<>|`&;\'";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  Path manifest_file(dataset_file_);
  if (!manifest_file.Exists()) {
    std::string err_msg = "dataset file: [" + dataset_file_ + "] is invalid or not exist";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetSampler("ManifestNode", sampler_));

  std::vector<std::string> usage_list = {"train", "eval", "inference"};
  if (find(usage_list.begin(), usage_list.end(), usage_) == usage_list.end()) {
    std::string err_msg = "usage should be train, eval or inference.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ManifestNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  std::shared_ptr<ManifestOp> manifest_op;
  manifest_op =
    std::make_shared<ManifestOp>(num_workers_, rows_per_buffer_, dataset_file_, connector_que_size_, decode_,
                                 class_index_, std::move(schema), std::move(sampler_->Build()), usage_);

  node_ops.push_back(manifest_op);
  return node_ops;
}
#endif

#ifndef ENABLE_ANDROID
MindDataNode::MindDataNode(const std::vector<std::string> &dataset_files, const std::vector<std::string> &columns_list,
                           const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded)
    : dataset_file_(std::string()),
      dataset_files_(dataset_files),
      search_for_pattern_(false),
      columns_list_(columns_list),
      sampler_(sampler),
      padded_sample_(padded_sample),
      sample_bytes_({}),
      num_padded_(num_padded) {}

MindDataNode::MindDataNode(const std::string &dataset_file, const std::vector<std::string> &columns_list,
                           const std::shared_ptr<SamplerObj> &sampler, nlohmann::json padded_sample, int64_t num_padded)
    : dataset_file_(dataset_file),
      dataset_files_({}),
      search_for_pattern_(true),
      columns_list_(columns_list),
      sampler_(sampler),
      padded_sample_(padded_sample),
      sample_bytes_({}),
      num_padded_(num_padded) {}

Status MindDataNode::ValidateParams() {
  if (!search_for_pattern_ && dataset_files_.size() > 4096) {
    std::string err_msg =
      "MindDataNode: length of dataset_file must be less than or equal to 4096, dataset_file length: " +
      std::to_string(dataset_file_.size());
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::vector<std::string> dataset_file_vec =
    search_for_pattern_ ? std::vector<std::string>{dataset_file_} : dataset_files_;
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("MindDataNode", dataset_file_vec));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("MindDataNode", sampler_));

  if (!columns_list_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MindDataNode", "columns_list", columns_list_));
  }

  if (padded_sample_ != nullptr) {
    if (num_padded_ < 0) {
      std::string err_msg =
        "MindDataNode: num_padded must be greater than or equal to zero, num_padded: " + std::to_string(num_padded_);
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (columns_list_.empty()) {
      std::string err_msg = "MindDataNode: padded_sample is specified and requires columns_list as well";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    for (std::string &column : columns_list_) {
      if (padded_sample_.find(column) == padded_sample_.end()) {
        std::string err_msg = "MindDataNode: " + column + " in columns_list does not match any column in padded_sample";
        MS_LOG(ERROR) << err_msg << ", padded_sample: " << padded_sample_;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
    }
  }
  if (num_padded_ > 0) {
    if (padded_sample_ == nullptr) {
      std::string err_msg = "MindDataNode: num_padded is specified but padded_sample is not";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  return Status::OK();
}

// Helper function to create runtime sampler for minddata dataset
Status MindDataNode::BuildMindDatasetSamplerChain(const std::shared_ptr<SamplerObj> &sampler,
                                                  std::vector<std::shared_ptr<mindrecord::ShardOperator>> *operators_,
                                                  int64_t num_padded) {
  std::shared_ptr<mindrecord::ShardOperator> op = sampler->BuildForMindDataset();
  if (op == nullptr) {
    std::string err_msg =
      "MindDataNode: Unsupported sampler is supplied for MindDataset. Supported sampler list: "
      "SubsetRandomSampler, PkSampler, RandomSampler, SequentialSampler and DistributedSampler";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  std::stack<std::shared_ptr<mindrecord::ShardOperator>> stack_ops;
  while (op != nullptr) {
    auto sampler_op = std::dynamic_pointer_cast<mindrecord::ShardDistributedSample>(op);
    if (sampler_op && num_padded > 0) {
      sampler_op->SetNumPaddedSamples(num_padded);
      stack_ops.push(sampler_op);
    } else {
      stack_ops.push(op);
    }
    op = op->GetChildOp();
  }
  while (!stack_ops.empty()) {
    operators_->push_back(stack_ops.top());
    stack_ops.pop();
  }
  return Status::OK();
}

// Helper function to set sample_bytes from py::byte type
void MindDataNode::SetSampleBytes(std::map<std::string, std::string> *sample_bytes) { sample_bytes_ = *sample_bytes; }

std::vector<std::shared_ptr<DatasetOp>> MindDataNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::vector<std::shared_ptr<ShardOperator>> operators_;
  RETURN_EMPTY_IF_ERROR(BuildMindDatasetSamplerChain(sampler_, &operators_, num_padded_));

  std::shared_ptr<MindRecordOp> mindrecord_op;
  // If pass a string to MindData(), it will be treated as a pattern to search for matched files,
  // else if pass a vector to MindData(), it will be treated as specified files to be read
  if (search_for_pattern_) {
    std::vector<std::string> dataset_file_vec_ = {dataset_file_};
    mindrecord_op = std::make_shared<MindRecordOp>(num_workers_, rows_per_buffer_, dataset_file_vec_,
                                                   search_for_pattern_, connector_que_size_, columns_list_, operators_,
                                                   num_padded_, padded_sample_, sample_bytes_);
  } else {
    mindrecord_op = std::make_shared<MindRecordOp>(num_workers_, rows_per_buffer_, dataset_files_, search_for_pattern_,
                                                   connector_que_size_, columns_list_, operators_, num_padded_,
                                                   padded_sample_, sample_bytes_);
  }

  RETURN_EMPTY_IF_ERROR(mindrecord_op->Init());
  node_ops.push_back(mindrecord_op);

  return node_ops;
}
#endif

MnistNode::MnistNode(std::string dataset_dir, std::string usage, std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir), usage_(usage), sampler_(sampler) {}

Status MnistNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("MnistNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("MnistNode", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue(usage_, {"train", "test", "all"}));

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> MnistNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  node_ops.push_back(std::make_shared<MnistOp>(usage_, num_workers_, rows_per_buffer_, dataset_dir_,
                                               connector_que_size_, std::move(schema), std::move(sampler_->Build())));
  return node_ops;
}

// ValideParams for RandomNode
Status RandomNode::ValidateParams() {
  if (total_rows_ < 0) {
    std::string err_msg = "RandomNode: total_rows must be greater than or equal 0, now get " + total_rows_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetSampler("RandomNode", sampler_));

  if (!columns_list_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("RandomNode", "columns_list", columns_list_));
  }

  return Status::OK();
}

int32_t RandomNode::GenRandomInt(int32_t min, int32_t max) {
  std::uniform_int_distribution<int32_t> uniDist(min, max);
  return uniDist(rand_gen_);
}

// Build for RandomNode
std::vector<std::shared_ptr<DatasetOp>> RandomNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  rand_gen_.seed(GetSeed());  // seed the random generator
  // If total rows was not given, then randomly pick a number
  std::shared_ptr<SchemaObj> schema_obj;
  if (!schema_path_.empty()) {
    schema_obj = Schema(schema_path_);
    if (schema_obj == nullptr) {
      return {};
    }
  }

  std::string schema_json_string, schema_file_path;
  if (schema_ != nullptr) {
    schema_->set_dataset_type("Random");
    if (total_rows_ != 0) {
      schema_->set_num_rows(total_rows_);
    }
    schema_json_string = schema_->to_json();
  } else {
    schema_file_path = schema_path_;
  }

  std::unique_ptr<DataSchema> data_schema;
  std::vector<std::string> columns_to_load;
  if (columns_list_.size() > 0) {
    columns_to_load = columns_list_;
  }
  if (!schema_file_path.empty() || !schema_json_string.empty()) {
    data_schema = std::make_unique<DataSchema>();
    if (!schema_file_path.empty()) {
      data_schema->LoadSchemaFile(schema_file_path, columns_to_load);
    } else if (!schema_json_string.empty()) {
      data_schema->LoadSchemaString(schema_json_string, columns_to_load);
    }
  }
  std::shared_ptr<RandomDataOp> op;
  op = std::make_shared<RandomDataOp>(num_workers_, connector_que_size_, rows_per_buffer_, total_rows_,
                                      std::move(data_schema), std::move(sampler_->Build()));
  node_ops.push_back(op);
  return node_ops;
}

// Constructor for TextFileNode
TextFileNode::TextFileNode(std::vector<std::string> dataset_files, int32_t num_samples, ShuffleMode shuffle,
                           int32_t num_shards, int32_t shard_id)
    : dataset_files_(dataset_files),
      num_samples_(num_samples),
      shuffle_(shuffle),
      num_shards_(num_shards),
      shard_id_(shard_id) {}

Status TextFileNode::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("TextFileNode", dataset_files_));

  if (num_samples_ < 0) {
    std::string err_msg = "TextFileNode: Invalid number of samples: " + num_samples_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetShardParams("TextFileNode", num_shards_, shard_id_));

  return Status::OK();
}

// Function to build TextFileNode
std::vector<std::shared_ptr<DatasetOp>> TextFileNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Sort the dataset files in a lexicographical order
  std::vector<std::string> sorted_dataset_files = dataset_files_;
  std::sort(sorted_dataset_files.begin(), sorted_dataset_files.end());

  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_EMPTY_IF_ERROR(
    schema->AddColumn(ColDescriptor("text", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));

  // Create and initalize TextFileOp
  std::shared_ptr<TextFileOp> text_file_op = std::make_shared<TextFileOp>(
    num_workers_, rows_per_buffer_, num_samples_, worker_connector_size_, std::move(schema), sorted_dataset_files,
    connector_que_size_, shuffle_files, num_shards_, shard_id_, nullptr);
  RETURN_EMPTY_IF_ERROR(text_file_op->Init());

  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp
    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(TextFileOp::CountAllFileRows(sorted_dataset_files, &num_rows));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(sorted_dataset_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }

  // Add TextFileOp
  node_ops.push_back(text_file_op);
  return node_ops;
}

#ifndef ENABLE_ANDROID
// Validator for TFRecordNode
Status TFRecordNode::ValidateParams() { return Status::OK(); }

// Function to build TFRecordNode
std::vector<std::shared_ptr<DatasetOp>> TFRecordNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  // Sort the datasets file in a lexicographical order
  std::vector<std::string> sorted_dir_files = dataset_files_;
  std::sort(sorted_dir_files.begin(), sorted_dir_files.end());

  // Create Schema Object
  std::unique_ptr<DataSchema> data_schema = std::make_unique<DataSchema>();
  if (!schema_path_.empty()) {
    RETURN_EMPTY_IF_ERROR(data_schema->LoadSchemaFile(schema_path_, columns_list_));
  } else if (schema_obj_ != nullptr) {
    std::string schema_json_string = schema_obj_->to_json();
    RETURN_EMPTY_IF_ERROR(data_schema->LoadSchemaString(schema_json_string, columns_list_));
  }

  bool shuffle_files = (shuffle_ == ShuffleMode::kGlobal || shuffle_ == ShuffleMode::kFiles);

  // Create and initialize TFReaderOp
  std::shared_ptr<TFReaderOp> tf_reader_op = std::make_shared<TFReaderOp>(
    num_workers_, worker_connector_size_, rows_per_buffer_, num_samples_, sorted_dir_files, std::move(data_schema),
    connector_que_size_, columns_list_, shuffle_files, num_shards_, shard_id_, shard_equal_rows_, nullptr);

  RETURN_EMPTY_IF_ERROR(tf_reader_op->Init());

  if (shuffle_ == ShuffleMode::kGlobal) {
    // Inject ShuffleOp

    std::shared_ptr<DatasetOp> shuffle_op = nullptr;
    int64_t num_rows = 0;

    // First, get the number of rows in the dataset
    RETURN_EMPTY_IF_ERROR(TFReaderOp::CountTotalRows(&num_rows, sorted_dir_files));

    // Add the shuffle op after this op
    RETURN_EMPTY_IF_ERROR(AddShuffleOp(sorted_dir_files.size(), num_shards_, num_rows, 0, connector_que_size_,
                                       rows_per_buffer_, &shuffle_op));
    node_ops.push_back(shuffle_op);
  }

  // Add TFReaderOp
  node_ops.push_back(tf_reader_op);
  return node_ops;
}

// Constructor for VOCNode
VOCNode::VOCNode(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                 const std::map<std::string, int32_t> &class_indexing, bool decode, std::shared_ptr<SamplerObj> sampler)
    : dataset_dir_(dataset_dir),
      task_(task),
      usage_(usage),
      class_index_(class_indexing),
      decode_(decode),
      sampler_(sampler) {}

Status VOCNode::ValidateParams() {
  Path dir(dataset_dir_);
  if (!dir.IsDirectory()) {
    std::string err_msg = "Invalid dataset path or no dataset path is specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetSampler("VOCNode", sampler_));

  if (task_ == "Segmentation") {
    if (!class_index_.empty()) {
      std::string err_msg = "class_indexing is invalid in Segmentation task.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    Path imagesets_file = dir / "ImageSets" / "Segmentation" / usage_ + ".txt";
    if (!imagesets_file.Exists()) {
      std::string err_msg = "Invalid usage: " + usage_ + ", file does not exist";
      MS_LOG(ERROR) << "Invalid usage: " << usage_ << ", file \"" << imagesets_file << "\" does not exist!";
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else if (task_ == "Detection") {
    Path imagesets_file = dir / "ImageSets" / "Main" / usage_ + ".txt";
    if (!imagesets_file.Exists()) {
      std::string err_msg = "Invalid usage: " + usage_ + ", file does not exist";
      MS_LOG(ERROR) << "Invalid usage: " << usage_ << ", file \"" << imagesets_file << "\" does not exist!";
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else {
    std::string err_msg = "Invalid task: " + task_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Function to build VOCNode
std::vector<std::shared_ptr<DatasetOp>> VOCNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

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
  voc_op = std::make_shared<VOCOp>(task_type_, usage_, dataset_dir_, class_index_, num_workers_, rows_per_buffer_,
                                   connector_que_size_, decode_, std::move(schema), std::move(sampler_->Build()));
  node_ops.push_back(voc_op);
  return node_ops;
}
#endif

// DERIVED DATASET CLASSES LEAF-NODE DATASETS
// (In alphabetical order)

BatchNode::BatchNode(std::shared_ptr<Dataset> child, int32_t batch_size, bool drop_remainder, bool pad,
                     std::vector<std::string> cols_to_map,
                     std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> pad_map)
    : batch_size_(batch_size),
      drop_remainder_(drop_remainder),
      pad_(pad),
      cols_to_map_(cols_to_map),
      pad_map_(pad_map) {
  this->children.push_back(child);
}

std::vector<std::shared_ptr<DatasetOp>> BatchNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

#ifdef ENABLE_PYTHON
  py::function noop;
  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               cols_to_map_, cols_to_map_, noop, noop, pad_map_));
#else
  node_ops.push_back(std::make_shared<BatchOp>(batch_size_, drop_remainder_, pad_, connector_que_size_, num_workers_,
                                               cols_to_map_, pad_map_));
#endif

  // Until py::function is implemented for C++ API, there is no need for a project op to be inserted after batch
  // because project is only needed when batch op performs per_batch_map. This per_batch_map is a pyfunc
  return node_ops;
}

Status BatchNode::ValidateParams() {
  if (batch_size_ <= 0) {
    std::string err_msg = "Batch: batch_size should be positive integer, but got: " + batch_size_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (!cols_to_map_.empty()) {
    std::string err_msg = "cols_to_map functionality is not implemented in C++; this should be left empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

#ifndef ENABLE_ANDROID
BucketBatchByLengthNode::BucketBatchByLengthNode(
  std::shared_ptr<Dataset> child, const std::vector<std::string> &column_names,
  const std::vector<int32_t> &bucket_boundaries, const std::vector<int32_t> &bucket_batch_sizes,
  std::function<TensorRow(TensorRow)> element_length_function,
  const std::map<std::string, std::pair<TensorShape, std::shared_ptr<Tensor>>> &pad_info, bool pad_to_bucket_boundary,
  bool drop_remainder)
    : column_names_(column_names),
      bucket_boundaries_(bucket_boundaries),
      bucket_batch_sizes_(bucket_batch_sizes),
      element_length_function_(element_length_function),
      pad_info_(pad_info),
      pad_to_bucket_boundary_(pad_to_bucket_boundary),
      drop_remainder_(drop_remainder) {
  this->children.push_back(child);
}

std::vector<std::shared_ptr<DatasetOp>> BucketBatchByLengthNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::shared_ptr<TensorOp> c_func;
  if (element_length_function_ != nullptr) {
    c_func = std::make_shared<CFuncOp>(element_length_function_);
  } else {
    c_func = nullptr;
  }
  node_ops.push_back(std::make_shared<BucketBatchByLengthOp>(column_names_, bucket_boundaries_, bucket_batch_sizes_,
                                                             c_func, pad_info_, pad_to_bucket_boundary_,
                                                             drop_remainder_, connector_que_size_));
  return node_ops;
}

Status BucketBatchByLengthNode::ValidateParams() {
  if (element_length_function_ == nullptr && column_names_.size() != 1) {
    std::string err_msg =
      "BucketBatchByLength: element_length_function not specified, but not one column name: " + column_names_.size();
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // Check bucket_boundaries: must be positive and strictly increasing
  if (bucket_boundaries_.empty()) {
    std::string err_msg = "BucketBatchByLength: bucket_boundaries cannot be empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (int i = 0; i < bucket_boundaries_.size(); i++) {
    if (bucket_boundaries_[i] <= 0) {
      std::string err_msg = "BucketBatchByLength: Invalid non-positive bucket_boundaries, index: ";
      MS_LOG(ERROR)
        << "BucketBatchByLength: bucket_boundaries must only contain positive numbers. However, the element at index: "
        << i << " was: " << bucket_boundaries_[i];
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (i > 0 && bucket_boundaries_[i - 1] >= bucket_boundaries_[i]) {
      std::string err_msg = "BucketBatchByLength: Invalid bucket_boundaries not be strictly increasing.";
      MS_LOG(ERROR)
        << "BucketBatchByLength: bucket_boundaries must be strictly increasing. However, the elements at index: "
        << i - 1 << " and " << i << " were: " << bucket_boundaries_[i - 1] << " and " << bucket_boundaries_[i]
        << " respectively.";
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  // Check bucket_batch_sizes: must be positive
  if (bucket_batch_sizes_.empty()) {
    std::string err_msg = "BucketBatchByLength: bucket_batch_sizes must be non-empty";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (bucket_batch_sizes_.size() != bucket_boundaries_.size() + 1) {
    std::string err_msg = "BucketBatchByLength: bucket_batch_sizes's size must equal the size of bucket_boundaries + 1";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (std::any_of(bucket_batch_sizes_.begin(), bucket_batch_sizes_.end(), [](int i) { return i <= 0; })) {
    std::string err_msg = "BucketBatchByLength: bucket_batch_sizes must only contain positive numbers.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

BuildVocabNode::BuildVocabNode(std::shared_ptr<Dataset> child, std::shared_ptr<Vocab> vocab,
                               const std::vector<std::string> &columns, const std::pair<int64_t, int64_t> &freq_range,
                               int64_t top_k, const std::vector<std::string> &special_tokens, bool special_first)
    : vocab_(vocab),
      columns_(columns),
      freq_range_(freq_range),
      top_k_(top_k),
      special_tokens_(special_tokens),
      special_first_(special_first) {
  this->children.push_back(child);
}

// Function to build BuildVocabNode
std::vector<std::shared_ptr<DatasetOp>> BuildVocabNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  std::shared_ptr<BuildVocabOp> build_vocab_op;
  build_vocab_op = std::make_shared<BuildVocabOp>(vocab_, columns_, freq_range_, top_k_, special_tokens_,
                                                  special_first_, num_workers_, connector_que_size_);
  node_ops.push_back(build_vocab_op);
  return node_ops;
}

Status BuildVocabNode::ValidateParams() {
  if (vocab_ == nullptr) {
    std::string err_msg = "BuildVocab: vocab is null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (top_k_ <= 0) {
    std::string err_msg = "BuildVocab: top_k should be positive, but got: " + top_k_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (freq_range_.first < 0 || freq_range_.second > kDeMaxFreq || freq_range_.first > freq_range_.second) {
    std::string err_msg = "BuildVocab: frequency_range [a,b] violates 0 <= a <= b (a,b are inclusive)";
    MS_LOG(ERROR) << "BuildVocab: frequency_range [a,b] should be 0 <= a <= b (a,b are inclusive), "
                  << "but got [" << freq_range_.first << ", " << freq_range_.second << "]";
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("BuildVocab", "columns", columns_));
  }

  return Status::OK();
}
#endif

// Function to build ConcatOp
ConcatNode::ConcatNode(const std::vector<std::shared_ptr<Dataset>> &datasets) : datasets_(datasets) {
  this->children = datasets_;
}

Status ConcatNode::ValidateParams() {
  if (datasets_.empty()) {
    std::string err_msg = "Concat: concatenated datasets are not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (find(datasets_.begin(), datasets_.end(), nullptr) != datasets_.end()) {
    std::string err_msg = "Concat: concatenated datasets should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ConcatNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ConcatOp>(connector_que_size_));
  return node_ops;
}

MapNode::MapNode(std::shared_ptr<Dataset> child, std::vector<std::shared_ptr<TensorOperation>> operations,
                 std::vector<std::string> input_columns, std::vector<std::string> output_columns,
                 const std::vector<std::string> &project_columns)
    : operations_(operations),
      input_columns_(input_columns),
      output_columns_(output_columns),
      project_columns_(project_columns) {
  this->children.push_back(child);
}

std::vector<std::shared_ptr<DatasetOp>> MapNode::Build() {
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

Status MapNode::ValidateParams() {
  if (operations_.empty()) {
    std::string err_msg = "Map: No operation is specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (!input_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "input_columns", input_columns_));
  }

  if (!output_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "output_columns", output_columns_));
  }

  if (!project_columns_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("MapNode", "project_columns", project_columns_));
  }

  return Status::OK();
}

// Function to build ProjectOp
ProjectNode::ProjectNode(std::shared_ptr<Dataset> child, const std::vector<std::string> &columns) : columns_(columns) {
  this->children.push_back(child);
}

Status ProjectNode::ValidateParams() {
  if (columns_.empty()) {
    std::string err_msg = "ProjectNode: No columns are specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("ProjectNode", "columns", columns_));

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ProjectNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ProjectOp>(columns_));
  return node_ops;
}

// Function to build RenameOp
RenameNode::RenameNode(std::shared_ptr<Dataset> child, const std::vector<std::string> &input_columns,
                       const std::vector<std::string> &output_columns)
    : input_columns_(input_columns), output_columns_(output_columns) {
  this->children.push_back(child);
}

Status RenameNode::ValidateParams() {
  if (input_columns_.size() != output_columns_.size()) {
    std::string err_msg = "RenameNode: input and output columns must be the same size";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("RenameNode", "input_columns", input_columns_));

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("RenameNode", "output_columns", output_columns_));

  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> RenameNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<RenameOp>(input_columns_, output_columns_, connector_que_size_));
  return node_ops;
}

RepeatNode::RepeatNode(std::shared_ptr<Dataset> child, int32_t count) : repeat_count_(count) {
  this->children.push_back(child);
}

std::vector<std::shared_ptr<DatasetOp>> RepeatNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<RepeatOp>(repeat_count_));
  return node_ops;
}

Status RepeatNode::ValidateParams() {
  if (repeat_count_ <= 0 && repeat_count_ != -1) {
    std::string err_msg =
      "Repeat: repeat_count should be either -1 or positive integer, repeat_count_: " + repeat_count_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Constructor for ShuffleNode
ShuffleNode::ShuffleNode(std::shared_ptr<Dataset> child, int32_t shuffle_size, bool reset_every_epoch)
    : shuffle_size_(shuffle_size), shuffle_seed_(GetSeed()), reset_every_epoch_(reset_every_epoch) {
  this->children.push_back(child);
}

// Function to build the ShuffleOp
std::vector<std::shared_ptr<DatasetOp>> ShuffleNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ShuffleOp>(shuffle_size_, shuffle_seed_, connector_que_size_, reset_every_epoch_,
                                                 rows_per_buffer_));
  return node_ops;
}

// Function to validate the parameters for ShuffleNode
Status ShuffleNode::ValidateParams() {
  if (shuffle_size_ <= 1) {
    std::string err_msg = "ShuffleNode: Invalid input, shuffle_size: " + shuffle_size_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Constructor for SkipNode
SkipNode::SkipNode(std::shared_ptr<Dataset> child, int32_t count) : skip_count_(count) {
  this->children.push_back(child);
}

// Function to build the SkipOp
std::vector<std::shared_ptr<DatasetOp>> SkipNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<SkipOp>(skip_count_, connector_que_size_));
  return node_ops;
}

// Function to validate the parameters for SkipNode
Status SkipNode::ValidateParams() {
  if (skip_count_ <= -1) {
    std::string err_msg = "Skip: skip_count should not be negative, skip_count: " + skip_count_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Constructor for TakeNode
TakeNode::TakeNode(std::shared_ptr<Dataset> child, int32_t count) : take_count_(count) {
  this->children.push_back(child);
}

// Function to build the TakeOp
std::vector<std::shared_ptr<DatasetOp>> TakeNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<TakeOp>(take_count_, connector_que_size_));
  return node_ops;
}

// Function to validate the parameters for TakeNode
Status TakeNode::ValidateParams() {
  if (take_count_ <= 0 && take_count_ != -1) {
    std::string err_msg = "Take: take_count should be either -1 or positive integer, take_count: " + take_count_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Function to build ZipOp
ZipNode::ZipNode(const std::vector<std::shared_ptr<Dataset>> &datasets) : datasets_(datasets) {
  for (auto dataset : datasets_) {
    this->children.push_back(dataset);
  }
}

Status ZipNode::ValidateParams() {
  if (datasets_.empty()) {
    std::string err_msg = "Zip: datasets to zip are not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (find(datasets_.begin(), datasets_.end(), nullptr) != datasets_.end()) {
    std::string err_msg = "ZipNode: zip datasets should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ZipNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ZipOp>(rows_per_buffer_, connector_que_size_));
  return node_ops;
}

}  // namespace api
}  // namespace dataset
}  // namespace mindspore
