/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/serdes.h"

#include "debug/common.h"
#include "utils/utils.h"

namespace mindspore {
namespace dataset {

std::map<std::string, Status (*)(nlohmann::json json_obj, std::shared_ptr<TensorOperation> *operation)>
  Serdes::func_ptr_ = Serdes::InitializeFuncPtr();

Status Serdes::SaveToJSON(std::shared_ptr<DatasetNode> node, const std::string &filename, nlohmann::json *out_json) {
  // Dump attributes of current node to json string
  nlohmann::json args;
  RETURN_IF_NOT_OK(node->to_json(&args));
  args["op_type"] = node->Name();

  // If the current node isn't leaf node, visit all its children and get all attributes
  std::vector<nlohmann::json> children_pipeline;
  if (!node->IsLeaf()) {
    for (auto child : node->Children()) {
      nlohmann::json child_args;
      RETURN_IF_NOT_OK(SaveToJSON(child, "", &child_args));
      children_pipeline.push_back(child_args);
    }
  }
  args["children"] = children_pipeline;

  // Save json string into file if filename is given.
  if (!filename.empty()) {
    RETURN_IF_NOT_OK(SaveJSONToFile(args, filename));
  }

  *out_json = args;
  return Status::OK();
}

Status Serdes::SaveJSONToFile(nlohmann::json json_string, const std::string &file_name) {
  try {
    auto realpath = Common::GetRealPath(file_name);
    if (!realpath.has_value()) {
      MS_LOG(ERROR) << "Get real path failed, path=" << file_name;
      RETURN_STATUS_UNEXPECTED("Get real path failed, path=" + file_name);
    }

    std::ofstream file(realpath.value());
    file << json_string;
    file.close();

    ChangeFileMode(realpath.value(), S_IRUSR | S_IWUSR);
  } catch (const std::exception &err) {
    RETURN_STATUS_UNEXPECTED("Save json string into " + file_name + " failed!");
  }
  return Status::OK();
}

Status Serdes::Deserialize(std::string json_filepath, std::shared_ptr<DatasetNode> *ds) {
  nlohmann::json json_obj;
  CHECK_FAIL_RETURN_UNEXPECTED(json_filepath.size() != 0, "Json path is null");
  std::ifstream json_in(json_filepath);
  CHECK_FAIL_RETURN_UNEXPECTED(json_in, "Json path is not valid");
  try {
    json_in >> json_obj;
  } catch (const std::exception &e) {
    return Status(StatusCode::kMDSyntaxError, "Json object is not valid");
  }
  RETURN_IF_NOT_OK(ConstructPipeline(json_obj, ds));
  return Status::OK();
}

Status Serdes::ConstructPipeline(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("children") != json_obj.end(), "Fail to find children");
  std::shared_ptr<DatasetNode> child_ds;

  if (json_obj["children"].size() == 0) {
    // If the JSON object has no child, then this node is a leaf node. Call create node to construct the corresponding
    // leaf node
    RETURN_IF_NOT_OK(CreateNode(nullptr, json_obj, ds));
  } else if (json_obj["children"].size() == 1) {
    // This node only has one child, construct the sub-tree under it first, and then call create node to construct the
    // corresponding node
    RETURN_IF_NOT_OK(ConstructPipeline(json_obj["children"][0], &child_ds));
    RETURN_IF_NOT_OK(CreateNode(child_ds, json_obj, ds));
  } else {
    // if json object has more than 1 children, the operation must be zip.
    CHECK_FAIL_RETURN_UNEXPECTED((json_obj["op_type"] == "Zip"), "Fail to find right op_type - zip");
    std::vector<std::shared_ptr<DatasetNode>> datasets;
    for (auto child_json_obj : json_obj["children"]) {
      RETURN_IF_NOT_OK(ConstructPipeline(child_json_obj, &child_ds));
      datasets.push_back(child_ds);
    }
    CHECK_FAIL_RETURN_UNEXPECTED(datasets.size() > 1, "Should zip more than 1 dataset");
    *ds = std::make_shared<ZipNode>(datasets);
  }
  return Status::OK();
}

Status Serdes::CreateNode(std::shared_ptr<DatasetNode> child_ds, nlohmann::json json_obj,
                          std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("op_type") != json_obj.end(), "Fail to find op_type");
  std::string op_type = json_obj["op_type"];
  if (child_ds == nullptr) {
    // if dataset doesn't have any child, then create a source dataset IR. e.g., ImageFolderNode, CocoNode
    RETURN_IF_NOT_OK(CreateDatasetNode(json_obj, op_type, ds));
  } else {
    // if the dataset has at least one child, then create an operation dataset IR, e.g., BatchNode, MapNode
    RETURN_IF_NOT_OK(CreateDatasetOperationNode(child_ds, json_obj, op_type, ds));
  }
  return Status::OK();
}

Status Serdes::CreateCelebADatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Fail to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Fail to find decode");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("extensions") != json_obj.end(), "Fail to find extension");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  bool decode = json_obj["decode"];
  std::set<std::string> extension = json_obj["extensions"];
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<CelebANode>(dataset_dir, usage, sampler, decode, extension, cache);
  return Status::OK();
}

Status Serdes::CreateCifar10DatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Fail to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<Cifar10Node>(dataset_dir, usage, sampler, cache);
  return Status::OK();
}

Status Serdes::CreateCifar100DatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Fail to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<Cifar100Node>(dataset_dir, usage, sampler, cache);
  return Status::OK();
}

Status Serdes::CreateCLUEDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("task") != json_obj.end(), "Fail to find task");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Fail to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_samples") != json_obj.end(), "Fail to find num_samples");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Fail to find shuffle");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_shards") != json_obj.end(), "Fail to find num_shards");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_id") != json_obj.end(), "Fail to find shard_id");
  std::vector<std::string> dataset_files = json_obj["dataset_dir"];
  std::string task = json_obj["task"];
  std::string usage = json_obj["usage"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<CLUENode>(dataset_files, task, usage, num_samples, shuffle, num_shards, shard_id, cache);
  return Status::OK();
}

Status Serdes::CreateCocoDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("annotation_file") != json_obj.end(), "Fail to find annotation_file");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("task") != json_obj.end(), "Fail to find task");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Fail to find decode");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string annotation_file = json_obj["annotation_file"];
  std::string task = json_obj["task"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  // default value for cache and extra_metadata - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  bool extra_metadata = false;
  *ds = std::make_shared<CocoNode>(dataset_dir, annotation_file, task, decode, sampler, cache, extra_metadata);
  return Status::OK();
}

Status Serdes::CreateCSVDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_files") != json_obj.end(), "Fail to find dataset_files");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("field_delim") != json_obj.end(), "Fail to find field_delim");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("column_names") != json_obj.end(), "Fail to find column_names");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_samples") != json_obj.end(), "Fail to find num_samples");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Fail to find shuffle");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_shards") != json_obj.end(), "Fail to find num_shards");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_id") != json_obj.end(), "Fail to find shard_id");
  std::vector<std::string> dataset_files = json_obj["dataset_files"];
  std::string field_delim = json_obj["field_delim"];
  std::vector<std::shared_ptr<CsvBase>> column_defaults = {};
  std::vector<std::string> column_names = json_obj["column_names"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<CSVNode>(dataset_files, field_delim.c_str()[0], column_defaults, column_names, num_samples,
                                  shuffle, num_shards, shard_id, cache);
  return Status::OK();
}

Status Serdes::CreateImageFolderDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Fail to find decode");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("extensions") != json_obj.end(), "Fail to find extension");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("class_indexing") != json_obj.end(), "Fail to find class_indexing");
  std::string dataset_dir = json_obj["dataset_dir"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  // This arg exists in ImageFolderOp, but not externalized (in Python API). The default value is false.
  bool recursive = false;
  std::set<std::string> extension = json_obj["extensions"];
  std::map<std::string, int32_t> class_indexing;
  nlohmann::json class_map = json_obj["class_indexing"];
  for (const auto &class_map_child : class_map) {
    std::string class_ = class_map_child[0];
    int32_t indexing = class_map_child[1];
    class_indexing.insert({class_, indexing});
  }
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<ImageFolderNode>(dataset_dir, decode, sampler, recursive, extension, class_indexing, cache);
  return Status::OK();
}

Status Serdes::CreateManifestDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_file") != json_obj.end(), "Fail to find dataset_file");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Fail to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("class_indexing") != json_obj.end(), "Fail to find class_indexing");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Fail to find decode");
  std::string dataset_file = json_obj["dataset_file"];
  std::string usage = json_obj["usage"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  std::map<std::string, int32_t> class_indexing;
  nlohmann::json class_map = json_obj["class_indexing"];
  for (const auto &class_map_child : class_map) {
    std::string class_ = class_map_child[0];
    int32_t indexing = class_map_child[1];
    class_indexing.insert({class_, indexing});
  }
  bool decode = json_obj["decode"];
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<ManifestNode>(dataset_file, usage, sampler, class_indexing, decode, cache);
  return Status::OK();
}

Status Serdes::CreateMnistDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Fail to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<MnistNode>(dataset_dir, usage, sampler, cache);
  return Status::OK();
}

Status Serdes::CreateTextFileDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_files") != json_obj.end(), "Fail to find dataset_files");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_samples") != json_obj.end(), "Fail to find num_samples");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Fail to find shuffle");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_shards") != json_obj.end(), "Fail to find num_shards");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_id") != json_obj.end(), "Fail to find shard_id");
  std::vector<std::string> dataset_files = json_obj["dataset_files"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<TextFileNode>(dataset_files, num_samples, shuffle, num_shards, shard_id, cache);
  return Status::OK();
}

Status Serdes::CreateTFRecordDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_files") != json_obj.end(), "Fail to find dataset_files");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("schema") != json_obj.end(), "Fail to find schema");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("columns_list") != json_obj.end(), "Fail to find columns_list");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_samples") != json_obj.end(), "Fail to find num_samples");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Fail to find shuffle");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_shards") != json_obj.end(), "Fail to find num_shards");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_id") != json_obj.end(), "Fail to find shard_id");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_equal_rows") != json_obj.end(), "Fail to find shard_equal_rows");
  std::vector<std::string> dataset_files = json_obj["dataset_files"];
  std::string schema = json_obj["schema"];
  std::vector<std::string> columns_list = json_obj["columns_list"];
  int64_t num_samples = json_obj["num_samples"];
  ShuffleMode shuffle = static_cast<ShuffleMode>(json_obj["shuffle"]);
  int32_t num_shards = json_obj["num_shards"];
  int32_t shard_id = json_obj["shard_id"];
  bool shard_equal_rows = json_obj["shard_equal_rows"];
  // default value for cache - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  *ds = std::make_shared<TFRecordNode>(dataset_files, schema, columns_list, num_samples, shuffle, num_shards, shard_id,
                                       shard_equal_rows, cache);
  return Status::OK();
}

Status Serdes::CreateVOCDatasetNode(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Fail to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("task") != json_obj.end(), "Fail to find task");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Fail to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("class_indexing") != json_obj.end(), "Fail to find class_indexing");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Fail to find decode");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Fail to find sampler");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string task = json_obj["task"];
  std::string usage = json_obj["usage"];
  std::map<std::string, int32_t> class_indexing;
  nlohmann::json class_map = json_obj["class_indexing"];
  for (const auto &class_map_child : class_map) {
    std::string class_ = class_map_child[0];
    int32_t indexing = class_map_child[1];
    class_indexing.insert({class_, indexing});
  }
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(ConstructSampler(json_obj["sampler"], &sampler));
  // default value for cache and extra_metadata - to_json function does not have the output
  std::shared_ptr<DatasetCache> cache = nullptr;
  bool extra_metadata = false;
  *ds = std::make_shared<VOCNode>(dataset_dir, task, usage, class_indexing, decode, sampler, cache, extra_metadata);
  return Status::OK();
}

Status Serdes::CreateDatasetNode(nlohmann::json json_obj, std::string op_type, std::shared_ptr<DatasetNode> *ds) {
  if (op_type == kCelebANode) {
    RETURN_IF_NOT_OK(CreateCelebADatasetNode(json_obj, ds));
  } else if (op_type == kCifar10Node) {
    RETURN_IF_NOT_OK(CreateCifar10DatasetNode(json_obj, ds));
  } else if (op_type == kCifar100Node) {
    RETURN_IF_NOT_OK(CreateCifar100DatasetNode(json_obj, ds));
  } else if (op_type == kCLUENode) {
    RETURN_IF_NOT_OK(CreateCLUEDatasetNode(json_obj, ds));
  } else if (op_type == kCocoNode) {
    RETURN_IF_NOT_OK(CreateCocoDatasetNode(json_obj, ds));
  } else if (op_type == kCSVNode) {
    RETURN_IF_NOT_OK(CreateCSVDatasetNode(json_obj, ds));
  } else if (op_type == kImageFolderNode) {
    RETURN_IF_NOT_OK(CreateImageFolderDatasetNode(json_obj, ds));
  } else if (op_type == kManifestNode) {
    RETURN_IF_NOT_OK(CreateManifestDatasetNode(json_obj, ds));
  } else if (op_type == kMnistNode) {
    RETURN_IF_NOT_OK(CreateMnistDatasetNode(json_obj, ds));
  } else if (op_type == kTextFileNode) {
    RETURN_IF_NOT_OK(CreateTextFileDatasetNode(json_obj, ds));
  } else if (op_type == kTFRecordNode) {
    RETURN_IF_NOT_OK(CreateTFRecordDatasetNode(json_obj, ds));
  } else if (op_type == kVOCNode) {
    RETURN_IF_NOT_OK(CreateVOCDatasetNode(json_obj, ds));
  } else {
    return Status(StatusCode::kMDUnexpectedError, op_type + " is not supported");
  }
  return Status::OK();
}

Status Serdes::CreateBatchOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                        std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("batch_size") != json_obj.end(), "Fail to find batch_size");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("drop_remainder") != json_obj.end(), "Fail to find drop_remainder");
  int32_t batch_size = json_obj["batch_size"];
  bool drop_remainder = json_obj["drop_remainder"];
  *result = std::make_shared<BatchNode>(ds, batch_size, drop_remainder);
  return Status::OK();
}

Status Serdes::CreateMapOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                      std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("input_columns") != json_obj.end(), "Fail to find input_columns");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("output_columns") != json_obj.end(), "Fail to find output_columns");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("project_columns") != json_obj.end(), "Fail to find project_columns");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("operations") != json_obj.end(), "Fail to find operations");
  std::vector<std::string> input_columns = json_obj["input_columns"];
  std::vector<std::string> output_columns = json_obj["output_columns"];
  std::vector<std::string> project_columns = json_obj["project_columns"];
  std::vector<std::shared_ptr<TensorOperation>> operations;
  RETURN_IF_NOT_OK(ConstructTensorOps(json_obj["operations"], &operations));
  *result = std::make_shared<MapNode>(ds, operations, input_columns, output_columns, project_columns);
  return Status::OK();
}

Status Serdes::CreateProjectOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                          std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("columns") != json_obj.end(), "Fail to find columns");
  std::vector<std::string> columns = json_obj["columns"];
  *result = std::make_shared<ProjectNode>(ds, columns);
  return Status::OK();
}

Status Serdes::CreateRenameOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                         std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("input_columns") != json_obj.end(), "Fail to find input_columns");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("output_columns") != json_obj.end(), "Fail to find output_columns");
  std::vector<std::string> input_columns = json_obj["input_columns"];
  std::vector<std::string> output_columns = json_obj["output_columns"];
  *result = std::make_shared<RenameNode>(ds, input_columns, output_columns);
  return Status::OK();
}

Status Serdes::CreateRepeatOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                         std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("count") != json_obj.end(), "Fail to find count");
  int32_t count = json_obj["count"];
  *result = std::make_shared<RepeatNode>(ds, count);
  return Status::OK();
}

Status Serdes::CreateShuffleOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                          std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("buffer_size") != json_obj.end(), "Fail to find buffer_size");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("reshuffle_each_epoch") != json_obj.end(),
                               "Fail to find reshuffle_each_epoch");
  int32_t buffer_size = json_obj["buffer_size"];
  bool reset_every_epoch = json_obj["reshuffle_each_epoch"];
  *result = std::make_shared<ShuffleNode>(ds, buffer_size, reset_every_epoch);
  return Status::OK();
}

Status Serdes::CreateSkipOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                       std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("count") != json_obj.end(), "Fail to find count");
  int32_t count = json_obj["count"];
  *result = std::make_shared<SkipNode>(ds, count);
  return Status::OK();
}

Status Serdes::CreateTakeOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj,
                                       std::shared_ptr<DatasetNode> *result) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("count") != json_obj.end(), "Fail to find count");
  int32_t count = json_obj["count"];
  *result = std::make_shared<TakeNode>(ds, count);
  return Status::OK();
}

Status Serdes::CreateDatasetOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj, std::string op_type,
                                          std::shared_ptr<DatasetNode> *result) {
  if (op_type == kBatchNode) {
    RETURN_IF_NOT_OK(CreateBatchOperationNode(ds, json_obj, result));
  } else if (op_type == kMapNode) {
    RETURN_IF_NOT_OK(CreateMapOperationNode(ds, json_obj, result));
  } else if (op_type == kProjectNode) {
    RETURN_IF_NOT_OK(CreateProjectOperationNode(ds, json_obj, result));
  } else if (op_type == kRenameNode) {
    RETURN_IF_NOT_OK(CreateRenameOperationNode(ds, json_obj, result));
  } else if (op_type == kRepeatNode) {
    RETURN_IF_NOT_OK(CreateRepeatOperationNode(ds, json_obj, result));
  } else if (op_type == kShuffleNode) {
    RETURN_IF_NOT_OK(CreateShuffleOperationNode(ds, json_obj, result));
  } else if (op_type == kSkipNode) {
    RETURN_IF_NOT_OK(CreateSkipOperationNode(ds, json_obj, result));
  } else if (op_type == kTakeNode) {
    RETURN_IF_NOT_OK(CreateTakeOperationNode(ds, json_obj, result));
  } else {
    return Status(StatusCode::kMDUnexpectedError, op_type + " operation is not supported");
  }
  return Status::OK();
}

Status Serdes::ConstructDistributedSampler(nlohmann::json json_obj, int64_t num_samples,
                                           std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_shards") != json_obj.end(), "Fail to find num_shards");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shard_id") != json_obj.end(), "Fail to find shard_id");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Fail to find shuffle");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("seed") != json_obj.end(), "Fail to find seed");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("offset") != json_obj.end(), "Fail to find offset");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("even_dist") != json_obj.end(), "Fail to find even_dist");
  int64_t num_shards = json_obj["num_shards"];
  int64_t shard_id = json_obj["shard_id"];
  bool shuffle = json_obj["shuffle"];
  uint32_t seed = json_obj["seed"];
  int64_t offset = json_obj["offset"];
  bool even_dist = json_obj["even_dist"];
  *sampler =
    std::make_shared<DistributedSamplerObj>(num_shards, shard_id, shuffle, num_samples, seed, offset, even_dist);
  if (json_obj.find("child_sampler") != json_obj.end()) {
    std::shared_ptr<SamplerObj> parent_sampler = *sampler;
    RETURN_IF_NOT_OK(ChildSamplerFromJson(json_obj, parent_sampler, sampler));
  }
  return Status::OK();
}

Status Serdes::ConstructPKSampler(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_val") != json_obj.end(), "Fail to find num_val");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("shuffle") != json_obj.end(), "Fail to find shuffle");
  int64_t num_val = json_obj["num_val"];
  bool shuffle = json_obj["shuffle"];
  *sampler = std::make_shared<PKSamplerObj>(num_val, shuffle, num_samples);
  if (json_obj.find("child_sampler") != json_obj.end()) {
    std::shared_ptr<SamplerObj> parent_sampler = *sampler;
    RETURN_IF_NOT_OK(ChildSamplerFromJson(json_obj, parent_sampler, sampler));
  }
  return Status::OK();
}

Status Serdes::ConstructRandomSampler(nlohmann::json json_obj, int64_t num_samples,
                                      std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("replacement") != json_obj.end(), "Fail to find replacement");
  bool replacement = json_obj["replacement"];
  *sampler = std::make_shared<RandomSamplerObj>(replacement, num_samples);
  if (json_obj.find("child_sampler") != json_obj.end()) {
    std::shared_ptr<SamplerObj> parent_sampler = *sampler;
    RETURN_IF_NOT_OK(ChildSamplerFromJson(json_obj, parent_sampler, sampler));
  }
  return Status::OK();
}

Status Serdes::ConstructSequentialSampler(nlohmann::json json_obj, int64_t num_samples,
                                          std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("start_index") != json_obj.end(), "Fail to find start_index");
  int64_t start_index = json_obj["start_index"];
  *sampler = std::make_shared<SequentialSamplerObj>(start_index, num_samples);
  if (json_obj.find("child_sampler") != json_obj.end()) {
    std::shared_ptr<SamplerObj> parent_sampler = *sampler;
    RETURN_IF_NOT_OK(ChildSamplerFromJson(json_obj, parent_sampler, sampler));
  }
  return Status::OK();
}

Status Serdes::ConstructSubsetRandomSampler(nlohmann::json json_obj, int64_t num_samples,
                                            std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("indices") != json_obj.end(), "Fail to find indices");
  std::vector<int64_t> indices = json_obj["indices"];
  *sampler = std::make_shared<SubsetRandomSamplerObj>(indices, num_samples);
  if (json_obj.find("child_sampler") != json_obj.end()) {
    std::shared_ptr<SamplerObj> parent_sampler = *sampler;
    RETURN_IF_NOT_OK(ChildSamplerFromJson(json_obj, parent_sampler, sampler));
  }
  return Status::OK();
}

Status Serdes::ConstructWeightedRandomSampler(nlohmann::json json_obj, int64_t num_samples,
                                              std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("replacement") != json_obj.end(), "Fail to find replacement");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("weights") != json_obj.end(), "Fail to find weights");
  bool replacement = json_obj["replacement"];
  std::vector<double> weights = json_obj["weights"];
  *sampler = std::make_shared<WeightedRandomSamplerObj>(weights, num_samples, replacement);
  if (json_obj.find("child_sampler") != json_obj.end()) {
    std::shared_ptr<SamplerObj> parent_sampler = *sampler;
    RETURN_IF_NOT_OK(ChildSamplerFromJson(json_obj, parent_sampler, sampler));
  }
  return Status::OK();
}

Status Serdes::ConstructSampler(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_samples") != json_obj.end(), "Fail to find num_samples");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler_name") != json_obj.end(), "Fail to find sampler_name");
  int64_t num_samples = json_obj["num_samples"];
  std::string sampler_name = json_obj["sampler_name"];
  if (sampler_name == "DistributedSampler") {
    RETURN_IF_NOT_OK(ConstructDistributedSampler(json_obj, num_samples, sampler));
  } else if (sampler_name == "PKSampler") {
    RETURN_IF_NOT_OK(ConstructPKSampler(json_obj, num_samples, sampler));
  } else if (sampler_name == "RandomSampler") {
    RETURN_IF_NOT_OK(ConstructRandomSampler(json_obj, num_samples, sampler));
  } else if (sampler_name == "SequentialSampler") {
    RETURN_IF_NOT_OK(ConstructSequentialSampler(json_obj, num_samples, sampler));
  } else if (sampler_name == "SubsetRandomSampler") {
    RETURN_IF_NOT_OK(ConstructSubsetRandomSampler(json_obj, num_samples, sampler));
  } else if (sampler_name == "WeightedRandomSampler") {
    RETURN_IF_NOT_OK(ConstructWeightedRandomSampler(json_obj, num_samples, sampler));
  } else {
    return Status(StatusCode::kMDUnexpectedError, sampler_name + "Sampler is not supported");
  }
  return Status::OK();
}

Status Serdes::ChildSamplerFromJson(nlohmann::json json_obj, std::shared_ptr<SamplerObj> parent_sampler,
                                    std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("child_sampler") != json_obj.end(), "Fail to find child_sampler");
  for (nlohmann::json child : json_obj["child_sampler"]) {
    std::shared_ptr<SamplerObj> child_sampler;
    RETURN_IF_NOT_OK(ConstructSampler(child, &child_sampler));
    parent_sampler.get()->AddChildSampler(child_sampler);
  }
  return Status::OK();
}

Status Serdes::BoundingBoxAugmentFromJson(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("transform") != op_params.end(), "Fail to find transform");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("ratio") != op_params.end(), "Fail to find ratio");
  std::vector<std::shared_ptr<TensorOperation>> transforms;
  std::vector<nlohmann::json> json_operations = {};
  json_operations.push_back(op_params["transform"]);
  RETURN_IF_NOT_OK(ConstructTensorOps(json_operations, &transforms));
  float ratio = op_params["ratio"];
  CHECK_FAIL_RETURN_UNEXPECTED(transforms.size() == 1,
                               "Expect size one of transforms parameter, but got:" + std::to_string(transforms.size()));
  *operation = std::make_shared<vision::BoundingBoxAugmentOperation>(transforms[0], ratio);
  return Status::OK();
}

Status Serdes::RandomSelectSubpolicyFromJson(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("policy") != op_params.end(), "Fail to find policy");
  nlohmann::json policy_json = op_params["policy"];
  std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> policy;
  std::vector<std::pair<std::shared_ptr<TensorOperation>, double>> policy_items;
  for (nlohmann::json item : policy_json) {
    for (nlohmann::json item_pair : item) {
      CHECK_FAIL_RETURN_UNEXPECTED(item_pair.find("prob") != item_pair.end(), "Fail to find prob");
      CHECK_FAIL_RETURN_UNEXPECTED(item_pair.find("tensor_op") != item_pair.end(), "Fail to find tensor_op");
      std::vector<std::shared_ptr<TensorOperation>> operations;
      std::pair<std::shared_ptr<TensorOperation>, double> policy_pair;
      std::shared_ptr<TensorOperation> operation;
      nlohmann::json tensor_op_json;
      double prob = item_pair["prob"];
      tensor_op_json.push_back(item_pair["tensor_op"]);
      RETURN_IF_NOT_OK(ConstructTensorOps(tensor_op_json, &operations));
      CHECK_FAIL_RETURN_UNEXPECTED(operations.size() == 1, "There should be only 1 tensor operation");
      policy_pair = std::make_pair(operations[0], prob);
      policy_items.push_back(policy_pair);
    }
    policy.push_back(policy_items);
  }
  *operation = std::make_shared<vision::RandomSelectSubpolicyOperation>(policy);
  return Status::OK();
}

Status Serdes::UniformAugFromJson(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("transforms") != op_params.end(), "Fail to find transforms");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("num_ops") != op_params.end(), "Fail to find num_ops");
  std::vector<std::shared_ptr<TensorOperation>> transforms = {};
  RETURN_IF_NOT_OK(ConstructTensorOps(op_params["transforms"], &transforms));
  int32_t num_ops = op_params["num_ops"];
  *operation = std::make_shared<vision::UniformAugOperation>(transforms, num_ops);
  return Status::OK();
}

Status Serdes::ConstructTensorOps(nlohmann::json operations, std::vector<std::shared_ptr<TensorOperation>> *result) {
  std::vector<std::shared_ptr<TensorOperation>> output;
  for (auto op : operations) {
    CHECK_FAIL_RETURN_UNEXPECTED(op.find("tensor_op_name") != op.end(), "Fail to find tensor_op_name");
    CHECK_FAIL_RETURN_UNEXPECTED(op.find("tensor_op_params") != op.end(), "Fail to find tensor_op_params");
    std::string op_name = op["tensor_op_name"];
    nlohmann::json op_params = op["tensor_op_params"];
    std::shared_ptr<TensorOperation> operation = nullptr;
    CHECK_FAIL_RETURN_UNEXPECTED(func_ptr_.find(op_name) != func_ptr_.end(), "Fail to find " + op_name);
    RETURN_IF_NOT_OK(func_ptr_[op_name](op_params, &operation));
    output.push_back(operation);
  }
  *result = output;
  return Status::OK();
}

std::map<std::string, Status (*)(nlohmann::json json_obj, std::shared_ptr<TensorOperation> *operation)>
Serdes::InitializeFuncPtr() {
  std::map<std::string, Status (*)(nlohmann::json json_obj, std::shared_ptr<TensorOperation> * operation)> ops_ptr;
  ops_ptr[vision::kAffineOperation] = &(vision::AffineOperation::from_json);
  ops_ptr[vision::kAutoContrastOperation] = &(vision::AutoContrastOperation::from_json);
  ops_ptr[vision::kBoundingBoxAugmentOperation] = &(BoundingBoxAugmentFromJson);
  ops_ptr[vision::kCenterCropOperation] = &(vision::CenterCropOperation::from_json);
  ops_ptr[vision::kCutMixBatchOperation] = &(vision::CutMixBatchOperation::from_json);
  ops_ptr[vision::kCutOutOperation] = &(vision::CutOutOperation::from_json);
  ops_ptr[vision::kDecodeOperation] = &(vision::DecodeOperation::from_json);
  ops_ptr[vision::kEqualizeOperation] = &(vision::EqualizeOperation::from_json);
  ops_ptr[vision::kGaussianBlurOperation] = &(vision::GaussianBlurOperation::from_json);
  ops_ptr[vision::kHorizontalFlipOperation] = &(vision::HorizontalFlipOperation::from_json);
  ops_ptr[vision::kHwcToChwOperation] = &(vision::HwcToChwOperation::from_json);
  ops_ptr[vision::kInvertOperation] = &(vision::InvertOperation::from_json);
  ops_ptr[vision::kMixUpBatchOperation] = &(vision::MixUpBatchOperation::from_json);
  ops_ptr[vision::kNormalizeOperation] = &(vision::NormalizeOperation::from_json);
  ops_ptr[vision::kNormalizePadOperation] = &(vision::NormalizePadOperation::from_json);
  ops_ptr[vision::kPadOperation] = &(vision::PadOperation::from_json);
  ops_ptr[vision::kRandomAffineOperation] = &(vision::RandomAffineOperation::from_json);
  ops_ptr[vision::kRandomColorOperation] = &(vision::RandomColorOperation::from_json);
  ops_ptr[vision::kRandomColorAdjustOperation] = &(vision::RandomColorAdjustOperation::from_json);
  ops_ptr[vision::kRandomCropDecodeResizeOperation] = &(vision::RandomCropDecodeResizeOperation::from_json);
  ops_ptr[vision::kRandomCropOperation] = &(vision::RandomCropOperation::from_json);
  ops_ptr[vision::kRandomCropWithBBoxOperation] = &(vision::RandomCropWithBBoxOperation::from_json);
  ops_ptr[vision::kRandomHorizontalFlipOperation] = &(vision::RandomHorizontalFlipOperation::from_json);
  ops_ptr[vision::kRandomHorizontalFlipWithBBoxOperation] = &(vision::RandomHorizontalFlipWithBBoxOperation::from_json);
  ops_ptr[vision::kRandomPosterizeOperation] = &(vision::RandomPosterizeOperation::from_json);
  ops_ptr[vision::kRandomResizeOperation] = &(vision::RandomResizeOperation::from_json);
  ops_ptr[vision::kRandomResizeWithBBoxOperation] = &(vision::RandomResizeWithBBoxOperation::from_json);
  ops_ptr[vision::kRandomResizedCropOperation] = &(vision::RandomResizedCropOperation::from_json);
  ops_ptr[vision::kRandomResizedCropWithBBoxOperation] = &(vision::RandomResizedCropWithBBoxOperation::from_json);
  ops_ptr[vision::kRandomRotationOperation] = &(vision::RandomRotationOperation::from_json);
  ops_ptr[vision::kRandomSelectSubpolicyOperation] = &(RandomSelectSubpolicyFromJson);
  ops_ptr[vision::kRandomSharpnessOperation] = &(vision::RandomSharpnessOperation::from_json);
  ops_ptr[vision::kRandomSolarizeOperation] = &(vision::RandomSolarizeOperation::from_json);
  ops_ptr[vision::kRandomVerticalFlipOperation] = &(vision::RandomVerticalFlipOperation::from_json);
  ops_ptr[vision::kRandomVerticalFlipWithBBoxOperation] = &(vision::RandomVerticalFlipWithBBoxOperation::from_json);
  ops_ptr[vision::kRandomSharpnessOperation] = &(vision::RandomSharpnessOperation::from_json);
  ops_ptr[vision::kRandomSolarizeOperation] = &(vision::RandomSolarizeOperation::from_json);
  ops_ptr[vision::kRescaleOperation] = &(vision::RescaleOperation::from_json);
  ops_ptr[vision::kResizeOperation] = &(vision::ResizeOperation::from_json);
  ops_ptr[vision::kResizePreserveAROperation] = &(vision::ResizePreserveAROperation::from_json);
  ops_ptr[vision::kResizeWithBBoxOperation] = &(vision::ResizeWithBBoxOperation::from_json);
  ops_ptr[vision::kRgbaToBgrOperation] = &(vision::RgbaToBgrOperation::from_json);
  ops_ptr[vision::kRgbaToRgbOperation] = &(vision::RgbaToRgbOperation::from_json);
  ops_ptr[vision::kRotateOperation] = &(vision::RotateOperation::from_json);
  ops_ptr[vision::kSoftDvppDecodeRandomCropResizeJpegOperation] =
    &(vision::SoftDvppDecodeRandomCropResizeJpegOperation::from_json);
  ops_ptr[vision::kSoftDvppDecodeResizeJpegOperation] = &(vision::SoftDvppDecodeResizeJpegOperation::from_json);
  ops_ptr[vision::kSwapRedBlueOperation] = &(vision::SwapRedBlueOperation::from_json);
  ops_ptr[vision::kUniformAugOperation] = &(UniformAugFromJson);
  return ops_ptr;
}

}  // namespace dataset
}  // namespace mindspore
