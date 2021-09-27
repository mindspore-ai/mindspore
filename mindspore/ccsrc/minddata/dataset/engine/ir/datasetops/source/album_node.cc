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

#include "minddata/dataset/engine/ir/datasetops/source/album_node.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/album_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for AlbumNode
AlbumNode::AlbumNode(const std::string &dataset_dir, const std::string &data_schema,
                     const std::vector<std::string> &column_names, bool decode,
                     const std::shared_ptr<SamplerObj> &sampler, const std::shared_ptr<DatasetCache> &cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      schema_path_(data_schema),
      column_names_(column_names),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> AlbumNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<AlbumNode>(dataset_dir_, schema_path_, column_names_, decode_, sampler, cache_);
  return node;
}

void AlbumNode::Print(std::ostream &out) const {
  out << (Name() + "(cache:" + ((cache_ != nullptr) ? "true" : "false") + ")");
}

Status AlbumNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("AlbumNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetFilesParam("AlbumNode", {schema_path_}));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("AlbumNode", sampler_));

  if (!column_names_.empty()) {
    RETURN_IF_NOT_OK(ValidateDatasetColumnParam("AlbumNode", "column_names", column_names_));
  }

  return Status::OK();
}

// Function to build AlbumNode
Status AlbumNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->LoadSchemaFile(schema_path_, column_names_));

  // Argument that is not exposed to user in the API.
  std::set<std::string> extensions = {".json", ".JSON"};
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto album_op = std::make_shared<AlbumOp>(num_workers_, dataset_dir_, connector_que_size_, decode_, extensions,
                                            std::move(schema), std::move(sampler_rt));
  album_op->SetTotalRepeats(GetTotalRepeats());
  album_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(album_op);
  return Status::OK();
}

// Get the shard id of node
Status AlbumNode::GetShardId(int32_t *const shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status AlbumNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                 int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size = -1;
  int64_t num_rows = 0;
  // iterate over the files in the directory and count files to initiate num_rows
  Path folder(dataset_dir_);
  std::shared_ptr<Path::DirIterator> dirItr = Path::DirIterator::OpenDirectory(&folder);
  if (!folder.Exists() || dirItr == nullptr) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open folder: " + dataset_dir_);
  }
  std::set<std::string> extensions = {".json", ".JSON"};

  while (dirItr->HasNext()) {
    Path file = dirItr->Next();
    if (extensions.empty() || extensions.find(file.Extension()) != extensions.end()) {
      num_rows += 1;
    }
  }
  // give sampler the total number of files and check if num_samples is smaller
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  // We cache dataset size so as to not duplicated run
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status AlbumNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["decode"] = decode_;
  args["data_schema"] = schema_path_;
  args["column_names"] = column_names_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status AlbumNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_parallel_workers") != json_obj.end(),
                               "Failed to find num_parallel_workers");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Failed to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("data_schema") != json_obj.end(), "Failed to find data_schema");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("column_names") != json_obj.end(), "Failed to find column_names");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Failed to find decode");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Failed to find sampler");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string data_schema = json_obj["data_schema"];
  std::vector<std::string> column_names = json_obj["column_names"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<AlbumNode>(dataset_dir, data_schema, column_names, decode, sampler, cache);
  (void)((*ds)->SetNumWorkers(json_obj["num_parallel_workers"]));
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
