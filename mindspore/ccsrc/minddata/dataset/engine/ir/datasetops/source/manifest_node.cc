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

#include "minddata/dataset/engine/ir/datasetops/source/manifest_node.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/manifest_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

ManifestNode::ManifestNode(const std::string &dataset_file, const std::string &usage,
                           const std::shared_ptr<SamplerObj> &sampler,
                           const std::map<std::string, int32_t> &class_indexing, bool decode,
                           std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_file_(dataset_file),
      usage_(usage),
      decode_(decode),
      class_index_(class_indexing),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> ManifestNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<ManifestNode>(dataset_file_, usage_, sampler, class_index_, decode_, cache_);
  return node;
}

void ManifestNode::Print(std::ostream &out) const {
  out << (Name() + "(file:" + dataset_file_);
  if (sampler_ != nullptr) {
    out << ",sampler";
  }
  if (cache_ != nullptr) {
    out << ",cache";
  }
  out << ")";
}

Status ManifestNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  std::vector<char> forbidden_symbols = {':', '*', '?', '"', '<', '>', '|', '`', '&', '\'', ';'};
  for (char c : dataset_file_) {
    auto p = std::find(forbidden_symbols.begin(), forbidden_symbols.end(), c);
    if (p != forbidden_symbols.end()) {
      std::string err_msg = "ManifestNode: filename should not contain :*?\"<>|`&;\'";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  Path manifest_file(dataset_file_);
  if (!manifest_file.Exists()) {
    std::string err_msg = "ManifestNode: dataset file: [" + dataset_file_ + "] is invalid or not exist";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (access(dataset_file_.c_str(), R_OK) == -1) {
    std::string err_msg = "ManifestNode: No access to specified annotation file: " + dataset_file_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetSampler("ManifestNode", sampler_));

  RETURN_IF_NOT_OK(ValidateStringValue("ManifestNode", usage_, {"train", "eval", "inference"}));

  return Status::OK();
}

Status ManifestNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));

  std::shared_ptr<ManifestOp> manifest_op;
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  manifest_op = std::make_shared<ManifestOp>(num_workers_, dataset_file_, connector_que_size_, decode_, class_index_,
                                             std::move(schema), std::move(sampler_rt), usage_);
  manifest_op->SetTotalRepeats(GetTotalRepeats());
  manifest_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(manifest_op);

  return Status::OK();
}

// Get the shard id of node
Status ManifestNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status ManifestNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                    int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows, sample_size;
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(Build(&ops));
  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "Unable to build op.");
  auto op = std::dynamic_pointer_cast<ManifestOp>(ops.front());
  RETURN_IF_NOT_OK(op->CountTotalRows(&num_rows));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status ManifestNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_file"] = dataset_file_;
  args["usage"] = usage_;
  args["class_indexing"] = class_index_;
  args["decode"] = decode_;

  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status ManifestNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_parallel_workers") != json_obj.end(),
                               "Failed to find num_parallel_workers");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_file") != json_obj.end(), "Failed to find dataset_file");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("usage") != json_obj.end(), "Failed to find usage");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler") != json_obj.end(), "Failed to find sampler");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("class_indexing") != json_obj.end(), "Failed to find class_indexing");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Failed to find decode");
  std::string dataset_file = json_obj["dataset_file"];
  std::string usage = json_obj["usage"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::map<std::string, int32_t> class_indexing;
  nlohmann::json class_map = json_obj["class_indexing"];
  for (const auto &class_map_child : class_map) {
    std::string class_ = class_map_child[0];
    int32_t indexing = class_map_child[1];
    class_indexing.insert({class_, indexing});
  }
  bool decode = json_obj["decode"];
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<ManifestNode>(dataset_file, usage, sampler, class_indexing, decode, cache);
  (*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
