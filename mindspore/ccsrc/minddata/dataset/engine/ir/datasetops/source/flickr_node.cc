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

#include "minddata/dataset/engine/ir/datasetops/source/flickr_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/flickr_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif

#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
// Constructor for FlickrNode
FlickrNode::FlickrNode(const std::string &dataset_dir, const std::string &annotation_file, bool decode,
                       std::shared_ptr<SamplerObj> sampler, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      annotation_file_(annotation_file),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> FlickrNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<FlickrNode>(dataset_dir_, annotation_file_, decode_, sampler, cache_);
  return node;
}

void FlickrNode::Print(std::ostream &out) const {
  out << Name() + "(dataset dir:" + dataset_dir_;
  out << ", annotation file:" + annotation_file_;
  if (sampler_ != nullptr) {
    out << ", sampler";
  }
  if (cache_ != nullptr) {
    out << ", cache";
  }
  out << ")";
}

Status FlickrNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("FlickrNode", dataset_dir_));

  if (annotation_file_.empty()) {
    std::string err_msg = "FlickrNode: annotation_file is not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::vector<char> forbidden_symbols = {':', '*', '?', '"', '<', '>', '|', '`', '&', '\'', ';'};
  for (char c : annotation_file_) {
    auto p = std::find(forbidden_symbols.begin(), forbidden_symbols.end(), c);
    if (p != forbidden_symbols.end()) {
      std::string err_msg = "FlickrNode: annotation_file: [" + annotation_file_ + "] should not contain :*?\"<>|`&;\'.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  Path annotation_file(annotation_file_);
  if (!annotation_file.Exists()) {
    std::string err_msg = "FlickrNode: annotation_file: [" + annotation_file_ + "] is invalid or not exist.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetSampler("FlickrNode", sampler_));
  return Status::OK();
}

// Function to build FlickrOp for Flickr
Status FlickrNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  auto schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kCv, 1)));
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("annotation", DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  auto flickr_op = std::make_shared<FlickrOp>(num_workers_, dataset_dir_, annotation_file_, decode_,
                                              connector_que_size_, std::move(schema), std::move(sampler_rt));
  flickr_op->SetTotalRepeats(GetTotalRepeats());
  flickr_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(flickr_op);
  return Status::OK();
}

// Get the shard id of node
Status FlickrNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size
Status FlickrNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                  int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }

  int64_t num_rows, sample_size;
  RETURN_IF_NOT_OK(FlickrOp::CountTotalRows(dataset_dir_, annotation_file_, &num_rows));
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

Status FlickrNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["annotation_file"] = annotation_file_;
  args["decode"] = decode_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

Status FlickrNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_parallel_workers") != json_obj.end(),
                               "Failed to find num_parallel_workers");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("dataset_dir") != json_obj.end(), "Failed to find dataset_dir");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("annotation_file") != json_obj.end(), "Failed to find annotation_file");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("decode") != json_obj.end(), "Failed to find decode");
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string annotation_file = json_obj["annotation_file"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<FlickrNode>(dataset_dir, annotation_file, decode, sampler, cache);
  (*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
