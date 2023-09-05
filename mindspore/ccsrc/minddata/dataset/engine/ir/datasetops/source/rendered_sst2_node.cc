/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/engine/ir/datasetops/source/rendered_sst2_node.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/rendered_sst2_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const std::set<std::string> kExts = {".png"};

RenderedSST2Node::RenderedSST2Node(const std::string &dataset_dir, const std::string &usage, bool decode,
                                   const std::shared_ptr<SamplerObj> &sampler, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      usage_(usage),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> RenderedSST2Node::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<RenderedSST2Node>(dataset_dir_, usage_, decode_, sampler, cache_);
  return node;
}

void RenderedSST2Node::Print(std::ostream &out) const {
  out << (Name() + "(path: " + dataset_dir_ + ", decode: " + (decode_ ? "true" : "false") + ")");
}

Status RenderedSST2Node::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("RenderedSST2Node", dataset_dir_));
  RETURN_IF_NOT_OK(ValidateStringValue("RenderedSST2Node", usage_, {"val", "train", "all", "test"}));
  RETURN_IF_NOT_OK(ValidateDatasetSampler("RenderedSST2Node", sampler_));
  return Status::OK();
}

Status RenderedSST2Node::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  // This arg exists in RenderedSST2Op, but is not externalized (in Python API).
  RETURN_UNEXPECTED_IF_NULL(node_ops);
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  const std::map<std::string, uint32_t> kClassIndex = {};
  auto op = std::make_shared<RenderedSST2Op>(num_workers_, dataset_dir_, usage_, connector_que_size_, decode_, kExts,
                                             kClassIndex, std::move(schema), std::move(sampler_rt));
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Get the shard id of node.
Status RenderedSST2Node::GetShardId(int32_t *shard_id) {
  RETURN_UNEXPECTED_IF_NULL(shard_id);
  *shard_id = sampler_->ShardId();
  return Status::OK();
}

// Get Dataset size.
Status RenderedSST2Node::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                        int64_t *dataset_size) {
  RETURN_UNEXPECTED_IF_NULL(dataset_size);
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size;
  int64_t num_rows;
  RETURN_IF_NOT_OK(RenderedSST2Op::CountRowsAndClasses(dataset_dir_, usage_, kExts, &num_rows, nullptr));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));
  sample_size = sampler_rt->CalculateNumSamples(num_rows);
  RETURN_UNEXPECTED_IF_NULL(size_getter);
  if (sample_size == -1) {
    RETURN_IF_NOT_OK(size_getter->DryRun(shared_from_this(), &sample_size));
  }
  *dataset_size = sample_size;
  dataset_size_ = *dataset_size;
  return Status::OK();
}

Status RenderedSST2Node::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  nlohmann::json sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["usage"] = usage_;
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
Status RenderedSST2Node::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kRenderedSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kRenderedSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "usage", kRenderedSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "decode", kRenderedSST2Node));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kRenderedSST2Node));
  std::string dataset_dir = json_obj["dataset_dir"];
  std::string usage = json_obj["usage"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<RenderedSST2Node>(dataset_dir, usage, decode, sampler, cache);
  (*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
