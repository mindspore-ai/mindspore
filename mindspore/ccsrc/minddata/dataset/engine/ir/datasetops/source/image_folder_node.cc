/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/ir/datasetops/source/image_folder_node.h"

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/image_folder_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

#ifdef ENABLE_PYTHON
ImageFolderNode::ImageFolderNode(std::string dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler,
                                 bool recursive, std::set<std::string> extensions,
                                 std::map<std::string, int32_t> class_indexing,
                                 std::shared_ptr<DatasetCache> cache = nullptr, py::function decrypt)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      decode_(decode),
      sampler_(sampler),
      recursive_(recursive),
      class_indexing_(class_indexing),
      exts_(extensions),
      decrypt_(decrypt) {}
#else
ImageFolderNode::ImageFolderNode(std::string dataset_dir, bool decode, std::shared_ptr<SamplerObj> sampler,
                                 bool recursive, std::set<std::string> extensions,
                                 std::map<std::string, int32_t> class_indexing,
                                 std::shared_ptr<DatasetCache> cache = nullptr)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      decode_(decode),
      sampler_(sampler),
      recursive_(recursive),
      class_indexing_(class_indexing),
      exts_(extensions) {}
#endif

std::shared_ptr<DatasetNode> ImageFolderNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
#ifdef ENABLE_PYTHON
  auto node = std::make_shared<ImageFolderNode>(dataset_dir_, decode_, sampler, recursive_, exts_, class_indexing_,
                                                cache_, decrypt_);
#else
  auto node =
    std::make_shared<ImageFolderNode>(dataset_dir_, decode_, sampler, recursive_, exts_, class_indexing_, cache_);
#endif
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void ImageFolderNode::Print(std::ostream &out) const {
  out << (Name() + "(path:" + dataset_dir_ + ",decode:" + (decode_ ? "true" : "false") + ",...)");
}

Status ImageFolderNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("ImageFolderDataset", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("ImageFolderDataset", sampler_));

  return Status::OK();
}

Status ImageFolderNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // Do internal Schema generation.
  // This arg is exist in ImageFolderOp, but not externalized (in Python API).
  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  TensorShape scalar = TensorShape::CreateScalar();
  RETURN_IF_NOT_OK(schema->AddColumn(ColDescriptor("image", DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor("label", DataType(DataType::DE_INT32), TensorImpl::kFlexible, 0, &scalar)));
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

#ifdef ENABLE_PYTHON
  auto op = std::make_shared<ImageFolderOp>(num_workers_, dataset_dir_, connector_que_size_, recursive_, decode_, exts_,
                                            class_indexing_, std::move(schema), std::move(sampler_rt), decrypt_);
#else
  auto op = std::make_shared<ImageFolderOp>(num_workers_, dataset_dir_, connector_que_size_, recursive_, decode_, exts_,
                                            class_indexing_, std::move(schema), std::move(sampler_rt));
#endif
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Get the shard id of node
Status ImageFolderNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status ImageFolderNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                       int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t sample_size, num_rows;
  RETURN_IF_NOT_OK(ImageFolderOp::CountRowsAndClasses(dataset_dir_, exts_, &num_rows, nullptr, {}));
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

Status ImageFolderNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["recursive"] = recursive_;
  args["decode"] = decode_;
  args["extensions"] = exts_;
  args["class_indexing"] = class_indexing_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status ImageFolderNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kImageFolderNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kImageFolderNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kImageFolderNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "decode", kImageFolderNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kImageFolderNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "recursive", kImageFolderNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "extensions", kImageFolderNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "class_indexing", kImageFolderNode));
  std::string dataset_dir = json_obj["dataset_dir"];
  bool decode = json_obj["decode"];
  std::shared_ptr<SamplerObj> sampler;
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  bool recursive = json_obj["recursive"];
  std::set<std::string> extension = json_obj["extensions"];
  std::map<std::string, int32_t> class_indexing;
  nlohmann::json class_map = json_obj["class_indexing"];
  for (const auto &class_map_child : class_map) {
    std::string class_ = class_map_child[0];
    int32_t indexing = class_map_child[1];
    class_indexing.insert({class_, indexing});
  }
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<ImageFolderNode>(dataset_dir, decode, sampler, recursive, extension, class_indexing, cache);
  (void)(*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  (void)(*ds)->SetConnectorQueueSize(json_obj["connector_queue_size"]);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
