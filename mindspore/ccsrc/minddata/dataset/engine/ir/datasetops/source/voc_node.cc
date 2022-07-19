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

#include "minddata/dataset/engine/ir/datasetops/source/voc_node.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/voc_op.h"
#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#endif

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

#ifdef ENABLE_PYTHON
// Constructor for VOCNode
VOCNode::VOCNode(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                 const std::map<std::string, int32_t> &class_indexing, bool decode, std::shared_ptr<SamplerObj> sampler,
                 std::shared_ptr<DatasetCache> cache, bool extra_metadata, py::function decrypt)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      task_(task),
      usage_(usage),
      class_index_(class_indexing),
      decode_(decode),
      sampler_(sampler),
      extra_metadata_(extra_metadata),
      decrypt_(decrypt) {}
#else
// Constructor for VOCNode
VOCNode::VOCNode(const std::string &dataset_dir, const std::string &task, const std::string &usage,
                 const std::map<std::string, int32_t> &class_indexing, bool decode, std::shared_ptr<SamplerObj> sampler,
                 std::shared_ptr<DatasetCache> cache, bool extra_metadata)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      task_(task),
      usage_(usage),
      class_index_(class_indexing),
      decode_(decode),
      sampler_(sampler),
      extra_metadata_(extra_metadata) {}
#endif

std::shared_ptr<DatasetNode> VOCNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
#ifdef ENABLE_PYTHON
  auto node = std::make_shared<VOCNode>(dataset_dir_, task_, usage_, class_index_, decode_, sampler, cache_,
                                        extra_metadata_, decrypt_);
#else
  auto node =
    std::make_shared<VOCNode>(dataset_dir_, task_, usage_, class_index_, decode_, sampler, cache_, extra_metadata_);
#endif
  (void)node->SetNumWorkers(num_workers_);
  (void)node->SetConnectorQueueSize(connector_que_size_);
  return node;
}

void VOCNode::Print(std::ostream &out) const { out << Name(); }

Status VOCNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  Path dir(dataset_dir_);

  RETURN_IF_NOT_OK(ValidateDatasetDirParam("VOCDataset", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("VOCDataset", sampler_));

  if (task_ == "Segmentation") {
    if (!class_index_.empty()) {
      std::string err_msg = "VOCDataset: 'class_indexing' is invalid in Segmentation task.";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    Path imagesets_file = dir / "ImageSets" / "Segmentation" / usage_ + ".txt";
    if (!imagesets_file.Exists()) {
      std::string err_msg = "VOCDataset: Invalid 'usage': " + usage_ + ", file does not exist";
      MS_LOG(ERROR) << "VOCDataset: Invalid 'usage': " << usage_ << ", file \"" << imagesets_file
                    << "\" does not exist!";
      return Status(StatusCode::kMDSyntaxError, err_msg);
    }
  } else if (task_ == "Detection") {
    Path imagesets_file = dir / "ImageSets" / "Main" / usage_ + ".txt";
    if (!imagesets_file.Exists()) {
      std::string err_msg = "VOCDataset: Invalid 'usage': " + usage_ + ", file does not exist";
      MS_LOG(ERROR) << "VOCDataset: Invalid 'usage': " << usage_ << ", file \"" << imagesets_file
                    << "\" does not exist!";
      return Status(StatusCode::kMDSyntaxError, err_msg);
    }
  } else {
    std::string err_msg = "VOCDataset: Invalid 'task': " + task_ + ", expected Segmentation or Detection.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

// Function to build VOCNode
Status VOCNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto schema = std::make_unique<DataSchema>();
  VOCOp::TaskType task_type_;

  if (task_ == "Segmentation") {
    task_type_ = VOCOp::TaskType::Segmentation;
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string(kColumnTarget), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  } else if (task_ == "Detection") {
    task_type_ = VOCOp::TaskType::Detection;
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string(kColumnImage), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string(kColumnBbox), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string(kColumnLabel), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string(kColumnDifficult), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(std::string(kColumnTruncate), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
  }
  if (extra_metadata_) {
    std::string meta_file_name = std::string(kDftMetaColumnPrefix) + std::string(kColumnFileName);
    TensorShape scalar = TensorShape::CreateScalar();
    RETURN_IF_NOT_OK(schema->AddColumn(
      ColDescriptor(meta_file_name, DataType(DataType::DE_STRING), TensorImpl::kFlexible, 0, &scalar)));
  }
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  std::shared_ptr<VOCOp> voc_op;
#ifdef ENABLE_PYTHON
  voc_op = std::make_shared<VOCOp>(task_type_, usage_, dataset_dir_, class_index_, num_workers_, connector_que_size_,
                                   decode_, std::move(schema), std::move(sampler_rt), extra_metadata_, decrypt_);

#else
  voc_op = std::make_shared<VOCOp>(task_type_, usage_, dataset_dir_, class_index_, num_workers_, connector_que_size_,
                                   decode_, std::move(schema), std::move(sampler_rt), extra_metadata_);
#endif
  voc_op->SetTotalRepeats(GetTotalRepeats());
  voc_op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(voc_op);

  return Status::OK();
}

// Get the shard id of node
Status VOCNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status VOCNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                               int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows = 0, sample_size;
  std::vector<std::shared_ptr<DatasetOp>> ops;
  RETURN_IF_NOT_OK(Build(&ops));
  CHECK_FAIL_RETURN_UNEXPECTED(!ops.empty(), "[Internal ERROR] Unable to build VocOp.");
  auto op = std::dynamic_pointer_cast<VOCOp>(ops.front());
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

Status VOCNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["connector_queue_size"] = connector_que_size_;
  args["dataset_dir"] = dataset_dir_;
  args["task"] = task_;
  args["usage"] = usage_;
  args["class_indexing"] = class_index_;
  args["decode"] = decode_;
  args["extra_metadata"] = extra_metadata_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}

#ifndef ENABLE_ANDROID
Status VOCNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> *ds) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_parallel_workers", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "connector_queue_size", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "dataset_dir", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "task", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "usage", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "class_indexing", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "decode", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "sampler", kTFRecordNode));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "extra_metadata", kTFRecordNode));
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
  RETURN_IF_NOT_OK(Serdes::ConstructSampler(json_obj["sampler"], &sampler));
  bool extra_metadata = json_obj["extra_metadata"];
  std::shared_ptr<DatasetCache> cache = nullptr;
  RETURN_IF_NOT_OK(DatasetCache::from_json(json_obj, &cache));
  *ds = std::make_shared<VOCNode>(dataset_dir, task, usage, class_indexing, decode, sampler, cache, extra_metadata);
  (*ds)->SetNumWorkers(json_obj["num_parallel_workers"]);
  return Status::OK();
}
#endif
}  // namespace dataset
}  // namespace mindspore
