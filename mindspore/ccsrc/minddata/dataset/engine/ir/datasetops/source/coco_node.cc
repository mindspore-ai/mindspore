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

#include "minddata/dataset/engine/ir/datasetops/source/coco_node.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/engine/datasetops/source/coco_op.h"

#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

// Constructor for CocoNode
CocoNode::CocoNode(const std::string &dataset_dir, const std::string &annotation_file, const std::string &task,
                   const bool &decode, const std::shared_ptr<SamplerObj> &sampler, std::shared_ptr<DatasetCache> cache)
    : MappableSourceNode(std::move(cache)),
      dataset_dir_(dataset_dir),
      annotation_file_(annotation_file),
      task_(task),
      decode_(decode),
      sampler_(sampler) {}

std::shared_ptr<DatasetNode> CocoNode::Copy() {
  std::shared_ptr<SamplerObj> sampler = (sampler_ == nullptr) ? nullptr : sampler_->SamplerCopy();
  auto node = std::make_shared<CocoNode>(dataset_dir_, annotation_file_, task_, decode_, sampler, cache_);
  return node;
}

void CocoNode::Print(std::ostream &out) const { out << Name(); }

Status CocoNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  RETURN_IF_NOT_OK(ValidateDatasetDirParam("CocoNode", dataset_dir_));

  RETURN_IF_NOT_OK(ValidateDatasetSampler("CocoNode", sampler_));

  Path annotation_file(annotation_file_);
  if (!annotation_file.Exists()) {
    std::string err_msg = "CocoNode: annotation_file is invalid or does not exist.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateStringValue("CocoNode", task_, {"Detection", "Stuff", "Panoptic", "Keypoint"}));

  return Status::OK();
}

// Function to build CocoNode
Status CocoNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  CocoOp::TaskType task_type = CocoOp::TaskType::Detection;
  if (task_ == "Detection") {
    task_type = CocoOp::TaskType::Detection;
  } else if (task_ == "Stuff") {
    task_type = CocoOp::TaskType::Stuff;
  } else if (task_ == "Keypoint") {
    task_type = CocoOp::TaskType::Keypoint;
  } else if (task_ == "Panoptic") {
    task_type = CocoOp::TaskType::Panoptic;
  } else {
    std::string err_msg = "Task type:'" + task_ + "' is not supported.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
  RETURN_IF_NOT_OK(
    schema->AddColumn(ColDescriptor(std::string("image"), DataType(DataType::DE_UINT8), TensorImpl::kFlexible, 1)));
  switch (task_type) {
    case CocoOp::TaskType::Detection:
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("bbox"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("category_id"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("iscrowd"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case CocoOp::TaskType::Stuff:
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("segmentation"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("iscrowd"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case CocoOp::TaskType::Keypoint:
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("keypoints"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("num_keypoints"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    case CocoOp::TaskType::Panoptic:
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("bbox"), DataType(DataType::DE_FLOAT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("category_id"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(schema->AddColumn(
        ColDescriptor(std::string("iscrowd"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      RETURN_IF_NOT_OK(
        schema->AddColumn(ColDescriptor(std::string("area"), DataType(DataType::DE_UINT32), TensorImpl::kFlexible, 1)));
      break;
    default:
      std::string err_msg = "CocoNode::Build : Invalid task type";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<SamplerRT> sampler_rt = nullptr;
  RETURN_IF_NOT_OK(sampler_->SamplerBuild(&sampler_rt));

  std::shared_ptr<CocoOp> op =
    std::make_shared<CocoOp>(task_type, dataset_dir_, annotation_file_, num_workers_, rows_per_buffer_,
                             connector_que_size_, decode_, std::move(schema), std::move(sampler_rt));
  op->set_total_repeats(GetTotalRepeats());
  op->set_num_repeats_per_epoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);

  return Status::OK();
}

// Get the shard id of node
Status CocoNode::GetShardId(int32_t *shard_id) {
  *shard_id = sampler_->ShardId();

  return Status::OK();
}

// Get Dataset size
Status CocoNode::GetDatasetSize(const std::shared_ptr<DatasetSizeGetter> &size_getter, bool estimate,
                                int64_t *dataset_size) {
  if (dataset_size_ > 0) {
    *dataset_size = dataset_size_;
    return Status::OK();
  }
  int64_t num_rows = 0, sample_size;
  RETURN_IF_NOT_OK(CocoOp::CountTotalRows(dataset_dir_, annotation_file_, task_, &num_rows));
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

Status CocoNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args, sampler_args;
  RETURN_IF_NOT_OK(sampler_->to_json(&sampler_args));
  args["sampler"] = sampler_args;
  args["num_parallel_workers"] = num_workers_;
  args["dataset_dir"] = dataset_dir_;
  args["annotation_file"] = annotation_file_;
  args["task"] = task_;
  args["decode"] = decode_;
  if (cache_ != nullptr) {
    nlohmann::json cache_args;
    RETURN_IF_NOT_OK(cache_->to_json(&cache_args));
    args["cache"] = cache_args;
  }
  *out_json = args;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
