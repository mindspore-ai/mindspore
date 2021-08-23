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
  RETURN_UNEXPECTED_IF_NULL(node);
  RETURN_UNEXPECTED_IF_NULL(out_json);
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
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("children") != json_obj.end(), "Failed to find children");
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
    std::vector<std::shared_ptr<DatasetNode>> datasets;
    for (auto child_json_obj : json_obj["children"]) {
      RETURN_IF_NOT_OK(ConstructPipeline(child_json_obj, &child_ds));
      datasets.push_back(child_ds);
    }
    if (json_obj["op_type"] == "Zip") {
      CHECK_FAIL_RETURN_UNEXPECTED(datasets.size() > 1, "Should zip more than 1 dataset");
      RETURN_IF_NOT_OK(ZipNode::from_json(datasets, ds));
    } else if (json_obj["op_type"] == "Concat") {
      CHECK_FAIL_RETURN_UNEXPECTED(datasets.size() > 1, "Should concat more than 1 dataset");
      RETURN_IF_NOT_OK(ConcatNode::from_json(json_obj, datasets, ds));
    } else {
      return Status(StatusCode::kMDUnexpectedError, "Operation is not supported");
    }
  }
  return Status::OK();
}

Status Serdes::CreateNode(std::shared_ptr<DatasetNode> child_ds, nlohmann::json json_obj,
                          std::shared_ptr<DatasetNode> *ds) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("op_type") != json_obj.end(), "Failed to find op_type");
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

Status Serdes::CreateDatasetNode(nlohmann::json json_obj, std::string op_type, std::shared_ptr<DatasetNode> *ds) {
  if (op_type == kAlbumNode) {
    RETURN_IF_NOT_OK(AlbumNode::from_json(json_obj, ds));
  } else if (op_type == kCelebANode) {
    RETURN_IF_NOT_OK(CelebANode::from_json(json_obj, ds));
  } else if (op_type == kCifar10Node) {
    RETURN_IF_NOT_OK(Cifar10Node::from_json(json_obj, ds));
  } else if (op_type == kCifar100Node) {
    RETURN_IF_NOT_OK(Cifar100Node::from_json(json_obj, ds));
  } else if (op_type == kCLUENode) {
    RETURN_IF_NOT_OK(CLUENode::from_json(json_obj, ds));
  } else if (op_type == kCocoNode) {
    RETURN_IF_NOT_OK(CocoNode::from_json(json_obj, ds));
  } else if (op_type == kCSVNode) {
    RETURN_IF_NOT_OK(CSVNode::from_json(json_obj, ds));
  } else if (op_type == kFlickrNode) {
    RETURN_IF_NOT_OK(FlickrNode::from_json(json_obj, ds));
  } else if (op_type == kImageFolderNode) {
    RETURN_IF_NOT_OK(ImageFolderNode::from_json(json_obj, ds));
  } else if (op_type == kManifestNode) {
    RETURN_IF_NOT_OK(ManifestNode::from_json(json_obj, ds));
  } else if (op_type == kMnistNode) {
    RETURN_IF_NOT_OK(MnistNode::from_json(json_obj, ds));
  } else if (op_type == kTextFileNode) {
    RETURN_IF_NOT_OK(TextFileNode::from_json(json_obj, ds));
  } else if (op_type == kTFRecordNode) {
    RETURN_IF_NOT_OK(TFRecordNode::from_json(json_obj, ds));
  } else if (op_type == kVOCNode) {
    RETURN_IF_NOT_OK(VOCNode::from_json(json_obj, ds));
  } else {
    return Status(StatusCode::kMDUnexpectedError, op_type + " is not supported");
  }
  return Status::OK();
}

Status Serdes::CreateDatasetOperationNode(std::shared_ptr<DatasetNode> ds, nlohmann::json json_obj, std::string op_type,
                                          std::shared_ptr<DatasetNode> *result) {
  if (op_type == kBatchNode) {
    RETURN_IF_NOT_OK(BatchNode::from_json(json_obj, ds, result));
  } else if (op_type == kMapNode) {
    RETURN_IF_NOT_OK(MapNode::from_json(json_obj, ds, result));
  } else if (op_type == kProjectNode) {
    RETURN_IF_NOT_OK(ProjectNode::from_json(json_obj, ds, result));
  } else if (op_type == kRenameNode) {
    RETURN_IF_NOT_OK(RenameNode::from_json(json_obj, ds, result));
  } else if (op_type == kRepeatNode) {
    RETURN_IF_NOT_OK(RepeatNode::from_json(json_obj, ds, result));
  } else if (op_type == kShuffleNode) {
    RETURN_IF_NOT_OK(ShuffleNode::from_json(json_obj, ds, result));
  } else if (op_type == kSkipNode) {
    RETURN_IF_NOT_OK(SkipNode::from_json(json_obj, ds, result));
  } else if (op_type == kTransferNode) {
    RETURN_IF_NOT_OK(TransferNode::from_json(json_obj, ds, result));
  } else if (op_type == kTakeNode) {
    RETURN_IF_NOT_OK(TakeNode::from_json(json_obj, ds, result));
  } else {
    return Status(StatusCode::kMDUnexpectedError, op_type + " operation is not supported");
  }
  return Status::OK();
}

Status Serdes::ConstructSampler(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *sampler) {
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("num_samples") != json_obj.end(), "Failed to find num_samples");
  CHECK_FAIL_RETURN_UNEXPECTED(json_obj.find("sampler_name") != json_obj.end(), "Failed to find sampler_name");
  int64_t num_samples = json_obj["num_samples"];
  std::string sampler_name = json_obj["sampler_name"];
  if (sampler_name == "DistributedSampler") {
    RETURN_IF_NOT_OK(DistributedSamplerObj::from_json(json_obj, num_samples, sampler));
  } else if (sampler_name == "PKSampler") {
    RETURN_IF_NOT_OK(PKSamplerObj::from_json(json_obj, num_samples, sampler));
  } else if (sampler_name == "RandomSampler") {
    RETURN_IF_NOT_OK(RandomSamplerObj::from_json(json_obj, num_samples, sampler));
  } else if (sampler_name == "SequentialSampler") {
    RETURN_IF_NOT_OK(SequentialSamplerObj::from_json(json_obj, num_samples, sampler));
  } else if (sampler_name == "SubsetSampler") {
    RETURN_IF_NOT_OK(SubsetSamplerObj::from_json(json_obj, num_samples, sampler));
  } else if (sampler_name == "SubsetRandomSampler") {
    RETURN_IF_NOT_OK(SubsetRandomSamplerObj::from_json(json_obj, num_samples, sampler));
  } else if (sampler_name == "WeightedRandomSampler") {
    RETURN_IF_NOT_OK(WeightedRandomSamplerObj::from_json(json_obj, num_samples, sampler));
  } else {
    return Status(StatusCode::kMDUnexpectedError, sampler_name + "Sampler is not supported");
  }
  return Status::OK();
}

Status Serdes::ConstructTensorOps(nlohmann::json json_obj, std::vector<std::shared_ptr<TensorOperation>> *result) {
  std::vector<std::shared_ptr<TensorOperation>> output;
  for (nlohmann::json item : json_obj) {
    CHECK_FAIL_RETURN_UNEXPECTED(item.find("is_python_front_end_op") == item.end(),
                                 "python operation is not yet supported");
    CHECK_FAIL_RETURN_UNEXPECTED(item.find("tensor_op_name") != item.end(), "Failed to find tensor_op_name");
    CHECK_FAIL_RETURN_UNEXPECTED(item.find("tensor_op_params") != item.end(), "Failed to find tensor_op_params");
    std::string op_name = item["tensor_op_name"];
    nlohmann::json op_params = item["tensor_op_params"];
    std::shared_ptr<TensorOperation> operation = nullptr;
    CHECK_FAIL_RETURN_UNEXPECTED(func_ptr_.find(op_name) != func_ptr_.end(), "Failed to find " + op_name);
    RETURN_IF_NOT_OK(func_ptr_[op_name](op_params, &operation));
    output.push_back(operation);
  }
  *result = output;
  return Status::OK();
}

std::map<std::string, Status (*)(nlohmann::json json_obj, std::shared_ptr<TensorOperation> *operation)>
Serdes::InitializeFuncPtr() {
  std::map<std::string, Status (*)(nlohmann::json json_obj, std::shared_ptr<TensorOperation> * operation)> ops_ptr;
  ops_ptr[vision::kAdjustGammaOperation] = &(vision::AdjustGammaOperation::from_json);
  ops_ptr[vision::kAffineOperation] = &(vision::AffineOperation::from_json);
  ops_ptr[vision::kAutoContrastOperation] = &(vision::AutoContrastOperation::from_json);
  ops_ptr[vision::kBoundingBoxAugmentOperation] = &(vision::BoundingBoxAugmentOperation::from_json);
  ops_ptr[vision::kCenterCropOperation] = &(vision::CenterCropOperation::from_json);
  ops_ptr[vision::kCropOperation] = &(vision::CropOperation::from_json);
  ops_ptr[vision::kCutMixBatchOperation] = &(vision::CutMixBatchOperation::from_json);
  ops_ptr[vision::kCutOutOperation] = &(vision::CutOutOperation::from_json);
  ops_ptr[vision::kDecodeOperation] = &(vision::DecodeOperation::from_json);
#ifdef ENABLE_ACL
  ops_ptr[vision::kDvppCropJpegOperation] = &(vision::DvppCropJpegOperation::from_json);
  ops_ptr[vision::kDvppDecodeResizeOperation] = &(vision::DvppDecodeResizeOperation::from_json);
  ops_ptr[vision::kDvppDecodeResizeCropOperation] = &(vision::DvppDecodeResizeCropOperation::from_json);
  ops_ptr[vision::kDvppNormalizeOperation] = &(vision::DvppNormalizeOperation::from_json);
  ops_ptr[vision::kDvppResizeJpegOperation] = &(vision::DvppResizeJpegOperation::from_json);
#endif
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
  ops_ptr[vision::kRandomSelectSubpolicyOperation] = &(vision::RandomSelectSubpolicyOperation::from_json);
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
  ops_ptr[vision::kRgbToBgrOperation] = &(vision::RgbToBgrOperation::from_json);
  ops_ptr[vision::kRgbToGrayOperation] = &(vision::RgbToGrayOperation::from_json);
  ops_ptr[vision::kRotateOperation] = &(vision::RotateOperation::from_json);
  ops_ptr[vision::kSlicePatchesOperation] = &(vision::SlicePatchesOperation::from_json);
  ops_ptr[vision::kSoftDvppDecodeRandomCropResizeJpegOperation] =
    &(vision::SoftDvppDecodeRandomCropResizeJpegOperation::from_json);
  ops_ptr[vision::kSoftDvppDecodeResizeJpegOperation] = &(vision::SoftDvppDecodeResizeJpegOperation::from_json);
  ops_ptr[vision::kSwapRedBlueOperation] = &(vision::SwapRedBlueOperation::from_json);
  ops_ptr[vision::kUniformAugOperation] = &(vision::UniformAugOperation::from_json);
  ops_ptr[vision::kVerticalFlipOperation] = &(vision::VerticalFlipOperation::from_json);
  ops_ptr[transforms::kFillOperation] = &(transforms::FillOperation::from_json);
  ops_ptr[transforms::kOneHotOperation] = &(transforms::OneHotOperation::from_json);
  ops_ptr[transforms::kTypeCastOperation] = &(transforms::TypeCastOperation::from_json);
  ops_ptr[text::kToNumberOperation] = &(text::ToNumberOperation::from_json);
  return ops_ptr;
}

}  // namespace dataset
}  // namespace mindspore
