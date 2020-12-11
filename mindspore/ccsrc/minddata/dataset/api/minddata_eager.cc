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

#include <unistd.h>
#include <unordered_map>

#include "minddata/dataset/include/minddata_eager.h"
#include "minddata/dataset/include/vision.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace api {

MindDataEager::MindDataEager(std::vector<std::shared_ptr<dataset::TensorOperation>> ops) : ops_(ops) {}

// Helper function to convert Type from DE to MS
DataType ToMSType(dataset::DataType type) {
  switch (dataset::DataType::Type(type)) {
    case dataset::DataType::DE_BOOL:
      return DataType::kMsBool;
    case dataset::DataType::DE_UINT8:
      return DataType::kMsUint8;
    case dataset::DataType::DE_INT32:
      return DataType::kMsInt32;
    case dataset::DataType::DE_INT64:
      return DataType::kMsInt64;
    case dataset::DataType::DE_FLOAT32:
      return DataType::kMsFloat32;
    default:
      return DataType::kMsUnknown;
  }
}

// Helper function to convert Type from MS to DE
dataset::DataType ToDEType(DataType type) {
  switch (type) {
    case DataType::kMsBool:
      return dataset::DataType(dataset::DataType::DE_BOOL);
    case DataType::kMsUint8:
      return dataset::DataType(dataset::DataType::DE_UINT8);
    case DataType::kMsInt32:
      return dataset::DataType(dataset::DataType::DE_INT32);
    case DataType::kMsInt64:
      return dataset::DataType(dataset::DataType::DE_INT64);
    case DataType::kMsFloat32:
      return dataset::DataType(dataset::DataType::DE_FLOAT32);
    default:
      return dataset::DataType(dataset::DataType::DE_UNKNOWN);
  }
}

Status MindDataEager::LoadImageFromDir(const std::string &image_dir, std::vector<std::shared_ptr<Tensor>> *images) {
  // Check target directory
  dataset::Path image_dir_(image_dir);
  if (!image_dir_.Exists() || !image_dir_.IsDirectory()) {
    std::string err_msg = "Target directory: " + image_dir + " does not exist or not a directory.";
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::FAILED, err_msg);
  }
  if (access(image_dir_.toString().c_str(), R_OK) == -1) {
    std::string err_msg = "No access to target directory: " + image_dir;
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::FAILED, err_msg);
  }

  // Start reading images and constructing tensors
  auto path_itr = dataset::Path::DirIterator::OpenDirectory(&image_dir_);
  while (path_itr->hasNext()) {
    dataset::Path file = path_itr->next();
    std::shared_ptr<dataset::Tensor> image;
    dataset::Tensor::CreateFromFile(file.toString(), &image);

    std::shared_ptr<Tensor> ms_image = std::make_shared<Tensor>("image", DataType(kMsUint8), image->shape().AsVector(),
                                                                image->GetBuffer(), image->SizeInBytes());
    images->push_back(ms_image);
  }

  // Check if read images or not
  if (images->empty()) {
    std::string err_msg = "No images found in target directory: " + image_dir;
    MS_LOG(ERROR) << err_msg;
    return Status(StatusCode::FAILED, err_msg);
  }

  return Status(StatusCode::SUCCESS);
}

std::shared_ptr<Tensor> MindDataEager::operator()(std::shared_ptr<Tensor> input) {
  // Validate ops
  if (ops_.empty()) {
    MS_LOG(ERROR) << "Input TensorOperation should be provided";
    return nullptr;
  }
  for (int32_t i = 0; i < ops_.size(); i++) {
    if (ops_[i] == nullptr) {
      MS_LOG(ERROR) << "Input TensorOperation[" << i << "] is invalid or null";
      return nullptr;
    }
  }
  // Validate input tensor
  if (input == nullptr) {
    MS_LOG(ERROR) << "Input Tensor should not be null";
    return nullptr;
  }

  // Start applying transforms in ops
  std::shared_ptr<dataset::Tensor> de_input;
  dataset::Tensor::CreateFromMemory(dataset::TensorShape(input->Shape()), ToDEType(input->DataType()),
                                    (const uchar *)(input->Data()), &de_input);

  for (int32_t i = 0; i < ops_.size(); i++) {
    // Build runtime op and run
    std::shared_ptr<dataset::Tensor> de_output;
    std::shared_ptr<dataset::TensorOp> transform = ops_[i]->Build();
    dataset::Status rc = transform->Compute(de_input, &de_output);

    // check execution failed
    if (rc.IsError()) {
      MS_LOG(ERROR) << "Operation execution failed : " << rc.ToString();
      return nullptr;
    }

    // For next transform
    de_input = std::move(de_output);
  }

  // Convert DETensor to Tensor
  if (!de_input->HasData()) {
    MS_LOG(ERROR) << "Apply transform failed, output tensor has no data";
    return nullptr;
  }
  std::shared_ptr<Tensor> output =
    std::make_shared<Tensor>("transfomed", ToMSType(de_input->type()), de_input->shape().AsVector(),
                             de_input->GetBuffer(), de_input->SizeInBytes());
  return output;
}

}  // namespace api
}  // namespace mindspore
