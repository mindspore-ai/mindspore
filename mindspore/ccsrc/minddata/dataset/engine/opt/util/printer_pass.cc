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

#include <memory>
#include "minddata/dataset/engine/opt/util/printer_pass.h"

namespace mindspore {
namespace dataset {

Status PrinterPass::RunOnNode(std::shared_ptr<DatasetOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting DatasetOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<BatchOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting BatchOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<MapOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting MapOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<ProjectOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting ProjectOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<RenameOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting RenameOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<SkipOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting SkipOp" << '\n';
  return Status::OK();
}
Status PrinterPass::RunOnNode(std::shared_ptr<ShuffleOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting ShuffleOp" << '\n';
  return Status::OK();
}
#ifndef ENABLE_ANDROID
Status PrinterPass::RunOnNode(std::shared_ptr<MindRecordOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting MindRecordOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<TFReaderOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting TFReaderOp" << '\n';
  return Status::OK();
}
#endif

#ifdef ENABLE_PYTHON
Status PrinterPass::RunOnNode(std::shared_ptr<FilterOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting FilterOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<GeneratorOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting GeneratorOp" << '\n';
  return Status::OK();
}
#endif

Status PrinterPass::RunOnNode(std::shared_ptr<TakeOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting TakeOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<ZipOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting ZipOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting DeviceQueueOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting ImageFolderOp" << '\n';
  return Status::OK();
}

Status PrinterPass::RunOnNode(std::shared_ptr<AlbumOp> node, bool *const modified) {
  *modified = false;
  std::cout << "Visiting ImageFolderOp" << '\n';
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
