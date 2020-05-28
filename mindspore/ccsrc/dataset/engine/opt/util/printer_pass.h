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

#ifndef DATASET_ENGINE_OPT_PASS_UTIL_PRINTER_H
#define DATASET_ENGINE_OPT_PASS_UTIL_PRINTER_H

#include <memory>
#include "dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class PrinterPass : public NodePass {
 public:
  Status RunOnNode(std::shared_ptr<DatasetOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<BatchOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<MapOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<ProjectOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<RenameOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<FilterOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<SkipOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<ShuffleOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<GeneratorOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<MindRecordOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<TFReaderOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<TakeOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<ZipOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *modified) override;

  Status RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *modified) override;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_ENGINE_OPT_PASS_UTIL_PRINTER_H
