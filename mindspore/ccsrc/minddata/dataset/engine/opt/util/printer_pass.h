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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_UTIL_PRINTER_H
#define MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_UTIL_PRINTER_H

#include <memory>
#include "minddata/dataset/engine/opt/pass.h"

namespace mindspore {
namespace dataset {

class PrinterPass : public NodePass {
 public:
  Status RunOnNode(std::shared_ptr<DatasetOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<BatchOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<MapOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<ProjectOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<RenameOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<SkipOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<ShuffleOp> node, bool *const modified) override;

#ifndef ENABLE_ANDROID
  Status RunOnNode(std::shared_ptr<MindRecordOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<TFReaderOp> node, bool *const modified) override;
#endif

#ifdef ENABLE_PYTHON
  Status RunOnNode(std::shared_ptr<FilterOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<GeneratorOp> node, bool *const modified) override;
#endif

  Status RunOnNode(std::shared_ptr<TakeOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<ZipOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<DeviceQueueOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<ImageFolderOp> node, bool *const modified) override;

  Status RunOnNode(std::shared_ptr<AlbumOp> node, bool *const modified) override;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_ENGINE_OPT_PASS_UTIL_PRINTER_H
