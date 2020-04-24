/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef DATASET_API_DE_PIPELINE_H_
#define DATASET_API_DE_PIPELINE_H_

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "dataset/core/client.h"  // DE client
#include "dataset/engine/dataset_iterator.h"
#include "dataset/util/status.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {
using DsOpPtr = std::shared_ptr<DatasetOp>;

// enum for the dataset operator names
enum OpName {
  kStorage = 0,
  kShuffle,
  kMindrecord,
  kBatch,
  kBarrier,
  kCache,
  kRepeat,
  kSkip,
  kTake,
  kZip,
  kMap,
  kFilter,
  kDeviceQueue,
  kGenerator,
  kRename,
  kTfReader,
  kProject,
  kImageFolder,
  kMnist,
  kManifest,
  kVoc,
  kCifar10,
  kCifar100,
  kCelebA,
  kTextFile
};

// The C++ binder class that we expose to the python script.
class DEPipeline {
 public:
  DEPipeline();

  ~DEPipeline();

  // Function to add a Node to the Execution Tree.
  Status AddNodeToTree(const OpName &op_name, const py::dict &args, DsOpPtr *out);

  // Function to add a child and parent relationship.
  static Status AddChildToParentNode(const DsOpPtr &child_op, const DsOpPtr &parent_op);

  // Function to assign the node as root.
  Status AssignRootNode(const DsOpPtr &dataset_op);

  // Function to launch the tree execution.
  Status LaunchTreeExec();

  // Get a row of data as dictionary of column name to the value.
  Status GetNextAsMap(py::dict *output);

  // Get a row of data as list.
  Status GetNextAsList(py::list *output);

  Status GetOutputShapes(py::list *output);

  Status GetOutputTypes(py::list *output);

  int GetDatasetSize() const;

  int GetBatchSize() const;

  int GetRepeatCount() const;

  Status ParseStorageOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseShuffleOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status CheckMindRecordPartitionInfo(const py::dict &args, std::vector<int> *ptr);

  Status ParseMindRecordOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseMapOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseFilterOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseRepeatOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseSkipOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseBatchOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseBarrierOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseGeneratorOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseRenameOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseTakeOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseZipOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseDeviceQueueOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseTFReaderOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseProjectOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseImageFolderOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseManifestOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseVOCOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseCifar10Op(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseCifar100Op(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  void PrintTree();

  int32_t GetNumClasses() const;

  Status ParseMnistOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status SetBatchParameters(const py::dict &args);

  Status ParseCelebAOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseTextFileOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

 private:
  // Execution tree that links the dataset operators.
  std::shared_ptr<ExecutionTree> tree_;

  std::unique_ptr<DatasetIterator> iterator_;

  // Validate required args passed to storage op.
  Status ValidateArgStorageOp(const py::dict &args);

  int batch_size_;
  int repeat_num_;
  int num_rows_;
  int num_classes_;

  int temp_batch_size_;
  bool temp_drop_remainder_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // DATASET_API_DE_PIPELINE_H_
