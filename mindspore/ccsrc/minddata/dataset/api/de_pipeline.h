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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_API_DE_PIPELINE_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_API_DE_PIPELINE_H_

#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "minddata/dataset/core/client.h"  // DE client
#include "minddata/dataset/engine/dataset_iterator.h"
#include "minddata/dataset/util/status.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace mindspore {
namespace dataset {
using json = nlohmann::json;
using DsOpPtr = std::shared_ptr<DatasetOp>;

class CacheClient;

// enum for the dataset operator names
enum OpName {
  kShuffle,
  kMindrecord,
  kBatch,
  kBucketBatch,
  kBarrier,
  kCache,
  kRepeat,
  kSkip,
  kTake,
  kZip,
  kConcat,
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
  kCoco,
  kCifar10,
  kCifar100,
  kCelebA,
  kRandomData,
  kTextFile,
  kBuildVocab,
  kClue,
  kEpochCtrl,
  kSentencePieceVocab,
  kCsv
};

// The C++ binder class that we expose to the python script.
class DEPipeline {
 public:
  DEPipeline();

  ~DEPipeline();

  // Function to add a Node to the Execution Tree.
  Status AddNodeToTree(const OpName &op_name, const py::dict &args, py::dict *output);

  // Function to add a child and parent relationship.
  static Status AddChildToParentNode(const DsOpPtr &child_op, const DsOpPtr &parent_op);

  // Function to assign the node as root.
  Status AssignRootNode(const DsOpPtr &dataset_op);

  // Function to launch the tree execution.
  Status LaunchTreeExec(int32_t num_epochs);

  // Get a row of data as dictionary of column name to the value.
  Status GetNextAsMap(py::dict *output);

  // Get a row of data as list.
  Status GetNextAsList(py::list *output);

  Status GetOutputShapes(py::list *output);

  Status GetOutputTypes(py::list *output);

  Status SaveDataset(const std::vector<std::string> &file_names, const std::string &file_type);

  int GetDatasetSize() const;

  int GetBatchSize() const;

  int GetRepeatCount() const;

  Status ParseShuffleOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseMindRecordOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  template <typename T, typename S>
  Status TransfromTensor(const unsigned char *src, const TensorShape &shape, const int64_t num_of_elements,
                         std::unique_ptr<T> *data, std::unique_ptr<std::vector<uint8_t>> *data_ptr,
                         std::unique_ptr<S> *s, bool need_convert = false);

  Status FetchMetaFromTensorRow(const std::unordered_map<std::string, int32_t> &column_name_id_map,
                                const TensorRow &row, json *schema, std::vector<std::string> *index_fields);

  Status FetchDataFromTensorRow(const TensorRow &row,
                                const std::unordered_map<std::string, int32_t> &column_name_id_map, json *row_raw_data,
                                std::map<std::string, std::unique_ptr<std::vector<uint8_t>>> *row_bin_data);

  Status BuildMindrecordSamplerChain(const py::handle &handle,
                                     std::vector<std::shared_ptr<mindrecord::ShardOperator>> *operators,
                                     int num_padded);

  Status ParseMapOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseFilterOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseRepeatOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseSkipOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseBatchOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseBucketBatchByLengthOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                    std::shared_ptr<DatasetOp> *bottom);

  Status ParseEpochCtrlOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseBatchOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr);

  Status ParseBarrierOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseGeneratorOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseRenameOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseTakeOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseZipOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseConcatOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseDeviceQueueOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseTFReaderOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseProjectOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseImageFolderOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseManifestOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseVOCOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseCocoOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseCifar10Op(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseCifar100Op(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseRandomDataOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  void PrintTree();

  int32_t GetNumClasses() const;

  Status ParseMnistOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status SetBatchParameters(const py::dict &args);

  Status ParseCelebAOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseTextFileOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseBuildVocabOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status StopSend();
  Status ParseBuildSentencePieceVocabOp(const py::dict &args, std::shared_ptr<DatasetOp> *top,
                                        std::shared_ptr<DatasetOp> *bottom);

  Status ParseClueOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

  Status ParseCsvOp(const py::dict &args, std::shared_ptr<DatasetOp> *top, std::shared_ptr<DatasetOp> *bottom);

 private:
  // Execution tree that links the dataset operators.
  std::shared_ptr<ExecutionTree> tree_;

  std::unique_ptr<DatasetIterator> iterator_;

  static Status ParsePadInfo(py::handle value, PadInfo *pad_info);

  /// \brief Helper function to inject a cache operator over top of the current operation being built.
  /// \param[in] cache_client The client to use for caching
  /// \param[in] num_workers The number of workers to use in the cache op
  /// \param[in] input_op The operator to build the cache on top of
  /// \param[out] cache_op The top node of the created subtree (subtree contains two nodes). In this case it will be
  ///     the cache operator
  /// \return Status return code
  Status AddCacheOp(std::shared_ptr<CacheClient> cache_client, int num_workers, std::shared_ptr<DatasetOp> input_op,
                    std::shared_ptr<DatasetOp> *cache_op);

  /// \brief Helper function to inject a shuffle operator over top of the current operation being built.
  /// \param[in] shuffle_size The size to use in the shuffle buffer
  /// \param[in] input_op The operator to build shuffle on top of
  /// \param[out] shuffle_op The top node of the created subtree (subtree contains two nodes). In this case it will be
  ///     the shuffle operator
  /// \return Status return code
  Status AddShuffleOp(int64_t shuffle_size, std::shared_ptr<DatasetOp> input_op,
                      std::shared_ptr<DatasetOp> *shuffle_op);

  /// \brief Helper function to compute the shuffle size
  /// \param[in] num_files The number of files in the dataset
  /// \param[in] num_devices The number of devices in the dataset
  /// \param[in] num_rows The number of rows in the dataset
  /// \param[in] total_rows An upper bound on the total rows in the dataset
  /// \param[out] shuffle_size The resultant computed shuffle size
  /// \return Status return code
  Status ComputeShuffleSize(int64_t num_files, int64_t num_devices, int64_t num_rows, int64_t total_rows,
                            int64_t *shuffle_size);

  int batch_size_;
  int repeat_num_;
  int num_rows_;
  int num_classes_;

  int temp_batch_size_;
  bool temp_drop_remainder_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_API_DE_PIPELINE_H_
