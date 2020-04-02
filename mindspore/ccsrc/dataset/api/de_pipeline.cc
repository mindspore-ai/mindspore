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
#include "dataset/api/de_pipeline.h"

#include <set>
#include <map>

#include "common/utils.h"
#include "dataset/kernels/py_func_op.h"
#include "dataset/engine/datasetops/source/image_folder_op.h"
#include "dataset/engine/datasetops/source/mnist_op.h"
#include "dataset/engine/datasetops/source/voc_op.h"
#include "dataset/core/tensor.h"
#include "dataset/engine/dataset_iterator.h"
#include "dataset/engine/datasetops/source/manifest_op.h"
#include "dataset/engine/datasetops/source/cifar_op.h"
#include "dataset/engine/datasetops/source/celeba_op.h"
#include "dataset/engine/datasetops/source/text_file_op.h"
#include "dataset/engine/datasetops/filter_op.h"
#include "mindrecord/include/shard_category.h"
#include "mindrecord/include/shard_sample.h"
#include "mindrecord/include/shard_shuffle.h"
#include "dataset/util/random.h"
#include "dataset/util/status.h"
#include "utils/log_adapter.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace dataset {
using pFunction = Status (DEPipeline::*)(const py::dict &, std::shared_ptr<DatasetOp> *);

static std::unordered_map<uint32_t, pFunction> g_parse_op_func_ = {{kStorage, &DEPipeline::ParseStorageOp},
                                                                   {kShuffle, &DEPipeline::ParseShuffleOp},
                                                                   {kMindrecord, &DEPipeline::ParseMindRecordOp},
                                                                   {kMap, &DEPipeline::ParseMapOp},
                                                                   {kFilter, &DEPipeline::ParseFilterOp},
                                                                   {kBatch, &DEPipeline::ParseBatchOp},
                                                                   {kBarrier, &DEPipeline::ParseBarrierOp},
                                                                   {kRepeat, &DEPipeline::ParseRepeatOp},
                                                                   {kSkip, &DEPipeline::ParseSkipOp},
                                                                   {kZip, &DEPipeline::ParseZipOp},
                                                                   {kRename, &DEPipeline::ParseRenameOp},
                                                                   {kDeviceQueue, &DEPipeline::ParseDeviceQueueOp},
                                                                   {kGenerator, &DEPipeline::ParseGeneratorOp},
                                                                   {kTfReader, &DEPipeline::ParseTFReaderOp},
                                                                   {kProject, &DEPipeline::ParseProjectOp},
                                                                   {kTake, &DEPipeline::ParseTakeOp},
                                                                   {kImageFolder, &DEPipeline::ParseImageFolderOp},
                                                                   {kMnist, &DEPipeline::ParseMnistOp},
                                                                   {kManifest, &DEPipeline::ParseManifestOp},
                                                                   {kVoc, &DEPipeline::ParseVOCOp},
                                                                   {kCifar10, &DEPipeline::ParseCifar10Op},
                                                                   {kCifar100, &DEPipeline::ParseCifar100Op},
                                                                   {kCelebA, &DEPipeline::ParseCelebAOp},
                                                                   {kTextFile, &DEPipeline::ParseTextFileOp}};

DEPipeline::DEPipeline() : iterator_(nullptr) {
  try {
    // One time init
    (void)GlobalInit();

    // Instantiate the execution tree
    tree_ = std::make_shared<ExecutionTree>();
    repeat_num_ = 1;
    batch_size_ = 1;
    num_rows_ = 0;
    num_classes_ = 0;
    temp_batch_size_ = 1;
    temp_drop_remainder_ = false;
  } catch (const std::exception &err) {
    MS_LOG(ERROR) << "Dataset pipeline exception caught on init: " << err.what() << ".";
    return;
  }
}

DEPipeline::~DEPipeline() {
  {
    // Release GIL before joining all threads
    py::gil_scoped_release gil_release;
    // Release tree
    tree_.reset();
  }
}

// Function to add a Node to the Execution Tree.
Status DEPipeline::AddNodeToTree(const OpName &op_name, const py::dict &args, DsOpPtr *out) {
  // For each operator, Parse through the list of arguments,
  // then call the respective builder/constructor.
  auto iter = g_parse_op_func_.find(op_name);
  if (iter != g_parse_op_func_.end()) {
    pFunction func = iter->second;
    RETURN_IF_NOT_OK((this->*func)(args, out));
  } else {
    RETURN_STATUS_UNEXPECTED("No such Op");
  }
  // Associate current dataset op node with the tree.
  RETURN_IF_NOT_OK(tree_->AssociateNode(*out));
  return Status::OK();
}
// Function to add a child and parent relationship.
Status DEPipeline::AddChildToParentNode(const DsOpPtr &child_op, const DsOpPtr &parent_op) {
  // Link this relationship.
  // Note parent node takes ownership of the child
  return (parent_op->AddChild(child_op));
}

// Function to assign the node as root.
Status DEPipeline::AssignRootNode(const DsOpPtr &dataset_op) { return (tree_->AssignRoot(dataset_op)); }

// Function to launch the tree execution.
Status DEPipeline::LaunchTreeExec() {
  RETURN_IF_NOT_OK(tree_->Prepare());
  RETURN_IF_NOT_OK(tree_->Launch());
  iterator_ = std::make_unique<DatasetIterator>(tree_);
  if (iterator_ == nullptr) RETURN_STATUS_UNEXPECTED("Cannot create an Iterator.");
  return Status::OK();
}

void DEPipeline::PrintTree() {
  for (auto itr = tree_->begin(); itr != tree_->end(); ++itr) {
    std::stringstream ss;
    ss << *itr;
    MS_LOG(INFO) << "Operator ID is " << itr->id() << ". Details: " << ss.str().c_str() << ".";
  }
}

Status DEPipeline::GetNextAsMap(py::dict *output) {
  TensorMap row;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->GetNextAsMap(&row);
  }
  RETURN_IF_NOT_OK(s);
  // Generate Python dict as return
  for (auto el : row) {
    (*output)[common::SafeCStr(el.first)] = el.second;
  }
  return Status::OK();
}

Status DEPipeline::GetNextAsList(py::list *output) {
  TensorRow row;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->FetchNextTensorRow(&row);
  }
  RETURN_IF_NOT_OK(s);
  // Generate Python list as return
  for (auto el : row) {
    output->append(el);
  }
  return Status::OK();
}

Status DEPipeline::GetOutputShapes(py::list *output) {
  std::vector<TensorShape> shapes;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->GetOutputShapes(&shapes);
  }
  RETURN_IF_NOT_OK(s);
  for (auto el : shapes) {
    py::list shape;
    for (auto dim : el.AsVector()) {
      shape.append(dim);
    }
    output->append(shape);
  }
  return Status::OK();
}

Status DEPipeline::GetOutputTypes(py::list *output) {
  std::vector<DataType> types;
  Status s;
  {
    py::gil_scoped_release gil_release;
    s = iterator_->GetOutputTypes(&types);
  }
  RETURN_IF_NOT_OK(s);
  for (auto el : types) {
    output->append(el.AsNumpyType());
  }
  return Status::OK();
}

int DEPipeline::GetDatasetSize() const { return num_rows_ / batch_size_; }

int DEPipeline::GetBatchSize() const { return batch_size_; }

int DEPipeline::GetRepeatCount() const { return repeat_num_; }

int ToInt(const py::handle &handle) { return py::reinterpret_borrow<py::int_>(handle); }

bool ToBool(const py::handle &handle) { return py::reinterpret_borrow<py::bool_>(handle); }

std::string ToString(const py::handle &handle) { return py::reinterpret_borrow<py::str>(handle); }

std::vector<std::string> ToStringVector(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::vector<std::string> vector;
  for (auto l : list) {
    if (!l.is_none())
      vector.push_back(py::str(l));
    else
      vector.emplace_back("");
  }
  return vector;
}

std::set<std::string> ToStringSet(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::set<std::string> set;
  for (auto l : list) {
    if (!l.is_none()) {
      (void)set.insert(py::str(l));
    }
  }
  return set;
}

std::map<std::string, int32_t> ToStringMap(const py::handle handle) {
  py::dict dict = py::reinterpret_borrow<py::dict>(handle);
  std::map<std::string, int32_t> map;
  for (auto p : dict) {
    (void)map.insert(std::make_pair(ToString(p.first), ToInt(p.second)));
  }
  return map;
}

std::vector<int> ToIntVector(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::vector<int> vector;
  for (auto l : list) {
    if (!l.is_none()) {
      vector.push_back(ToInt(l));
    }
  }
  return vector;
}

std::vector<DataType> ToTypeVector(const py::handle handle) {
  py::list list = py::reinterpret_borrow<py::list>(handle);
  std::vector<DataType> vector;
  for (auto l : list) {
    if (l.is_none()) {
      vector.emplace_back(DataType());
    } else {
      vector.push_back(l.cast<DataType>());
    }
  }
  return vector;
}

Status DEPipeline::SetBatchParameters(const py::dict &args) {
  if (args["batch_size"].is_none()) {
    std::string err_msg = "Error: batchSize is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  temp_batch_size_ = ToInt(args["batch_size"]);
  CHECK_FAIL_RETURN_UNEXPECTED(temp_batch_size_ > 0, "Error: batchSize is invalid.");
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "drop_remainder") {
        temp_drop_remainder_ = ToBool(value);
      }
    }
  }

  return Status::OK();
}

Status DEPipeline::ValidateArgStorageOp(const py::dict &args) {
  // Required arguments
  if (((args.contains("dataset_files") && args["dataset_files"].is_none()) || args["schema"].is_none()) &&
      ((args.contains("dataset_dir") && args["dataset_dir"].is_none()) ||
       (args["schema"].is_none() && args["schema_json_string"].is_none()))) {
    std::string err_msg = "Error: at least one of dataset_files or schema_file is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

Status DEPipeline::ParseStorageOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  RETURN_IF_NOT_OK(ValidateArgStorageOp(args));
  std::shared_ptr<StorageOp::Builder> builder;
  if (args.contains("dataset_files") && !args["dataset_files"].is_none()) {
    builder = std::make_shared<StorageOp::Builder>();
    (void)builder->SetDatasetFileList(ToStringVector(args["dataset_files"]));
    (void)builder->SetSchemaFile(ToString(args["schema"]));
  } else if (args.contains("dataset_dir") && !args["dataset_dir"].is_none()) {
    builder = std::make_shared<StorageOp::Builder>();
    (void)builder->SetDatasetFilesDir(ToString(args["dataset_dir"]));
    if (!args["schema"].is_none()) {
      (void)builder->SetSchemaFile(ToString(args["schema"]));
    } else if (!args["schema_json_string"].is_none()) {
      std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
      std::string s = ToString(args["schema_json_string"]);
      RETURN_IF_NOT_OK(schema->LoadSchemaString(s, std::vector<std::string>()));
      (void)builder->SetNumRows(schema->num_rows());
      (void)builder->SetSchema(std::move(schema));
    }
  }

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "prefetch_size") {
        (void)builder->SetOpConnectorSize(ToInt(value));
      } else if (key == "columns_list") {
        (void)builder->SetColumnsToLoad(ToStringVector(value));
      } else if (key == "distribution") {
        (void)builder->SetDataDistributionFile(ToString(value));
      } else if (key == "labels_filename") {
        (void)builder->setLabelsFileName(ToString(value));
      } else if (key == "dataset_usage") {
        (void)builder->SetDatasetUsage(ToString(value));
      }
    }
  }
  (void)builder->SetBatchSize(temp_batch_size_);
  (void)builder->SetDropRemainder(temp_drop_remainder_);

  std::shared_ptr<StorageOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  num_rows_ = op->num_rows();
  num_classes_ = op->num_classes();
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseShuffleOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::shared_ptr<ShuffleOp::Builder> builder = std::make_shared<ShuffleOp::Builder>();
  if (!args["buffer_size"].is_none()) {
    (void)builder->SetShuffleSize(ToInt(args["buffer_size"]));
  } else {
    std::string err_msg = "Error: Shuffle buffer size is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<ShuffleOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::CheckMindRecordPartitionInfo(const py::dict &args, std::vector<int> *in_partitions) {
  if (args["partitions"].is_none()) {
    std::string err_msg = "Error: partitions is not set (None)";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  py::list list = py::reinterpret_borrow<py::list>(args["partitions"]);
  for (auto l : list) {
    if (!l.is_none()) {
      in_partitions->push_back(ToInt(l));
    }
  }

  if (in_partitions->size() != 2) {
    std::string err_msg = "Error: partitions is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  constexpr int kMaxPartitions = 64;
  if (in_partitions->at(0) <= 0 || in_partitions->at(0) > kMaxPartitions) {
    std::string err_msg = "Error: partitions is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  if (in_partitions->at(1) < 0 || in_partitions->at(1) >= in_partitions->at(0)) {
    std::string err_msg = "Error: partitions is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}

Status DEPipeline::ParseMindRecordOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  if (args["dataset_file"].is_none()) {
    std::string err_msg = "Error: at least one of dataset_files is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::shared_ptr<MindRecordOp::Builder> builder = std::make_shared<MindRecordOp::Builder>();
  (void)builder->SetDatasetFile(ToString(args["dataset_file"]));

  std::vector<std::string> in_col_names;
  if (!args["columns_list"].is_none()) {
    in_col_names = ToStringVector(args["columns_list"]);
    if (in_col_names.empty() || in_col_names[0].empty()) {
      std::string err_msg = "Error: columns_list is invalid or not set.";
      RETURN_STATUS_UNEXPECTED(err_msg);
    }
    (void)builder->SetColumnsToLoad(in_col_names);
  }

  std::vector<std::shared_ptr<mindrecord::ShardOperator>> operators;
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumMindRecordWorkers(ToInt(value));
      } else if (key == "block_reader" && ToBool(value) == true) {
        (void)builder->SetBlockReader();
      } else if (key == "global_shuffle" && ToBool(value) == true) {
        uint32_t seed = args["partitions"].is_none() ? GetSeed() : 0;
        operators.push_back(std::make_shared<mindrecord::ShardShuffle>(seed));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("_create_for_minddataset");
        std::shared_ptr<mindrecord::ShardOperator> sample_op =
          create().cast<std::shared_ptr<mindrecord::ShardOperator>>();
        operators.push_back(sample_op);
      }
    }
  }

  std::vector<int> in_partitions;
  if (!args["partitions"].is_none()) {
    auto ret = CheckMindRecordPartitionInfo(args, &in_partitions);
    if (Status::OK() != ret) {
      return ret;
    }
    operators.push_back(std::make_shared<mindrecord::ShardSample>(1, in_partitions[0], in_partitions[1]));
  }

  if (!operators.empty()) {
    (void)builder->SetOperators(operators);
  }
  std::shared_ptr<MindRecordOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  num_rows_ = op->num_rows();
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseMapOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::shared_ptr<MapOp::Builder> builder = std::make_shared<MapOp::Builder>();
  std::vector<std::shared_ptr<TensorOp>> tensor_op_list;

  if (args["operations"].is_none()) RETURN_STATUS_UNEXPECTED("Error: 'operations' is not set. \n");

  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "input_columns") {
        std::vector<std::string> in_col_names = ToStringVector(args["input_columns"]);
        (void)builder->SetInColNames(in_col_names);
      } else if (key == "output_columns") {
        (void)builder->SetOutColNames(ToStringVector(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "prefetch_size") {
        (void)builder->SetOpConnectorSize(ToInt(value));
      } else if (key == "operations") {
        py::handle tensor_ops = args["operations"];
        // operation can be a list of TensorOps or a single TensorOp.
        if (py::isinstance<py::list>(tensor_ops)) {
          for (auto op : tensor_ops) {
            std::shared_ptr<TensorOp> tensor_op;
            if (py::isinstance<TensorOp>(op)) {
              tensor_op = op.cast<std::shared_ptr<TensorOp>>();
            } else if (py::isinstance<py::function>(op)) {
              tensor_op = std::make_shared<PyFuncOp>(op.cast<py::function>());
            } else {
              RETURN_STATUS_UNEXPECTED("Error: tensor_op is not recognised (not TensorOp and not pyfunc).");
            }
            tensor_op_list.push_back(tensor_op);
          }
        }
        if (tensor_op_list.empty()) RETURN_STATUS_UNEXPECTED("Error: tensor_op is invalid or not set.");
        (void)builder->SetTensorFuncs(std::move(tensor_op_list));
      } else {
        RETURN_STATUS_UNEXPECTED("Error: Unhandled key: " + key);
      }
    }
  }

  std::shared_ptr<MapOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseFilterOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::shared_ptr<FilterOp::Builder> builder = std::make_shared<FilterOp::Builder>();

  if (args["predicate"].is_none()) {
    RETURN_STATUS_UNEXPECTED("Error: 'predicate' is not set. \n");
  }

  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "predicate") {
        py::handle op = args["predicate"];
        if (!py::isinstance<py::function>(op)) {
          RETURN_STATUS_UNEXPECTED("Error: predicate is not recognised (not pyfunc).");
        }
        py::function predicate_func = op.cast<py::function>();
        (void)builder->SetPredicateFunc(std::move(predicate_func));
      } else if (key == "input_columns") {
        std::vector<std::string> in_col_names = ToStringVector(args["input_columns"]);
        (void)builder->SetInColNames(in_col_names);
      } else {
        RETURN_STATUS_UNEXPECTED("Error: Unhandled key: " + key);
      }
    }
  }

  std::shared_ptr<FilterOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseRepeatOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  if (args["count"].is_none()) {
    std::string err_msg = "Error: count is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  repeat_num_ = ToInt(args["count"]);
  std::shared_ptr<RepeatOp> op;
  RETURN_IF_NOT_OK(RepeatOp::Builder(ToInt(args["count"])).Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseSkipOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  if (args["count"].is_none()) {
    std::string err_msg = "Error: count is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<SkipOp> op;
  RETURN_IF_NOT_OK(SkipOp::Builder(ToInt(args["count"])).Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseGeneratorOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::shared_ptr<GeneratorOp::Builder> builder = std::make_shared<GeneratorOp::Builder>();
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "source") {
        py::object obj = py::cast(&value);
        if (!py::isinstance<py::function>(obj)) {
          std::string err_msg = "Error: generator is invalid or not set.";
          RETURN_STATUS_UNEXPECTED(err_msg);
        }
        (void)builder->SetGeneratorFunction(obj.cast<py::function>());
      } else if (key == "column_names") {
        (void)builder->SetColumnNames(ToStringVector(value));
      } else if (key == "column_types") {
        (void)builder->SetColumnTypes(ToTypeVector(value));
      }
    }
  }
  std::shared_ptr<GeneratorOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseBatchOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::shared_ptr<BatchOp::Builder> builder;
  if (py::isinstance<py::int_>(args["batch_size"])) {
    batch_size_ = ToInt(args["batch_size"]);
    CHECK_FAIL_RETURN_UNEXPECTED(batch_size_ > 0, "Error: batch_size is invalid.");
    builder = std::make_shared<BatchOp::Builder>(ToInt(args["batch_size"]));
  } else if (py::isinstance<py::function>(args["batch_size"])) {
    builder = std::make_shared<BatchOp::Builder>(1);
    (void)builder->SetBatchSizeFunc(args["batch_size"].cast<py::function>());
  } else {
    std::string err_msg = "Error: batch_size is neither an Integer nor a python function";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "drop_remainder") {
        (void)builder->SetDrop(ToBool(value));
      }
      if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      }
      if (key == "per_batch_map") {
        (void)builder->SetBatchMapFunc(value.cast<py::function>());
      }
      if (key == "input_columns") {
        (void)builder->SetColumnsToMap(ToStringVector(value));
      }
    }
  }

  std::shared_ptr<BatchOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseBarrierOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::shared_ptr<BarrierOp::Builder> builder = std::make_shared<BarrierOp::Builder>();
  // Right now barrier should only take num_rows_per_buffer = 1
  // The reason for this is because having it otherwise can lead to blocking issues
  // See barrier_op.h for more details
  (void)builder->SetRowsPerBuffer(1);
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "condition_name") {
        (void)builder->SetConditionName(ToString(value));
      } else if (key == "condition_func") {
        (void)builder->SetConditionFunc(value.cast<py::function>());
      }
    }
  }

  std::shared_ptr<BarrierOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseDeviceQueueOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  int32_t prefetch_size = 0;
  if (args.contains("prefetch_size")) {
    if (args["prefetch_size"].is_none()) {
      prefetch_size = 16;
    } else {
      prefetch_size = ToInt(args["prefetch_size"]);
    }
  }
  std::shared_ptr<DeviceQueueOp::Builder> builder = std::make_shared<DeviceQueueOp::Builder>(prefetch_size);
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "queue_name") {
        (void)builder->SetChannelName(ToString(value));
      } else if (key == "device_type") {
        (void)builder->SetDeviceType(ToString(value));
      } else if (key == "device_id") {
        (void)builder->SetDeviceId(ToInt(value));
      } else if (key == "num_batch") {
        (void)builder->SetNumBatch(ToInt(value));
      }
    }
  }
  std::shared_ptr<DeviceQueueOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseRenameOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::vector<std::string> in_col_names;
  std::vector<std::string> out_col_names;
  std::shared_ptr<RenameOp::Builder> builder = std::make_shared<RenameOp::Builder>();
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "input_columns") {
        in_col_names = ToStringVector(value);
      } else if (key == "output_columns") {
        out_col_names = ToStringVector(value);
      }
    }
  }
  if (in_col_names.empty() || in_col_names[0].empty()) {
    std::string err_msg = "Error: input_column_names is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  if (out_col_names.empty() || out_col_names[0].empty()) {
    std::string err_msg = "Error: output_column_names is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  (void)builder->SetInColNames(in_col_names);
  (void)builder->SetOutColNames(out_col_names);
  std::shared_ptr<RenameOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseTakeOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  if (args["count"].is_none()) {
    std::string err_msg = "Error: count is invalid or not set.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<TakeOp> op;
  RETURN_IF_NOT_OK(TakeOp::Builder(ToInt(args["count"])).Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseZipOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  std::shared_ptr<ZipOp::Builder> builder = std::make_shared<ZipOp::Builder>();
  std::shared_ptr<ZipOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseTFReaderOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  std::shared_ptr<TFReaderOp::Builder> builder = std::make_shared<TFReaderOp::Builder>();
  if (!args["dataset_files"].is_none()) {
    (void)builder->SetDatasetFilesList(ToStringVector(args["dataset_files"]));
  } else {
    std::string err_msg = "Error: at least one of dataset_files or schema_file is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::vector<std::string> columns_to_load;
  bool schema_exists = false;
  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "columns_list") {
        columns_to_load = ToStringVector(value);
        (void)builder->SetColumnsToLoad(columns_to_load);
      } else if (key == "shuffle_files") {
        (void)builder->SetShuffleFiles(ToBool(value));
      } else if (key == "schema_file_path" || key == "schema_json_string") {
        schema_exists = true;
      } else if (key == "num_samples") {
        (void)builder->setTotalRows(ToInt(value));
      } else if (key == "num_shards") {
        (void)builder->SetNumDevices(ToInt(value));
      } else if (key == "shard_id") {
        (void)builder->SetDeviceId(ToInt(value));
      } else if (key == "shard_equal_rows") {
        (void)builder->SetShardEqualRows(ToBool(value));
      }
    }
  }
  if (schema_exists) {
    std::unique_ptr<DataSchema> schema = std::make_unique<DataSchema>();
    if (args.contains("schema_file_path")) {
      RETURN_IF_NOT_OK(schema->LoadSchemaFile(ToString(args["schema_file_path"]), columns_to_load));
    } else {
      RETURN_IF_NOT_OK(schema->LoadSchemaString(ToString(args["schema_json_string"]), columns_to_load));
    }
    (void)builder->SetDataSchema(std::move(schema));
  }
  std::shared_ptr<TFReaderOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseProjectOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  if (args["columns"].is_none()) {
    std::string err_msg = "Error: columns is missing";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::vector<std::string> columns_to_project = ToStringVector(args["columns"]);
  std::shared_ptr<ProjectOp::Builder> builder = std::make_shared<ProjectOp::Builder>(columns_to_project);
  std::shared_ptr<ProjectOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseImageFolderOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<ImageFolderOp::Builder> builder = std::make_shared<ImageFolderOp::Builder>();
  (void)builder->SetImageFolderDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<Sampler> sampler = create().cast<std::shared_ptr<Sampler>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "extensions") {
        (void)builder->SetExtensions(ToStringSet(value));
      } else if (key == "class_indexing") {
        (void)builder->SetClassIndex(ToStringMap(value));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      }
    }
  }
  std::shared_ptr<ImageFolderOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseManifestOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  if (args["dataset_file"].is_none()) {
    std::string err_msg = "Error: No dataset files specified for manifest";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  std::shared_ptr<ManifestOp::Builder> builder = std::make_shared<ManifestOp::Builder>();
  (void)builder->SetManifestFile(ToString(args["dataset_file"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<Sampler> sampler = create().cast<std::shared_ptr<Sampler>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "class_indexing") {
        (void)builder->SetClassIndex(ToStringMap(value));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      } else if (key == "usage") {
        (void)builder->SetUsage(ToString(value));
      }
    }
  }
  std::shared_ptr<ManifestOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseVOCOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::shared_ptr<VOCOp::Builder> builder = std::make_shared<VOCOp::Builder>();
  (void)builder->SetDir(ToString(args["dataset_dir"]));
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<Sampler> sampler = create().cast<std::shared_ptr<Sampler>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      }
    }
  }
  std::shared_ptr<VOCOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseCifar10Op(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::shared_ptr<CifarOp::Builder> builder = std::make_shared<CifarOp::Builder>();
  (void)builder->SetCifarDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<Sampler> sampler = create().cast<std::shared_ptr<Sampler>>();
        (void)builder->SetSampler(std::move(sampler));
      }
    }
  }

  (void)builder->SetCifarType(true);

  std::shared_ptr<CifarOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseCifar100Op(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::shared_ptr<CifarOp::Builder> builder = std::make_shared<CifarOp::Builder>();
  (void)builder->SetCifarDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<Sampler> sampler = create().cast<std::shared_ptr<Sampler>>();
        (void)builder->SetSampler(std::move(sampler));
      }
    }
  }

  (void)builder->SetCifarType(false);

  std::shared_ptr<CifarOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

int32_t DEPipeline::GetNumClasses() const { return num_classes_; }

Status DEPipeline::ParseMnistOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  std::shared_ptr<MnistOp::Builder> builder = std::make_shared<MnistOp::Builder>();
  (void)builder->SetDir(ToString(args["dataset_dir"]));

  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<Sampler> sampler = create().cast<std::shared_ptr<Sampler>>();
        (void)builder->SetSampler(std::move(sampler));
      }
    }
  }
  std::shared_ptr<MnistOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseCelebAOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  if (args["dataset_dir"].is_none()) {
    std::string err_msg = "Error: No dataset path specified";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
  }

  std::shared_ptr<CelebAOp::Builder> builder = std::make_shared<CelebAOp::Builder>();
  if (builder == nullptr) {
    std::string err_msg = "Create celebaop builder failed";
    return Status(StatusCode::kUnexpectedError, __LINE__, __FILE__, err_msg);
  }
  (void)builder->SetCelebADir(ToString(args["dataset_dir"]));
  for (const auto &arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "sampler") {
        auto create = py::reinterpret_borrow<py::object>(value).attr("create");
        std::shared_ptr<Sampler> sampler = create().cast<std::shared_ptr<Sampler>>();
        (void)builder->SetSampler(std::move(sampler));
      } else if (key == "decode") {
        (void)builder->SetDecode(ToBool(value));
      } else if (key == "extensions") {
        (void)builder->SetExtensions(ToStringSet(value));
      } else if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "dataset_type") {
        (void)builder->SetDatasetType(ToString(value));
      }
    }
  }

  std::shared_ptr<CelebAOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}

Status DEPipeline::ParseTextFileOp(const py::dict &args, std::shared_ptr<DatasetOp> *ptr) {
  // Required arguments
  std::shared_ptr<TextFileOp::Builder> builder = std::make_shared<TextFileOp::Builder>();
  if (!args["dataset_files"].is_none()) {
    (void)builder->SetTextFilesList(ToStringVector(args["dataset_files"]));
  } else {
    RETURN_STATUS_UNEXPECTED("Error: dataset_files is missing");
  }
  // Optional arguments
  for (auto arg : args) {
    std::string key = py::str(arg.first);
    py::handle value = arg.second;
    if (!value.is_none()) {
      if (key == "num_parallel_workers") {
        (void)builder->SetNumWorkers(ToInt(value));
      } else if (key == "shuffle_files") {
        (void)builder->SetShuffleFiles(ToBool(value));
      } else if (key == "num_samples") {
        (void)builder->SetNumSamples(ToInt(value));
      } else if (key == "num_shards") {
        (void)builder->SetNumDevices(ToInt(value));
      } else if (key == "shard_id") {
        (void)builder->SetDeviceId(ToInt(value));
      }
    }
  }
  std::shared_ptr<TextFileOp> op;
  RETURN_IF_NOT_OK(builder->Build(&op));
  *ptr = op;
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
