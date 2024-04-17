/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "tools/benchmark_train/net_train_base.h"
#include <algorithm>
#include <cstring>
#ifdef ENABLE_NEON
#include <arm_neon.h>
#endif
#include "src/common/common.h"
#include "include/api/serialization.h"

namespace mindspore {
namespace lite {
const std::unordered_map<int, std::string> kTypeIdMap{
  {kNumberTypeFloat16, "Float16"}, {kNumberTypeFloat, "Float32"},    {kNumberTypeFloat32, "Float32"},
  {kNumberTypeInt8, "Int8"},       {kNumberTypeInt16, "Int16"},      {kNumberTypeInt, "Int32"},
  {kNumberTypeInt32, "Int32"},     {kNumberTypeUInt8, "UInt8"},      {kNumberTypeUInt16, "UInt16"},
  {kNumberTypeUInt, "UInt32"},     {kNumberTypeUInt32, "UInt32"},    {kObjectTypeString, "String"},
  {kNumberTypeBool, "Bool"},       {kObjectTypeTensorType, "Tensor"}};

const std::unordered_map<mindspore::Format, std::string> kTensorFormatMap{
  {mindspore::NCHW, "NCHW"}, {mindspore::NHWC, "NHWC"},     {mindspore::NHWC4, "NHWC4"}, {mindspore::HWKC, "HWKC"},
  {mindspore::HWCK, "HWCK"}, {mindspore::KCHW, "KCHW"},     {mindspore::CKHW, "CKHW"},   {mindspore::KHWC, "KHWC"},
  {mindspore::CHWK, "CHWK"}, {mindspore::HW, "HW"},         {mindspore::HW4, "HW4"},     {mindspore::NC, "NC"},
  {mindspore::NC4, "NC4"},   {mindspore::NC4HW4, "NC4HW4"}, {mindspore::NCDHW, "NCDHW"}};

std::function<int(NetTrainFlags *)> NetTrainBase::nr_cb_ = nullptr;

int NetTrainBase::SetNr(std::function<int(NetTrainFlags *)> param) {
  nr_cb_ = param;
  return 0;
}

float *NetTrainBase::ReadFileBuf(const std::string file, size_t *size) {
  if (file.empty()) {
    MS_LOG(ERROR) << "file is nullptr";
    return nullptr;
  }
  MS_ASSERT(size != nullptr);
  std::string real_path = RealPath(file.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return nullptr;
  }

  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << " open failed";
    return nullptr;
  }

  ifs.seekg(0, std::ios::end);
  *size = ifs.tellg();
  std::unique_ptr<float[]> buf = std::make_unique<float[]>(*size / sizeof(float) + 1);
  if (buf == nullptr) {
    MS_LOG(ERROR) << "malloc buf failed, file: " << real_path;
    ifs.close();
    return nullptr;
  }

  ifs.seekg(0, std::ios::beg);
  ifs.read(reinterpret_cast<char *>(buf.get()), *size);
  ifs.close();

  return buf.release();
}

int NetTrainBase::GenerateRandomData(mindspore::MSTensor *tensor) {
  auto input_data = tensor->MutableData();
  if (input_data == nullptr) {
    MS_LOG(ERROR) << "MallocData for inTensor failed";
    return RET_ERROR;
  }
  auto tensor_byte_size = tensor->DataSize();
  char *casted_data = static_cast<char *>(input_data);
  for (size_t i = 0; i < tensor_byte_size; i++) {
    casted_data[i] =
      (tensor->DataType() == mindspore::DataType::kNumberTypeFloat32) ? static_cast<char>(i) : static_cast<char>(0);
  }
  return RET_OK;
}

int NetTrainBase::LoadInput() {
  inputs_buf_.clear();
  inputs_size_.clear();
  batch_num_ = 0;
  if (flags_->in_data_file_.empty()) {
    auto status = GenerateInputData();
    if (status != RET_OK) {
      std::cerr << "Generate input data error " << status << std::endl;
      MS_LOG(ERROR) << "Generate input data error " << status;
      return status;
    }
  } else {
    auto status = ReadInputFile();
    if (status != RET_OK) {
      std::cerr << "Read Input File error, " << status << std::endl;
      MS_LOG(ERROR) << "Read Input File error, " << status;
      return status;
    }
  }
  return RET_OK;
}

int NetTrainBase::RunNetTrain() {
  auto file_name = flags_->model_file_.substr(flags_->model_file_.find_last_of(DELIM_SLASH) + 1);
  bool is_train = (file_name.find("train") != std::string::npos) || !flags_->bb_model_file_.empty();
  auto status = CreateAndRunNetwork(flags_->model_file_, flags_->bb_model_file_, is_train, flags_->epochs_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "CreateAndRunNetwork failed for model " << flags_->model_file_ << ". Status is " << status;
    std::cout << "CreateAndRunNetwork failed for model " << flags_->model_file_ << ". Status is " << status
              << std::endl;
    return status;
  }

  status = CheckExecutionOfSavedModels();  // re-initialize sessions according to flags
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Run CheckExecute error: " << status;
    std::cout << "Run CheckExecute error: " << status << std::endl;
    return status;
  }
  return RET_OK;
}

int NetTrainBase::CheckExecutionOfSavedModels() {
  int status = RET_OK;
  if (!flags_->export_file_.empty()) {
    status = CreateAndRunNetwork(flags_->export_file_, flags_->bb_model_file_, true, 0);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Run Exported model " << flags_->export_file_ << " error: " << status;
      std::cout << "Run Exported model " << flags_->export_file_ << " error: " << status << std::endl;
      return status;
    }
    if (flags_->bb_model_file_.empty()) {
      status = CreateAndRunNetwork(flags_->export_file_ + "_qt", "", true, 0, false);
      if (status != RET_OK) {
        MS_LOG(ERROR) << "Run Exported model " << flags_->export_file_ << "_qt.ms error: " << status;
        std::cout << "Run Exported model " << flags_->export_file_ << "_qt.ms error: " << status << std::endl;
        return status;
      }
    }
  }
  if (!flags_->inference_file_.empty()) {
    status = CreateAndRunNetwork(flags_->inference_file_, "", false, 0);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Running saved model " << flags_->inference_file_ << ".ms error: " << status;
      std::cout << "Running saved model " << flags_->inference_file_ << ".ms error: " << status << std::endl;
      return status;
    }
    status = CreateAndRunNetwork(flags_->inference_file_ + "_qt", "", false, 0, false);
    if (status != RET_OK) {
      MS_LOG(ERROR) << "Running saved model " << flags_->inference_file_ << "_qt.ms error: " << status;
      std::cout << "Running saved model " << flags_->inference_file_ << "_qt.ms error: " << status << std::endl;
      return status;
    }
  }
  return status;
}

void NetTrainBase::CheckSum(MSTensor *tensor, const std::string &node_type, int id, const std::string &in_out) {
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "input tensor is nullptr.";
    return;
  }
  int tensor_size = tensor->ElementNum();
  void *data = tensor->MutableData();
  auto *fdata = reinterpret_cast<float *>(tensor->MutableData());
  auto type = tensor->DataType();
  std::cout << node_type << " " << in_out << id << " shape=" << tensor->Shape() << " sum=";
  switch (type) {
    case mindspore::DataType::kNumberTypeFloat32:
      TensorNan(reinterpret_cast<float *>(data), tensor_size);
      std::cout << TensorSum<float>(data, tensor_size) << std::endl;
      std::cout << "tensor name: " << tensor->Name() << std::endl;
      std::cout << "data: ";
      for (int i = 0; i <= kPrintOffset && i < tensor_size; i++) {
        std::cout << static_cast<float>(fdata[i]) << ", ";
      }
      std::cout << std::endl;
      break;
    case mindspore::DataType::kNumberTypeInt32:
      std::cout << TensorSum<int>(data, tensor_size) << std::endl;
      break;
#ifdef ENABLE_FP16
    case mindspore::DataType::kNumberTypeFloat16:
      std::cout << TensorSum<float16_t>(data, tensor_size) << std::endl;
      TensorNan(reinterpret_cast<float16_t *>(data), tensor_size);
      break;
#endif
    default:
      std::cout << "unsupported type:" << static_cast<int>(type) << std::endl;
      break;
  }
}

std::string NetTrainBase::GenerateOutputFileName(mindspore::MSTensor *tensor, const std::string &op_name,
                                                 const std::string &file_type, const size_t &idx) {
  std::string file_name = op_name;
  auto pos = file_name.find_first_of('/');
  while (pos != std::string::npos) {
    file_name.replace(pos, 1, ".");
    pos = file_name.find_first_of('/');
  }
  file_name += "_" + file_type + "_" + std::to_string(idx) + "_shape_";
  for (const auto &dim : tensor->Shape()) {
    file_name += std::to_string(dim) + "_";
  }
  if (kTypeIdMap.find(static_cast<int>(tensor->DataType())) != kTypeIdMap.end()) {
    file_name += kTypeIdMap.at(static_cast<int>(tensor->DataType()));
  }
  auto tensor_format = tensor->format();
  if (kTensorFormatMap.find(tensor_format) != kTensorFormatMap.end()) {
    file_name += "_" + kTensorFormatMap.at(tensor_format) + ".bin";
  }

  file_name += ".bin";
  return file_name;
}

int NetTrainBase::InitCallbackParameter() {
  int ret = RET_OK;
  if (flags_->dump_tensor_data_) {
    ret = InitDumpTensorDataCallbackParameter();
  } else if (flags_->time_profiling_) {
    ret = InitTimeProfilingCallbackParameter();
  }
  return ret;
}

void NetTrainFlags::InitResizeDimsList() {
  std::string content = this->resize_dims_in_;
  if (content.empty()) {
    return;
  }
  std::vector<int> shape;
  auto shape_strs = StrSplit(content, std::string(DELIM_COLON));
  for (const auto &shape_str : shape_strs) {
    shape.clear();
    auto dim_strs = StrSplit(shape_str, std::string(DELIM_COMMA));
    std::cout << "Resize Dims: ";
    for (const auto &dim_str : dim_strs) {
      std::cout << dim_str << " ";
      shape.emplace_back(static_cast<int>(std::stoi(dim_str)));
    }
    std::cout << std::endl;
    this->resize_dims_.emplace_back(shape);
  }
}

int NetTrainBase::Init() {
  if (this->flags_ == nullptr) {
    return 1;
  }
  MS_LOG(INFO) << "ModelPath = " << this->flags_->model_file_;
  MS_LOG(INFO) << "InDataPath = " << this->flags_->in_data_file_;
  MS_LOG(INFO) << "InDataType = " << this->flags_->in_data_type_in_;
  MS_LOG(INFO) << "Epochs = " << this->flags_->epochs_;
  MS_LOG(INFO) << "AccuracyThreshold = " << this->flags_->accuracy_threshold_;
  MS_LOG(INFO) << "WarmUpLoopCount = " << this->flags_->warm_up_loop_count_;
  MS_LOG(INFO) << "NumThreads = " << this->flags_->num_threads_;
  MS_LOG(INFO) << "expectedDataFile = " << this->flags_->data_file_;
  MS_LOG(INFO) << "exportDataFile = " << this->flags_->export_file_;
  MS_LOG(INFO) << "enableFp16 = " << this->flags_->enable_fp16_;
  MS_LOG(INFO) << "virtualBatch = " << this->flags_->virtual_batch_;

  if (this->flags_->epochs_ < 0) {
    MS_LOG(ERROR) << "epochs:" << this->flags_->epochs_ << " must be equal/greater than 0";
    std::cerr << "epochs:" << this->flags_->epochs_ << " must be equal/greater than 0" << std::endl;
    return RET_ERROR;
  }

  if (this->flags_->num_threads_ < 1) {
    MS_LOG(ERROR) << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0";
    std::cerr << "numThreads:" << this->flags_->num_threads_ << " must be greater than 0" << std::endl;
    return RET_ERROR;
  }

  this->flags_->in_data_type_ = this->flags_->in_data_type_in_ == "img" ? kImage : kBinary;

  if (flags_->in_data_file_.empty() && !flags_->data_file_.empty()) {
    MS_LOG(ERROR) << "expectedDataFile not supported in case that inDataFile is not provided";
    std::cerr << "expectedDataFile is not supported in case that inDataFile is not provided" << std::endl;
    return RET_ERROR;
  }

  if (flags_->in_data_file_.empty() && !flags_->export_file_.empty()) {
    MS_LOG(ERROR) << "exportDataFile not supported in case that inDataFile is not provided";
    std::cerr << "exportDataFile is not supported in case that inDataFile is not provided" << std::endl;
    return RET_ERROR;
  }

  if (flags_->model_file_.empty()) {
    MS_LOG(ERROR) << "modelPath is required";
    std::cerr << "modelPath is required" << std::endl;
    return 1;
  }

  // get dump data output path
  auto dump_cfg_path = std::getenv(dump::kConfigPath);
  if (dump_cfg_path != nullptr) {
    flags_->dump_tensor_data_ = true;
    if (InitDumpConfigFromJson(dump_cfg_path) != RET_OK) {
      MS_LOG(ERROR) << "parse dump config file failed.";
      return RET_ERROR;
    }
  } else {
    MS_LOG(INFO) << "No MINDSPORE_DUMP_CONFIG in env, don't need to dump data";
  }

  auto status = InitCallbackParameter();
  if (status != RET_OK) {
    MS_LOG(ERROR) << "Init callback Parameter failed.";
    std::cerr << "Init callback Parameter failed." << std::endl;
    return RET_ERROR;
  }

  flags_->InitResizeDimsList();
  if (!flags_->resize_dims_.empty() && !flags_->input_data_list_.empty() &&
      flags_->resize_dims_.size() != flags_->input_data_list_.size()) {
    MS_LOG(ERROR) << "Size of input resizeDims should be equal to size of input inDataPath";
    std::cerr << "Size of input resizeDims should be equal to size of input inDataPath" << std::endl;
    return RET_ERROR;
  }
  return RET_OK;
}

int NetTrainBase::InitDumpConfigFromJson(std::string path) {
  auto real_path = RealPath(path.c_str());
  std::ifstream ifs(real_path);
  if (!ifs.good()) {
    MS_LOG(ERROR) << "file: " << real_path << " is not exist";
    return RET_ERROR;
  }
  if (!ifs.is_open()) {
    MS_LOG(ERROR) << "file: " << real_path << " open failed";
    return RET_ERROR;
  }

  try {
    dump_cfg_json_ = nlohmann::json::parse(ifs);
  } catch (const nlohmann::json::parse_error &error) {
    MS_LOG(ERROR) << "parse json file failed, please check your file.";
    return RET_ERROR;
  }
  if (dump_cfg_json_[dump::kSettings] == nullptr) {
    MS_LOG(ERROR) << "\"common_dump_settings\" is required.";
    return RET_ERROR;
  }
  if (dump_cfg_json_[dump::kSettings][dump::kMode] == nullptr) {
    MS_LOG(ERROR) << "\"dump_mode\" is required.";
    return RET_ERROR;
  }
  if (dump_cfg_json_[dump::kSettings][dump::kPath] == nullptr) {
    MS_LOG(ERROR) << "\"path\" is required.";
    return RET_ERROR;
  }
  if (dump_cfg_json_[dump::kSettings][dump::kNetName] == nullptr) {
    dump_cfg_json_[dump::kSettings][dump::kNetName] = "default";
  }
  if (dump_cfg_json_[dump::kSettings][dump::kInputOutput] == nullptr) {
    dump_cfg_json_[dump::kSettings][dump::kInputOutput] = 0;
  }
  if (dump_cfg_json_[dump::kSettings][dump::kKernels] != nullptr &&
      !dump_cfg_json_[dump::kSettings][dump::kKernels].empty()) {
    if (dump_cfg_json_[dump::kSettings][dump::kMode] == 0) {
      MS_LOG(ERROR) << R"("dump_mode" should be 1 when "kernels" isn't empty.)";
      return RET_ERROR;
    }
  }

  auto abs_path = dump_cfg_json_[dump::kSettings][dump::kPath].get<std::string>();
  auto net_name = dump_cfg_json_[dump::kSettings][dump::kNetName].get<std::string>();
  if (abs_path.back() == '\\' || abs_path.back() == '/') {
    dump_file_output_dir_ = abs_path + net_name;
  } else {
#ifdef _WIN32
    dump_file_output_dir_ = abs_path + "\\" + net_name;
#else
    dump_file_output_dir_ = abs_path + "/" + net_name;
#endif
  }

  auto status = CreateOutputDir(&dump_file_output_dir_);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "create data output directory failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

NetTrainBase::~NetTrainBase() {}
}  // namespace lite
}  // namespace mindspore
