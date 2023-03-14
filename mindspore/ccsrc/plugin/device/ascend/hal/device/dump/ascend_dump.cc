/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/dump/ascend_dump.h"
#include <set>
#include <utility>
#include <algorithm>
#include <map>
#include "include/backend/debug/data_dump/tensor_stat_dump.h"
#include "runtime/device/ms_device_shape_transfer.h"

namespace mindspore {
namespace ascend {
namespace {
constexpr int kDhaAtomicAddInfoSize = 128;
constexpr int kL2AtomicAddInfoSize = 128;
constexpr int kAiCoreInfoSize = 256;
constexpr int kDhaAtomicAddStatusSize = 256;
constexpr int kL2AtomicAddStatusSize = 256;
constexpr int kUint64Size = sizeof(uint64_t);
using ProtoFormat = debugger::dump::OutputFormat;
using ProtoDataType = debugger::dump::OutputDataType;
const std::set<std::pair<std::string, std::string>> kSuppTransFormatPair = {
  // {device format, host format}
  {kOpFormat_FRAC_Z, kOpFormat_NCHW},      {kOpFormat_FRAC_NZ, kOpFormat_NCHW},
  {kOpFormat_NC1HWC0, kOpFormat_NCHW},     {kOpFormat_C1HWNCoC0, kOpFormat_NCHW},
  {kOpFormat_NC1HWC0_C04, kOpFormat_NCHW}, {kOpFormat_NDC1HWC0, kOpFormat_NCHW},
  {kOpFormat_FRACTAL_Z_3D, kOpFormat_NCHW}};

const std::map<ProtoFormat, std::string> kFormatToStringMap = {
  {ProtoFormat::FORMAT_NCHW, kOpFormat_NCHW},
  {ProtoFormat::FORMAT_NHWC, kOpFormat_NHWC},
  {ProtoFormat::FORMAT_ND, kOpFormat_ND},
  {ProtoFormat::FORMAT_NC1HWC0, kOpFormat_NC1HWC0},
  {ProtoFormat::FORMAT_FRACTAL_Z, kOpFormat_FRAC_Z},
  {ProtoFormat::FORMAT_NC1HWC0_C04, kOpFormat_NC1HWC0_C04},
  {ProtoFormat::FORMAT_FRACTAL_Z_C04, kOpFormat_FRACTAL_Z_C04},
  {ProtoFormat::FORMAT_NC1KHKWHWC0, kOpFormat_NC1KHKWHWC0},
  {ProtoFormat::FORMAT_HWCN, kOpFormat_HWCN},
  {ProtoFormat::FORMAT_NDHWC, kOpFormat_NDHWC},
  {ProtoFormat::FORMAT_NCDHW, kOpFormat_NCDHW},
  {ProtoFormat::FORMAT_DHWCN, kOpFormat_DHWCN},
  {ProtoFormat::FORMAT_DHWNC, kOpFormat_DHWNC},
  {ProtoFormat::FORMAT_NDC1HWC0, kOpFormat_NDC1HWC0},
  {ProtoFormat::FORMAT_FRACTAL_Z_3D, kOpFormat_FRACTAL_Z_3D},
  {ProtoFormat::FORMAT_C1HWNCoC0, kOpFormat_C1HWNCoC0},
  {ProtoFormat::FORMAT_FRACTAL_NZ, kOpFormat_FRAC_NZ},
  {ProtoFormat::FORMAT_FRACTAL_ZN_LSTM, kOpFormat_FRACTAL_ZN_LSTM}};

const std::map<ProtoDataType, mindspore::TypeId> kDataTypetoMSTypeMap = {
  {ProtoDataType::DT_UNDEFINED, mindspore::TypeId::kTypeUnknown},
  {ProtoDataType::DT_FLOAT, mindspore::TypeId::kNumberTypeFloat32},
  {ProtoDataType::DT_FLOAT16, mindspore::TypeId::kNumberTypeFloat16},
  {ProtoDataType::DT_INT8, mindspore::TypeId::kNumberTypeInt8},
  {ProtoDataType::DT_UINT8, mindspore::TypeId::kNumberTypeUInt8},
  {ProtoDataType::DT_INT16, mindspore::TypeId::kNumberTypeInt16},
  {ProtoDataType::DT_UINT16, mindspore::TypeId::kNumberTypeUInt16},
  {ProtoDataType::DT_INT32, mindspore::TypeId::kNumberTypeInt32},
  {ProtoDataType::DT_INT64, mindspore::TypeId::kNumberTypeInt64},
  {ProtoDataType::DT_UINT32, mindspore::TypeId::kNumberTypeUInt32},
  {ProtoDataType::DT_UINT64, mindspore::TypeId::kNumberTypeUInt64},
  {ProtoDataType::DT_BOOL, mindspore::TypeId::kNumberTypeBool},
  {ProtoDataType::DT_DOUBLE, mindspore::TypeId::kNumberTypeFloat64},
  {ProtoDataType::DT_STRING, mindspore::TypeId::kObjectTypeString}};

inline uint64_t UnpackUint64Value(const char *ptr) {
#if defined(__APPLE__)
  return *reinterpret_cast<const uint64_t *>(ptr);
#else
  return le64toh(*reinterpret_cast<const uint64_t *>(ptr));
#endif
}

inline std::string IntToHexString(const uint64_t value) {
  std::stringstream ss;
  ss << "0x" << std::hex << value;
  return ss.str();
}

template <typename T>
dump_data_t ParseAttrsFromDumpData(const std::string &dump_path, char *data_ptr, const T &tensor, const std::string &io,
                                   uint32_t slot) {
  // get data type
  auto iter_dtype = kDataTypetoMSTypeMap.find(tensor.data_type());
  if (iter_dtype == kDataTypetoMSTypeMap.end()) {
    MS_LOG(INFO) << "Unsupported data type for tensor " << dump_path << ": unknown(" << tensor.data_type() << ")";
    return dump_data_t{};
  }
  auto data_type = iter_dtype->second;
  // get format
  auto iter_fmt = kFormatToStringMap.find(tensor.format());
  if (iter_fmt == kFormatToStringMap.end()) {
    MS_LOG(INFO) << "Unsupported tensor format for tensor " << dump_path << ": unknown(" << tensor.format() << ")";
    return dump_data_t{};
  }
  std::string device_format = iter_fmt->second;
  // get shape
  ShapeVector shape_d;
  (void)std::transform(tensor.shape().dim().begin(), tensor.shape().dim().end(), std::back_inserter(shape_d),
                       SizeToLong);
  ShapeVector shape_to;
  (void)std::transform(tensor.original_shape().dim().begin(), tensor.original_shape().dim().end(),
                       std::back_inserter(shape_to), SizeToLong);
  // get size and sub_format
  size_t data_size = static_cast<size_t>(tensor.size());
  int32_t sub_format = tensor.sub_format();
  return dump_data_t{dump_path, data_ptr, data_type, device_format, shape_d, shape_to, data_size, sub_format, io, slot};
}
}  // namespace

nlohmann::json AscendAsyncDump::ParseOverflowInfo(const char *data_ptr) {
  uint32_t index = 0;
  uint64_t model_id = UnpackUint64Value(data_ptr);
  index += kUint64Size;
  uint64_t stream_id = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t task_id = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t task_type = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t pc_start = UnpackUint64Value(data_ptr + index);
  index += kUint64Size;
  uint64_t para_base = UnpackUint64Value(data_ptr + index);

  nlohmann::json overflow_info;
  overflow_info["model_id"] = model_id;
  overflow_info["stream_id"] = stream_id;
  overflow_info["task_id"] = task_id;
  overflow_info["task_type"] = task_type;
  overflow_info["pc_start"] = IntToHexString(pc_start);
  overflow_info["para_base"] = IntToHexString(para_base);
  return overflow_info;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: It serves for A+M dump. Convert tensor from device format to host format if needed.
 */
bool AscendAsyncDump::ConvertFormatForOneTensor(dump_data_t *dump_tensor_info) {
  MS_EXCEPTION_IF_NULL(dump_tensor_info);
  bool trans_success = false;
  auto trans_buf = std::make_shared<tensor::Tensor>(dump_tensor_info->data_type, dump_tensor_info->host_shape);
  // convert format to host format. It can be either NCHW or ND (non 4-dimemsions).
  const uint8_t kNumFourDim = 4;
  std::string host_format;
  std::string device_format = dump_tensor_info->format;
  if (dump_tensor_info->host_shape.size() == kNumFourDim) {
    host_format = kOpFormat_NCHW;
  } else {
    host_format = kOpFormat_ND;
  }
  if (device_format != host_format) {
    std::set<std::pair<std::string, std::string>>::const_iterator iter =
      kSuppTransFormatPair.find(std::make_pair(device_format, host_format));
    if (iter == kSuppTransFormatPair.cend()) {
      MS_LOG(INFO) << "Do not support convert from format " << device_format << " to " << host_format << " for tensor "
                   << dump_tensor_info->dump_file_path << "." << dump_tensor_info->in_out_str << "."
                   << dump_tensor_info->slot;
    } else {
      const trans::FormatArgs format_args{dump_tensor_info->data_ptr,
                                          dump_tensor_info->data_size,
                                          host_format,
                                          device_format,
                                          dump_tensor_info->host_shape,
                                          dump_tensor_info->device_shape,
                                          dump_tensor_info->data_type};
      auto group = dump_tensor_info->sub_format > 1 ? dump_tensor_info->sub_format : 1;
      trans_success = trans::TransFormatFromDeviceToHost(format_args, trans_buf->data_c(), group);
      if (!trans_success) {
        MS_LOG(ERROR) << "Trans format failed.";
      }
    }
  } else {
    MS_LOG(INFO) << "The host_format and device_format are same, no need to convert format for file: "
                 << dump_tensor_info->dump_file_path;
    return true;
  }
  if (trans_success) {
    dump_tensor_info->format = host_format;
    dump_tensor_info->trans_buf = trans_buf;
  }
  return trans_success;
}

void AscendAsyncDump::ConvertFormatForTensors(std::vector<dump_data_t> *dump_tensor_vec, size_t start_idx,
                                              size_t end_idx) {
  for (size_t idx = start_idx; idx <= end_idx; idx++) {
    auto &dump_data_obj = dump_tensor_vec->at(idx);
    auto succ = ConvertFormatForOneTensor(&dump_data_obj);
    if (!succ) {
      MS_LOG(INFO) << "Failed to convert format for tensor " << dump_data_obj.dump_file_path << "."
                   << dump_data_obj.in_out_str << "." << dump_data_obj.slot;
    }
    (void)DumpTensorDataIfNeeded(dump_data_obj);
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: It serves for A+M dump. Save statistic of the tensor data into dump path as configured.
 */
bool AscendAsyncDump::DumpTensorStatsIfNeeded(const dump_data_t &dump_tensor_info) {
  // dump_path: dump_dir/op_type.op_name.task_id.stream_id.timestamp
  if (!DumpJsonParser::GetInstance().IsStatisticDump()) {
    return true;
  }
  std::string dump_path = dump_tensor_info.dump_file_path;
  size_t pos = dump_path.rfind("/");
  std::string file_name = dump_path.substr(pos + 1);
  size_t first_dot = file_name.find(".");
  size_t fourth_dot = file_name.rfind(".");
  size_t third_dot = file_name.rfind(".", fourth_dot - 1);
  size_t second_dot = file_name.rfind(".", third_dot - 1);
  if (first_dot == std::string::npos || second_dot == std::string::npos || third_dot == std::string::npos ||
      first_dot == second_dot) {
    MS_LOG(ERROR) << "Dump path " << dump_path << " received is not well formed";
    return false;
  }
  std::string op_type = file_name.substr(0, first_dot);
  std::string op_name = file_name.substr(first_dot + 1, second_dot - first_dot - 1);
  std::string task_id = file_name.substr(second_dot + 1, third_dot - second_dot - 1);
  std::string stream_id = file_name.substr(third_dot + 1, fourth_dot - third_dot - 1);
  std::string timestamp = file_name.substr(fourth_dot + 1);
  TensorStatDump stat_dump(op_type, op_name, task_id, stream_id, timestamp, dump_tensor_info.in_out_str,
                           dump_tensor_info.slot, dump_tensor_info.slot);
  std::shared_ptr<TensorData> data = std::make_shared<TensorData>();
  if (dump_tensor_info.data_type <= TypeId::kNumberTypeBegin ||
      dump_tensor_info.data_type >= TypeId::kNumberTypeComplex64) {
    MS_LOG(ERROR) << "Data type of operator " << file_name << " is not supported by statistic dump";
    return false;
  }
  std::shared_ptr<tensor::Tensor> trans_buf = dump_tensor_info.trans_buf;
  if (trans_buf) {
    data->SetByteSize(trans_buf->Size());
    data->SetDataPtr(static_cast<char *>(trans_buf->data_c()));
  } else {
    data->SetByteSize(dump_tensor_info.data_size);
    data->SetDataPtr(dump_tensor_info.data_ptr);
  }
  data->SetType(dump_tensor_info.data_type);
  data->SetShape(dump_tensor_info.host_shape);
  return stat_dump.DumpTensorStatsToFile(dump_path.substr(0, pos), data);
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: It serves for A+M dump. Save tensor into dump path as configured.
 */
bool AscendAsyncDump::DumpTensorDataIfNeeded(const dump_data_t &dump_tensor_info) {
  if (!DumpJsonParser::GetInstance().IsTensorDump()) {
    return true;
  }
  // dump_path: dump_dir/op_type.op_name.task_id.stream_id.timestamp
  std::ostringstream dump_path_ss;
  dump_path_ss << dump_tensor_info.dump_file_path << "." << dump_tensor_info.in_out_str << "." << dump_tensor_info.slot
               << "." << dump_tensor_info.format;
  std::string dump_path_slot = dump_path_ss.str();
  std::shared_ptr<tensor::Tensor> trans_buf = dump_tensor_info.trans_buf;
  bool dump_succ = false;
  if (trans_buf) {
    dump_succ = DumpJsonParser::DumpToFile(dump_path_slot, trans_buf->data_c(), trans_buf->Size(),
                                           dump_tensor_info.host_shape, dump_tensor_info.data_type);
  } else if (dump_tensor_info.data_size == 0) {
    MS_LOG(INFO) << "Data size is 0 for file: " << dump_tensor_info.dump_file_path << " no need to dump.";
    return true;
  } else {
    dump_succ = DumpJsonParser::DumpToFile(dump_path_slot, dump_tensor_info.data_ptr, dump_tensor_info.data_size,
                                           dump_tensor_info.host_shape, dump_tensor_info.data_type);
  }
  return dump_succ;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is for ascend A+M dump only. It parses and converts each slot of tensor in DumpData object
 * and dump the tensor data in npy file or statistic data in csv file.
 */
void AscendAsyncDump::DumpTensorToFile(const std::string &dump_path, const debugger::dump::DumpData &dump_data,
                                       char *data_ptr) {
  MS_EXCEPTION_IF_NULL(data_ptr);
  std::vector<dump_data_t> dump_tensor_vec;
  // dump input tensors
  std::vector<debugger::dump::OpInput> input_tensors(dump_data.input().begin(), dump_data.input().end());
  uint64_t offset = 0;
  for (uint32_t slot = 0; slot < input_tensors.size(); slot++) {
    auto in_tensor = input_tensors[slot];
    dump_tensor_vec.push_back(ParseAttrsFromDumpData(dump_path, data_ptr + offset, in_tensor, "input", slot));
    offset += in_tensor.size();
  }

  // dump output tensors
  std::vector<debugger::dump::OpOutput> output_tensors(dump_data.output().begin(), dump_data.output().end());
  for (uint32_t slot = 0; slot < output_tensors.size(); slot++) {
    auto out_tensor = output_tensors[slot];
    dump_tensor_vec.push_back(ParseAttrsFromDumpData(dump_path, data_ptr + offset, out_tensor, "output", slot));
    offset += out_tensor.size();
  }

  // assign slot conversion task to different thread.
  if (dump_tensor_vec.empty()) {
    return;
  }
  // The maximum tensor size to allow convert format in single thread to 1 MB.
  constexpr int kMaxTensorSize = 1048576;
  if (offset <= kMaxTensorSize) {
    // If the total tensor size is less than 1MB, do it in single thread.
    ConvertFormatForTensors(&dump_tensor_vec, 0, dump_tensor_vec.size() - 1);
  } else {
    // In multi_thread process, we only use 1/4 of the total concurrent threads.
    size_t ratio_divider = 4;
    auto default_num_workers = std::max<size_t>(1, std::thread::hardware_concurrency() / ratio_divider);
    auto num_threads = std::min<size_t>(default_num_workers, dump_tensor_vec.size());
    size_t task_size = dump_tensor_vec.size() / num_threads;
    size_t remainder = dump_tensor_vec.size() % num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    MS_LOG(INFO) << "Number of threads used for A+M dump: " << num_threads;
    for (size_t t = 0; t < num_threads; t++) {
      size_t start_idx = t * task_size;
      size_t end_idx = start_idx + task_size - 1;
      if (t == num_threads - 1) {
        end_idx += remainder;
      }
      (void)threads.emplace_back(
        std::thread(&AscendAsyncDump::ConvertFormatForTensors, &dump_tensor_vec, start_idx, end_idx));
    }
    for (auto &thd : threads) {
      if (thd.joinable()) {
        thd.join();
      }
    }
  }
  for (auto &dump_tensor_item : dump_tensor_vec) {
    (void)DumpTensorStatsIfNeeded(dump_tensor_item);
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: This function is for Ascend A+M dump. It parses and dump op overflow info in json file.
 */
void AscendAsyncDump::DumpOpDebugToFile(const std::string &dump_path, const debugger::dump::DumpData &dump_data,
                                        const char *data_ptr) {
  MS_EXCEPTION_IF_NULL(data_ptr);
  std::string out_path = dump_path + ".output.";
  std::vector<debugger::dump::OpOutput> op_debug(dump_data.output().begin(), dump_data.output().end());
  for (uint32_t slot = 0; slot < op_debug.size(); slot++) {
    uint32_t index = 0;
    // parse DHA Atomic Add info
    nlohmann::json dha_atomic_add_info = ParseOverflowInfo(data_ptr + index);
    index += kDhaAtomicAddInfoSize;
    // parse L2 Atomic Add info
    nlohmann::json l2_atomic_add_info = ParseOverflowInfo(data_ptr + index);
    index += kL2AtomicAddInfoSize;
    // parse AICore info
    nlohmann::json ai_core_info = ParseOverflowInfo(data_ptr + index);
    index += kAiCoreInfoSize;
    // parse DHA Atomic Add status
    dha_atomic_add_info["status"] = UnpackUint64Value(data_ptr + index);
    index += kDhaAtomicAddStatusSize;
    // parse L2 Atomic Add status
    l2_atomic_add_info["status"] = UnpackUint64Value(data_ptr + index);
    index += kL2AtomicAddStatusSize;
    // parse AICore status
    uint64_t kernel_code = UnpackUint64Value(data_ptr + index);
    index += kUint64Size;
    uint64_t block_idx = UnpackUint64Value(data_ptr + index);
    index += kUint64Size;
    uint64_t status = UnpackUint64Value(data_ptr + index);
    ai_core_info["kernel_code"] = IntToHexString(kernel_code);
    ai_core_info["block_idx"] = block_idx;
    ai_core_info["status"] = status;

    nlohmann::json opdebug_data;
    opdebug_data["DHA Atomic Add"] = dha_atomic_add_info;
    opdebug_data["L2 Atomic Add"] = l2_atomic_add_info;
    opdebug_data["AI Core"] = ai_core_info;

    // save json to file
    DumpToFile(out_path + std::to_string(slot) + ".json", opdebug_data.dump());
  }
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: It is a function to be registered to Adx server for a + m dump feature with the following steps:
 * 1) Merge chunks into one memory segment after receiving all the data for one node.
 * 2) Parse dump data object.
 * 3) Convert data from device to host format.
 * 4) Dump to disk based on configuration.
 */
int32_t DumpDataCallBack(const DumpChunk *dump_chunk, int32_t size) {
  MS_LOG(DEBUG) << "ADX DumpDataCallBack is called";
  MS_LOG(DEBUG) << "The dump_chunk size is: " << size;
  MS_EXCEPTION_IF_NULL(dump_chunk);
  string file_name = dump_chunk->fileName;
  uint32_t isLastChunk = dump_chunk->isLastChunk;

  // parse chunk header
  auto &manager = AscendAsyncDumpManager::GetInstance();
  auto dump_data_build = manager.LoadDumpDataBuilder(file_name);
  if (dump_data_build == nullptr) {
    MS_LOG(ERROR) << "Failed to load dump data builder for node " << file_name;
    return 0;
  }
  if (!dump_data_build->CopyDumpChunk(dump_chunk)) {
    return 1;
  }

  if (isLastChunk == 1) {
    // construct dump data object
    debugger::dump::DumpData dump_data;
    std::vector<char> data_buf;
    if (!dump_data_build->ConstructDumpData(&dump_data, &data_buf)) {
      MS_LOG(ERROR) << "Failed to parse data for node " << file_name;
      return 0;
    }

    // convert and save to files
    auto separator = file_name.rfind("/");
    auto path_name = file_name.substr(0, separator);
    auto file_base_name = file_name.substr(separator + 1);
    if (file_base_name.rfind("Opdebug.Node_OpDebug.") == 0) {
      // save overflow data
      AscendAsyncDump::DumpOpDebugToFile(file_name, dump_data, data_buf.data());
    } else {
      // save tensor data
      // generate fully qualified file name
      // before: op_type.op_name.task_id.stream_id.timestamp
      // after: op_type.op_name_no_scope.task_id.stream_id.timestamp
      size_t first_dot = file_base_name.find(".");
      size_t second_dot = file_base_name.size();
      const int kNumDots = 3;
      int nth_dot_from_back = 0;
      while (nth_dot_from_back != kNumDots && second_dot != std::string::npos) {
        second_dot = file_base_name.rfind(".", second_dot - 1);
        nth_dot_from_back++;
      }
      if (first_dot == std::string::npos || second_dot == std::string::npos) {
        MS_LOG(ERROR) << "Failed to generate fully qualified file name for " << file_name;
        return 0;
      }
      auto op_type = file_base_name.substr(0, first_dot);
      auto task_stream_timestamp = file_base_name.substr(second_dot);
      std::string op_name = dump_data.op_name();
      auto op_name_no_scope = GetOpNameWithoutScope(op_name, "/");
      AscendAsyncDump::DumpTensorToFile(path_name + "/" + op_type + "." + op_name_no_scope + task_stream_timestamp,
                                        dump_data, data_buf.data());
    }
    manager.ClearDumpDataBuilder(file_name);
  }
  return 0;
}

AscendAsyncDumpManager &AscendAsyncDumpManager::GetInstance() {
  static AscendAsyncDumpManager manager{};
  return manager;
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: Load DumpDataBuilder object from dump_data_construct_map_ for tracking data chunks of node_name. It's
 * for Ascend a + m dump. If not found, create a new one for it and add to dump_data_construct_map_.
 */
std::shared_ptr<DumpDataBuilder> AscendAsyncDumpManager::LoadDumpDataBuilder(const std::string &node_name) {
  std::map<std::string, std::shared_ptr<DumpDataBuilder>>::const_iterator iter =
    dump_data_construct_map_.find(node_name);
  if (iter == dump_data_construct_map_.cend()) {
    dump_data_construct_map_[node_name] = std::make_shared<DumpDataBuilder>();
  }
  return dump_data_construct_map_[node_name];
}

void AscendAsyncDumpManager::ClearDumpDataBuilder(const std::string &node_name) {
  (void)dump_data_construct_map_.erase(node_name);
}

/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: MindRT.
 * Description: This function is used for A+M dump to make sure training processing ends after tensor data have been
 * dumped to disk completely. Check if dump_data_construct_map_ is empty to see if no dump task is alive. If not, sleep
 * for 500ms and check again.
 */
void AscendAsyncDumpManager::WaitForWriteFileFinished() const {
  const int kRetryTimeInMilliseconds = 500;
  const int kMaxRecheckCount = 10;
  int recheck_cnt = 0;
  while (recheck_cnt < kMaxRecheckCount && !dump_data_construct_map_.empty()) {
    MS_LOG(INFO) << "Sleep for " << std::to_string(kRetryTimeInMilliseconds)
                 << " ms to wait for dumping files to finish. Retry count: " << std::to_string(recheck_cnt + 1) << "/"
                 << std::to_string(kMaxRecheckCount);
    std::this_thread::sleep_for(std::chrono::milliseconds(kRetryTimeInMilliseconds));
    recheck_cnt++;
  }
}
}  // namespace ascend
}  // namespace mindspore
