/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_CALIB_DATA_GENERATOR_H_
#define MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_CALIB_DATA_GENERATOR_H_

#include <fstream>
#include <numeric>
#include <vector>
#include <string>
#include <functional>
#include <map>
#include <memory>
#include <utility>
#include "third_party/securec/include/securec.h"
#include "include/registry/converter_context.h"
#include "common/data_transpose_utils.h"
#include "mindapi/ir/anf.h"
#include "include/errorcode.h"
#include "common/check_base.h"

namespace mindspore {
namespace dpico {
enum DumpMode : int { kDumpInputOutput = 0, kDumpInput = 1, kDumpOutput = 2 };
struct OpAttr {
  std::string data_type;
  size_t input_output_idx;
  std::vector<int32_t> shape;
  std::string format;
};
struct DumpOpInfo {
  std::string origin_op_name;
  std::string dump_op_name;
  int input_index;
  int output_index;
};
class CalibDataGenerator {
 public:
  explicit CalibDataGenerator(int dump_level = 0,
                              const std::map<api::AnfNodePtr, std::pair<api::CNodePtr, int>> &control_flow_inputs = {})
      : dump_level_(dump_level), control_flow_inputs_(control_flow_inputs) {}
  ~CalibDataGenerator() = default;
  int Run(const api::AnfNodePtrList &graph_inputs, const api::AnfNodePtrList &nodes);

 private:
  int GenerateDumpConfig(const std::string &dump_cfg_path, const std::vector<DumpOpInfo> &dump_infos);
  std::vector<std::vector<int64_t>> GetInputShapes(const api::AnfNodePtrList &graph_inputs);
  std::vector<std::string> GetInDataFileList(const api::AnfNodePtrList &graph_inputs);
  int DumpKernelsData(const std::string &dump_cfg_path, const std::vector<std::string> &in_data_file_list,
                      const std::vector<std::string> kernel_names,
                      const std::vector<std::vector<int64_t>> &input_shapes);
  STATUS ParseAttrFromFilename(struct OpAttr *op_attr, const std::string &file_name, bool is_input);
  int TransBinsToTxt(const std::vector<DumpOpInfo> &dump_infos);

  template <typename T>
  int ReadBinToOfstream(const std::string &file_path, const struct OpAttr &op_attr, std::ofstream &ofs) {
    std::ifstream ifs;
    ifs.open(file_path, std::ifstream::in | std::ios::binary);
    if (!ifs.is_open() || !ifs.good()) {
      MS_LOG(ERROR) << "open file failed. " << file_path;
      return RET_ERROR;
    }
    size_t shape_size = 1;
    for (size_t i = 0; i < op_attr.shape.size(); i++) {
      if (op_attr.shape.at(i) < 0) {
        MS_LOG(ERROR) << "dim val should be equal or greater than 0";
        return RET_ERROR;
      }
      if (SIZE_MUL_OVERFLOW(shape_size, static_cast<size_t>(op_attr.shape.at(i)))) {
        MS_LOG(ERROR) << "size_t mul overflow.";
        return RET_ERROR;
      }
      shape_size *= static_cast<size_t>(op_attr.shape.at(i));
    }
    (void)ifs.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(ifs.tellg());
    if (file_size != shape_size * sizeof(T)) {
      MS_LOG(ERROR) << "file size " << file_size << " is not equal to shape size " << shape_size;
      return RET_ERROR;
    }
    auto raw_datas = std::make_unique<T[]>(shape_size);
    if (raw_datas == nullptr) {
      MS_LOG(ERROR) << "new T failed.";
      return RET_ERROR;
    }
    (void)ifs.seekg(0, std::ios::beg);
    (void)ifs.read(reinterpret_cast<char *>(raw_datas.get()), shape_size * sizeof(T));
    ifs.close();
    if (op_attr.format == "NHWC" && op_attr.shape.size() == kDims4) {
      auto dst_datas = std::make_unique<T[]>(shape_size);
      if (dst_datas == nullptr) {
        MS_LOG(ERROR) << "new T failed.";
        return RET_ERROR;
      }
      if (memcpy_s(dst_datas.get(), shape_size * sizeof(T), raw_datas.get(), shape_size * sizeof(T)) != EOK) {
        MS_LOG(ERROR) << "memcpy_s failed.";
        return RET_ERROR;
      }
      if (NHWC2NCHW<T>(raw_datas.get(), dst_datas.get(), op_attr.shape) != RET_OK) {
        MS_LOG(ERROR) << "NHWC to NCHW failed.";
        return RET_ERROR;
      }
      for (size_t i = 0; i < shape_size; i++) {
        ofs << dst_datas.get()[i] << ' ';
      }
    } else {
      for (size_t i = 0; i < shape_size; i++) {
        ofs << raw_datas.get()[i] << ' ';
      }
    }

    return RET_OK;
  }
  int dump_level_;
  std::map<api::AnfNodePtr, std::pair<api::CNodePtr, int>> control_flow_inputs_;
};
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_TOOLS_CONVERTER_ADAPTER_DPICO_SRC_CALIB_DATA_GENERATOR_H_
