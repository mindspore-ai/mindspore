/**
 # Adapted from
 # https://gitee.com/ascend/samples/blob/master/cplusplus/common/acllite/src/AclLiteUtils.cpp
* Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File utils.cpp
* Description: handle file operations
*/
#include "minddata/dataset/vision/kernels/dvpp/utils/AclLiteUtils.h"

#include <dirent.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <vector>

#include "acl/ops/acl_dvpp.h"
#include "plugin/ascend/res_manager/symbol_interface/acl_rt_symbol.h"
#include "plugin/ascend/res_manager/symbol_interface/symbol_utils.h"

namespace {
const char COMMENT_CHAR = '#';
const char EQUALS_CHAR = '=';
const char BLANK_SPACE_CHAR = ' ';
const char TABLE_CHAR = '\t';

const std::string kImagePathSeparator = ",";
const int kStatSuccess = 0;
const std::string kFileSperator = "/";
const std::string kPathSeparator = "/";
// output image prefix
const std::string kOutputFilePrefix = "out_";

const std::string kRegexIpAddr =
  "^(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|[0-9])\\."
  "(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)\\."
  "(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)\\."
  "(1\\d{2}|2[0-4]\\d|25[0-5]|[1-9]\\d|\\d)"
  ":([1-9]|[1-9]\\d|[1-9]\\d{2}|[1-9]\\d{3}|[1-5]\\d{4}|"
  "6[0-4]\\d{3}|65[0-4]\\d{2}|655[0-2]\\d|6553[0-5])$";

// regex for verify video file name
const std::string kRegexVideoFile = "^.+\\.(mp4|h264|h265)$";

// regex for verify RTSP rtsp://ip:port/channelname
const std::string kRegexRtsp = "^rtsp://.*";
}  // namespace

bool IsDigitStr(const std::string &str) { return std::all_of(str.begin(), str.end(), isdigit); }

bool IsPathExist(const std::string &path) {
  std::ifstream file(path, std::ios::in);
  if (!file) {
    return false;
  }
  file.close();
  return true;
}

bool IsVideoFile(const std::string &path) {
  std::regex regexVideoFile(kRegexVideoFile.c_str());
  return regex_match(path, regexVideoFile);
}

bool IsRtspAddr(const std::string &str) {
  std::regex regexRtspAddress(kRegexRtsp.c_str());

  return regex_match(str, regexRtspAddress);
}

bool IsIpAddrWithPort(const std::string &addrStr) {
  std::regex regexIpAddr(kRegexIpAddr.c_str());

  return regex_match(addrStr, regexIpAddr);
}

void ParseIpAddr(std::string &ip, std::string &port, const std::string &addr) {
  std::string::size_type pos = addr.find(':');

  (void)ip.assign(addr.substr(0, pos));
  (void)port.assign(addr.substr(pos + 1));
}

bool IsDirectory(const std::string &path) {
  // get path stat
  struct stat buf {};
  if (stat(path.c_str(), &buf) != kStatSuccess) {
    return false;
  }

  // check
  return S_ISDIR(buf.st_mode);
}

void SplitPath(const std::string &path, std::vector<std::string> &pathVec) {
  char *imageFile = strtok(const_cast<char *>(path.c_str()), kImagePathSeparator.c_str());
  while (imageFile) {
    (void)pathVec.emplace_back(imageFile);
    imageFile = strtok(nullptr, kImagePathSeparator.c_str());
  }
}

void GetPathFiles(const std::string &path, std::vector<std::string> &fileVec) {
  if (IsDirectory(path)) {
    DIR *dir = opendir(path.c_str());
    struct dirent *direntPtr;
    while ((direntPtr = readdir(dir)) != nullptr) {
      // skip . and ..
      if (direntPtr->d_name[0] == '.') {
        continue;
      }

      // file path
      std::string fullPath = path + kPathSeparator + direntPtr->d_name;
      // directory need recursion
      if (IsDirectory(fullPath)) {
        GetPathFiles(fullPath, fileVec);
      } else {
        // put file
        (void)fileVec.emplace_back(fullPath);
      }
    }
    closedir(dir);
  } else {
    (void)fileVec.emplace_back(path);
  }
}

void GetAllFiles(const std::string &pathList, std::vector<std::string> &fileVec) {
  // split file path
  std::vector<std::string> pathVec;
  SplitPath(pathList, pathVec);

  for (const std::string &everyPath : pathVec) {
    // check path exist or not
    if (!IsPathExist(everyPath)) {
      ACLLITE_LOG_ERROR("Failed to deal path=%s. Reason: not exist or can not access.", everyPath.c_str());
      continue;
    }
    // get files in path and sub-path
    GetPathFiles(everyPath, fileVec);
  }
}

void *MallocMemory(uint32_t dataSize, MemoryType memType) {
  void *buffer = nullptr;
  aclError aclRet = ACL_SUCCESS;

  switch (memType) {
    case MemoryType::MEMORY_NORMAL:
      buffer = new uint8_t[dataSize];
      break;
    case MemoryType::MEMORY_HOST:
      aclRet = CALL_ASCEND_API(aclrtMallocHost, &buffer, dataSize);
      break;
    case MemoryType::MEMORY_DEVICE:
      aclRet = CALL_ASCEND_API(aclrtMalloc, &buffer, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
      break;
    case MemoryType::MEMORY_DVPP:
      aclRet = acldvppMalloc(&buffer, dataSize);
      break;
    default:
      ACLLITE_LOG_ERROR("Invalid memory type %d", memType);
      aclRet = ACL_ERROR_INVALID_PARAM;
      break;
  }

  if ((aclRet != ACL_SUCCESS) || (buffer == nullptr)) {
    ACLLITE_LOG_ERROR("Malloc memory failed, type: %d, errorno:%d", memType, aclRet);
    return nullptr;
  }

  return buffer;
}

void FreeMemory(void *mem, MemoryType memType) {
  if (mem == nullptr) {
    ACLLITE_LOG_ERROR("Invalid mem");
    return;
  }
  aclError ret = ACL_SUCCESS;
  switch (memType) {
    case MemoryType::MEMORY_NORMAL:
      delete[](reinterpret_cast<uint8_t *>(mem));
      break;
    case MemoryType::MEMORY_HOST:
      ret = CALL_ASCEND_API(aclrtFreeHost, mem);
      if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("aclrtFreeHost failed, errorno: %d", ret);
      }
      break;
    case MemoryType::MEMORY_DEVICE:
      ret = CALL_ASCEND_API(aclrtFree, mem);
      if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("aclrtFree failed, errorno: %d", ret);
      }
      break;
    case MemoryType::MEMORY_DVPP:
      ret = acldvppFree(mem);
      if (ret != ACL_SUCCESS) {
        ACLLITE_LOG_ERROR("acldvppFree failed, errorno: %d", ret);
      }
      break;
    default:
      ACLLITE_LOG_ERROR("Invalid memory type %d", memType);
      break;
  }
}

aclrtMemcpyKind GetCopyPolicy(aclrtRunMode srcDev, CopyDirection direct, MemoryType memType) {
  aclrtMemcpyKind policy = ACL_MEMCPY_HOST_TO_HOST;

  if (direct == CopyDirection::TO_DEVICE) {
    if (srcDev == ACL_HOST) {
      policy = ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
      policy = ACL_MEMCPY_DEVICE_TO_DEVICE;
    }
  } else {  // TO_HOST
    if (srcDev == ACL_DEVICE) {
      policy = ACL_MEMCPY_DEVICE_TO_HOST;
    }
  }

  return policy;
}

void *CopyDataToDevice(const void *data, uint32_t size, aclrtRunMode curRunMode, MemoryType memType) {
  if ((data == nullptr) || (size == 0) || ((curRunMode != ACL_HOST) && (curRunMode != ACL_DEVICE)) ||
      (memType >= MemoryType::MEMORY_INVALID_TYPE) || (memType == MemoryType::MEMORY_HOST)) {
    ACLLITE_LOG_ERROR(
      "Copy data args invalid, data %p, "
      "size %d, src dev %d, memory type %d",
      data, size, curRunMode, memType);
    return nullptr;
  }

  aclrtMemcpyKind policy = GetCopyPolicy(curRunMode, CopyDirection::TO_DEVICE, memType);

  return CopyData(data, size, policy, memType);
}

AclLiteError CopyDataToDeviceEx(void *dest, uint32_t destSize, const void *src, uint32_t srcSize,
                                aclrtRunMode runMode) {
  aclrtMemcpyKind policy = ACL_MEMCPY_HOST_TO_DEVICE;
  if (runMode == ACL_DEVICE) {
    policy = ACL_MEMCPY_DEVICE_TO_DEVICE;
  }

  aclError aclRet = CALL_ASCEND_API(aclrtMemcpy, dest, destSize, src, srcSize, policy);
  if (aclRet != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Copy data to device failed, aclRet is %d", aclRet);
    return ACLLITE_ERROR;
  }

  return ACLLITE_OK;
}

void *CopyDataToHost(const void *data, uint32_t size, aclrtRunMode curRunMode, MemoryType memType) {
  if ((data == nullptr) || (size == 0) || ((curRunMode != ACL_HOST) && (curRunMode != ACL_DEVICE)) ||
      ((memType != MemoryType::MEMORY_HOST) && (memType != MemoryType::MEMORY_NORMAL))) {
    ACLLITE_LOG_ERROR(
      "Copy data args invalid, data %p, "
      "size %d, src dev %d, memory type %d",
      data, size, curRunMode, memType);
    return nullptr;
  }

  aclrtMemcpyKind policy = GetCopyPolicy(curRunMode, CopyDirection::TO_HOST, memType);

  return CopyData(data, size, policy, memType);
}

AclLiteError CopyDataToHostEx(void *dest, uint32_t destSize, const void *src, uint32_t srcSize, aclrtRunMode runMode) {
  aclrtMemcpyKind policy = ACL_MEMCPY_DEVICE_TO_HOST;
  if (runMode == ACL_DEVICE) {
    policy = ACL_MEMCPY_DEVICE_TO_DEVICE;
  }

  aclError aclRet = CALL_ASCEND_API(aclrtMemcpy, dest, destSize, src, srcSize, policy);
  if (aclRet != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Copy data to device failed, aclRet is %d", aclRet);
    return ACLLITE_ERROR;
  }

  return ACLLITE_OK;
}

void *CopyData(const void *data, uint32_t size, aclrtMemcpyKind policy, MemoryType memType) {
  void *buffer = MallocMemory(size, memType);
  if (buffer == nullptr) {
    return nullptr;
  }

  aclError aclRet = CALL_ASCEND_API(aclrtMemcpy, buffer, size, data, size, policy);
  if (aclRet != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("Copy data to device failed, aclRet is %d", aclRet);
    FreeMemory(buffer, memType);
    return nullptr;
  }

  return buffer;
}

AclLiteError CopyImageToLocal(ImageData &destImage, ImageData &srcImage, aclrtRunMode curRunMode) {
  void *data = CopyDataToHost(srcImage.data.get(), srcImage.size, curRunMode, MemoryType::MEMORY_NORMAL);
  if (data == nullptr) {
    return ACLLITE_ERROR_COPY_DATA;
  }

  destImage.format = srcImage.format;
  destImage.width = srcImage.width;
  destImage.height = srcImage.height;
  destImage.size = srcImage.size;
  destImage.alignWidth = srcImage.alignWidth;
  destImage.alignHeight = srcImage.alignHeight;
  destImage.data = SHARED_PTR_U8_BUF(data);

  return ACLLITE_OK;
}

AclLiteError CopyImageToDevice(ImageData &destImage, ImageData &srcImage, aclrtRunMode curRunMode, MemoryType memType) {
  void *data = CopyDataToDevice(srcImage.data.get(), srcImage.size, curRunMode, memType);
  if (data == nullptr) {
    return ACLLITE_ERROR_COPY_DATA;
  }

  destImage.format = srcImage.format;
  destImage.width = srcImage.width;
  destImage.height = srcImage.height;
  destImage.size = srcImage.size;
  destImage.alignWidth = srcImage.alignWidth;
  destImage.alignHeight = srcImage.alignHeight;

  if (memType == MemoryType::MEMORY_DEVICE) {
    destImage.data = SHARED_PTR_DEV_BUF(data);
  } else {
    destImage.data = SHARED_PTR_DVPP_BUF(data);
  }

  return ACLLITE_OK;
}

AclLiteError ReadBinFile(const std::string &fileName, void *&data, uint32_t &size) {
  struct stat sBuf {};
  int fileStatus = stat(fileName.data(), &sBuf);
  if (fileStatus == -1) {
    ACLLITE_LOG_ERROR("failed to get file");
    return ACLLITE_ERROR_ACCESS_FILE;
  }
  if (S_ISREG(sBuf.st_mode) == 0) {
    ACLLITE_LOG_ERROR("%s is not a file, please enter a file", fileName.c_str());
    return ACLLITE_ERROR_INVALID_FILE;
  }
  std::ifstream binFile(fileName, std::ifstream::in | std::ifstream::binary);
  if (!binFile.is_open()) {
    ACLLITE_LOG_ERROR("open file %s failed", fileName.c_str());
    return ACLLITE_ERROR_OPEN_FILE;
  }

  (void)binFile.seekg(0, std::ifstream::end);
  uint32_t binFileBufferLen = binFile.tellg();
  if (binFileBufferLen == 0) {
    ACLLITE_LOG_ERROR("binfile is empty, filename is %s", fileName.c_str());
    binFile.close();
    return ACLLITE_ERROR_INVALID_FILE;
  }

  (void)binFile.seekg(0, std::ifstream::beg);

  auto *binFileBufferData = new (std::nothrow) uint8_t[binFileBufferLen];
  if (binFileBufferData == nullptr) {
    ACLLITE_LOG_ERROR("malloc binFileBufferData failed");
    binFile.close();
    return ACLLITE_ERROR_MALLOC;
  }
  (void)binFile.read(reinterpret_cast<char *>(binFileBufferData), binFileBufferLen);
  binFile.close();

  data = binFileBufferData;
  size = binFileBufferLen;

  return ACLLITE_OK;
}

AclLiteError ReadJpeg(ImageData &image, const std::string &fileName) {
  uint32_t size = 0;
  void *buf = nullptr;

  auto lite_ret = ReadBinFile(fileName, buf, size);
  if (lite_ret != ACLLITE_OK) {
    delete[](reinterpret_cast<uint8_t *>(buf));
    return lite_ret;
  }

  int32_t ch = 0;
  auto ret = acldvppJpegGetImageInfo(buf, size, &(image.width), &(image.height), &ch);
  if (ret != ACL_SUCCESS) {
    ACLLITE_LOG_ERROR("acldvppJpegGetImageInfo failed, errorno: %d", ret);
    delete[](reinterpret_cast<uint8_t *>(buf));
    return ACLLITE_ERROR;
  }
  if (image.width == 0 || image.height == 0) {
    ACLLITE_LOG_ERROR("unsupported format, only Baseline JPEG");
    delete[](reinterpret_cast<uint8_t *>(buf));
    return ACLLITE_ERROR;
  }
  image.data.reset(reinterpret_cast<uint8_t *>(buf), [](const uint8_t *p) { delete[](p); });
  image.size = size;

  return ACLLITE_OK;
}

void SaveBinFile(const std::string &filename, const void *data, uint32_t size) {
  FILE *outFileFp = fopen(filename.c_str(), "wb+");
  if (outFileFp == nullptr) {
    ACLLITE_LOG_ERROR("Save file %s failed for open error", filename.c_str());
    return;
  }
  (void)fwrite(data, 1, size, outFileFp);

  (void)fflush(outFileFp);
  (void)fclose(outFileFp);
}

bool IsSpace(char c) { return (c == BLANK_SPACE_CHAR || c == TABLE_CHAR); }

void Trim(std::string &str) {
  if (str.empty()) {
    return;
  }
  int32_t i;
  int32_t start_pos;
  int32_t end_pos;
  for (i = 0; i < str.size(); ++i) {
    if (!IsSpace(str[i])) {
      break;
    }
  }
  if (i == str.size()) {  // is all blank space
    str = "";
    return;
  }

  start_pos = i;

  for (i = str.size() - 1; i >= 0; --i) {
    if (!IsSpace(str[i])) {
      break;
    }
  }
  end_pos = i;

  str = str.substr(start_pos, end_pos - start_pos + 1);
}

bool AnalyseLine(const std::string &line, std::string &key, std::string &value) {
  if (line.empty()) {
    return false;
  }

  int start_pos = 0;
  auto end_pos = line.size() - 1;
  std::string::size_type pos = line.find(COMMENT_CHAR);
  if (pos != std::string::npos) {
    if (pos == 0) {  // the first character is #
      return false;
    }
    end_pos = pos - 1;
  }
  std::string new_line = line.substr(start_pos, end_pos - start_pos + 1);  // delete comment
  pos = new_line.find(EQUALS_CHAR);
  if (pos == std::string::npos) {  // has no =
    return false;
  }

  key = new_line.substr(0, pos);
  value = new_line.substr(pos + 1, end_pos + 1 - (pos + 1));

  Trim(key);
  if (key.empty()) {
    return false;
  }
  Trim(value);
  return true;
}

bool ReadConfig(std::map<std::string, std::string> &config, const char *configFile) {
  config.clear();
  std::ifstream infile(configFile, std::ifstream::in);
  if (!infile) {
    return false;
  }
  std::string line;
  std::string key;
  std::string value;
  while (getline(infile, line)) {
    if (AnalyseLine(line, key, value)) {
      config[key] = value;
    }
  }

  infile.close();
  return true;
}

void PrintConfig(const std::map<std::string, std::string> &config) {
  auto mIter = config.begin();
  for (; mIter != config.end(); ++mIter) {
    std::cout << mIter->first << "=" << mIter->second << std::endl;
  }
}
