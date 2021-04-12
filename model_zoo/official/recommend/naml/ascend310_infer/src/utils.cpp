/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "../inc/utils.h"

extern bool g_isDevice;

void* Utils::ReadBinFile(std::string fileName, uint32_t *fileSize) {
    struct stat sBuf;
    int fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file");
        return nullptr;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return nullptr;
    }

    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ERROR_LOG("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);

    void* binFileBufferData = nullptr;
    if (!g_isDevice) {
        aclrtMallocHost(&binFileBufferData, binFileBufferLen);
        if (binFileBufferData == nullptr) {
            ERROR_LOG("malloc binFileBufferData failed");
            binFile.close();
            return nullptr;
        }
    } else {
        aclError ret = aclrtMalloc(&binFileBufferData, binFileBufferLen, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed. size is %u", binFileBufferLen);
            binFile.close();
            return nullptr;
        }
    }
    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    *fileSize = binFileBufferLen;
    return binFileBufferData;
}

void* Utils::GetDeviceBufferOfFile(std::string fileName, uint32_t *fileSize) {
    uint32_t inputHostBuffSize = 0;
    void* inputHostBuff = Utils::ReadBinFile(fileName, &inputHostBuffSize);
    if (inputHostBuff == nullptr) {
        return nullptr;
    }
    if (!g_isDevice) {
        void *inBufferDev = nullptr;
        uint32_t inBufferSize = inputHostBuffSize;
        aclError ret = aclrtMalloc(&inBufferDev, inBufferSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("malloc device buffer failed. size is %u", inBufferSize);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }

        ret = aclrtMemcpy(inBufferDev, inBufferSize, inputHostBuff, inputHostBuffSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("memcpy failed. device buffer size is %u, input host buffer size is %u",
                inBufferSize, inputHostBuffSize);
            aclrtFree(inBufferDev);
            aclrtFreeHost(inputHostBuff);
            return nullptr;
        }
        aclrtFreeHost(inputHostBuff);
        *fileSize = inBufferSize;
        return inBufferDev;
    } else {
        *fileSize = inputHostBuffSize;
        return inputHostBuff;
    }
}

std::vector<std::vector<std::string>> Utils::GetAllInputData(std::string dir_name) {
    DIR *dir = OpenDir(dir_name);
    if (dir == nullptr) {
        return {};
    }
    struct dirent *filename;

    std::vector<std::string> sub_dirs;
    while ((filename = readdir(dir)) != nullptr) {
        std::string d_name = std::string(filename->d_name);
        if (d_name == "." || d_name == ".." || d_name.empty() || d_name[0] != '0') {
            continue;
        }

        std::string dir_path = RealPath(std::string(dir_name) + "/" + filename->d_name);
        struct stat s;
        lstat(dir_path.c_str(), &s);
        if (!S_ISDIR(s.st_mode)) {
            continue;
        }

        sub_dirs.emplace_back(dir_path);
    }
    std::sort(sub_dirs.begin(), sub_dirs.end());

    std::vector<std::vector<std::string>> result(sub_dirs.size());

    std::transform(sub_dirs.begin(), sub_dirs.end(), result.begin(), GetAllBins);

    return result;
}

DIR *Utils::OpenDir(std::string dir_name) {
    // check the parameter !
    if (dir_name.empty()) {
        std::cout << " dir_name is null ! " << std::endl;
        return nullptr;
    }

    std::string real_path = RealPath(dir_name);

    // check if dir_name is a valid dir
    struct stat s;
    lstat(real_path.c_str(), &s);
    if (!S_ISDIR(s.st_mode)) {
        std::cout << "dir_name is not a valid directory !" << std::endl;
        return nullptr;
    }

    DIR *dir;
    dir = opendir(real_path.c_str());
    if (dir == nullptr) {
        std::cout << "Can not open dir " << dir_name << std::endl;
        return nullptr;
    }
    std::cout << "Successfully opened the dir " << dir_name << std::endl;
    return dir;
}

std::string Utils::RealPath(std::string path) {
    char real_path_mem[PATH_MAX] = {0};
    char *real_path_ret = nullptr;
    real_path_ret = realpath(path.data(), real_path_mem);
    if (real_path_ret == nullptr) {
        std::cout << "File: " << path << " is not exist.";
        return "";
    }

    std::string real_path(real_path_mem);
    std::cout << path << " realpath is: " << real_path << std::endl;
    return real_path;
}

std::vector<std::string> Utils::GetAllBins(std::string dir_name) {
    struct dirent *filename;
    DIR *dir = OpenDir(dir_name);
    if (dir == nullptr) {
        return {};
    }

    std::vector<std::string> res;
    while ((filename = readdir(dir)) != nullptr) {
        std::string d_name = std::string(filename->d_name);
        if (d_name == "." || d_name == ".." || d_name.size() <= 3 || d_name.substr(d_name.size() - 4) != ".bin" ||
            filename->d_type != DT_REG) {
            continue;
        }
        res.emplace_back(std::string(dir_name) + "/" + filename->d_name);
    }

    std::sort(res.begin(), res.end());

    return res;
}

Result Utils::ReadFileToVector(std::string newsIdFileName, std::vector<int> *newsId) {
    int id;

    std::ifstream in(newsIdFileName, std::ios::in | std::ios::binary);
    while (in.read(reinterpret_cast<char *>(&id), sizeof(id))) {
        newsId->emplace_back(id);
    }
    in.close();

    return SUCCESS;
}

Result Utils::ReadFileToVector(std::string newsIdFileName, uint32_t batchSize, std::vector<int> *newsId) {
    int id;

    std::ifstream in(newsIdFileName, std::ios::in | std::ios::binary);
    for (int i = 0; i < batchSize; ++i) {
        in.read(reinterpret_cast<char *>(&id), sizeof(id));
        newsId->emplace_back(id);
    }
    in.close();

    return SUCCESS;
}

Result Utils::ReadFileToVector(std::string fileName, uint32_t batchSize,
                               uint32_t count, std::vector<std::vector<int>> *newsId) {
    int id;

    std::ifstream in(fileName, std::ios::in | std::ios::binary);
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < count; ++j) {
            in.read(reinterpret_cast<char *>(&id), sizeof(id));
            (*newsId)[i][j] = id;
        }
    }
    in.close();

    return SUCCESS;
}
