/*
 * Copyright (c) 2020.Huawei Technologies Co., Ltd. All rights reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <dirent.h>
#include <sys/stat.h>
#include <gflags/gflags.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include "../inc/AclProcess.h"
#include "../inc/CommonDataType.h"

DEFINE_string(om_path, "./maskrcnn.om", "om model path.");
DEFINE_string(data_path, "./test.jpg", "om model path.");
DEFINE_int32(width, 1280, "width");
DEFINE_int32(height, 768, "height");
DEFINE_int32(device_id, 0, "height");

static bool is_file(const std::string &filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

static bool is_dir(const std::string &filefodler) {
    struct stat buffer;
    return (stat(filefodler.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}
/*
 * @description Initialize and run AclProcess module
 * @param resourceInfo resource info of deviceIds, model info, single Operator Path, etc
 * @param file the absolute path of input file
 * @return int int code
 */
int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::cout << "OM File Path :" << FLAGS_om_path << std::endl;
    std::cout << "data Path :" << FLAGS_data_path << std::endl;
    std::cout << "width :" << FLAGS_width << std::endl;
    std::cout << "height :" << FLAGS_height << std::endl;
    std::cout << "deviceId :" << FLAGS_device_id << std::endl;

    char omAbsPath[PATH_MAX];
    if (realpath(FLAGS_om_path.c_str(), omAbsPath) == nullptr) {
        std::cout << "Failed to get the om real path." << std::endl;
        return INVALID_PARAM;
    }

    if (access(omAbsPath, R_OK) == -1) {
        std::cout << "ModelPath " << omAbsPath << " doesn't exist or read failed." << std::endl;
        return INVALID_PARAM;
    }

    char dataAbsPath[PATH_MAX];
    if (realpath(FLAGS_data_path.c_str(), dataAbsPath) == nullptr) {
        std::cout << "Failed to get the data real path." << std::endl;
        return INVALID_PARAM;
    }
    if (access(dataAbsPath, R_OK) == -1) {
        std::cout << "data paeh " << dataAbsPath << " doesn't exist or read failed." << std::endl;
        return INVALID_PARAM;
    }

    std::map<double, double> costTime_map;
    AclProcess aclProcess(FLAGS_device_id, FLAGS_om_path, FLAGS_width, FLAGS_height);
    int ret = aclProcess.InitResource();
    if (ret != OK) {
        aclProcess.Release();
        return ret;
    }
    if (is_file(FLAGS_data_path)) {
        ret = aclProcess.Process(FLAGS_data_path, &costTime_map);
        if (ret != OK) {
            std::cout << "model process failed, errno = " << ret << std::endl;
            return ret;
        }
    } else if (is_dir(FLAGS_data_path)) {
        struct dirent *filename;
        DIR *dir;
        dir = opendir(FLAGS_data_path.c_str());
        if (dir == nullptr) {
            return ERROR;
        }

        while ((filename = readdir(dir)) != nullptr) {
            if (strcmp(filename->d_name, ".") == 0 || strcmp(filename->d_name, "..") == 0) {
                continue;
            }
            std::string wholePath = FLAGS_data_path + "/" + filename->d_name;
            ret = aclProcess.Process(wholePath, &costTime_map);
            if (ret != OK) {
                std::cout << "model process failed, errno = " << ret << std::endl;
                return ret;
            }
        }
    } else {
        std::cout << " input image path error" << std::endl;
    }

    double average = 0.0;
    int infer_cnt = 0;

    for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
        double diff = 0.0;
        diff = iter->second - iter->first;
        average += diff;
        infer_cnt++;
    }
    average = average / infer_cnt;
    std::stringstream timeCost;
    timeCost << "NN inference cost average time: "<< average << " ms of infer_count " << infer_cnt << std::endl;
    std::cout << "NN inference cost average time: "<< average << "ms of infer_count " << infer_cnt << std::endl;
    std::string file_name = "./time_Result" + std::string("/test_perform_static.txt");
    std::ofstream file_stream(file_name.c_str(), std::ios::trunc);
    file_stream << timeCost.str();
    file_stream.close();
    costTime_map.clear();

    aclProcess.Release();
    return OK;
}
