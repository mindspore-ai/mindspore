
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

#include "load_input.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void *ReadInputData(const char *real_input_path, int *size) {
  if (real_input_path == NULL) {
    return NULL;
  }
  if (strstr(real_input_path, ".bin") || strstr(real_input_path, ".net")) {
    FILE *file;
    file = fopen(real_input_path, "rb+");
    if (!file) {
      printf("Can't find %s\n", real_input_path);
      return NULL;
    }
    int curr_file_posi = ftell(file);
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    unsigned char *buf = malloc((*size));
    (void)memset(buf, 0, (*size));
    fseek(file, curr_file_posi, SEEK_SET);
    int read_size = (int)(fread(buf, 1, *size, file));
    if (read_size != (*size)) {
      printf("read file failed, total file size: %d, read_size: %d\n", (*size), read_size);
      fclose(file);
      free(buf);
      return NULL;
    }
    fclose(file);
    return (void *)buf;
  } else {
    printf("input data file should be .bin , .net");
    return NULL;
  }
}

void SaveOutputData(char *final_name, unsigned char *output_data, unsigned int out_size) {
  FILE *output_file;
  output_file = fopen(final_name, "w");
  if (output_file == NULL) {
    printf("fopen output file: %s failed\n", final_name);
    return;
  }
  unsigned char str[out_size];
  for (unsigned int i = 0; i < out_size; ++i) {
    str[i] = output_data[i];
    fprintf(output_file, "%d\t", str[i]);
  }
  fclose(output_file);
}

int ReadInputsFile(char *path, void **buffers, const int *inputs_size, int inputs_num) {
  char *inputs_path[inputs_num];
  char *delim = ",";
  char *token;
  int i = 0;
  while ((token = strtok_r(path, delim, &path))) {
    if (i >= inputs_num) {
      printf("inputs num is error, need: %d\n", inputs_num);
      return -1;
    }
    inputs_path[i] = token;
    printf("input %d: %s\n", i, inputs_path[i]);
    i++;
  }

  for (i = 0; i < inputs_num; ++i) {
    int size = 0;
    buffers[i] = ReadInputData(inputs_path[i], &size);
    if (size != inputs_size[i] || buffers[i] == NULL) {
      printf("size mismatch, %s, input: %d, needed: %d\n", inputs_path[i], size, inputs_size[i]);
      return -1;
    }
  }
  return 0;
}

