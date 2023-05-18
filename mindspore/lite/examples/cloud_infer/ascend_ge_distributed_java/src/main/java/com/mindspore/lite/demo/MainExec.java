/*
 * Copyright 2023 Huawei Technologies Co., Ltd
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

package com.mindspore.lite.demo;

import java.util.ArrayList;
import java.io.*;

public class MainExec {
    public static void main(String[] args) {
        String pkgClassPath = "path to the project jar file";
        String msClassPath = "path to mindspore-lite-java.jar file";
        String classPaths = pkgClassPath + ":" + msClassPath;
        String configPath = "path to the model config file";
        String firstMainModelPath = "path to the first mindir main model";
        String firstIncModelPath = "path to the first mindir incremental model";
        String secondMainModelPath = "path to the second mindir main model";
        String secondIncModelPath = "path to the second mindir incremental model";
        String firstDeviceId = "4";
        String secondDeviceId = "5";
        String firstRankId = "0";
        String secondRankId = "1";
        String[] command1 = {"java", "-classpath", classPaths, "com.mindspore.lite.demo.Main", firstMainModelPath,
            firstIncModelPath, firstDeviceId, firstRankId, configPath};
        String[] command2 = {"java", "-classpath", classPaths, "com.mindspore.lite.demo.Main", secondMainModelPath,
            secondIncModelPath, secondDeviceId, secondRankId, configPath};
        ProcessBuilder pb1 = new ProcessBuilder().command(command1).redirectOutput(ProcessBuilder.Redirect.INHERIT);
        ProcessBuilder pb2 = new ProcessBuilder().command(command2).redirectOutput(ProcessBuilder.Redirect.INHERIT);
        pb1.redirectErrorStream(true);
        pb2.redirectErrorStream(true);
        ArrayList<ProcessBuilder> processBuilders = new ArrayList<>();
        ArrayList<Process> processes = new ArrayList<>();
        processBuilders.add(pb1);
        processBuilders.add(pb2);
        for (ProcessBuilder pb : processBuilders) {
            try {
                Process p = pb.start();
                processes.add(p);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        for (Process p : processes) {
            try {
                p.waitFor();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        System.out.println("All processes execute done!");
    }
}
