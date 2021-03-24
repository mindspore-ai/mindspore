package com.mindspore.lite.train_lenet;

import com.mindspore.lite.Version;

public class Main {
    public static void main(String[] args) {
        System.out.println(Version.version());
        if (args.length < 2) {
            System.err.println("model path and dataset path must be provided.");
            return;
        }
        String modelPath = args[0];
        String datasetPath = args[1];

        NetRunner net_runner = new NetRunner();
        net_runner.trainModel(modelPath, datasetPath);
    }


}
