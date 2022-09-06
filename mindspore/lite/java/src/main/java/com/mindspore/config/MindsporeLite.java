package com.mindspore.config;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.logging.FileHandler;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public final class MindsporeLite {
    private static final Object lock = new Object();
    private static Logger LOGGER = GetLogger();

    public static Logger GetLogger() {
        if (LOGGER == null) {
            synchronized (lock) {
                if (LOGGER != null) {
                    return LOGGER;
                }
                LOGGER = Logger.getLogger(MindsporeLite.class.toString());
                String logtostderr = System.getenv("GLOG_logtostderr");
                if ("0".equals(logtostderr)) {
                    String GLOG_log_dir = System.getenv("GLOG_log_dir");
                    if (GLOG_log_dir != null && !"".equals(GLOG_log_dir)) {
                        FileHandler fileHandler = null;
                        try {
                            DateTimeFormatter dtf = DateTimeFormatter.ofPattern("uuuuMMdd-HHmmss.SSSS");
                            LocalDateTime now = LocalDateTime.now();
                            fileHandler = new FileHandler(GLOG_log_dir + "/MinsSporeLite.java_log." + dtf.format(now) + ".log");
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                        SimpleFormatter formatter = new SimpleFormatter();
                        fileHandler.setFormatter(formatter);
                        LOGGER.addHandler(fileHandler);
                    }
                }
            }
        }
        return LOGGER;
    }

    /**
     * Init function.
     */
    public static void init() {
        LOGGER.info("MindsporeLite init load ...");
        try {
            NativeLibrary.load();
        } catch (Exception e) {
            LOGGER.severe("Failed to load MindSporLite native library.");
            throw e;
        }
    }

    static {
        LOGGER.info("MindsporeLite init ...");
        init();
    }
}
