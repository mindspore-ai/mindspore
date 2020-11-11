package com.mindspore.himindspore.net;

import java.util.concurrent.TimeUnit;

import io.reactivex.Observable;
import okhttp3.OkHttpClient;
import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;

public class RetrofitHelper {
    public static final String MS_BASE_HOST = "https://download.mindspore.cn/model_zoo/official/lite/apk/";

    private RetrofitService retrofitService;


    public RetrofitHelper() {

        OkHttpClient httpClient = new OkHttpClient.Builder()
                .retryOnConnectionFailure(true)
                .connectTimeout(30, TimeUnit.SECONDS)
                .build();

        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(MS_BASE_HOST)
                .client(httpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .addCallAdapterFactory(RxJava2CallAdapterFactory.create())
                .build();

        retrofitService = retrofit.create(RetrofitService.class);
    }


    public Call<UpdateInfoBean> getUpdateInfo() {
        return retrofitService.getUpdateInfo();
    }

    public Observable<ResponseBody> downlaodApk() {
        return retrofitService.downloadApk();
    }

}