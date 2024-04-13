#ifndef __TRT_MODEL_HPP__
#define __TRT_MODEL_HPP__

#include <memory>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "trt_timer.hpp"
#include "trt_logger.hpp"
#include "trt_preprocess.hpp"

#define WORKSPACESIZE 1<<28

namespace model{

enum task_type {        //->任务类型
    CLASSIFICATION,
    DETECTION,
    SEGMENTATION,
    MULTITASK
};

enum device {           //->设备类型
    CPU,
    GPU
};

enum precision {        //->精度
    FP32,
    FP16,
    INT8
};

struct image_info {     //->输入图像size
    int h;
    int w;
    int c;
    image_info(int height, int width, int channel) : h(height), w(width), c(channel) {}
};

struct Params {         //->构建模型参数
    device               dev           = GPU;
    int                  num_cls       = 1000;
    preprocess::tactics  tac           = preprocess::tactics::GPU_BILINEAR;
    image_info           img           = {224, 224, 3};
    task_type            task          = CLASSIFICATION;
    int                  ws_size       = WORKSPACESIZE;
    precision            prec          = FP32;
    std::string          cal_list;
    std::string          cal_table;
    int                  cal_batchsize = 64;
};

template<typename T>            //->构建trt智能指针释放函数. trt指针的释放通过ptr->destroy完成
void destroy_trt_ptr(T* ptr){
    if (ptr) {
        std::string type_name = typeid(T).name();
        LOGD("Destroy %s", type_name.c_str());
        ptr->destroy(); 
    };
}

class Model {

public:
    Model(std::string onnx_path, logger::Level level, Params params); 
    virtual ~Model() {};
    void load_image(std::string image_path);//->加载图像
    void init_model();                      //->初始化模型，包括build推理引擎, 分配内存，创建context, 设置bindings
    void inference();                       //->推理部分，preprocess ---> enqueue ---> postprocess
    std::string getPrec(precision prec);

public:
    bool build_engine();    //->创建engine文件，创建推理上下文context, 以及分配memory
    bool load_engine();     //->加载engine文件，创建推理上下文context, 以及分配memory
    void save_plan(nvinfer1::IHostMemory& plan);    //->保存序列化文件
    void print_network(nvinfer1::INetworkDefinition &network, bool optimized);  //->打印模型信息

    bool enqueue_bindings();    //->DNN推理，设定输入输出bingings,不同的任务的推理实现是一样的

    virtual void setup(void const* data, std::size_t size) = 0; //->虚函数，setup负责分配host/device的memory, bindings, 以及创建推理所需要的上下文

    virtual void reset_task() = 0;      //->推理一次后。需要清理m_bboxes

    //->不同任务的前处理/后处理是不同的，创建虚函数在子类中重写
    virtual bool preprocess_cpu()  = 0; //->虚函数，CPU前处理
    virtual bool preprocess_gpu()  = 0; //->虚函数，GPU前处理
    virtual bool postprocess_cpu() = 0; //->虚函数，CPU后处理
    virtual bool postprocess_gpu() = 0; //->虚函数，GPU后处理

public:
    std::string m_imagePath;    //->图像路径   
    std::string m_outputPath;   //->输出路径
    std::string m_onnxPath;     //->ONNX路径
    std::string m_enginePath;   //->engine文件路径

    cv::Mat m_inputImage;
    Params* m_params;

    int    m_workspaceSize;
    cudaStream_t   m_stream;

    std::shared_ptr<logger::Logger>               m_logger;
    std::shared_ptr<timer::Timer>                 m_timer;
    std::shared_ptr<nvinfer1::IRuntime>           m_runtime;
    std::shared_ptr<nvinfer1::ICudaEngine>        m_engine;
    std::shared_ptr<nvinfer1::IExecutionContext>  m_context;
    std::shared_ptr<nvinfer1::INetworkDefinition> m_network;

    std::vector<void*> m_DeviceBindings;   //->用来保存bingds memory空间
};

}; //->namespace model

#endif //->__TRT_MODEL_HPP__
