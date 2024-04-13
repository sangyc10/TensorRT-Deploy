#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "trt_calibrator.hpp"
#include <string>

using namespace std;
using namespace nvinfer1;
using namespace nvonnxparser;

namespace model{

Model::Model(string onnx_path, logger::Level level, Params params) {
    m_onnxPath      = onnx_path;
    m_workspaceSize = WORKSPACESIZE;
    m_logger        = make_shared<logger::Logger>(level);
    m_timer         = make_shared<timer::Timer>();
    m_params        = new Params(params);
    m_enginePath    = changePath(onnx_path, "../engine", ".engine", getPrec(params.prec));
}

void Model::load_image(string image_path) {
    if (!fileExists(image_path)){
        LOGE("%s not found", image_path.c_str());
    } else {
        m_imagePath = image_path;
        LOG("*********************INFERENCE INFORMATION***********************");
        LOG("\tModel:      %s", getFileName(m_onnxPath).c_str());
        LOG("\tImage:      %s", getFileName(m_imagePath).c_str());
        LOG("\tPrecision:  %s", getPrec(m_params->prec).c_str());
    }
}

void Model::init_model() {
    if (m_context == nullptr){              //->如果context存在，直接进行推理，无需创建或加载engine文件
        if (!fileExists(m_enginePath)){     //->检查engine文件是否存在
            LOG("%s not found. Building trt engine...", m_enginePath.c_str());
            build_engine();                 //->创建engine文件
        } else {
            LOG("%s has been generated! loading trt engine...", m_enginePath.c_str());
            load_engine();                  //->加载engine文件
        }
    }else{
        reset_task();
    }
}

bool Model::build_engine() {
    auto builder = shared_ptr<IBuilder>(createInferBuilder(*m_logger), destroy_trt_ptr<IBuilder>);
    auto network = shared_ptr<INetworkDefinition>(builder->createNetworkV2(1), destroy_trt_ptr<INetworkDefinition>);
    auto config  = shared_ptr<IBuilderConfig>(builder->createBuilderConfig(), destroy_trt_ptr<IBuilderConfig>);
    auto parser  = shared_ptr<IParser>(createParser(*network, *m_logger), destroy_trt_ptr<IParser>);

    config->setMaxWorkspaceSize(m_workspaceSize);                   //->设置最大工作空间大小
    config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);   //->设置创建engine时的层分析信息等级

    if (!parser->parseFromFile(m_onnxPath.c_str(), 1)){             //->加载ONNX文件
        return false;
    }

    if (builder->platformHasFastFp16() && m_params->prec == model::FP16) {        //->判断设备是否支持FP16；判断设置的模型推理精度
        config->setFlag(BuilderFlag::kFP16);                                      //->设置为FP16
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);              //->FP16精度构建失败则使用其他可行精度构建（根据需求设计参数）
    } else if (builder->platformHasFastInt8() && m_params->prec == model::INT8) { //->判断设备是否支持INT8；判断设置的模型推理精度
        config->setFlag(BuilderFlag::kINT8);                                      //->设置为INT8
        config->setFlag(BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);              //->INT8精度构建失败则使用其他可行精度构建（根据需求设计参数）
    }

    shared_ptr<Int8EntropyCalibrator> calibrator(new Int8EntropyCalibrator(       //->INT8校准
        m_params->cal_batchsize,                                                  //->校准batchsize
        m_params->cal_list,                                                       //->校准数据列表
        m_params->cal_table,                                                      //->校准参数缓存
        m_params->img.c * m_params->img.h * m_params->img.w, m_params->img.h, m_params->img.w, static_cast<int>(m_params->task)));
    config->setInt8Calibrator(calibrator.get());

    auto engine  = shared_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config), destroy_trt_ptr<ICudaEngine>);
    auto plan    = builder->buildSerializedNetwork(*network, *config);
    auto runtime = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);

    save_plan(*plan);                   //->保存序列化后的engine

    setup(plan->data(), plan->size());  //->根据runtime初始化engine, context, 以及memory

    //->打印优化前和优化后各个层的信息
    LOGV("Before TensorRT optimization");
    print_network(*network, false);
    LOGV("After TensorRT optimization");
    print_network(*network, true);

    return true;
}

bool Model::load_engine() {    
    if (!fileExists(m_enginePath)) {            //->判断engine文件是否存在
        LOGE("engine does not exits! Program terminated");
        return false;
    }

    vector<unsigned char> modelData;
    modelData = loadFile(m_enginePath);         //->反序列化engine文件
    
    setup(modelData.data(), modelData.size());  //->根据runtime初始化engine, context, 以及memory

    return true;
}

void Model::save_plan(IHostMemory& plan) {      //->保存序列化文件
    auto f = fopen(m_enginePath.c_str(), "wb");
    fwrite(plan.data(), 1, plan.size(), f);
    fclose(f);
}


void Model::inference() {       //->推理
    if (m_params->dev == CPU) {
        preprocess_cpu();       //->CPU前处理
    } else {
        preprocess_gpu();       //->GPU前处理
    }

    enqueue_bindings();         //->模型推理

    if (m_params->dev == CPU) {
        postprocess_cpu();      //->CPU后处理
    } else {
        postprocess_gpu();      //->实际上YOLOV5后处理在CPU做的
    }
}


bool Model::enqueue_bindings() {
    m_timer->start_gpu();   //->GPU计时
    if (!m_context->enqueueV2((void**)m_DeviceBindings.data(), m_stream, nullptr)){ //推理，m_DeviceBindings为绑定的输入输出
        LOG("Error happens during DNN inference part, program terminated");
        return false;
    }
    m_timer->stop_gpu();
    m_timer->duration_gpu("trt-inference(GPU)");
    return true;
}

void Model::print_network(INetworkDefinition &network, bool optimized) {

    int inputCount = network.getNbInputs();
    int outputCount = network.getNbOutputs();
    string layer_info;

    for (int i = 0; i < inputCount; i++) {
        auto input = network.getInput(i);
        LOGV("Input info: %s:%s", input->getName(), printTensorShape(input).c_str());
    }

    for (int i = 0; i < outputCount; i++) {
        auto output = network.getOutput(i);
        LOGV("Output info: %s:%s", output->getName(), printTensorShape(output).c_str());
    }
    
    int layerCount = optimized ? m_engine->getNbLayers() : network.getNbLayers();
    LOGV("network has %d layers", layerCount);

    if (!optimized) {
        for (int i = 0; i < layerCount; i++) {
            char layer_info[1000];
            auto layer   = network.getLayer(i);
            auto input   = layer->getInput(0);
            int n = 0;
            if (input == nullptr){
                continue;
            }
            auto output  = layer->getOutput(0);

            LOGV("layer_info: %-40s:%-25s->%-25s[%s]", 
                layer->getName(),
                printTensorShape(input).c_str(),
                printTensorShape(output).c_str(),
                getPrecision(layer->getPrecision()).c_str());
        }

    } else {
        auto inspector = shared_ptr<IEngineInspector>(m_engine->createEngineInspector());
        for (int i = 0; i < layerCount; i++) {
            LOGV("layer_info: %s", inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
        }
    }
}

string Model::getPrec(model::precision prec) {
    switch(prec) {
        case model::precision::FP16:   return "fp16";
        case model::precision::INT8:   return "int8";
        default:                       return "fp32";
    }
}

} //->namespace model
