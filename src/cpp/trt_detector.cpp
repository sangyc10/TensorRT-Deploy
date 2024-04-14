#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include "trt_model.hpp"
#include "utils.hpp" 
#include "trt_logger.hpp"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <algorithm>
#include <string>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc//imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "trt_detector.hpp"
#include "trt_preprocess.hpp"
#include "traffic_vehicle_labels.hpp"

using namespace std;
using namespace nvinfer1;

namespace model{

namespace detector {

float iou_calc(bbox bbox1, bbox bbox2){             //->计算两个检测框的IOU
    auto inter_x0 = std::max(bbox1.x0, bbox2.x0);
    auto inter_y0 = std::max(bbox1.y0, bbox2.y0);
    auto inter_x1 = std::min(bbox1.x1, bbox2.x1);
    auto inter_y1 = std::min(bbox1.y1, bbox2.y1);

    float inter_w = inter_x1 - inter_x0;
    float inter_h = inter_y1 - inter_y0;
    
    float inter_area = inter_w * inter_h;
    float union_area = 
        (bbox1.x1 - bbox1.x0) * (bbox1.y1 - bbox1.y0) + 
        (bbox2.x1 - bbox2.x0) * (bbox2.y1 - bbox2.y0) - 
        inter_area;
    
    return inter_area / union_area;
}


void Detector::setup(void const* data, size_t size) {
    m_runtime   = shared_ptr<IRuntime>(createInferRuntime(*m_logger), destroy_trt_ptr<IRuntime>);
    m_engine    = shared_ptr<ICudaEngine>(m_runtime->deserializeCudaEngine(data, size), destroy_trt_ptr<ICudaEngine>);
    m_context   = shared_ptr<IExecutionContext>(m_engine->createExecutionContext(), destroy_trt_ptr<IExecutionContext>);
    m_inputDims = m_context->getBindingDimensions(0);
   
    m_DecodeNumDims     = m_context->getBindingDimensions(1);        // 获取输出1维度
    m_DecodeBoxesDims   = m_context->getBindingDimensions(2);        // 获取输出2维度
    m_DecodeScoresDims  = m_context->getBindingDimensions(3);        // 获取输出3维度
    m_DecodeClassesDims = m_context->getBindingDimensions(4);        // 获取输出4维度

    CUDA_CHECK(cudaStreamCreate(&m_stream));
    
    m_inputSize     = m_params->img.h * m_params->img.w * m_params->img.c * sizeof(float);  //->图像size
    m_imgArea       = m_params->img.h * m_params->img.w;                                    //->图像 高×宽
    m_DecodeNumSize     = m_DecodeNumDims.d[0]     * m_DecodeNumDims.d[1]     * sizeof(int);                            // 输出1 size
    m_DecodeBoxesSize   = m_DecodeBoxesDims.d[0]   * m_DecodeBoxesDims.d[1]   * m_DecodeBoxesDims.d[2] * sizeof(float); // 输出2 size
    m_DecodeScoresSize  = m_DecodeScoresDims.d[0]  * m_DecodeScoresDims.d[1]  * sizeof(float);                          // 输出3 size
    m_DecodeClassesSize = m_DecodeClassesDims.d[0] * m_DecodeClassesDims.d[1] * sizeof(int);                            // 输出4 size

    CUDA_CHECK(cudaMallocHost(&m_inputMemoryHost, m_inputSize));                //->输入分配Host pinned memory
    CUDA_CHECK(cudaMalloc(&m_inputMemoryDevice, m_inputSize));                  //->输入分配Device memory
    m_DeviceBindings.emplace_back(m_inputMemoryDevice);                         //->添加到binding

    CUDA_CHECK(cudaMallocHost(&m_outputMemoryNumHost, m_DecodeNumSize));        //->输出1分配Host pinned memory
    CUDA_CHECK(cudaMalloc(&m_outputMemoryNumDevice, m_DecodeNumSize));          //->输出1分配Device memory
    m_DeviceBindings.emplace_back(m_outputMemoryNumDevice);                     //->添加到binding

    CUDA_CHECK(cudaMallocHost(&m_outputMemoryBoxesHost, m_DecodeBoxesSize));    //->输出2分配Host pinned memory
    CUDA_CHECK(cudaMalloc(&m_outputMemoryBoxesDevice, m_DecodeBoxesSize));      //->输出2分配Device memory
    m_DeviceBindings.emplace_back(m_outputMemoryBoxesDevice);                   //->添加到binding

    CUDA_CHECK(cudaMallocHost(&m_outputMemoryScoresHost, m_DecodeScoresSize));  //->输出3分配Host pinned memory
    CUDA_CHECK(cudaMalloc(&m_outputMemoryScoresDevice, m_DecodeScoresSize));    //->输出3分配Device memory
    m_DeviceBindings.emplace_back(m_outputMemoryScoresDevice);                  //->添加到binding

    CUDA_CHECK(cudaMallocHost(&m_outputMemoryClassesHost, m_DecodeClassesSize));//->输出4分配Host pinned memory
    CUDA_CHECK(cudaMalloc(&m_outputMemoryClassesDevice, m_DecodeClassesSize));  //->输出4分配Device memory
    m_DeviceBindings.emplace_back(m_outputMemoryClassesDevice);                 //->添加到binding
}

void Detector::reset_task(){
    m_bboxes.clear();
}

bool Detector::preprocess_cpu() {
    m_inputImage = cv::imread(m_imagePath);     //->读取数据
    if (m_inputImage.data == nullptr) {
        LOGE("ERROR: Image file not founded! Program terminated"); 
        return false;
    }

    m_timer->start_cpu();                       //->CPU测速
    cv::resize(m_inputImage, m_inputImage, cv::Size(m_params->img.w, m_params->img.h), 0, 0, cv::INTER_LINEAR); //->opencv函数resize

    //->host端进行normalization和BGR2RGB, NHWC->NCHW
    int index;
    int offset_ch0 = m_imgArea * 0;
    int offset_ch1 = m_imgArea * 1;
    int offset_ch2 = m_imgArea * 2;
    for (int i = 0; i < m_inputDims.d[2]; i++) {
        for (int j = 0; j < m_inputDims.d[3]; j++) {
            index = i * m_inputDims.d[3] * m_inputDims.d[1] + j * m_inputDims.d[1];
            ((float*)m_inputMemoryHost)[offset_ch2++] = m_inputImage.data[index + 0] / 255.0f;
            ((float*)m_inputMemoryHost)[offset_ch1++] = m_inputImage.data[index + 1] / 255.0f;
            ((float*)m_inputMemoryHost)[offset_ch0++] = m_inputImage.data[index + 2] / 255.0f;
        }
    }

    //->将host的数据移动到device上
    CUDA_CHECK(cudaMemcpyAsync(m_inputMemoryDevice, m_inputMemoryHost, m_inputSize, cudaMemcpyKind::cudaMemcpyHostToDevice, m_stream));

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("preprocess(CPU)");
    return true;
}

bool Detector::preprocess_gpu() {
    m_inputImage = cv::imread(m_imagePath);     //->读取数据
    if (m_inputImage.data == nullptr) {
        LOGE("ERROR: file not founded! Program terminated"); return false;
    }
    
    m_timer->start_gpu();                       //->GPU测速

    //->使用GPU进行warpAffine, 并将结果返回到m_inputMemoryDevice中
    preprocess::preprocess_resize_gpu(m_inputImage, (float*)m_inputMemoryDevice,
                                   m_params->img.h, m_params->img.w, 
                                   preprocess::tactics::GPU_WARP_AFFINE);

    m_timer->stop_gpu();
    m_timer->duration_gpu("preprocess(GPU)");
    return true;
}


bool Detector::postprocess_cpu() {
    m_timer->start_cpu();

    //->将device上的数据移动到host上，YOLOV5 decode使用plugin重写，因此有4个输出
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemoryNumHost,     m_outputMemoryNumDevice,     m_DecodeNumSize,     cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemoryBoxesHost,   m_outputMemoryBoxesDevice,   m_DecodeBoxesSize,   cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemoryScoresHost,  m_outputMemoryScoresDevice,  m_DecodeScoresSize,  cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaMemcpyAsync(m_outputMemoryClassesHost, m_outputMemoryClassesDevice, m_DecodeClassesSize, cudaMemcpyKind::cudaMemcpyDeviceToHost, m_stream));
    CUDA_CHECK(cudaStreamSynchronize(m_stream));

    int32_t *num_det = (int32_t *)m_outputMemoryNumHost;   //->检测到的目标个数，数据类型为指针
    float *boxes = (float *)m_outputMemoryBoxesHost;       //->检测到的目标框
    float *confs = (float *)m_outputMemoryScoresHost;      //->检测到的目标置信度
    int32_t *cls = (int32_t *)m_outputMemoryClassesHost;   //->检测到的目标类别

    float conf_threshold = 0.4;     //->用来过滤decode时的bboxes
    float nms_threshold  = 0.7;     //->用来过滤nms时的bboxes

    //->YOLOV5检测框的数量，正常情况下检测框数量为25200个，plugin内做了处理，检测框最大类别概率小于0.25已经被剔除
    int    boxes_count = num_det[0];    

    float conf;             
    float x0, y0, x1, y1;   
    int   label;            

    for (int i = 0; i < boxes_count; i ++){
        label = cls[i];         //->检测框类别
        conf  = confs[i];       //->置信度
        if (conf < conf_threshold) continue;    //->置信度小于conf_threshold则认为检测框目标不存在

        x0 = boxes[i * 4 + 0];  //->图像变换原坐标前，检测框左上x坐标
        y0 = boxes[i * 4 + 1];  //->图像变换原坐标前，检测框左上y坐标
        x1 = boxes[i * 4 + 2];  //->图像变换原坐标前，检测框右下x坐标
        y1 = boxes[i * 4 + 3];  //->图像变换原坐标前，检测框右下y坐标
        
        //->通过warpaffine的逆变换得到yolo feature中的x0, y0, x1, y1在原图上的坐标
        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x0, y0, &x0, &y0);
        preprocess::affine_transformation(preprocess::affine_matrix.reverse, x1, y1, &x1, &y1);
        
        bbox yolo_box(x0, y0, x1, y1, conf, label);
        m_bboxes.emplace_back(yolo_box);        //->所有检测框存入vector
    }
    LOGD("the count of decoded bbox is %d", m_bboxes.size());
    
    //************** NMS ****************//
    vector<bbox> final_bboxes;
    final_bboxes.reserve(m_bboxes.size());
    std::sort(m_bboxes.begin(), m_bboxes.end(),     //->按照置信度从高到低排序
              [](bbox& box1, bbox& box2){return box1.confidence > box2.confidence;});

    for(int i = 0; i < m_bboxes.size(); i ++){
        if (m_bboxes[i].flg_remove)         //->检测框设置为移除则跳过当前循环
            continue;
        
        final_bboxes.emplace_back(m_bboxes[i]);
        for (int j = i + 1; j < m_bboxes.size(); j ++) {
            if (m_bboxes[j].flg_remove)     //->检测框设置为移除则跳过当前循环
                continue;

            //->当检测框为同一类别时，两检测框IOU计算，大于nms_threshold，则只保留置信度大的检测框
            if (m_bboxes[i].label == m_bboxes[j].label){    
                if (iou_calc(m_bboxes[i], m_bboxes[j]) > nms_threshold)
                    m_bboxes[j].flg_remove = true;
            }
        }
    }
    LOGD("the count of bbox after NMS is %d", final_bboxes.size());


    /***************** 绘制检测框 *********************/
    string tag   = "detect-" + getPrec(m_params->prec);
    m_outputPath = changePath(m_imagePath, "../result", ".png", tag);

    int   font_face  = 0;
    float font_scale = 0.001 * MIN(m_inputImage.cols, m_inputImage.rows);
    int   font_thick = 2;
    int   baseline;
    TrafficVehicleLabels labels;

    LOG("\tResult:");
    for (int i = 0; i < final_bboxes.size(); i ++){
        auto box = final_bboxes[i];
        auto name = labels.trafficVehicle_get_label(box.label);
        auto rec_color = labels.trafficVehicle_get_color(box.label);
        auto txt_color = labels.get_inverse_color(rec_color);
        auto txt = cv::format({"%s: %.2f%%"}, name.c_str(), box.confidence * 100);
        auto txt_size = cv::getTextSize(txt, font_face, font_scale, font_thick, &baseline);

        int txt_height = txt_size.height + baseline + 10;
        int txt_width  = txt_size.width + 3;

        cv::Point txt_pos(round(box.x0), round(box.y0 - (txt_size.height - baseline + font_thick)));
        cv::Rect  txt_rec(round(box.x0 - font_thick), round(box.y0 - txt_height), txt_width, txt_height);
        cv::Rect  box_rec(round(box.x0), round(box.y0), round(box.x1 - box.x0), round(box.y1 - box.y0));

        cv::rectangle(m_inputImage, box_rec, rec_color, 3);
        cv::rectangle(m_inputImage, txt_rec, rec_color, -1);
        cv::putText(m_inputImage, txt, txt_pos, font_face, font_scale, txt_color, font_thick, 16);

        LOG("%+20s detected. Confidence: %.2f%%. Cord: (x0, y0):(%6.2f, %6.2f), (x1, y1)(%6.2f, %6.2f)", 
            name.c_str(), box.confidence * 100, box.x0, box.y0, box.x1, box.y1);

    }
    LOG("\tSummary:");
    LOG("\t\tDetected Objects: %d", final_bboxes.size());
    LOG("");

    m_timer->stop_cpu();
    m_timer->duration_cpu<timer::Timer::ms>("postprocess(CPU)");

    cv::imwrite(m_outputPath, m_inputImage);
    LOG("\tsave image to %s\n", m_outputPath.c_str());

    return true;
}


bool Detector::postprocess_gpu() {
    return postprocess_cpu();
}


shared_ptr<Detector> make_detector(
    std::string onnx_path, logger::Level level, Params params)
{
    return make_shared<Detector>(onnx_path, level, params);
}

}; //->namespace detector
}; //->namespace model
