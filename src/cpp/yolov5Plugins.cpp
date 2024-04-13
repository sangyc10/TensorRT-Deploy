/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "yolov5Plugins.hpp"
#include "NvInferPlugin.h"
#include <cassert>
#include <iostream>
#include <memory>
#define NANCHORS 3
#define NFEATURES 3

namespace
{
    template <typename T>
    void write(char *&buffer, const T &val)
    {
        *reinterpret_cast<T *>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char *&buffer, T &val)
    {
        val = *reinterpret_cast<const T *>(buffer);
        buffer += sizeof(T);
    }
}

//->静态变量声明
PluginFieldCollection YoloLayerPluginCreator::mFC{};
std::vector<PluginField> YoloLayerPluginCreator::mPluginAttributes;

YoloLayerPluginCreator::YoloLayerPluginCreator() noexcept
{
    //->从ONNX获取参数
    mPluginAttributes.emplace_back(PluginField("max_stride", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_classes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("anchors", nullptr, PluginFieldType::kFLOAT32, NFEATURES * NANCHORS * 2));
    mPluginAttributes.emplace_back(PluginField("prenms_score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();  //->PluginField的数量
    mFC.fields = mPluginAttributes.data();    //->指向PluginField vector数组的指针
}

YoloLayerPluginCreator::~YoloLayerPluginCreator() noexcept {}   //->一般不做任何操作

const char* YoloLayerPluginCreator::getPluginName() const noexcept { 
    return YOLOLAYER_PLUGIN_NAME;             //->获取插件名字
}

const char* YoloLayerPluginCreator::getPluginVersion() const noexcept { 
    return YOLOLAYER_PLUGIN_VERSION;          //->获取插件版本
}

const PluginFieldCollection* YoloLayerPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

const char* YoloLayerPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}


IPluginV2DynamicExt* YoloLayerPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept {
    /******** 从fc获取参数，实例化一个plugin *********/
    const PluginField *fields = fc->fields;
    int max_stride = 0;
    int num_classes = 0;
    std::vector<float> anchors;
    float score_threshold = 0.0;

    for (int i = 0; i < fc->nbFields; ++i) {                                    //->根据名字解析参数
        const char *attrName = fields[i].name;                                  //->获取名字
        if (!strcmp(attrName, "max_stride")) {
            assert(fields[i].type == PluginFieldType::kINT32);                  //->断言类型
            max_stride = *(static_cast<const int*>(fields[i].data));            //->变量赋值
        }
        if (!strcmp(attrName, "num_classes")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            num_classes = *(static_cast<const int*>(fields[i].data));
        }
        if (!strcmp(attrName, "anchors")) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            const auto anchors_ptr = static_cast<const float*>(fields[i].data);
            anchors.assign(anchors_ptr, anchors_ptr + NFEATURES * NANCHORS * 2);
        }
        if (!strcmp(attrName, "prenms_score_threshold")) {
            assert(fields[i].type == PluginFieldType::kFLOAT32);
            score_threshold = *(static_cast<const float *>(fields[i].data));
        }
    }
    return new YoloLayer(max_stride, num_classes, anchors, score_threshold);    //->实例化plugin
}

IPluginV2DynamicExt* YoloLayerPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept {
    std::cout << "Deserialize yoloLayer plugin: " << name << std::endl;
    return new YoloLayer(serialData, serialLength);
}

void YoloLayerPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}



cudaError_t cudaYoloLayer_nc(
    const void *input, void *num_detections, void *detection_boxes, void *detection_scores, void *detection_classes,
    const uint &batchSize, uint64_t &inputSize, uint64_t &outputSize, const float &scoreThreshold, const uint &netWidth,
    const uint &netHeight, const uint &gridSizeX, const uint &gridSizeY, const uint &numOutputClasses, const uint &numBBoxes,
    const float &scaleXY, const void *anchors, cudaStream_t stream);

YoloLayer::YoloLayer(const void *data, size_t length)
{
    const char *d = static_cast<const char *>(data);

    read(d, m_NetWidth);
    read(d, m_NetHeight);
    read(d, m_MaxStride);
    read(d, m_NumClasses);
    read(d, m_ScoreThreshold);
    read(d, m_OutputSize);

    m_Anchors.resize(NFEATURES * NANCHORS * 2);
    for (uint i = 0; i < m_Anchors.size(); i++)
    {
        read(d, m_Anchors[i]);
    }

    for (uint i = 0; i < NFEATURES; i++)
    {
        int height;
        int width;
        read(d, height);
        read(d, width);
        m_FeatureSpatialSize.push_back(DimsHW(height, width));
    }
};

YoloLayer::YoloLayer(
    const uint &maxStride, const uint &numClasses,
    const std::vector<float> &anchors, const float &scoreThreshold) : m_MaxStride(maxStride),
                                                                      m_NumClasses(numClasses),
                                                                      m_Anchors(anchors),
                                                                      m_ScoreThreshold(scoreThreshold){

                                                                      };

const char* YoloLayer::getPluginType() const noexcept { 
    return YOLOLAYER_PLUGIN_NAME;    //->获取名字
}

const char* YoloLayer::getPluginVersion() const noexcept {
    return YOLOLAYER_PLUGIN_VERSION; //->获取版本
}

int YoloLayer::getNbOutputs() const noexcept {
    return 4;                        //->输出数量为4
}

size_t YoloLayer::getSerializationSize() const noexcept {   //->序列化参数的size大小
    size_t totalSize = 0;

    totalSize += sizeof(m_NetWidth);
    totalSize += sizeof(m_NetHeight);
    totalSize += sizeof(m_MaxStride);
    totalSize += sizeof(m_NumClasses);
    totalSize += sizeof(m_ScoreThreshold);
    totalSize += sizeof(m_OutputSize);

    //->anchors
    totalSize += m_Anchors.size() * sizeof(m_Anchors[0]);

    //->feature size
    totalSize += m_FeatureSpatialSize.size() * 2 * sizeof(m_FeatureSpatialSize[0].h());

    return totalSize;
}

const char* YoloLayer::getPluginNamespace() const noexcept {
    return m_Namespace.c_str();         
}

DataType YoloLayer::getOutputDataType(int index, const DataType *inputType, int nbInputs) const noexcept {
    //->获取输出数据类型
    if (index == 0 || index == 3) {   //->DecodeNumDetection和DecodeDetectionClasses是int32类型
        return DataType::kINT32;
    }
    return inputType[0];              //->DecodeDetectionBoxes和DecodeDetectionScores数据类型与输入相同
}

DimsExprs YoloLayer::getOutputDimensions(int outputIndex, const DimsExprs *inputs, int nbInputs, IExprBuilder &exprBuilder) noexcept {
    assert(outputIndex < 4);    //->断言输出数量
    DimsExprs out_dim;
    const IDimensionExpr *batch_size = inputs[0].d[0];

    const IDimensionExpr *output_num_boxes = exprBuilder.constant(0);
    //->输入特征维度 [batch_size, (nc+5) * nanchor, height, width]
    //->输入特征维度指的是输入到plugin的维度
    for (int32_t i = 0; i < NFEATURES; i++) {
        output_num_boxes = exprBuilder.operation(DimensionOperation::kSUM, *output_num_boxes,
                                                 *exprBuilder.operation(DimensionOperation::kPROD,
                                                 *inputs[i].d[2], *inputs[i].d[3]));
    }
    output_num_boxes = exprBuilder.operation(DimensionOperation::kPROD, *output_num_boxes, *exprBuilder.constant(NANCHORS));

    if (outputIndex == 0) {       //->num_detections: [batch_size, 1]
        out_dim.nbDims = 2;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = exprBuilder.constant(1);
    }
    else if (outputIndex == 1) {  //->detection_boxes: [batch_size, numboxes, 4]
        out_dim.nbDims = 3;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
        out_dim.d[2] = exprBuilder.constant(4);
    }
    else {                        //->detection_scores:  [batch_size, numboxes]
        out_dim.nbDims = 2;       //->detection_classes: [batch_size, numboxes]
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
    }
    return out_dim;
}

size_t YoloLayer::getWorkspaceSize(const PluginTensorDesc *inputs, int nbInputs, const PluginTensorDesc *outputs, int nbOutputs) const noexcept {
    return 0;           //->一般返回0
};

int YoloLayer::initialize() noexcept {
    return 0;           //->一般返回0
}
void YoloLayer::terminate() noexcept {
    return;             //->一般什么也不做
}    

void YoloLayer::serialize(void *buffer) const noexcept {
    char *d = static_cast<char *>(buffer);

    write(d, m_NetWidth);
    write(d, m_NetHeight);
    write(d, m_MaxStride);
    write(d, m_NumClasses);
    write(d, m_ScoreThreshold);
    write(d, m_OutputSize);

    for (int i = 0; i < m_Anchors.size(); i++) {
        write(d, m_Anchors[i]);
    }

    uint yoloTensorsSize = m_FeatureSpatialSize.size();
    for (uint i = 0; i < yoloTensorsSize; ++i) {
        write(d, m_FeatureSpatialSize[i].h());
        write(d, m_FeatureSpatialSize[i].w());
    }
}

void YoloLayer::destroy() noexcept {
    delete this;    
    return;
}

int32_t YoloLayer::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
    const int batchSize = inputDesc[0].dims.d[0];
    void *num_detections = outputs[0];
    void *detection_boxes = outputs[1];
    void *detection_scores = outputs[2];
    void *detection_classes = outputs[3];

    CUDA_CHECK(cudaMemsetAsync((int *)num_detections, 0, sizeof(int) * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync((float *)detection_boxes, 0, sizeof(float) * m_OutputSize * 4 * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync((float *)detection_scores, 0, sizeof(float) * m_OutputSize * batchSize, stream));
    CUDA_CHECK(cudaMemsetAsync((int *)detection_classes, 0, sizeof(int) * m_OutputSize * batchSize, stream));

    uint yoloTensorsSize = NFEATURES;
    //-> input p3: [batch_size, 78, 80, 80] 
    //-> input p4: [batch_size, 78, 40, 40]
    //-> input p5: [batch_size, 78, 20, 20]
    for (uint i = 0; i < yoloTensorsSize; ++i) {
        const DimsHW &gridSize = m_FeatureSpatialSize[i];

        uint numBBoxes = NANCHORS;
        float scaleXY = 2.0;
        uint gridSizeX = gridSize.w();
        uint gridSizeY = gridSize.h();
        std::vector<float> anchors(m_Anchors.begin() + i * NANCHORS * 2, m_Anchors.begin() + (i + 1) * NANCHORS * 2);

        void *v_anchors;
        if (anchors.size() > 0) {
            float *f_anchors = anchors.data();
            CUDA_CHECK(cudaMalloc(&v_anchors, sizeof(float) * anchors.size()));
            CUDA_CHECK(cudaMemcpyAsync(v_anchors, f_anchors, sizeof(float) * anchors.size(), cudaMemcpyHostToDevice, stream));
        }
        uint64_t inputSize = gridSizeX * gridSizeY * (numBBoxes * (4 + 1 + m_NumClasses));

        CUDA_CHECK(cudaYoloLayer_nc(
            inputs[i], num_detections, detection_boxes, detection_scores, detection_classes, batchSize,
            inputSize, m_OutputSize, m_ScoreThreshold, m_NetWidth, m_NetHeight, gridSizeX, gridSizeY,
            m_NumClasses, numBBoxes, scaleXY, v_anchors, stream));

        if (anchors.size() > 0) {
            CUDA_CHECK(cudaFree(v_anchors));
        }
    }

    return 0;
}

IPluginV2DynamicExt *YoloLayer::clone() const noexcept {      //->克隆插件
    return new YoloLayer(m_MaxStride, m_NumClasses, m_Anchors, m_ScoreThreshold);
}

bool YoloLayer::supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) noexcept {
    //->设置这个Plugin支持的Datatype以及TensorFormat
    //->如果需要扩展到FP16以及INT8，需要在这里设置
    if (inOut[pos].format != PluginFormat::kLINEAR) {
        return false;
    }

    const int posOut = pos - nbInputs;
    if (posOut == 0 || posOut == 3) {   //->DecodeNumDetection和DecodeDetectionClasses是int32类型
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == PluginFormat::kLINEAR;
    }

    return (inOut[pos].type == DataType::kFLOAT) && (inOut[0].type == inOut[pos].type);
}

void YoloLayer::configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, const DynamicPluginTensorDesc *out, int nbOutputs) noexcept {
    assert(nbInputs == NFEATURES);
    //->input feature [batch_size, (nc+5) * nanchor, height, width]
    m_OutputSize = 0;
    m_FeatureSpatialSize.clear();
    for (int i = 0; i < NFEATURES; i++)
    {
        m_FeatureSpatialSize.push_back(DimsHW(in[i].desc.dims.d[2], in[i].desc.dims.d[3]));
        m_OutputSize += in[i].desc.dims.d[2] * in[i].desc.dims.d[3] * NANCHORS;
    }
    // Compute the network input by last feature map and max stride
    m_NetHeight = in[NFEATURES - 1].desc.dims.d[2] * m_MaxStride;
    m_NetWidth = in[NFEATURES - 1].desc.dims.d[3] * m_MaxStride;
}

void YoloLayer::setPluginNamespace(const char *pluginNamespace) noexcept {
    m_Namespace = pluginNamespace;
}

void YoloLayer::attachToContext(cudnnContext* contextCudnn, cublasContext* contextCublas, IGpuAllocator *gpuAllocator) noexcept {
    return;
}

void YoloLayer::detachFromContext() noexcept {
    return;
}

//->注册插件
REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);

