#include <torch/script.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "nanodet.h"

int resizeUniform(cv::Mat src, cv::Mat& dst, cv::Size dstSize, ObjectRect& effectArea)
{
    int w = src.cols;
    int h = src.rows;
    int dstW = dstSize.width;
    int dstH = dstSize.height;
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dstW, dstH), CV_8UC3, cv::Scalar(0));

    float ratioSrc = w * 1.0 / h;
    float ratioDst = dstW * 1.0 / dstH;

    int tmpW = 0;
    int tmpH = 0;

    if (ratioSrc > ratioDst) {
        tmpW = dstW;
        tmpH = floor((dstW * 1.0 / w) * h);
    }
    else if (ratioSrc < ratioDst) {
        tmpH = dstH;
        tmpW = floor((dstH * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dstSize);
        effectArea.x = 0;
        effectArea.y = 0;
        effectArea.width = dstW;
        effectArea.height = dstH;
        return 0;
    }

    cv::Mat tmp;
    cv::resize(src, tmp, cv::Size(tmpW, tmpH));

    if (tmpW != dstW) {
        int indexW = floor((dstW - tmpW) / 2.0);
        for (int i = 0; i < dstH; i++) {
            memcpy(dst.data + i * dstW * 3 + indexW * 3, tmp.data + i * tmpW * 3, tmpW * 3);
        }
        effectArea.x = indexW;
        effectArea.y = 0;
        effectArea.width = tmpW;
        effectArea.height = tmpH;
    }
    else if (tmpH != dstH) {
        int indexH = floor((dstH - tmpH) / 2.0);
        memcpy(dst.data + indexH * dstW * 3, tmp.data, tmpW * tmpH * 3);
        effectArea.x = 0;
        effectArea.y = indexH;
        effectArea.width = tmpW;
        effectArea.height = tmpH;
    }
    else {
        printf("error\n");
        return 1;
    }
    return 0;
}

inline float fastExp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

template<typename T>
int activationFunctionSoftmax(const T* srcs, T* dsts, int length)
{
    const T alpha = *std::max_element(srcs, srcs + length);
    T denominator{ 0 };

    for (int i = 0; i < length; ++i)
    {
        dsts[i] = fastExp(srcs[i] - alpha);
        denominator += dsts[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dsts[i] /= denominator;
    }

    return 0;
}

static void generateGridCenterPriors(const int inputHeight, const int inputWidth, std::vector<int>& strides, std::vector<CenterPrior>& centerPriors)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int featW = ceil((float)inputWidth / stride);
        int featH = ceil((float)inputHeight / stride);
        for (int y = 0; y < featH; y++)
        {
            for (int x = 0; x < featW; x++)
            {
                CenterPrior ct;
                ct.x = x;
                ct.y = y;
                ct.stride = stride;
                centerPriors.push_back(ct);
            }
        }
    }
}


NanoDet::NanoDet(const char* modelPath, const int* inputSize, int regMax, std::vector<int> strides, torch::Device device)
{
    this->net = torch::jit::load(modelPath);
    this->net.eval();
    this->net.to(device);

    this->inputSize[0] = inputSize[0];
    this->inputSize[1] = inputSize[1];
    this->regMax = regMax;
    this->strides = strides;

    this->meanValues = torch::tensor({{{-103.53f}}, {{-116.28f}}, {{-123.675f}}}).expand({3, inputSize[0], inputSize[1]}).unsqueeze(0);
    this->normValues = torch::tensor({{{0.017429f}}, {{0.017507f}}, {{0.017125f}}}).expand({3, inputSize[0], inputSize[1]}).unsqueeze(0);
}

NanoDet::~NanoDet()
{
}

torch::Tensor NanoDet::mPreprocess(cv::Mat image)
{
    int imgW = image.cols;
    int imgH = image.rows;
    torch::Tensor tensorImage = torch::from_blob(image.data, {1, imgW, imgH,3}, torch::kByte);
    tensorImage = tensorImage.permute({0,3,1,2});
    tensorImage = tensorImage.toType(torch::kFloat);
    tensorImage = tensorImage.add(this->meanValues);
    tensorImage = tensorImage.mul(this->normValues);
    return tensorImage;
}

std::vector<opendr_detection_target_t> NanoDet::detect(opendr_image_t *image, float scoreThreshold, float nmsThreshold, torch::Device device)
{
    std::vector<opendr_detection_target_t> dets;

    cv::Mat *opencv_image = static_cast<cv::Mat *>(image->data);
    if (!opencv_image) {
      std::cerr << "Cannot load image for inference." << std::endl;
      return dets;
    }

    cv::Mat resizedImg;
    resizeUniform(*opencv_image, resizedImg, cv::Size(this->inputSize[0], this->inputSize[1]), this->effectRoiValue);

    std::cout<<"input cols:" << resizedImg.cols << " input rows: " <<resizedImg.rows<<std::endl;
    torch::Tensor input = mPreprocess(resizedImg);
    input = input.to(device);
    torch::Tensor outputs = this->net.forward({input}).toTensor();
    outputs = outputs.to(torch::Device(torch::kCPU, 0));
    torch::Tensor clsPreds = outputs.slice(2, 0, this->numClass, 1);
    torch::Tensor boxPreds = outputs.slice(2, this->numClass,  this->numClass+(4*((this->regMax)+1)),1);
    std::vector<std::vector<opendr_detection_target_t>> results;
    results.resize(this->numClass);
    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> centerPriors;
    generateGridCenterPriors(this->inputSize[0], this->inputSize[1], this->strides, centerPriors);
    // post process
    this->mDecodeInfer(clsPreds, boxPreds, centerPriors, scoreThreshold, results);
    // nms

    for (int i = 0; i < (int)results.size(); i++)
    {
        this->cNms(results[i], nmsThreshold);

        for (int j = 0; j < results[i].size(); i++)
        {
            dets.push_back(results[i][j]);
        }
    }
    return dets;
}

void NanoDet::mDecodeInfer(torch::Tensor clsPreds, torch::Tensor disPreds, std::vector<CenterPrior>& centerPriors, float scoreThreshold, std::vector<std::vector<opendr_detection_target_t>>& results)
{
    const int nPoints = centerPriors.size();

    std::tuple<torch::Tensor, torch::Tensor> resultMax = clsPreds[0].max(1);
    torch::Tensor scores = std::get<0>(resultMax);
    float* scoresFloat = scores.contiguous().data_ptr<float>();
    torch::Tensor maxLabelsIdx = std::get<1>(resultMax);
    long* maxLabelsIdxLong = maxLabelsIdx.contiguous().data_ptr<long>();

    for (int idx = 0; idx < nPoints; idx++)
    {
        int label = maxLabelsIdxLong[idx];
        float score = scoresFloat[idx];
        if (score > scoreThreshold)
        {
            torch::Tensor curDis = disPreds[0][idx].contiguous();
            const float* bboxPred = curDis.data_ptr<float>();
            int x =  centerPriors[idx].x;
            int y =  centerPriors[idx].y;
			      int stride =  centerPriors[idx].stride;
            const float* dflDet= bboxPred;

            float ctX = x * stride;
            float ctY = y * stride;
            std::vector<float> disPred;
            disPred.resize(4);
            for (int i = 0; i < 4; i++)
            {
                float dis = 0;
                float* distancesAfterSm = new float[this->regMax + 1];
                activationFunctionSoftmax(dflDet + i * (this->regMax + 1), distancesAfterSm, this->regMax + 1);
                for (int j = 0; j < this->regMax + 1; j++)
                {
                    dis += j * distancesAfterSm[j];
                }
                dis *= stride;
                disPred[i] = dis;
                delete[] distancesAfterSm;
            }
            float xmin = (std::max)(ctX - disPred[0], .0f);
            float ymin = (std::max)(ctY - disPred[1], .0f);
            float xmax = (std::min)(ctX + disPred[2], (float)this->inputSize[0]);
            float ymax = (std::min)(ctY + disPred[3], (float)this->inputSize[1]);
            results[label].push_back(opendr_detection_target_t {label, xmin, ymin, (xmax-xmin), (ymax-ymin), score});
        }
    }
}

void NanoDet::cNms(std::vector<opendr_detection_target_t>& inputBoxes, float nmsThreshold)
{
    std::sort(inputBoxes.begin(), inputBoxes.end(), [](opendr_detection_target_t a, opendr_detection_target_t b) { return a.score > b.score; });
    std::vector<float> boxesArea(inputBoxes.size());
    for (int i = 0; i < int(inputBoxes.size()); ++i)
    {
        boxesArea[i] = (inputBoxes.at(i).width + 1) * (inputBoxes.at(i).height + 1);
    }
    for (int i = 0; i < int(inputBoxes.size()); ++i)
    {
        for (int j = i + 1; j < int(inputBoxes.size());)
        {
            float xx1 = (std::max)(inputBoxes[i].left, inputBoxes[j].left);
            float yy1 = (std::max)(inputBoxes[i].top, inputBoxes[j].top);
            float xx2 = (std::min)((inputBoxes[i].left + inputBoxes[i].width), (inputBoxes[j].left + inputBoxes[j].width));
            float yy2 = (std::min)((inputBoxes[i].top + inputBoxes[i].height), (inputBoxes[j].height + inputBoxes[j].height));
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (boxesArea[i] + boxesArea[j] - inter);
            if (ovr >= nmsThreshold)
            {
                inputBoxes.erase(inputBoxes.begin() + j);
                boxesArea.erase(boxesArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

ObjectRect NanoDet::effectRoi()
{
    return this->effectRoiValue;
}
