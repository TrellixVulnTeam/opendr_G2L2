#include <torch/script.h>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//#include "nanodet.h"
#include "nanodet_c.h"

int resizeUniform(cv::Mat src, cv::Mat& dst, cv::Size dstSize, objectRect_t& effectArea)
{
    int w = src.cols;
    int h = src.rows;
    int dstW = dstSize.width;
    int dstH = dstSize.height;

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

static void generateGridCenterPriors(nanodet_model_t *model, std::vector<CenterPrior>& centerPriors)
{
    const int inputWidth = model->inputSize[0];
    const int inputHeight = model->inputSize[1];
    std::vector<int> strides = *(((std::vector<int>*) (model->strides)));
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

torch::Tensor mPreprocess(cv::Mat image, nanodet_model_t *model)
{
    int imgW = image.cols;
    int imgH = image.rows;
    torch::Tensor tensorImage = torch::from_blob(image.data, {1, imgW, imgH,3}, torch::kByte);
    tensorImage = tensorImage.permute({0,3,1,2});
    tensorImage = tensorImage.toType(torch::kFloat);
    tensorImage = tensorImage.add(*((torch::Tensor*) (model->meanValues)));
    tensorImage = tensorImage.mul(*((torch::Tensor*) (model->stdValues)));

    return tensorImage;
}

void mDecodeInfer(torch::Tensor clsPreds, torch::Tensor disPreds, std::vector<CenterPrior>& centerPriors, std::vector<std::vector<opendr_bounding_box_target>>& results, nanodet_model_t *model)
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
        if (score > model->scoreThreshold)
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
                float* distancesAfterSm = new float[model->regMax + 1];
                activationFunctionSoftmax(dflDet + i * (model->regMax + 1), distancesAfterSm, model->regMax + 1);
                for (int j = 0; j < model->regMax + 1; j++)
                {
                    dis += j * distancesAfterSm[j];
                }
                dis *= stride;
                disPred[i] = dis;
                delete[] distancesAfterSm;
            }
            float xmin = (std::max)(ctX - disPred[0], .0f);
            float ymin = (std::max)(ctY - disPred[1], .0f);
            float xmax = (std::min)(ctX + disPred[2], (float)model->inputSize[0]);
            float ymax = (std::min)(ctY + disPred[3], (float)model->inputSize[1]);
            results[label].push_back(opendr_bounding_box_target {label, xmin, ymin, (xmax-xmin), (ymax-ymin), score});
        }
    }
}

void cNms(std::vector<opendr_bounding_box_target>& inputBoxes, float nmsThreshold)
{
    std::sort(inputBoxes.begin(), inputBoxes.end(), [](opendr_bounding_box_target a, opendr_bounding_box_target b) { return a.score > b.score; });
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

//void load_nanodet_model(const char *modelPath, const int *inputSize, int regMax, std::vector<int> strides, const char* deviceString, float scoreThreshold, float nmsThreshold, nanodet_model_t *model)
void load_nanodet_model(char **argv, int argc, float scoreThreshold, float nmsThreshold, nanodet_model_t *model)
{
    std::vector<int> strides;
    for(int i=7; i<argc; i++)
    {
        strides.push_back(atoi(argv[i]));
    }

    // Initialize model
    model->inputSize[0] = atoi(argv[4]);
    model->inputSize[1] = atoi(argv[5]);
    model->regMax = atoi(argv[6]);

    model->scoreThreshold = scoreThreshold;
    model->nmsThreshold = nmsThreshold;
    model->numClass = 80;

    torch::DeviceType deviceType;
    if (std::string(argv[2]) == "cuda")
    {
        fprintf(stderr, "to cuda\n");
        deviceType = torch::kCUDA;
    }
    else
    {
        fprintf(stderr, "to cpu\n");
        deviceType = torch::kCPU;
    }

    torch::Device device = torch::Device(deviceType, 0);
    torch::jit::script::Module net = torch::jit::load(argv[1], device);
    net.eval();

    torch::Tensor tempMeanValues  = torch::tensor({{{-103.53f}}, {{-116.28f}}, {{-123.675f}}}).expand({3, model->inputSize[0], model->inputSize[1]}).unsqueeze(0);
    torch::Tensor tempStdValues = torch::tensor({{{0.017429f}}, {{0.017507f}}, {{0.017125f}}}).expand({3, model->inputSize[0], model->inputSize[1]}).unsqueeze(0);

    std::vector<std::string> labels{ "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                                     "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                     "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                     "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                     "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
                                     "hair drier", "toothbrush" };


    model->device = (void*) &device;
    model->net = (void*) &net;
    model->meanValues = (void*) &tempMeanValues;
    model->stdValues = (void*) &tempStdValues;
    model->classes = (void*) &labels;
    model->bboxesList = NULL;
    model->strides = (void*) &strides;
}

void infer_nanodet(opendr_image_t *image, nanodet_model_t *model)
{
    std::vector<opendr_bounding_box_target> dets;

    cv::Mat *opencv_image = static_cast<cv::Mat *>(image->data);
    if (!opencv_image) {
        std::cerr << "Cannot load image for inference." << std::endl;
        model->bboxesList = (void*) &dets;
        return;
    }

    torch::jit::script::Module* net = ((torch::jit::script::Module*) (model->net));
    torch::Device* device = ((torch::Device*) (model->device));
    cv::Mat resizedImg;
    resizeUniform(*opencv_image, resizedImg, cv::Size(model->inputSize[0], model->inputSize[1]), model->effectRoi);

    torch::Tensor input = mPreprocess(resizedImg, model);
    input = input.to(*device);
    torch::Tensor outputs = (*net).forward({input}).toTensor();
    outputs = outputs.to(torch::Device(torch::kCPU, 0));
    torch::Tensor clsPreds = outputs.slice(2, 0, model->numClass, 1);
    torch::Tensor boxPreds = outputs.slice(2, model->numClass,  model->numClass+(4*((model->regMax)+1)),1);
    std::vector<std::vector<opendr_bounding_box_target>> results;
    results.resize(model->numClass);
    // generate center priors in format of (x, y, stride)
    std::vector<CenterPrior> centerPriors;
    generateGridCenterPriors(model, centerPriors);
    // post process
    mDecodeInfer(clsPreds, boxPreds, centerPriors, results, model);
    // nms

    for (int i = 0; i < (int)results.size(); i++)
    {
        cNms(results[i], model->nmsThreshold);

        for (int j = 0; j < results[i].size(); i++)
        {
            dets.push_back(results[i][j]);
        }
    }
    model->bboxesList = (void*) &dets;
//    return dets;
}

void drawBboxes(opendr_image_t *opendr_image, nanodet_model_t *model)
{
    std::vector<opendr_bounding_box_target> bboxes = *((std::vector<opendr_bounding_box_target>*) (model->bboxesList));
    objectRect_t effectRoi = model->effectRoi;
    std::vector<std::string> classNames = *((std::vector<std::string>*) (model->classes));
    cv::Mat *opencv_image = static_cast<cv::Mat *>(opendr_image->data);
    if (!opencv_image) {
      std::cerr << "Cannot load image for inference." << std::endl;
      return;
    }

    cv::Mat image = (*opencv_image).clone();
    int srcW = image.cols;
    int srcH = image.rows;
    int dstW = effectRoi.width;
    int dstH = effectRoi.height;
    float widthRatio = (float)srcW / (float)dstW;
    float heightRatio = (float)srcH / (float)dstH;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const opendr_bounding_box_target& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(colorList[bbox.name][0], colorList[bbox.name][1], colorList[bbox.name][2]);

        cv::rectangle(image, cv::Rect(cv::Point((bbox.left - effectRoi.x) * widthRatio, (bbox.top - effectRoi.y) * heightRatio),
                                      cv::Point(((bbox.left + bbox.width) - effectRoi.x) * widthRatio, ((bbox.top + bbox.height) - effectRoi.y) * heightRatio)), color);

        char text[256];
        float score = bbox.score > 1 ? 1 : bbox.score;
        sprintf(text, "%s %.1f%%", classNames[bbox.name].c_str(), score * 100);

        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (bbox.left - effectRoi.x) * widthRatio;
        int y = (bbox.top - effectRoi.y) * heightRatio - labelSize.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + labelSize.width > image.cols)
            x = image.cols - labelSize.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)),
            color, -1);

        cv::putText(image, text, cv::Point(x, y + labelSize.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}
