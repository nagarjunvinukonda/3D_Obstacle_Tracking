# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Task FP.1 Match 3D Objects

Implement the method "matchBoundingBoxes", which takes both the previous and the current data frames as input and provides the IDs of the matched regions of interest. Here I use ```multimap<int, int> mmap``` to store the pairs of bounding Box ID. Then find the best match with the highest number of keypoints correspondences.

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    multimap<int, int> mmap;
    for (auto match: matches)
    {
        int prevBoxId = -1, currBoxId = -1;

        for (auto bbox: prevFrame.boundingBoxes)
        {
            if (bbox.roi.contains(prevFrame.keypoints[match.queryIdx].pt))
                prevBoxId = bbox.boxID;
        }

        for (auto bbox: currFrame.boundingBoxes)
        {
            if (bbox.roi.contains(currFrame.keypoints[match.trainIdx].pt))
                currBoxId = bbox.boxID;
        }

        mmap.insert({currBoxId, prevBoxId});
    }

    int prevBoxSize = prevFrame.boundingBoxes.size();
    for (int i = 0; i < prevBoxSize; i++)
    {
        auto mmapPair = mmap.equal_range(i);
        vector<int> currBoxCount(prevBoxSize, 0);
        for (auto pr = mmapPair.first; pr != mmapPair.second; pr++)
        {
            if (pr->second >= 0)
                currBoxCount[pr->second]++;
        }

        int maxPosition = std::distance(currBoxCount.begin(), std::max_element(currBoxCount.begin(), currBoxCount.end()));
        bbBestMatches.insert({maxPosition, i});
    }
}
```

## Task FP.2 Compute Lidar-based TTC

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous data frame. I calculate the mean value of x-distance and set a distance threshold at 0.1. Any point that is out of the threshold will not be considered for TTC calculation.

```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // Find the mean dist if all lidar points
    double meanPrev = 0.0;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
    {
        meanPrev += it->x;
    }
    meanPrev /= lidarPointsPrev.size();

    double meanCurr = 0.0;
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
    {
        meanCurr += it->x;
    }
    meanCurr /= lidarPointsCurr.size();

    // Base on mean value, reject the outliers
    std::vector<LidarPoint> inliersPrev;
    std::vector<LidarPoint> inliersCurr;
    double distTol = 0.1;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
    {
        double dist = fabs(it->x - meanPrev);
        if (dist <= distTol)
            inliersPrev.push_back(*it);
    }
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
    {
        double dist = fabs(it->x - meanCurr);
        if (dist <= distTol)
            inliersCurr.push_back(*it);
    }

    if (inliersCurr.size() && inliersPrev.size())
    {
        // Find the min value for inliers
        double minXPrev = 1e8, minXCurr = 1e8;
        for (auto it = inliersPrev.begin(); it != inliersPrev.end(); it++)
            minXPrev = minXPrev < it->x ? minXPrev : it->x;

        for (auto it = inliersCurr.begin(); it != inliersCurr.end(); it++)
            minXCurr = minXCurr < it->x ? minXCurr : it->x;

        // Calculate the time to collision

        TTC = minXCurr * (1.0/frameRate) / (minXPrev - minXCurr);
        if (TTC < 0)
            TTC = NAN;
    }
    else
    {
        TTC = NAN;
    }
}
```

## Task FP.3 Associate Keypoint Correspondences with Bounding Boxes

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box. I calculate the mean and the standard deviation of the distribution of distance between match points. After that I remove the points that are too far away from mean.

```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    float shrinkFactor = 0.1;
    for (auto &match: kptMatches)
    {
        cv::Rect smallerBox;
        smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
        smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
        smallerBox.width = boundingBox.roi.width * (1 - shrinkFactor);
        smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);

        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
        {
            boundingBox.kptMatches.push_back(match);
        }
    }

    // Remove outliers
    double meanDist = 0.0;
    std::vector<double> distances;
    for (auto &match: boundingBox.kptMatches)
    {
        auto kptCurr = kptsCurr[match.trainIdx].pt;
        auto kptPrev = kptsPrev[match.queryIdx].pt;
        double dist = cv::norm(kptCurr - kptPrev);
        meanDist += dist;
        distances.push_back(dist);
    }
    meanDist /= boundingBox.kptMatches.size();

    double sigma = 0.0;
    for (int i = 0; i < distances.size(); i++)
    {
        sigma += pow(distances[i] - meanDist, 2);
    }
    sigma = sqrt(sigma / distances.size());

    for (int i = 0; i < kptMatches.size(); i++) {
    	if (abs(distances[i] - meanDist) < sigma) {
    		boundingBox.kptMatches.push_back(kptMatches[i]);
    	}
    }
}
```

## Task FP.4 Compute Camera-based TTC

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous data frame. I compute the median of distance ratios to reduce the impact of outliers of keypoints.

```c++
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}
```

## Task FP.5 Performance Evaluation 1

Taking the mean value of x-distance and reject the outliers seem works pretty fine. I didn't find any unreasonable result.

## Task FP.6 Performance Evaluation 2

**Total Time cost (s)**

|Detector/Descriptor|BRISK|BRIEF|ORB|FREAK|AKAZE|SIFT|
|---|:---:|:---:|:---:|:---:|:---:|:---:|
|Shi-Tomasi|26.13|**14.68**|**15.76**|52.39|x|39.19|
|Harris|**12.72**|**10.9**|**14.05**|44.93|x|24.36|
|FAST|49.14|16.63|**13.08**|61.17|x|169.77|
|BRISK|315|279.1|288.86|320.78|x|475.76|
|ORB|**11.93**|**8.98**|21.2|41.67|x|115.2|
|AKAZE|71.94|62.95|72.11|105.61|117.4|112.15|
|SIFT|117.17|187.33|x|142.81|x|202.83|

Lidar TTC: 12.56 s

**Camera TTC (s)**

|Detector/Descriptor|BRISK|BRIEF|ORB|FREAK|AKAZE|SIFT|
|---|:---:|:---:|:---:|:---:|:---:|:---:|
|Shi-Tomasi|15.9|16.03|15.72|16.82|x|13.28|
|Harris|8.69|24.2|24.2|22.99|x|16.84|
|FAST|13.91|13.76|13.69|14.72|x|14.46|
|BRISK|17.67|16.53|14.97|17.98|x|18.52|
|ORB|20.55|19.61|16.78|-inf|x|25.86|
|AKAZE|13.81|17.28|14.85|15.69|17.09|15.92|
|SIFT|13.57|16.56|x|13.17|x|15.34|

According to the test result, Shi-Tomasi detector with BRIEF/ORB descriptor, Harris detector with BRISK/BRIEF/ORB descriptor, FAST detector with BRIEF/ORB descriptor, and ORB detector with BRISK/BRIEF descriptor will cost less run time than others. ORB detector with BRIEF descriptor is the fastest combination. Since the average TTC of all combination is 16.7s, and Lidar TTC is 12.56s, any reading that is close to these two number will be considered a good TTC result.

Hence, my Top 3 choices will be:
1. FAST + ORB
2. Shi-Tomasi + BRIEF
3. Shi-Tomasi + ORB
