#include <iostream>

#include <string>
#include <vector>
#include "assert.h"
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

class TrafficVehicleLabels {
public:
    TrafficVehicleLabels() {
        for(int i = 0 ; i < 21; ++i) {
            string x = "NA";
            mLabels.push_back( x );
        }

        mLabels[0]  = "ambulance";
        mLabels[1]  = "auto rickshaw";
        mLabels[2]  = "bicycle";
        mLabels[3]  = "bus";
        mLabels[4]  = "car";
        mLabels[5]  = "garbagevan";
        mLabels[6]  = "human hauler";
        mLabels[7]  = "minibus";
        mLabels[8]  = "minivan";
        mLabels[9]  = "motorbike";
        mLabels[10] = "Pickup";
        mLabels[11] = "army vehicle";
        mLabels[12] = "policecar";
        mLabels[13] = "rickshaw";
        mLabels[14] = "scooter";
        mLabels[15] = "suv";
        mLabels[16] = "taxi";
        mLabels[17] = "three wheelers (CNG)";
        mLabels[18] = "truck";
        mLabels[19] = "van";
        mLabels[20] = "wheelbarrow";
    }

    string trafficVehicle_get_label(int i) {
        assert( i >= 0 && i < 80 );
        return mLabels[i];
    }

    cv::Scalar trafficVehicle_get_color(int i) {
        float r;
        srand(i);
        r = (float)rand() / RAND_MAX;
        int red    = int(r * 255);

        srand(i + 1);
        r = (float)rand() / RAND_MAX;
        int green    = int(r * 255);

        srand(i + 2);
        r = (float)rand() / RAND_MAX;
        int blue    = int(r * 255);

        return cv::Scalar(blue, green, red);
    }

    cv::Scalar get_inverse_color(cv::Scalar color) {
        int blue = 255 - color[0];
        int green = 255 - color[1];
        int red = 255 - color[2];
        return cv::Scalar(blue, green, red);
    }


private:
  vector<string> mLabels;

};
