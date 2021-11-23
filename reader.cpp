
#include "reader.hpp"

Reader::Reader(){

    x=0;
    y=0;
    r=0.f;
    min_value=0.f;
    max_value=0.f ;
    min_angle=0.f;
    max_angle=0.f ;
};

Mat Reader::readimg(string src_image_path){

    this->srcimg = imread(src_image_path, 1);
    return this->srcimg.clone();
};

Mat Reader::readimg(Mat &src_img,Vec4i x){
    Mat zhizhen = src_img(cv::Rect(int(x[0])-15,int(x[1])-15,int(x[2])-int(x[0])+30,int(x[3])-int(x[1])+30));
    this->srcimg = zhizhen;
    return this->srcimg.clone();
};

Mat Reader::readimg(Mat &src_img){
    this->srcimg = src_img;
    return this->srcimg.clone();
};


Vec3d Reader::avg_circles(vector<cv::Vec3f> circles, int b){
    int avg_x=0;
    int avg_y=0;
    int avg_r=0;
    for (int i=0;  i< b; i++ )
    {
        //平均圆心 半径
        avg_x = avg_x + circles[i][0];
        avg_y = avg_y + circles[i][1];
        avg_r = avg_r + circles[i][2];
    }
    //半径为啥int
    avg_x = int(avg_x/(b));
    avg_y = int(avg_y/(b));
    avg_r = int(avg_r/(b));

    Vec3d xyr = Vec3d(avg_x, avg_y, avg_r);
    return xyr;

}

float Reader::getDist_P2L(cv::Point2f pointP, cv::Point2f pointA, cv::Point2f pointB)
{
    float A = 0, B = 0, C = 0;
    A = pointA.y - pointB.y;
    B = pointB.x - pointA.x;
    C = pointA.x*pointB.y - pointA.y*pointB.x;

    float distance = 0.0;
    distance = ((float)abs(A*pointP.x + B*pointP.y + C)) / ((float)sqrtf(A*A + B*B));
    return distance;
}


float Reader::dist_2_pts(int x1, int y1, int x2, int y2){
    int pp = pow(x2-x1,2)+pow(y2-y1,2);
    float tmp = sqrt(pp);
    return tmp;
}
Mat Reader::region_of_interest(Mat &img,vector<vector<cv::Point>> &vertices){
    Mat mask = Mat::zeros(img.size(), img.type());
    int match_mask_color= 255;

    fillPoly(mask, vertices, Scalar(match_mask_color));

    Mat masked_image;
    bitwise_and(img, mask,masked_image);

    return masked_image.clone();;


}


Mat Reader::detectCircles() {
    if (this->srcimg.empty()) {
        cv::Mat a;
        return a;
    }
    Mat midd_img = this->srcimg.clone();
    int wight = midd_img.rows;
    int height = midd_img.cols;

    Mat gray_img;
    cvtColor(midd_img, gray_img, COLOR_RGB2GRAY);
    medianBlur(gray_img, gray_img, 5);

    std::vector<Vec3f> circles;

    // HoughCircles(gray_img, circles, HOUGH_GRADIENT, 1,
    //     gray_img.rows / 16,     // change this value to detect circles with different distances to each other
    //     100, 30, 127, 138		// change the last two parameters
    //                             // (min_radius & max_radius) to detect larger circles
    // );
    //HoughCircles(gray_img, circles,cv2.HOUGH_GRADIENT, 1, 120,  100, 50, int(height*0.35), int(height*0.48));
    HoughCircles(gray_img, circles, HOUGH_GRADIENT, 1, 120, 100, 50, int(height * 0.35),int(height * 0.48));

    int b = circles.size();
    if (b == 0) {
        cv::Mat a;
        return a;
    } else {
        Vec3d xyr = this->avg_circles(circles, b);

        this->x = xyr[0];
        this->y = xyr[1];
        this->r = xyr[2];

        //画圆和圆心
        //circle(midd_img, Point(this->x, this->y), this->r, (0, 0, 255), 3, LINE_AA);
        //circle(midd_img, Point(this->x, this->y), 2, (0, 255, 0), 3, LINE_AA);

        //imwrite("jianceyuan.jpg", midd_img);

        float separation = 10.0;
        int interval = int(360 / separation);
/**
        Mat imgtt = midd_img.clone();

        float separation = 10.0;
        int interval = int(360 / separation);
        //p1 = np.zeros((interval,2))
        vector<Point> p1;
        vector<Point> p2;
        vector<Point> p_text;
        //p_text = np.zeros((interval,2))
        for (int i = 0; i < interval; i++) {
            Point pp;
            for (int j = 0; j < 2; j++) {
                if (j % 2 == 0) {
                    pp.x = this->x + 0.88 * r * cos(separation * i * CV_PI / 180);
                } else {
                    pp.y = this->y + 0.88 * r * sin(separation * i * CV_PI / 180);
                }
            }
            p1.push_back(pp);
        }


        int text_offset_x = 10;
        int text_offset_y = 5;
        for (int i = 0; i < interval; i++) {
            Point pp, p_t;
            for (int j = 0; j < 2; j++) {
                if (j % 2 == 0) {
                    pp.x = this->x + r * cos(separation * i * 3.14 / 180);
                    p_t.x = this->x - text_offset_x + 1.2 * r * cos((separation) * (i + 9) * 3.14 /
                                                                    180); //point for text labels, i+9 rotates the labels by 90 degrees
                } else {
                    pp.y = this->y + r * sin(separation * i * 3.14 / 180);
                    p_t.y = this->y + text_offset_y + 1.2 * r * sin((separation) * (i + 9) * 3.14 /
                                                                    180);//point for text labels, i+9 rotates the labels by 90 degrees
                }
            }
            p2.push_back(pp);
            p_text.push_back(p_t);

        }

        for (int i = 0; i < interval; i++) {
            cv::line(imgtt, Point(int(p1[i].x), int(p1[i].y)), Point(int(p2[i].x), int(p2[i].y)),
                     Scalar(0, 255, 0), 2);
            putText(imgtt, to_string(int(i * separation)),
                    Point(int(p_text[i].x), int(p_text[i].y)), FONT_HERSHEY_SIMPLEX, 0.3,
                    Scalar(0, 0, 0), 1, LINE_AA);
        }

        imwrite("calibrate.jpg", imgtt);
**/

        //separation=10;
        //interval = int(360/separation);

        std::vector<Point> pts;

        for (int i = 0; i < interval; i++) {
            Point pp;
            for (int j = 0; j < 2; j++) {
                if (j % 2 == 0)
                    pp.x = this->x + 1.0 * r * cos(separation * i * CV_PI / 180);
                else
                    pp.y = this->y + 1.0 * r * sin(separation * i * CV_PI / 180);
            }
            pts.push_back(pp);
        }

        Mat canny;
        Canny(gray_img, canny, 200, 20);
        //Mat region_of_interest_vertices= p3;
        //imwrite("canny.jpg", canny);

        std::vector<std::vector<Point>> region_of_interest_vertices;
        region_of_interest_vertices.push_back(pts);

        Mat cropped_image = region_of_interest(canny, region_of_interest_vertices);


        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;

        //findContours(cropped_image,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());
        findContours(cropped_image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

        //Mat imageContours=Mat::zeros(image.size(),CV_8UC1);
        //Mat Contours=Mat::zeros(image.size(),CV_8UC1);  //绘制
        //std::vector<int> int_cnt;
        vector<vector<cv::Point> > int_cnt;

        for (int i = 0; i < contours.size(); i++) {
            float area = contourArea(contours[i]);
            Rect prect = boundingRect(contours[i]);

            float cpd = dist_2_pts(prect.x + prect.width / 2, prect.y + prect.height / 2, this->x,this->y);

            if ((area < 500) && (cpd < this->r * 4 / 4) && (cpd > this->r * 2 / 4)) {
                //drawContours(contours3, vector<vector<Point> >(1, contours[i]), -1,Scalar(255, 0, 0), 3);
                int_cnt.push_back(contours[i]);
            }
        }
        //imwrite("contours3.jpg", contours3);


        //10 350
        float reference_zero_angle = 20;
        float reference_end_angle = 340;
        float min_angle = 90;
        float max_angle = 270;

        std::vector<int> frth_quad_index;
        std::vector<int> thrd_quad_index;
        std::vector<float> frth_quad_angle;
        std::vector<float> thrd_quad_angle;

        for (int i = 0; i < int_cnt.size(); i++) {
            //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
            vector<cv::Point> conPoints;
            float x1, y1;
            float sx1 = 0, sy1 = 0;
            for (int j = 0; j < contours[i].size(); j++) {
                //绘制出contours向量内所有的像素点
                //Point P=Point(contours[i][j].x,contours[i][j].y);
                //conPoints.push_back(P);
                sx1 += contours[i][j].x;
                sy1 += contours[i][j].y;
            }
            x1 = sx1 / contours[i].size();
            y1 = sy1 / contours[i].size();

            float xlen = x1 - this->x;
            float ylen = this->y - y1;

            //double res = atan2(float(ylen), float(xlen));
            //res = res * 180.0 / M_PI;

            if ((xlen < 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_start_angle = 90 - res;

                frth_quad_index.push_back(i);
                frth_quad_angle.push_back(final_start_angle);

                if (final_start_angle > reference_zero_angle)
                    if (final_start_angle < min_angle)
                        min_angle = final_start_angle;

            }

            if ((xlen > 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_end_angle = 270 + res;

                thrd_quad_index.push_back(i);
                thrd_quad_angle.push_back(final_end_angle);

                if (final_end_angle < reference_end_angle)
                    if (final_end_angle > max_angle)
                        max_angle = final_end_angle;
            }

        }


        std::vector<float> frth_angle_(frth_quad_angle);
        std::vector<float> thrd_angle_(thrd_quad_angle);

        //升序 ，，降序
        std::sort(frth_angle_.begin(), frth_angle_.end(), std::less<float>());
        std::sort(thrd_angle_.begin(), thrd_angle_.end(), std::greater<float>());

        std::vector<float> frth_sub;
        std::vector<float> thrd_sub;
        for (int i = 0; i < frth_angle_.size() - 1; i++)
            frth_sub.push_back(frth_angle_[i + 1] - frth_angle_[i]);
        for (int i = 0; i < thrd_angle_.size() - 1; i++)
            thrd_sub.push_back(thrd_angle_[i + 1] - thrd_angle_[i]);


        std::vector<float>::iterator maxPosition = max_element(frth_sub.begin(), frth_sub.end());
        min_angle = frth_angle_[maxPosition - frth_sub.begin() + 1];
        //min_angle = *(max_element(frth_sub.begin(), frth_sub.end())+1);

        std::vector<float>::iterator minPosition = min_element(thrd_sub.begin(), thrd_sub.end());
        max_angle = thrd_angle_[minPosition - thrd_sub.begin() + 1];

        this->min_angle = min_angle;
        this->max_angle = max_angle;

        return midd_img;
    }
}

float Reader::detectLines() {
        Mat gray_img;
        if((this->min_angle==0) || (this->max_angle==0)) return float(10086.111f);
        if(this->srcimg.empty()) return float(10086.111f);
        Mat midd_img = this->srcimg.clone();

        cvtColor(midd_img, gray_img, COLOR_RGB2GRAY);
        //50cm 模糊3像素
        //cv::Ptr<cv::CLAHE> clahe = createCLAHE(40.0, Size(8, 8));
        //Mat dstcle;
        //限制对比度的自适应阈值
        //clahe->apply(gray_img, dstcle);
        //原图一定屏蔽掉，模糊的要添加，原图添加，识别不了， 模糊的 不添加 识别不了
        //gray2 =dst

        int thresh = 166;
        int maxValue = 255;
        Mat midd_img2;

        //Canny(gray_img, midd_img2, 23, 55, 3);

        // convert cannied image to a gray one
        //cvtColor(midd_img2, dst_img, CV_GRAY2BGR);

        // define a vector to collect all possible lines
        vector<Vec4i> mylines;
        int g_nthreshold = 39;

        //minLineLength = 10
        //maxLineGap = 0
        //image=dst2, rho=3, theta=np.pi / 180, threshold=100,minLineLength=minLineLength, maxLineGap=0
        //HoughLinesP(midd_img, mylines, 1, CV_PI / 180, g_nthreshold + 1, 20, 5);
        //imwrite("gray_img.jpg", gray_img);
        threshold(gray_img, midd_img2, thresh, maxValue, CV_THRESH_BINARY_INV);
        //imwrite("midd_img2.jpg", midd_img2);
        HoughLinesP(midd_img2, mylines, 3, CV_PI / 180, 100, 10, 0);
        //HoughLinesP(gray_img, mylines, 3, CV_PI / 180, 100, 10, 0);
        if (mylines.empty()) return float(10086.111f);
        Point circle_center = Point2f(this->x, this->y);
        float circle_radius = this->r;

/**
        // draw every line by using for
        Mat midd_img3 = midd_img.clone();
        for (size_t i = 0; i < mylines.size(); i++) {
            Vec4i l = mylines[i];
            //直线是四元组，直线的起点坐标 终点坐标，
            //起点 与圆心 xy坐标 10以内
            //终点 在圆边以内，
            if (((circle_center.x - 10) <= l[0]) && (l[0] <= (circle_center.x + 10)))
                if (((circle_center.y - 10) <= l[1]) && (l[1] <= (circle_center.y + 10)))
                    if (((circle_center.x - circle_radius) <= l[2]) &&
                        (l[2] <= (circle_center.x + circle_radius)))
                        if (((circle_center.y - circle_radius) <= l[3]) &&
                            (l[3] <= (circle_center.y + circle_radius))) {
                            //cout << Point(l[0], l[1]) << " " << Point(l[2], l[3]) << " " << l[0] << " " << circle_center.x - circle_radius << " " << circle_center.x + circle_radius << endl;
                            cv::line(midd_img3, Point(l[0], l[1]), Point(l[2], l[3]),
                                     Scalar(23, 180, 55), 2, CV_AA);
                            Vec4i cho_l = l;
                            //cv::line(midd_img3, Point(circle_center.x, circle_center.y), Point(cho_l[2], cho_l[3]), Scalar(23, 180, 55), 2, CV_AA);
                        }
        }
        imwrite("midd_img3.jpg", midd_img3);
**/
        float diff1LowerBound = 0.05;
        float diff1UpperBound = 0.25;
        float diff2LowerBound = 0.05;
        float diff2UpperBound = 1.0;


        vector<Vec4i> final_line_list;
        vector<float> distance_list;
        vector<float> line_length_list;
        Mat midd_img6 = midd_img.clone();

        for (size_t i = 0; i < mylines.size(); i++) {
            Vec4i l = mylines[i];
            float diff1 = this->dist_2_pts(circle_center.x, circle_center.y, l[0], l[1]);
            float diff2 = this->dist_2_pts(circle_center.x, circle_center.y, l[2], l[3]);
            if (diff1 > diff2) {
                float temp = diff1;
                diff1 = diff2;
                diff2 = temp;
            }

            if (((diff1 < diff1UpperBound * circle_radius) && (diff1 > diff1LowerBound * circle_radius)) &&
                ((diff2 < diff2UpperBound * circle_radius) && (diff2 > diff2LowerBound * circle_radius))) {

                float line_length = this->dist_2_pts(l[0], l[1], l[2], l[3]);
                float distance = this->getDist_P2L(Point2f(circle_center.x, circle_center.y), Point2f(l[0], l[1]),
                                             Point2f(l[2], l[3]));

                if ((line_length>0.1*circle_radius)  && (distance>-20) && (distance <10)){
                final_line_list.push_back(Vec4i(l[0], l[1], l[2], l[3]));
                distance_list.push_back(distance);
                line_length_list.push_back(line_length);
                //cv::line(midd_img6, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(23, 180, 55), 2,CV_AA);
                }

            }

        };
        //imwrite("midd_img6.jpg", midd_img6);
        if (final_line_list.empty()) return float(10086.111f);
        //输出第一个线，点到直线的距离，点到两个端点的距离，线的长度；最短距离的位置，线最长的位置
        vector<float>::iterator maxPosition = max_element(line_length_list.begin(),line_length_list.end());

        vector<float>::iterator minPosition = min_element(distance_list.begin(),distance_list.end());

        Vec4i final_line;

        final_line = final_line_list[maxPosition - line_length_list.begin() + 1];


        float x1 = final_line[0];
        float y1 = final_line[1];
        float x2 = final_line[2];
        float y2 = final_line[3];


        //find the farthest point from the center to be what is used to determine the angle
        float dist_pt_0 = this->dist_2_pts(circle_center.x, circle_center.y, x1, y1);
        float dist_pt_1 = this->dist_2_pts(circle_center.x, circle_center.y, x2, y2);

        float x_angle = 0.0;
        float y_angle = 0.0;
        if (dist_pt_0 > dist_pt_1) {
            x_angle = x1 - circle_center.x;
            y_angle = circle_center.y - y1;
        } else {
            x_angle = x2 - circle_center.x;
            y_angle = circle_center.y - y2;
        }

        x_angle = (x1 + x2) / 2 - circle_center.x;
        y_angle = circle_center.y - (y1 + y2) / 2;


        double res = atan2(float(y_angle), float(x_angle));

        //these were determined by trial and error
        res = res * 180.0 / M_PI;

        float final_angle = 0.0;

        if ((x_angle > 0) && (y_angle > 0))//in quadrant I
            final_angle = 270 - res;
        if (x_angle < 0 && y_angle > 0) //in quadrant II
            final_angle = 90 - res;
        if (x_angle < 0 && y_angle < 0)  //in quadrant III
            final_angle = 90 - res;
        if (x_angle > 0 && y_angle < 0)  //in quadrant IV
            final_angle = 270 - res;


        //vector<float> final_value_list;
        //for (int i = 0; i < 10; i++) {

            float old_min = float(this->min_angle) ;
            float old_max = float(this->max_angle) ;

            float new_min = float( this->min_value);
            float new_max = float( this->max_value);

            float old_value = final_angle;

            float old_range = (old_max - old_min);
            float new_range = (new_max - new_min);
            float final_value = (((old_value - old_min) * new_range) / old_range) + new_min;
            //final_value_list.push_back(final_value);
        //}

//        float dushu = accumulate(final_value_list.begin(), final_value_list.end(), 0.0);
//        dushu = dushu / float(final_value_list.size());
        if ((final_value< float(this->min_value) ) || (final_value> float(this->max_value)))
            return float(10086.111f);

        return final_value;
    }
