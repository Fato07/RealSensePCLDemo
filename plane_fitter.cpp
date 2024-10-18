#include <iostream>
#include <librealsense2/rs.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>

#include "opencv2/opencv.hpp"
#include "AHCPlaneFitter.hpp"

using ahc::utils::Timer;

typedef pcl::PointCloud<pcl::PointXYZRGBA> PointCloudRGBA;

#define MACRO_DEG2RAD(d) ((d)*M_PI/180.0)
#define MACRO_RAD2DEG(r) ((r)*180.0/M_PI)

// pcl::PointCloud interface for our ahc::PlaneFitter
template<class PointT>
struct OrganizedImage3D {
    const pcl::PointCloud<PointT>& cloud;
    const double unitScaleFactor;

    OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1000) {}
    int width() const { return cloud.width; }
    int height() const { return cloud.height; }
    bool get(const int row, const int col, double& x, double& y, double& z) const {
        if (col >= cloud.width || row >= cloud.height) {
            return false;  // out of bounds
        }
        const PointT& pt = cloud.at(col, row);
        x = pt.x; y = pt.y; z = pt.z;
        return !std::isnan(z);  // Use std::isnan to check for NaN
    }
};
typedef OrganizedImage3D<pcl::PointXYZRGBA> RGBDImage;
typedef ahc::PlaneFitter<RGBDImage> PlaneFitter;

class MainLoop {
protected:
    PlaneFitter pf;
    cv::Mat rgb, seg;
    bool done;
    rs2::pipeline pipe;

    // Filters
    rs2::temporal_filter temporal_filter;
    rs2::disparity_transform depth_to_disparity{ true };
    rs2::disparity_transform disparity_to_depth{ false };

public:
    MainLoop() : done(false) {
        try {
            // Configure and start the RealSense pipeline
            rs2::config cfg;
            //cfg.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_BGR8);
            //cfg.enable_stream(RS2_STREAM_DEPTH, RS2_FORMAT_Z16);
            cfg.enable_stream(RS2_STREAM_INFRARED, 1, 424, 240, RS2_FORMAT_Y8, 90);  // Infrared stream
            cfg.enable_stream(RS2_STREAM_DEPTH, 424, 240, RS2_FORMAT_Z16, 90);
            std::cout << "Starting RealSense pipeline..." << std::endl;
            pipe.start(cfg);
            std::cout << "RealSense pipeline started successfully!" << std::endl;

            // Set temporal filter options
            temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.01f);
            temporal_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 100);
            temporal_filter.set_option(RS2_OPTION_HOLES_FILL, 7);

            // Adjust plane fitting parameters
            pf.params.stdTol_merge = 0.06;      // Adjust MSE tolerance for merging planes
            pf.windowHeight = 5;               // Adjust window size for better plane fitting
            pf.windowWidth = 5;
            pf.minSupport = 12500;                // Adjust minimum support (number of points)
            pf.doRefine = true;                 // Enable refinement for better accuracy

            pf.params.angle_near = MACRO_DEG2RAD(35.0);
            pf.params.angle_far = MACRO_DEG2RAD(90.0);
            pf.params.similarityTh_merge = std::cos(MACRO_DEG2RAD(30.0));
            pf.params.similarityTh_refine = std::cos(MACRO_DEG2RAD(60.0));
            pf.params.depthAlpha = 0.005;
            pf.params.depthChangeTol = 0.005;
        }
        catch (const rs2::error& e) {
            std::cerr << "RealSense error: " << e.what() << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "Exception: " << e.what() << std::endl;
        }
    }

    // Process a new frame of point cloud
    void onNewCloud(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& cloud) {
        if (rgb.empty() || rgb.rows != cloud->height || rgb.cols != cloud->width) {
            rgb.create(cloud->height, cloud->width, CV_8UC3);
            seg.create(cloud->height, cloud->width, CV_8UC3);
        }
        for (int i = 0; i < (int)cloud->height; ++i) {
            for (int j = 0; j < (int)cloud->width; ++j) {
                const pcl::PointXYZRGBA& p = cloud->at(j, i);
                if (!std::isnan(p.z)) {
                    rgb.at<cv::Vec3b>(i, j) = cv::Vec3b(p.b, p.g, p.r);
                }
                else {
                    rgb.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255); // whiten invalid area
                }
            }
        }

        // Run PlaneFitter on the current frame of point cloud
        RGBDImage rgbd(*cloud);
        Timer timer(1000);
        timer.tic();
        pf.run(&rgbd, 0, &seg);
        double process_ms = timer.toc();

        // Check if any valid plane was found
        bool has_planes = !pf.extractedPlanes.empty();

        // If no plane detected, clear the segmentation image to black
        if (!has_planes) {
            seg.setTo(cv::Scalar(0, 0, 0));
        }

        cv::imshow("raw seg", seg);
        // Blend segmentation with rgb
        cv::cvtColor(seg, seg, cv::COLOR_RGB2BGR);  // Updated color conversion constant
        seg = (rgb + seg) / 2.0;

        // Show frame rate
        std::stringstream stext;
        stext << "Frame Rate: " << (1000.0 / process_ms) << "Hz";
        cv::putText(seg, stext.str(), cv::Point(15, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255, 1));

        //cv::imshow("rgb", rgb);
        cv::imshow("seg", seg);
    }

    // Start the main loop
    void run() {
        rs2::align align_to_infrared(RS2_STREAM_INFRARED);  // Align depth to infrared

        // Create the ROI for auto exposure (adjust these values to suit your needs)
        rs2::region_of_interest roi;
        roi.min_x = 100;
        roi.min_y = 10;
        roi.max_x = 424;
        roi.max_y = 240;

        // Get the active profile from the pipeline
        rs2::pipeline_profile active_profile = pipe.get_active_profile();

        // Get the device from the active profile
        rs2::device dev = active_profile.get_device();

        // Query the sensors from the device
        std::vector<rs2::sensor> sensors_started = dev.query_sensors();

        // Iterate over all the sensors
        for (rs2::sensor sensor_in : sensors_started) {
            // Check if the sensor supports the auto-exposure ROI
            if (sensor_in.is<rs2::roi_sensor>()) {
                rs2::roi_sensor roi_sensor = sensor_in.as<rs2::roi_sensor>();

                try {
                    // Set the ROI for auto exposure
                    roi_sensor.set_region_of_interest(roi);
                    std::cout << "ROI set successfully for sensor." << std::endl;
                }
                catch (const rs2::invalid_value_error& e) {
                    std::cout << "Invalid value error: " << e.what() << std::endl;
                }
                catch (const rs2::error& e) {
                    std::cout << "RealSense error calling " << e.get_failed_function() << "("
                        << e.get_failed_args() << "): " << e.what() << std::endl;
                }
            }
        }

        cv::namedWindow("Control", cv::WINDOW_NORMAL);
        int mergeMSETol = static_cast<int>(pf.params.stdTol_merge * 1000);
        int minSupport = pf.minSupport;
        int doRefine = static_cast<int>(pf.doRefine);
        int windowHeight = pf.windowHeight;

        int angle_near = MACRO_RAD2DEG(pf.params.angle_near);
        int angle_far = MACRO_RAD2DEG(pf.params.angle_far);
        int similarityTh_merge = static_cast<int>(MACRO_RAD2DEG(std::acos(pf.params.similarityTh_merge)));
        int similarityTh_refine = static_cast<int>(MACRO_RAD2DEG(std::acos(pf.params.similarityTh_refine)));
        int depthAlpha = static_cast<int>(pf.params.depthAlpha * 1000);
        int depthChangeTol = static_cast<int>(pf.params.depthChangeTol * 1000);

        cv::createTrackbar("MSE Tol Merge", "Control", &mergeMSETol, 100);
        cv::createTrackbar("Min Support", "Control", &minSupport, 100000);
        cv::createTrackbar("Refine", "Control", &doRefine, 1);
        cv::createTrackbar("PF Window Size", "Control", &windowHeight, 100);

        cv::createTrackbar("Angle Near", "Control", &angle_near, 100);//pf.params.angle_near = MACRO_DEG2RAD(10.0);
        cv::createTrackbar("Angle Far", "Control", &angle_far, 100);//pf.params.angle_far = MACRO_DEG2RAD(90.0);
        cv::createTrackbar("Th Merge", "Control", &similarityTh_merge, 100);//pf.params.similarityTh_merge = std::cos(MACRO_DEG2RAD(120.0));
        cv::createTrackbar("Th Refine", "Control", &similarityTh_refine, 100);//pf.params.similarityTh_refine = std::cos(MACRO_DEG2RAD(30.0));
        cv::createTrackbar("Depth Alpha", "Control", &depthAlpha, 100);//pf.params.depthAlpha = 0.04;
        cv::createTrackbar("Depth Change Tolerance", "Control", &depthChangeTol, 100);//pf.params.depthChangeTol = 0.001;

        while (!done) {
            try {
                pf.params.stdTol_merge = mergeMSETol / 1000.0;
                pf.minSupport = minSupport;
                pf.doRefine = (doRefine != 0);
                pf.windowHeight = windowHeight;
                pf.windowWidth = windowHeight;

                pf.params.angle_near = MACRO_DEG2RAD(angle_near)/100;
                pf.params.angle_far = MACRO_DEG2RAD(angle_far)/100;
                pf.params.similarityTh_merge = std::cos(MACRO_DEG2RAD(similarityTh_merge))/100;
                pf.params.similarityTh_refine = std::cos(MACRO_DEG2RAD(similarityTh_refine))/100;
                pf.params.depthAlpha = depthAlpha / 1000.0;
                pf.params.depthChangeTol = depthChangeTol / 1000.0;
                
                // Capture frames from RealSense
                rs2::frameset frames = pipe.wait_for_frames();
                frames = align_to_infrared.process(frames);  // Align frames

                rs2::video_frame ir_frame = frames.get_infrared_frame();
                rs2::depth_frame depth_frame = frames.get_depth_frame();

                if (!ir_frame || !depth_frame) {
                    std::cout << "Failed to retrieve infrared or depth frame!" << std::endl;
                    continue;
                }
                std::cout << "Captured frames successfully!" << std::endl;


                // Convert RealSense infrared frame to OpenCV matrix (grayscale)
                cv::Mat ir_img(cv::Size(ir_frame.get_width(), ir_frame.get_height()), CV_8UC1, (void*)ir_frame.get_data(), cv::Mat::AUTO_STEP);

                // Apply median blur and bilateral filter for noise reduction
                /*cv::Mat ir_img_blurred;
                cv::medianBlur(ir_img, ir_img_blurred, 7);
                cv::Mat ir_img_bilateral;
                cv::bilateralFilter(ir_img_blurred, ir_img_bilateral, 10, 100, 100);

                // Edge detection using Canny with adjusted thresholds
                cv::Mat edges;
                cv::Canny(ir_img_blurred, edges, 120, 60);

                // Dilate the edges to make them thicker
                cv::Mat edges_dilated;
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
                cv::dilate(edges, edges_dilated, kernel, cv::Point(-1, -1), 2);

                // Convert RealSense depth frame to OpenCV matrix (depth image)
                cv::Mat depth_img(cv::Size(depth_frame.get_width(), depth_frame.get_height()), CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

                // Highlight the edges on the depth image by modifying the depth image
                cv::Mat depth_with_edges = depth_img.clone();
                depth_with_edges.setTo(0, edges_dilated > 0);  // Set depth to a minimum value where edges are detected

                // Get the data buffer of the original depth frame
                const uint16_t* original_depth_data = reinterpret_cast<const uint16_t*>(depth_frame.get_data());

                // Allocate a new buffer for the modified depth frame
                size_t frame_size = depth_with_edges.total() * depth_with_edges.elemSize();
                std::vector<uint16_t> new_depth_data(depth_with_edges.begin<uint16_t>(), depth_with_edges.end<uint16_t>());
                rs2::frame modified_depth_frame = depth_frame;

                memcpy((void*)original_depth_data, new_depth_data.data(), frame_size);*/


                // Convert RealSense infrared frames to PCL point cloud
                PointCloudRGBA::Ptr cloud(new PointCloudRGBA);
                cloud->width = ir_frame.get_width();
                cloud->height = ir_frame.get_height();
                cloud->is_dense = false;
                cloud->points.resize(cloud->width * cloud->height);

                int depth_width = depth_frame.get_width();
                int depth_height = depth_frame.get_height();

                for (int y = 0; y < cloud->height; y++) {
                    for (int x = 0; x < cloud->width; x++) {
                        pcl::PointXYZRGBA& p = cloud->at(x, y);

                        if (x < depth_width && y < depth_height) {
                            float depth = depth_frame.get_distance(x, y);
                            p.x = static_cast<float>(x) * depth;
                            p.y = static_cast<float>(y) * depth;
                            p.z = depth;
                        }
                        else {
                            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                        }

                        const uint8_t* ir_data = reinterpret_cast<const uint8_t*>(ir_frame.get_data());
                        uint8_t intensity = ir_data[y * cloud->width + x];  // Grayscale infrared intensity
                        p.r = p.g = p.b = intensity;  // Use grayscale value for RGB
                    }
                }

                onNewCloud(cloud);
                onKey(cv::waitKey(10));
            }
            catch (const rs2::error& e) {
                std::cerr << "RealSense error during frame capture: " << e.what() << std::endl;
            }
            catch (const std::exception& e) {
                std::cerr << "Exception during frame capture: " << e.what() << std::endl;
            }
        }
    }

    // Handle keyboard commands
    void onKey(const unsigned char key) {
        if (key == 'q') {
            done = true;
        }
    }
};

int main() {
    MainLoop loop;
    loop.run();
    std::cout << "Press any key to exit..." << std::endl;
    std::cin.get();  // Wait for user input before exiting
    return 0;
}
