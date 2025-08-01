#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>

struct TrackedPoint {
    cv::Point2f pos;
    int life;
    int size;
    std::string label;
    double font_scale;
    cv::Scalar text_color;
    bool vertical;
    bool is_bright;
    bool is_dark;
};

struct EdgePoint {
    cv::Point2f pos;
    int life;
    int size;
};

// Global parameters for edge detection
bool show_edge_boxes = false;
int edge_box_amount = 130;
int edge_box_life = 1;
bool is_fullscreen = false;

// QR code image
cv::Mat qr_code;
bool qr_loaded = false;

std::string random_label(std::mt19937& gen) {
    std::uniform_real_distribution<> r(0, 1);
    double rand_val = r(gen);

    if (rand_val < 0.33) {
        // Hex-like string
        static const char charset[] = "ABCDEF0123456789";
        std::uniform_int_distribution<> dist(0, 15);
        std::string label;
        for (int i = 0; i < 6; ++i) label += charset[dist(gen)];
        return label;
    } else if (rand_val < 0.66) {
        // Random number
        std::uniform_int_distribution<> dist(1, 999);
        return std::to_string(dist(gen));
    } else {
        // UUID-like string
        static const char charset[] = "abcdef0123456789";
        std::uniform_int_distribution<> dist(0, 15);
        std::string label;
        for (int i = 0; i < 8; ++i) label += charset[dist(gen)];
        return label;
    }
}

int sample_size_bell(std::mt19937& gen, int min_size = 7, int max_size = 60, double width_div = 4.0) {
    double mean = (min_size + max_size) / 2.0;
    double sigma = (max_size - min_size) / width_div;
    std::normal_distribution<> dist(mean, sigma);

    for (int i = 0; i < 10; ++i) {
        int val = static_cast<int>(dist(gen));
        if (val >= min_size && val <= max_size) {
            return val;
        }
    }
    return std::clamp(static_cast<int>(dist(gen)), min_size, max_size);
}

cv::Scalar random_text_color(std::mt19937& gen) {
    std::uniform_int_distribution<> choice(0, 2);
    switch (choice(gen)) {
        case 0: return cv::Scalar(255, 255, 255); // White
        case 1: return cv::Scalar(0, 0, 0);       // Black
        default: return cv::Scalar(255, 0, 255);  // Magenta
    }
}

void draw_vertical_text(cv::Mat& frame, const std::string& text, cv::Point org,
                       int font_face, double font_scale, cv::Scalar color, int thickness) {
    int y_cursor = org.y;
    int line_height = static_cast<int>(12 * font_scale);

    for (char ch : text) {
        cv::putText(frame, std::string(1, ch), cv::Point(org.x, y_cursor),
                   font_face, font_scale, color, thickness, cv::LINE_AA);
        y_cursor += line_height;
    }
}

void processEdgeDetection(cv::Mat& frame, const cv::Mat& gray, std::vector<EdgePoint>& edge_points, std::mt19937& gen) {
    if (!show_edge_boxes) return;

    // Detect edges using Canny
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);

    // Find edge pixels
    std::vector<cv::Point> edge_pixels;
    cv::findNonZero(edges, edge_pixels);

    // Update existing edge points life
    std::vector<EdgePoint> new_edge_points;
    for (auto& ep : edge_points) {
        if (ep.life > 0) {
            ep.life--;
            new_edge_points.push_back(ep);
        }
    }

    // Add new edge points if we have fewer than desired amount
    if (new_edge_points.size() < edge_box_amount && !edge_pixels.empty()) {
        std::shuffle(edge_pixels.begin(), edge_pixels.end(), gen);

        int to_add = std::min(edge_box_amount - static_cast<int>(new_edge_points.size()),
                             static_cast<int>(edge_pixels.size()));

        for (int i = 0; i < to_add; ++i) {
            EdgePoint ep;
            ep.pos = cv::Point2f(edge_pixels[i]);
            ep.life = edge_box_life;
            ep.size = static_cast<int>(sample_size_bell(gen) * 0.1); // 10% of normal size
            ep.size = std::max(ep.size, 3); // Minimum size of 3
            new_edge_points.push_back(ep);
        }
    }

    edge_points = new_edge_points;

    // Draw connections between edge points (each connects to 2 others)
    for (size_t i = 0; i < edge_points.size(); ++i) {
        std::vector<std::pair<float, size_t>> dists;
        for (size_t j = 0; j < edge_points.size(); ++j) {
            if (i == j) continue;
            float d = cv::norm(edge_points[i].pos - edge_points[j].pos);
            dists.emplace_back(d, j);
        }
        std::sort(dists.begin(), dists.end());

        // Connect to 2 nearest neighbors
        for (int k = 0; k < std::min(2, static_cast<int>(dists.size())); ++k) {
            cv::line(frame, edge_points[i].pos, edge_points[dists[k].second].pos, cv::Scalar(0, 0, 255), 1);
        }
    }

    // Draw red boxes for edge points
    for (const auto& ep : edge_points) {
        int x = static_cast<int>(ep.pos.x);
        int y = static_cast<int>(ep.pos.y);
        int s = ep.size;

        cv::Point tl(x - s/2, y - s/2);
        cv::Point br(x + s/2, y + s/2);

        // Ensure bounds
        tl.x = std::max(0, tl.x);
        tl.y = std::max(0, tl.y);
        br.x = std::min(frame.cols - 1, br.x);
        br.y = std::min(frame.rows - 1, br.y);

        // Draw red hollow square
        cv::rectangle(frame, tl, br, cv::Scalar(0, 0, 255), 1);
    }
}

void handleKeyboardInput(int key) {
    switch (key) {
        case 'f':
        case 'F':
            show_edge_boxes = !show_edge_boxes;
            std::cout << "Edge boxes: " << (show_edge_boxes ? "ON" : "OFF") << std::endl;
            break;
        case 'q':
        case 'Q':
            is_fullscreen = !is_fullscreen;
            if (is_fullscreen) {
                cv::setWindowProperty("Live Feed with Beat-Synced Effects", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            } else {
                cv::setWindowProperty("Live Feed with Beat-Synced Effects", cv::WND_PROP_FULLSCREEN, cv::WINDOW_NORMAL);
            }
            std::cout << "Fullscreen: " << (is_fullscreen ? "ON" : "OFF") << std::endl;
            break;
        case 'g':
        case 'G':
            edge_box_amount = std::min(600, edge_box_amount + 10);
            std::cout << "Edge box amount: " << edge_box_amount << std::endl;
            break;
        case 'h':
        case 'H':
            edge_box_amount = std::max(10, edge_box_amount - 10);
            std::cout << "Edge box amount: " << edge_box_amount << std::endl;
            break;
        default:
            break;
    }
}

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera\n";
        return -1;
    }

    // Load QR code image
    qr_code = cv::imread("../images/qr.png", cv::IMREAD_UNCHANGED);
    if (!qr_code.empty()) {
        qr_loaded = true;
        // Resize QR code to reasonable size (e.g., 150x150 pixels)
        cv::resize(qr_code, qr_code, cv::Size(150, 150));
        std::cout << "QR code loaded successfully" << std::endl;
    } else {
        std::cout << "Warning: Could not load QR code from images/qr.png" << std::endl;
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create(1500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31, 20);
    cv::Mat prev_gray, gray;

    std::vector<TrackedPoint> active;
    std::vector<EdgePoint> edge_points;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> uniform01(0, 1);
    std::normal_distribution<float> jitter_dist(0, 0.5f);
    std::poisson_distribution<int> ambient_spawn(5.0 / 30.0); // 5 per second at 30fps

    // Beat simulation parameters
    int frame_count = 2;
    int beat_interval = 10; // Simulate beat every 60 frames (~2 seconds at 30fps)
    int pts_per_beat = 20;
    int life_frames = 10;
    int neighbor_links = 3;
    double ambient_rate = 5.0;
    int min_points = 30;   // Minimum points to maintain
    int max_points = 300; // Maximum points allowed

    // Bright/dark spot detection parameters
    int bright_dark_check_interval = 30; // Check every 30 frames

    std::cout << "Controls: F - Toggle edge boxes, Q - Toggle fullscreen, G/H - Increase/Decrease edge box amount" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        frame_count++;

        // Track existing points with optical flow
        if (!prev_gray.empty() && !active.empty()) {
            std::vector<cv::Point2f> prev_pts, next_pts;
            for (const auto& tp : active) {
                prev_pts.push_back(tp.pos);
            }

            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, next_pts, status, err);

            std::vector<TrackedPoint> new_active;
            for (size_t i = 0; i < active.size(); ++i) {
                if (status[i] && active[i].life > 0) {
                    TrackedPoint tp = active[i];
                    tp.pos = next_pts[i];
                    tp.life--;

                    // Apply jitter
                    tp.pos.x += jitter_dist(gen);
                    tp.pos.y += jitter_dist(gen);
                    tp.pos.x = std::clamp(tp.pos.x, 0.0f, static_cast<float>(frame.cols - 1));
                    tp.pos.y = std::clamp(tp.pos.y, 0.0f, static_cast<float>(frame.rows - 1));

                    new_active.push_back(tp);
                }
            }
            active = new_active;
        }

        // Bright and dark spot detection
        if (frame_count % bright_dark_check_interval == 0) {
            // Find bright spots (local maxima)
            cv::Mat bright_mask;
            cv::threshold(gray, bright_mask, 200, 255, cv::THRESH_BINARY);

            std::vector<cv::Point> bright_locations;
            cv::findNonZero(bright_mask, bright_locations);

            // Sort by brightness and take top 10
            std::vector<std::pair<int, cv::Point>> bright_with_intensity;
            for (const auto& pt : bright_locations) {
                if (pt.x >= 0 && pt.y >= 0 && pt.x < gray.cols && pt.y < gray.rows) {
                    bright_with_intensity.emplace_back(gray.at<uchar>(pt), pt);
                }
            }
            std::sort(bright_with_intensity.begin(), bright_with_intensity.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

            // Spawn points for top 10 brightest spots
            int bright_spawned = 0;
            for (const auto& [intensity, pt] : bright_with_intensity) {
                if (bright_spawned >= 10 || static_cast<int>(active.size()) >= max_points) break;

                // Check if too close to existing points
                bool too_close = false;
                for (const auto& tp : active) {
                    if (cv::norm(tp.pos - cv::Point2f(pt)) < 20) {
                        too_close = true;
                        break;
                    }
                }
                if (too_close) continue;

                TrackedPoint tp;
                tp.pos = cv::Point2f(pt);
                tp.life = life_frames + 10; // Longer life for bright spots
                tp.size = sample_size_bell(gen, 18, 90); // Larger size for bright spots
                tp.label = "BRIGHT";
                tp.font_scale = std::uniform_real_distribution<>(1.5, 2.2)(gen);
                tp.text_color = cv::Scalar(0, 255, 255); // Yellow
                tp.vertical = uniform01(gen) < 0.3;
                tp.is_bright = true;
                tp.is_dark = false;

                active.push_back(tp);
                bright_spawned++;
            }

            // Find dark spots (local minima)
            cv::Mat dark_mask;
            cv::threshold(gray, dark_mask, 50, 255, cv::THRESH_BINARY_INV);

            std::vector<cv::Point> dark_locations;
            cv::findNonZero(dark_mask, dark_locations);

            // Sort by darkness and take some dark spots
            std::vector<std::pair<int, cv::Point>> dark_with_intensity;
            for (const auto& pt : dark_locations) {
                if (pt.x >= 0 && pt.y >= 0 && pt.x < gray.cols && pt.y < gray.rows) {
                    dark_with_intensity.emplace_back(gray.at<uchar>(pt), pt);
                }
            }
            std::sort(dark_with_intensity.begin(), dark_with_intensity.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

            // Spawn points for dark spots
            int dark_spawned = 0;
            int max_dark = std::uniform_int_distribution<>(5, 15)(gen);
            for (const auto& [intensity, pt] : dark_with_intensity) {
                if (dark_spawned >= max_dark || static_cast<int>(active.size()) >= max_points) break;

                // Check if too close to existing points
                bool too_close = false;
                for (const auto& tp : active) {
                    if (cv::norm(tp.pos - cv::Point2f(pt)) < 20) {
                        too_close = true;
                        break;
                    }
                }
                if (too_close) continue;

                TrackedPoint tp;
                tp.pos = cv::Point2f(pt);
                tp.life = life_frames + 5;
                tp.size = sample_size_bell(gen, 15, 67); // Medium size for dark spots
                tp.label = "DARK";
                tp.font_scale = std::uniform_real_distribution<>(1.2, 1.8)(gen);
                tp.text_color = cv::Scalar(128, 0, 128); // Purple
                tp.vertical = uniform01(gen) < 0.2;
                tp.is_bright = false;
                tp.is_dark = true;

                active.push_back(tp);
                dark_spawned++;
            }
        }

        // Simulate beat detection - spawn points on "beats"
        if (frame_count % beat_interval == 0) {
            std::vector<cv::KeyPoint> kps;
            orb->detect(gray, kps);

            // Sort by response strength
            std::sort(kps.begin(), kps.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                return a.response > b.response;
            });

            int target_spawn = std::uniform_int_distribution<>(1, pts_per_beat)(gen);
            // Limit spawning based on current count and max limit
            target_spawn = std::min(target_spawn, max_points - static_cast<int>(active.size()));
            int spawned = 0;

            for (const auto& kp : kps) {
                if (spawned >= target_spawn || static_cast<int>(active.size()) >= max_points) break;

                // Check if too close to existing points
                bool too_close = false;
                for (const auto& tp : active) {
                    if (cv::norm(tp.pos - kp.pt) < 10) {
                        too_close = true;
                        break;
                    }
                }
                if (too_close) continue;

                TrackedPoint tp;
                tp.pos = kp.pt;
                tp.life = life_frames;
                tp.size = sample_size_bell(gen);
                tp.label = random_label(gen);
                tp.font_scale = std::uniform_real_distribution<>(1.0, 1.8)(gen);
                tp.text_color = random_text_color(gen);
                tp.vertical = uniform01(gen) < 0.25;
                tp.is_bright = false;
                tp.is_dark = false;

                active.push_back(tp);
                spawned++;
            }
        }

        // Ambient random spawns - but respect limits
        if (static_cast<int>(active.size()) < max_points) {
            int noise_spawns = ambient_spawn(gen);
            noise_spawns = std::min(noise_spawns, max_points - static_cast<int>(active.size()));

            for (int i = 0; i < noise_spawns; ++i) {
                TrackedPoint tp;
                tp.pos = cv::Point2f(uniform01(gen) * frame.cols, uniform01(gen) * frame.rows);
                tp.life = life_frames;
                tp.size = sample_size_bell(gen);
                tp.label = random_label(gen);
                tp.font_scale = std::uniform_real_distribution<>(1.0, 1.8)(gen);
                tp.text_color = random_text_color(gen);
                tp.vertical = uniform01(gen) < 0.25;
                tp.is_bright = false;
                tp.is_dark = false;

                active.push_back(tp);
            }
        }

        // Force spawn if below minimum
        if (static_cast<int>(active.size()) < min_points) {
            int needed = min_points - static_cast<int>(active.size());
            for (int i = 0; i < needed; ++i) {
                TrackedPoint tp;
                tp.pos = cv::Point2f(uniform01(gen) * frame.cols, uniform01(gen) * frame.rows);
                tp.life = life_frames;
                tp.size = sample_size_bell(gen);
                tp.label = random_label(gen);
                tp.font_scale = std::uniform_real_distribution<>(1.0, 1.8)(gen);
                tp.text_color = random_text_color(gen);
                tp.vertical = uniform01(gen) < 0.25;
                tp.is_bright = false;
                tp.is_dark = false;

                active.push_back(tp);
            }
        }

        // Process edge detection and draw edge boxes
        processEdgeDetection(frame, gray, edge_points, gen);

        // Draw edges to nearest neighbors
        for (size_t i = 0; i < active.size(); ++i) {
            std::vector<std::pair<float, size_t>> dists;
            for (size_t j = 0; j < active.size(); ++j) {
                if (i == j) continue;
                float d = cv::norm(active[i].pos - active[j].pos);
                dists.emplace_back(d, j);
            }
            std::sort(dists.begin(), dists.end());

            for (int k = 0; k < std::min(neighbor_links, static_cast<int>(dists.size())); ++k) {
                cv::line(frame, active[i].pos, active[dists[k].second].pos, cv::Scalar(255, 255, 255), 1);
            }
        }

        // Draw squares and labels for each point
        for (const auto& tp : active) {
            int x = static_cast<int>(tp.pos.x);
            int y = static_cast<int>(tp.pos.y);
            int s = tp.size;

            cv::Point tl(x - s/2, y - s/2);
            cv::Point br(x + s/2, y + s/2);

            // Ensure bounds
            tl.x = std::max(0, tl.x);
            tl.y = std::max(0, tl.y);
            br.x = std::min(frame.cols - 1, br.x);
            br.y = std::min(frame.rows - 1, br.y);

            // Invert colors inside the box for pop effect
            if (br.x > tl.x && br.y > tl.y) {
                cv::Mat roi = frame(cv::Rect(tl, br));
                cv::bitwise_not(roi, roi);
            }

            // Draw white hollow square
            cv::rectangle(frame, tl, br, cv::Scalar(255, 255, 255), 1);

            // Draw label text
            if (tp.vertical) {
                draw_vertical_text(frame, tp.label, cv::Point(tl.x + 2, tl.y + 12),
                                 cv::FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1);
            } else {
                cv::putText(frame, tp.label, cv::Point(tl.x + 2, br.y - 4),
                           cv::FONT_HERSHEY_PLAIN, tp.font_scale, tp.text_color, 1, cv::LINE_AA);
            }
        }

        // Draw "ROBOTUPRISING.FI" text at the top of the frame with black banner background
        int banner_height = 50;
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(frame.cols, banner_height), cv::Scalar(0, 0, 0), -1);
        cv::putText(frame, "https://ROBOTUPRISING.FI  - github.com/robot-uprising-hq", cv::Point(20, 35),
                   cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

        // Overlay QR code in lower right corner
        if (qr_loaded && !qr_code.empty()) {
            int qr_margin = 20;
            int qr_x = frame.cols - qr_code.cols - qr_margin;
            int qr_y = frame.rows - qr_code.rows - qr_margin;

            // Ensure QR code fits within frame boundaries
            if (qr_x >= 0 && qr_y >= 0) {
                cv::Rect roi(qr_x, qr_y, qr_code.cols, qr_code.rows);

                if (qr_code.channels() == 4) {
                    // Handle PNG with alpha channel
                    cv::Mat qr_bgr, qr_alpha;
                    std::vector<cv::Mat> qr_channels;
                    cv::split(qr_code, qr_channels);
                    cv::merge(std::vector<cv::Mat>{qr_channels[0], qr_channels[1], qr_channels[2]}, qr_bgr);
                    qr_alpha = qr_channels[3];

                    // Apply alpha blending
                    cv::Mat frame_roi = frame(roi);
                    for (int y = 0; y < qr_bgr.rows; ++y) {
                        for (int x = 0; x < qr_bgr.cols; ++x) {
                            float alpha = qr_alpha.at<uchar>(y, x) / 255.0f;
                            frame_roi.at<cv::Vec3b>(y, x) = alpha * qr_bgr.at<cv::Vec3b>(y, x) +
                                                           (1.0f - alpha) * frame_roi.at<cv::Vec3b>(y, x);
                        }
                    }
                } else {
                    // Simple copy for images without alpha channel
                    qr_code.copyTo(frame(roi));
                }
            }
        }

        cv::imshow("Live Feed with Beat-Synced Effects", frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break; // ESC key
        handleKeyboardInput(key);

        prev_gray = gray.clone();
    }

    return 0;
}
