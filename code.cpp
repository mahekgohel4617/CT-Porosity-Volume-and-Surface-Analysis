#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

// -----------------------------------------------------------------------------
//  User-tunable parameters (all settable from the command line)
// -----------------------------------------------------------------------------
struct Params {
    // physical voxel size (so volume comes out in unit^3)
    double dx = 1.0, dy = 1.0, dz = 1.0;
    std::string unit = "px";

    // --- ROI (ring) detection ---
    double minContrast   = 30;    // slice max intensity must exceed this, else "empty"
    int    minRingPixels = 200;   // minimum bright pixels needed to attempt a fit
    double ringTol       = 3.0;   // inlier distance to fitted circle (px), grows with MAD
    double minInlierFrac = 0.6;   // inliers / bright pixels
    double minCoverage   = 0.7;   // fraction of the 72 angular sectors containing ring
    double minRadius     = 20.0;  // px
    double maxRadiusFrac = 0.6;   // max radius as fraction of min(width,height)
    double maxThickFrac  = 0.3;   // (rOut - rIn) must be <= this * radius

    // --- pore thresholding ---
    double poreFrac  = 0.5;       // pore pixel if gray < poreFrac * ring brightness
    double bandShrink = 0.6;      // px ignored at each edge of the ring band (edge pixels of a
                                  // thin, discretised ring are noisy and give 1-px false pores)
    int    openIters = 0;         // 3x3 opening iterations on the 2-D pore mask (0 = off)

    // --- 3-D filtering / classification ---
    int    minVoxels     = 4;     // discard pores smaller than this
    int    minSlices     = 2;     // pores must span at least this many slices
    double minSphericity = 0.70;  // flag "spherical" if sphericity >= this ...
    double minAspect     = 0.50;  // ... and shortest/longest principal axis >= this

    // --- I/O ---
    std::string csvPath;          // write CSV here if non-empty
    std::string debugDir;         // write overlay PNGs here if non-empty
    bool verbose = false;
};

// -----------------------------------------------------------------------------
//  Data structures
// -----------------------------------------------------------------------------
struct RingInfo {
    bool   valid = false;
    double cx = 0, cy = 0, r = 0;       // fitted circle
    double rIn = 0, rOut = 0;           // band occupied by the ring
    double brightness = 0;              // median intensity of ring pixels
    double coverage = 0;                // fraction of circumference seen
    std::string reason;                 // why invalid (for the log)
};

// One 2-D pore cross-section inside a single slice
struct Node {
    int z = 0;
    std::vector<cv::Point> px;
};

struct Pore {
    int id = 0;
    long long voxels = 0;
    double volume = 0, eqDiam = 0, surface = 0, sphericity = 0;
    double axisA = 0, axisB = 0, axisC = 0;   // principal axis full-lengths (a>=b>=c)
    double aspect = 0;                        // c / a
    double fill = 0;                          // V / V(sphere of radius Rmax)
    double cx = 0, cy = 0, cz = 0;            // centroid in voxel index units
    int x0 = 0, x1 = 0, y0 = 0, y1 = 0, z0 = 0, z1 = 0;
    double angleDeg = 0;                      // angular position on the ring
    bool spherical = false;
};

// -----------------------------------------------------------------------------
//  Image loading
// -----------------------------------------------------------------------------
static cv::Mat toGray8(const cv::Mat& src) {
    cv::Mat g;
    if (src.channels() == 3)      cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
    else if (src.channels() == 4) cv::cvtColor(src, g, cv::COLOR_BGRA2GRAY);
    else                          g = src;
    if (g.depth() == CV_16U)      g.convertTo(g, CV_8U, 255.0 / 65535.0);
    else if (g.depth() != CV_8U)  { cv::normalize(g, g, 0, 255, cv::NORM_MINMAX); g.convertTo(g, CV_8U); }
    return g;
}

// Loads a stack from a folder (sorted by file name) or a multi-page TIFF.
static std::vector<cv::Mat> loadStack(const std::string& path, std::vector<std::string>& names) {
    std::vector<cv::Mat> slices;
    if (fs::is_directory(path)) {
        std::vector<fs::path> files;
        for (auto& e : fs::directory_iterator(path)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            for (auto& c : ext) c = (char)std::tolower(c);
            if (ext == ".tif" || ext == ".tiff" || ext == ".png" || ext == ".bmp" ||
                ext == ".jpg" || ext == ".jpeg")
                files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());      // slice order = file-name order
        for (auto& f : files) {
            cv::Mat m = cv::imread(f.string(), cv::IMREAD_UNCHANGED);
            if (m.empty()) { std::cerr << "  warning: cannot read " << f << "\n"; continue; }
            slices.push_back(toGray8(m));
            names.push_back(f.filename().string());
        }
    } else {
        std::vector<cv::Mat> pages;
        if (!cv::imreadmulti(path, pages, cv::IMREAD_UNCHANGED)) return slices;
        for (size_t i = 0; i < pages.size(); ++i) {
            slices.push_back(toGray8(pages[i]));
            names.push_back(fs::path(path).filename().string() + "#" + std::to_string(i));
        }
    }
    return slices;
}

// -----------------------------------------------------------------------------
//  Step 2 : automatic ROI = ring detection
// -----------------------------------------------------------------------------

// Algebraic (Kasa) least-squares circle fit
static bool fitCircle(const std::vector<cv::Point2d>& p, double& cx, double& cy, double& r) {
    if (p.size() < 3) return false;
    double mx = 0, my = 0;
    for (auto& q : p) { mx += q.x; my += q.y; }
    mx /= p.size(); my /= p.size();
    // minimise sum (u^2+v^2 + D u + E v + F)^2 with u=x-mx, v=y-my
    cv::Mat A(3, 3, CV_64F, cv::Scalar(0)), b(3, 1, CV_64F, cv::Scalar(0));
    for (auto& q : p) {
        double u = q.x - mx, v = q.y - my, w = u * u + v * v;
        double row[3] = {u, v, 1.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) A.at<double>(i, j) += row[i] * row[j];
            b.at<double>(i) += -w * row[i];
        }
    }
    cv::Mat sol;
    if (!cv::solve(A, b, sol, cv::DECOMP_SVD)) return false;
    double D = sol.at<double>(0), E = sol.at<double>(1), F = sol.at<double>(2);
    double ucx = -D / 2, ucy = -E / 2;
    double rr = ucx * ucx + ucy * ucy - F;
    if (rr <= 0) return false;
    cx = ucx + mx; cy = ucy + my; r = std::sqrt(rr);
    return true;
}

static double median(std::vector<double> v) {
    if (v.empty()) return 0;
    size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    return v[m];
}

static RingInfo detectRing(const cv::Mat& gray, const Params& P) {
    RingInfo ri;
    double mn, mx;
    cv::minMaxLoc(gray, &mn, &mx);
    if (mx < P.minContrast) { ri.reason = "slice is empty / no bright structure"; return ri; }

    // Otsu separates the bright ring from the dark background
    cv::Mat bw;
    cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    std::vector<cv::Point2d> pts;
    for (int y = 0; y < bw.rows; ++y) {
        const uchar* row = bw.ptr<uchar>(y);
        for (int x = 0; x < bw.cols; ++x)
            if (row[x]) pts.emplace_back(x, y);
    }
    if ((int)pts.size() < P.minRingPixels) { ri.reason = "too few bright pixels"; return ri; }

    // Robust fit: fit -> measure residuals -> keep inliers -> refit
    std::vector<char> inl(pts.size(), 1);
    double cx = 0, cy = 0, r = 0;
    std::vector<double> res(pts.size());
    for (int it = 0; it < 6; ++it) {
        std::vector<cv::Point2d> sel;
        for (size_t i = 0; i < pts.size(); ++i) if (inl[i]) sel.push_back(pts[i]);
        if (!fitCircle(sel, cx, cy, r)) { ri.reason = "circle fit failed"; return ri; }
        std::vector<double> absr;
        for (size_t i = 0; i < pts.size(); ++i) {
            res[i] = std::hypot(pts[i].x - cx, pts[i].y - cy) - r;
            if (inl[i]) absr.push_back(std::fabs(res[i]));
        }
        double mad = median(absr);
        double tol = std::max(P.ringTol, 3.0 * 1.4826 * mad);
        for (size_t i = 0; i < pts.size(); ++i) inl[i] = std::fabs(res[i]) < tol;
    }

    // Gather inlier statistics
    std::vector<double> radii, vals;
    std::vector<char> sector(72, 0);
    size_t nin = 0;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!inl[i]) continue;
        ++nin;
        double dx = pts[i].x - cx, dy = pts[i].y - cy;
        radii.push_back(std::hypot(dx, dy));
        vals.push_back(gray.at<uchar>((int)pts[i].y, (int)pts[i].x));
        double ang = std::atan2(dy, dx) + CV_PI;                 // 0..2pi
        sector[std::min(71, (int)(ang / (2 * CV_PI) * 72))] = 1;
    }
    ri.coverage = std::accumulate(sector.begin(), sector.end(), 0) / 72.0;

    if (nin < (size_t)(P.minInlierFrac * pts.size())) { ri.reason = "bright pixels do not form a circle"; return ri; }
    if (ri.coverage < P.minCoverage)                  { ri.reason = "circle only partly visible"; return ri; }
    if (r < P.minRadius || r > P.maxRadiusFrac * std::min(gray.cols, gray.rows)) {
        ri.reason = "radius out of range"; return ri;
    }

    std::sort(radii.begin(), radii.end());
    double rIn  = radii[(size_t)(0.02 * (radii.size() - 1))];    // robust ring edges
    double rOut = radii[(size_t)(0.98 * (radii.size() - 1))];
    if (rOut - rIn > P.maxThickFrac * r) { ri.reason = "ring too thick for its radius"; return ri; }

    ri.valid = true;
    ri.cx = cx; ri.cy = cy; ri.r = r;
    ri.rIn = rIn; ri.rOut = rOut;
    ri.brightness = median(vals);
    return ri;
}

// -----------------------------------------------------------------------------
//  Step 3 : threshold dark pixels inside the ring band
// -----------------------------------------------------------------------------
static cv::Mat poreMask(const cv::Mat& gray, const RingInfo& ri, const Params& P) {
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8U);
    const double T = P.poreFrac * ri.brightness;
    const double shrink = std::min(P.bandShrink, 0.4 * (ri.rOut - ri.rIn));
    const double bIn = ri.rIn + shrink, bOut = ri.rOut - shrink;   // core of the ring
    int xa = std::max(0, (int)std::floor(ri.cx - ri.rOut - 1));
    int xb = std::min(gray.cols - 1, (int)std::ceil(ri.cx + ri.rOut + 1));
    int ya = std::max(0, (int)std::floor(ri.cy - ri.rOut - 1));
    int yb = std::min(gray.rows - 1, (int)std::ceil(ri.cy + ri.rOut + 1));
    for (int y = ya; y <= yb; ++y)
        for (int x = xa; x <= xb; ++x) {
            double d = std::hypot(x - ri.cx, y - ri.cy);
            if (d >= bIn && d <= bOut && gray.at<uchar>(y, x) < T)
                mask.at<uchar>(y, x) = 255;
        }
    if (P.openIters > 0) {
        cv::Mat k = cv::getStructuringElement(cv::MORPH_CROSS, {3, 3});
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, k, {-1, -1}, P.openIters);
    }
    return mask;
}

// -----------------------------------------------------------------------------
//  Union-find for the 3-D linking
// -----------------------------------------------------------------------------
struct DSU {
    std::vector<int> p;
    int add() { p.push_back((int)p.size()); return (int)p.size() - 1; }
    int find(int a) { while (p[a] != a) { p[a] = p[p[a]]; a = p[a]; } return a; }
    void unite(int a, int b) { a = find(a); b = find(b); if (a != b) p[b] = a; }
};

// -----------------------------------------------------------------------------
//  Step 6 : measurements for one 3-D pore
// -----------------------------------------------------------------------------
static Pore measurePore(const std::vector<cv::Point3i>& vox, int W, int H, const Params& P,
                        double angleDeg) {
    Pore p;
    p.voxels = (long long)vox.size();
    p.angleDeg = angleDeg;
    const double dx = P.dx, dy = P.dy, dz = P.dz;
    const double voxVol = dx * dy * dz;
    p.volume = p.voxels * voxVol;
    p.eqDiam = std::cbrt(6.0 * p.volume / CV_PI);

    // centroid + bounding box
    double sx = 0, sy = 0, sz = 0;
    p.x0 = p.y0 = p.z0 = INT32_MAX; p.x1 = p.y1 = p.z1 = INT32_MIN;
    for (auto& v : vox) {
        sx += v.x; sy += v.y; sz += v.z;
        p.x0 = std::min(p.x0, v.x); p.x1 = std::max(p.x1, v.x);
        p.y0 = std::min(p.y0, v.y); p.y1 = std::max(p.y1, v.y);
        p.z0 = std::min(p.z0, v.z); p.z1 = std::max(p.z1, v.z);
    }
    p.cx = sx / vox.size(); p.cy = sy / vox.size(); p.cz = sz / vox.size();

    // surface area from exposed voxel faces
    std::unordered_set<int64_t> S;
    S.reserve(vox.size() * 2);
    auto key = [&](int x, int y, int z) { return ((int64_t)z * H + y) * W + x; };
    for (auto& v : vox) S.insert(key(v.x, v.y, v.z));
    long long Fx = 0, Fy = 0, Fz = 0;
    for (auto& v : vox) {
        if (!S.count(key(v.x - 1, v.y, v.z))) ++Fx;
        if (!S.count(key(v.x + 1, v.y, v.z))) ++Fx;
        if (!S.count(key(v.x, v.y - 1, v.z))) ++Fy;
        if (!S.count(key(v.x, v.y + 1, v.z))) ++Fy;
        if (!S.count(key(v.x, v.y, v.z - 1))) ++Fz;
        if (!S.count(key(v.x, v.y, v.z + 1))) ++Fz;
    }
    // Raw staircase area over-estimates a smooth surface. For a sphere the mean
    // of |n_x|+|n_y|+|n_z| is 1.5, so dividing by 1.5 (=> *2/3) gives an unbiased
    // area estimate for round objects (isotropic or anisotropic voxels).
    double Afaces = Fx * dy * dz + Fy * dx * dz + Fz * dx * dy;
    p.surface = Afaces * 2.0 / 3.0;
    // Wadell sphericity = area of equal-volume sphere / actual area
    double Asph = std::cbrt(CV_PI) * std::pow(6.0 * p.volume, 2.0 / 3.0);
    p.sphericity = std::min(1.0, Asph / p.surface);

    // principal axes from the covariance matrix (physical units)
    cv::Matx33d C = cv::Matx33d::zeros();
    double mx = p.cx * dx, my = p.cy * dy, mz = p.cz * dz;
    double Rmax = 0;
    for (auto& v : vox) {
        double a = v.x * dx - mx, b = v.y * dy - my, c = v.z * dz - mz;
        C(0, 0) += a * a; C(0, 1) += a * b; C(0, 2) += a * c;
        C(1, 1) += b * b; C(1, 2) += b * c; C(2, 2) += c * c;
        Rmax = std::max(Rmax, std::sqrt(a * a + b * b + c * c));
    }
    C(1, 0) = C(0, 1); C(2, 0) = C(0, 2); C(2, 1) = C(1, 2);
    C = C * (1.0 / vox.size());
    // add the variance of a voxel itself (d^2/12) so 1-voxel-thick pores stay sane
    C(0, 0) += dx * dx / 12; C(1, 1) += dy * dy / 12; C(2, 2) += dz * dz / 12;
    cv::Mat evals, evecs;
    cv::eigen(cv::Mat(C), evals, evecs);                    // descending
    double l1 = std::max(evals.at<double>(0), 1e-12);
    double l2 = std::max(evals.at<double>(1), 1e-12);
    double l3 = std::max(evals.at<double>(2), 1e-12);
    // solid ellipsoid: variance = (semi-axis)^2 / 5  ->  full length = 2*sqrt(5*lambda)
    p.axisA = 2 * std::sqrt(5 * l1);
    p.axisB = 2 * std::sqrt(5 * l2);
    p.axisC = 2 * std::sqrt(5 * l3);
    p.aspect = std::sqrt(l3 / l1);

    // fill ratio: how much of the smallest enclosing-from-centroid sphere is pore
    double Rs = Rmax + 0.5 * std::cbrt(voxVol);
    p.fill = p.volume / ((4.0 / 3.0) * CV_PI * Rs * Rs * Rs);

    p.spherical = (p.sphericity >= P.minSphericity) && (p.aspect >= P.minAspect);
    return p;
}

// -----------------------------------------------------------------------------
//  Process one stack
// -----------------------------------------------------------------------------
static int processStack(const std::string& path, const Params& P, std::ofstream* csv) {
    std::cout << "\n==============================================================\n"
              << "Stack: " << path << "\n"
              << "==============================================================\n";
    std::vector<std::string> names;
    std::vector<cv::Mat> keep;                              // only filled for debug overlays
    std::vector<cv::Mat> slices = loadStack(path, names);
    if (slices.empty()) { std::cerr << "  no readable images found\n"; return 0; }
    const int nZ = (int)slices.size();
    const int W = slices[0].cols, H = slices[0].rows;

    Params Q = P;
    if (nZ == 1 && Q.minSlices > 1) {
        std::cout << "  (single slice -> min-slices lowered to 1; result is a 2-D analysis)\n";
        Q.minSlices = 1;
    }

    std::vector<RingInfo> rings(nZ);
    std::vector<Node> nodes;
    DSU dsu;
    cv::Mat prevLabels; std::vector<int> prevNodeId; bool prevValid = false;
    int validSlices = 0;

    // ---- slice loop: ROI -> threshold -> 2-D components -> link to previous slice
    for (int z = 0; z < nZ; ++z) {
        cv::Mat& g = slices[z];
        if (g.cols != W || g.rows != H) { std::cerr << "  slice size mismatch at " << names[z] << "\n"; rings[z].reason = "size mismatch"; prevValid = false; continue; }
        if (!Q.debugDir.empty()) keep.push_back(g.clone());

        RingInfo ri = detectRing(g, Q);
        rings[z] = ri;
        if (!ri.valid) {
            if (Q.verbose) std::cout << "  slice " << std::setw(4) << z << " [" << names[z] << "]: no ROI (" << ri.reason << ")\n";
            prevValid = false;            // pores cannot be linked across a slice with no ROI
            continue;
        }
        ++validSlices;

        cv::Mat mask = poreMask(g, ri, Q);
        cv::Mat labels, stats, cent;
        int nl = cv::connectedComponentsWithStats(mask, labels, stats, cent, 8, CV_32S);

        std::vector<int> nodeId(nl, -1);
        for (int l = 1; l < nl; ++l) {
            nodeId[l] = dsu.add();
            Node nd; nd.z = z; nodes.push_back(nd);          // nodes index == dsu index
        }
        for (int y = 0; y < H; ++y) {
            const int* lr = labels.ptr<int>(y);
            for (int x = 0; x < W; ++x)
                if (lr[x] > 0) nodes[nodeId[lr[x]]].px.emplace_back(x, y);
        }
        // link with previous slice: any shared XY pixel => same 3-D pore
        if (prevValid) {
            for (int y = 0; y < H; ++y) {
                const int* a = labels.ptr<int>(y);
                const int* b = prevLabels.ptr<int>(y);
                for (int x = 0; x < W; ++x)
                    if (a[x] > 0 && b[x] > 0) dsu.unite(nodeId[a[x]], prevNodeId[b[x]]);
            }
        }
        if (Q.verbose)
            std::cout << "  slice " << std::setw(4) << z << " [" << names[z] << "]: ring c=("
                      << std::fixed << std::setprecision(1) << ri.cx << "," << ri.cy << ") R=" << ri.r
                      << " band=[" << ri.rIn << "," << ri.rOut << "] I=" << ri.brightness
                      << "  2-D pore sections: " << (nl - 1) << "\n";
        prevLabels = labels; prevNodeId = nodeId; prevValid = true;
    }

    std::cout << "  slices: " << nZ << "   with ROI: " << validSlices
              << "   without ROI: " << (nZ - validSlices) << "\n";

    // ---- group nodes by 3-D pore
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < (int)nodes.size(); ++i) groups[dsu.find(i)].push_back(i);

    std::vector<Pore> pores;
    std::vector<int> nodeToPore(nodes.size(), -1);
    // deterministic order: sort groups by first slice, then first pixel
    std::vector<std::pair<int, std::vector<int>>> ordered(groups.begin(), groups.end());
    std::sort(ordered.begin(), ordered.end(), [&](auto& a, auto& b) {
        int za = nodes[*std::min_element(a.second.begin(), a.second.end())].z;
        int zb = nodes[*std::min_element(b.second.begin(), b.second.end())].z;
        if (za != zb) return za < zb;
        return a.first < b.first;
    });

    for (auto& kv : ordered) {
        std::vector<cv::Point3i> vox;
        double sumCx = 0, sumCy = 0; int nr = 0;
        std::unordered_set<int> zs;
        for (int ni : kv.second) {
            for (auto& q : nodes[ni].px) vox.emplace_back(q.x, q.y, nodes[ni].z);
            if (!zs.count(nodes[ni].z)) {
                zs.insert(nodes[ni].z);
                sumCx += rings[nodes[ni].z].cx; sumCy += rings[nodes[ni].z].cy; ++nr;
            }
        }
        if ((int)vox.size() < Q.minVoxels) continue;
        if ((int)zs.size() < Q.minSlices) continue;

        // angular position of the pore on the circumference (0 deg = +x, CCW as seen
        // on screen with y pointing down => clockwise on screen)
        double mx = 0, my = 0;
        for (auto& v : vox) { mx += v.x; my += v.y; }
        mx /= vox.size(); my /= vox.size();
        double ang = std::atan2(my - sumCy / nr, mx - sumCx / nr) * 180.0 / CV_PI;
        if (ang < 0) ang += 360.0;

        Pore p = measurePore(vox, W, H, Q, ang);
        p.id = (int)pores.size() + 1;
        for (int ni : kv.second) nodeToPore[ni] = p.id;
        pores.push_back(p);
    }

    // ---- report
    const std::string u = Q.unit;
    std::cout << "\n  NUMBER OF PORES FOUND: " << pores.size() << "\n\n";
    if (!pores.empty()) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  " << std::setw(4) << "ID" << std::setw(9) << "z-range" << std::setw(8) << "voxels"
                  << std::setw(13) << ("vol[" + u + "^3]") << std::setw(11) << ("d_eq[" + u + "]")
                  << std::setw(10) << "spheric." << std::setw(9) << "c/a" << std::setw(8) << "fill"
                  << std::setw(21) << "centroid x,y,z[vox]" << std::setw(9) << "angle" << "  shape\n";
        for (auto& p : pores) {
            std::ostringstream zr; zr << p.z0 << "-" << p.z1;
            std::ostringstream cc; cc << std::fixed << std::setprecision(1) << p.cx << "," << p.cy << "," << p.cz;
            std::cout << "  " << std::setw(4) << p.id << std::setw(9) << zr.str() << std::setw(8) << p.voxels
                      << std::setw(13) << p.volume << std::setw(11) << p.eqDiam
                      << std::setw(10) << p.sphericity << std::setw(9) << p.aspect << std::setw(8) << p.fill
                      << std::setw(21) << cc.str() << std::setw(8) << std::setprecision(1) << p.angleDeg << "d"
                      << std::setprecision(3) << "  " << (p.spherical ? "spherical" : "irregular") << "\n";
        }
        double totV = 0, meanD = 0; int ns = 0;
        for (auto& p : pores) { totV += p.volume; meanD += p.eqDiam; ns += p.spherical; }
        std::cout << "\n  total pore volume : " << totV << " " << u << "^3"
                  << "\n  mean equiv. diam. : " << meanD / pores.size() << " " << u
                  << "\n  spherical pores   : " << ns << " / " << pores.size()
                  << "  (sphericity >= " << Q.minSphericity << " and c/a >= " << Q.minAspect << ")\n";
        std::cout << "  note: sphericity/area are estimates; pores of only a few voxels are unreliable.\n";
    }

    // ---- CSV (one row per pore = feature vector for ML)
    if (csv && csv->is_open()) {
        for (auto& p : pores)
            *csv << path << "," << p.id << "," << p.z0 << "," << p.z1 << "," << p.voxels << ","
                 << p.volume << "," << p.eqDiam << "," << p.surface << "," << p.sphericity << ","
                 << p.axisA << "," << p.axisB << "," << p.axisC << "," << p.aspect << "," << p.fill << ","
                 << p.cx << "," << p.cy << "," << p.cz << "," << p.angleDeg << "," << (p.spherical ? 1 : 0) << "\n";
    }

    // ---- debug overlays: green = fitted ring, red = kept pores (numbered)
    if (!Q.debugDir.empty() && (int)keep.size() == validSlices + (nZ - validSlices)) {
        fs::create_directories(Q.debugDir);
        std::string tag = fs::path(path).filename().string();
        for (int z = 0; z < nZ; ++z) {
            cv::Mat col;
            cv::cvtColor(keep[z], col, cv::COLOR_GRAY2BGR);
            if (rings[z].valid)
                cv::circle(col, {(int)std::lround(rings[z].cx), (int)std::lround(rings[z].cy)},
                           (int)std::lround(rings[z].r), {0, 160, 0}, 1, cv::LINE_AA);
            else
                cv::putText(col, "no ROI", {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 165, 255}, 2);
            for (size_t i = 0; i < nodes.size(); ++i) {
                if (nodes[i].z != z || nodeToPore[i] < 0) continue;
                for (auto& q : nodes[i].px) col.at<cv::Vec3b>(q) = {0, 0, 255};
                cv::putText(col, std::to_string(nodeToPore[i]), nodes[i].px[0] + cv::Point(6, -6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, {0, 255, 255}, 1, cv::LINE_AA);
            }
            std::ostringstream fn;
            fn << Q.debugDir << "/" << tag << "_z" << std::setw(4) << std::setfill('0') << z << ".png";
            cv::imwrite(fn.str(), col);
        }
        std::cout << "  overlays written to " << Q.debugDir << "/\n";
    }
    return (int)pores.size();
}

// -----------------------------------------------------------------------------
//  Command line
// -----------------------------------------------------------------------------
static void usage() {
    std::cout <<
        "usage: pore_analyzer <stack_dir | multipage.tif> [more stacks] [options]\n"
        "  --dx/--dy/--dz <v>     voxel size (default 1 1 1)       --unit <str>   unit label\n"
        "  --pore-frac <f>        pore if gray < f*ring brightness (default 0.5)\n"
        "  --band-shrink <px>     ignore this many px at both ring edges (default 0.6)\n"
        "  --min-aspect <f>       also require c/a >= f for the spherical flag (default 0.5)\n"
        "  --open <n>             3x3 opening iterations on 2-D pore mask (default 0)\n"
        "  --min-voxels <n>       drop pores smaller than n voxels (default 4)\n"
        "  --min-slices <n>       pores must span >= n slices (default 2)\n"
        "  --min-sphericity <f>   'spherical' flag threshold (default 0.70)\n"
        "  --ring-tol <px>        ring inlier tolerance (default 3)\n"
        "  --min-coverage <f>     fraction of circumference that must be visible (default 0.7)\n"
        "  --csv <file>           write per-pore features to CSV\n"
        "  --debug <dir>          write overlay PNGs\n"
        "  --verbose              per-slice log\n";
}

int main(int argc, char** argv) {
    Params P;
    std::vector<std::string> stacks;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << a << "\n"; std::exit(1); }
            return argv[++i];
        };
        if      (a == "--dx") P.dx = std::stod(next());
        else if (a == "--dy") P.dy = std::stod(next());
        else if (a == "--dz") P.dz = std::stod(next());
        else if (a == "--unit") P.unit = next();
        else if (a == "--pore-frac") P.poreFrac = std::stod(next());
        else if (a == "--band-shrink") P.bandShrink = std::stod(next());
        else if (a == "--min-aspect") P.minAspect = std::stod(next());
        else if (a == "--open") P.openIters = std::stoi(next());
        else if (a == "--min-voxels") P.minVoxels = std::stoi(next());
        else if (a == "--min-slices") P.minSlices = std::stoi(next());
        else if (a == "--min-sphericity") P.minSphericity = std::stod(next());
        else if (a == "--ring-tol") P.ringTol = std::stod(next());
        else if (a == "--min-coverage") P.minCoverage = std::stod(next());
        else if (a == "--csv") P.csvPath = next();
        else if (a == "--debug") P.debugDir = next();
        else if (a == "--verbose") P.verbose = true;
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (!a.empty() && a[0] == '-') { std::cerr << "unknown option " << a << "\n"; usage(); return 1; }
        else stacks.push_back(a);
    }
    if (stacks.empty()) { usage(); return 1; }

    std::ofstream csv;
    if (!P.csvPath.empty()) {
        csv.open(P.csvPath);
        csv << std::fixed << std::setprecision(5);
        csv << "stack,pore_id,z_start,z_end,voxels,volume,eq_diameter,surface_area,sphericity,"
               "axis_a,axis_b,axis_c,aspect_c_over_a,fill_ratio,cx,cy,cz,angle_deg,is_spherical\n";
    }

    int total = 0;
    for (auto& s : stacks) total += processStack(s, P, P.csvPath.empty() ? nullptr : &csv);
    if (stacks.size() > 1) std::cout << "\nTOTAL pores over all stacks: " << total << "\n";
    return 0;
}