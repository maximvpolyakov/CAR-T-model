// Source.cpp
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cerrno>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

struct Params {
    double D_C = 0.014;
    double D_T = 5e-5;
    double D_E = 0.001;
    double D_S = 0.0045;
    double D_A = 0.02;
    double D_H = 1.5;

    double rho_C = 0.55;
    double r_T = 0.13;

    double delta_CT = 0.41;
    double gamma = 0.41;

    double d_C = 0.08;
    double k_e0 = 0.05;
    double K_C = 2e9;

    double K_T = 2.4e8;

    double sigma_S = 5e-4;
    double lambda_S = 0.46;

    double sigma_A = 5e-3;
    double lambda_A = 0.75;

    double eta = 5e-10;
    double lambda_H = 0.43;

    double chi_0 = 0.1;
    double p_r = 0.10;

    double d_E = 0.04;
    double d_TA = 0.01;
    double d_TB = 0.01;

    double mu_0 = 0.13;
    double mu_1 = 0.07;

    double nu = 1e-10;
    double xi = 1e-9;

    double alpha_A = 1.0;
    double beta_A = 1.0;
    double beta_H = 1.0;

    double K_H = 0.5;
    double sigma_M = 0.002;
};

struct Grid3D {
    int nx = 40;
    int ny = 40;
    int nz = 40;

    double Lx = 1.0;
    double Ly = 1.0;
    double Lz = 1.0;

    double dx = 0.0;
    double dy = 0.0;
    double dz = 0.0;

    std::size_t size() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    }

    void finalize() {
        if (nx < 2 || ny < 2 || nz < 2) throw std::runtime_error("nx, ny, nz must be >= 2");
        if (Lx <= 0 || Ly <= 0 || Lz <= 0) throw std::runtime_error("Lx, Ly, Lz must be > 0");
        dx = Lx / static_cast<double>(nx - 1);
        dy = Ly / static_cast<double>(ny - 1);
        dz = Lz / static_cast<double>(nz - 1);
    }
};

static inline int clamp_int(int v, int lo, int hi) { return std::max(lo, std::min(hi, v)); }

struct Field {
    Grid3D g;
    std::vector<double> v;

    Field() = default;
    explicit Field(const Grid3D& grid) : g(grid), v(grid.size(), 0.0) {}

    std::size_t idx(int i, int j, int k) const {
        return (static_cast<std::size_t>(k) * static_cast<std::size_t>(g.ny) + static_cast<std::size_t>(j)) *
            static_cast<std::size_t>(g.nx) +
            static_cast<std::size_t>(i);
    }

    double get(int i, int j, int k) const {
        i = clamp_int(i, 0, g.nx - 1);
        j = clamp_int(j, 0, g.ny - 1);
        k = clamp_int(k, 0, g.nz - 1);
        return v[idx(i, j, k)];
    }

    void set(int i, int j, int k, double val) { v[idx(i, j, k)] = val; }

    void fill(double val) { std::fill(v.begin(), v.end(), val); }

    void clamp_nonneg() {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(v.size()); ++i) v[static_cast<std::size_t>(i)] = std::max(0.0, v[static_cast<std::size_t>(i)]);
    }
};

struct State {
    Field C, E, TA, TB, S, A, H;
    explicit State(const Grid3D& g) : C(g), E(g), TA(g), TB(g), S(g), A(g), H(g) {}
    void clamp_nonneg() {
        C.clamp_nonneg();
        E.clamp_nonneg();
        TA.clamp_nonneg();
        TB.clamp_nonneg();
        S.clamp_nonneg();
        A.clamp_nonneg();
        H.clamp_nonneg();
    }
};

struct Deriv {
    Field dC, dE, dTA, dTB, dS, dA, dH;
    explicit Deriv(const Grid3D& g) : dC(g), dE(g), dTA(g), dTB(g), dS(g), dA(g), dH(g) {}
};

static std::map<std::string, std::string> parse_kv_args(int argc, char** argv) {
    std::map<std::string, std::string> kv;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s.rfind("--", 0) == 0) s = s.substr(2);
        auto pos = s.find('=');
        if (pos == std::string::npos) {
            kv[s] = "1";
        }
        else {
            kv[s.substr(0, pos)] = s.substr(pos + 1);
        }
    }
    return kv;
}

template <class T>
static T parse_num(const std::map<std::string, std::string>& kv, const std::string& key, T def) {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    std::istringstream iss(it->second);
    T x{};
    iss >> x;
    if (!iss || !iss.eof()) throw std::runtime_error("Failed to parse arg: " + key + "=" + it->second);
    return x;
}

static std::string parse_str(const std::map<std::string, std::string>& kv, const std::string& key,
    const std::string& def) {
    auto it = kv.find(key);
    if (it == kv.end()) return def;
    return it->second;
}

static bool ensure_dir(const std::string& dir) {
    if (dir.empty() || dir == "." || dir == "./") return true;
#ifdef _WIN32
    int r = _mkdir(dir.c_str());
    if (r == 0) return true;
    return errno == EEXIST;
#else
    int r = mkdir(dir.c_str(), 0755);
    if (r == 0) return true;
    return errno == EEXIST;
#endif
}

static std::string join_path(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir == "." || dir == "./") return name;
#ifdef _WIN32
    char sep = '\\';
#else
    char sep = '/';
#endif
    if (!dir.empty() && (dir.back() == '/' || dir.back() == '\\')) return dir + name;
    return dir + sep + name;
}

static double chi_C(const Params& p, double A) { return p.chi_0 / (1.0 + p.alpha_A * A); }
static double k_e(const Params& p, double A, double H) { return p.k_e0 * (1.0 + p.beta_A * A + p.beta_H * H); }
static double mu_AB(const Params& p, double C) { return p.mu_0 + p.nu * C; }
static double mu_BA(const Params& p, double C) { return p.mu_1 * std::exp(-p.xi * C); }
static double f_H(const Params& p, double H) { return H / (p.K_H + H); }

static Field compute_chemotaxis_divergence(const Grid3D& g, const Params& p, const Field& C, const Field& S,
    const Field& A) {
    Field out(g);
    const double inv_dx = 1.0 / g.dx;
    const double inv_dy = 1.0 / g.dy;
    const double inv_dz = 1.0 / g.dz;

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int k = 0; k < g.nz; ++k) {
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                auto flux_x = [&](int ia, int ib) -> double {
                    const double Sa = S.get(ia, j, k);
                    const double Sb = S.get(ib, j, k);
                    const double Ca = C.get(ia, j, k);
                    const double Cb = C.get(ib, j, k);
                    const double Aa = A.get(ia, j, k);
                    const double Ab = A.get(ib, j, k);
                    const double chi_f = 0.5 * (chi_C(p, Aa) + chi_C(p, Ab));
                    const double C_f = 0.5 * (Ca + Cb);
                    const double dSdx = (Sb - Sa) * inv_dx;
                    return chi_f * C_f * dSdx;
                };

                auto flux_y = [&](int ja, int jb) -> double {
                    const double Sa = S.get(i, ja, k);
                    const double Sb = S.get(i, jb, k);
                    const double Ca = C.get(i, ja, k);
                    const double Cb = C.get(i, jb, k);
                    const double Aa = A.get(i, ja, k);
                    const double Ab = A.get(i, jb, k);
                    const double chi_f = 0.5 * (chi_C(p, Aa) + chi_C(p, Ab));
                    const double C_f = 0.5 * (Ca + Cb);
                    const double dSdy = (Sb - Sa) * inv_dy;
                    return chi_f * C_f * dSdy;
                };

                auto flux_z = [&](int ka, int kb) -> double {
                    const double Sa = S.get(i, j, ka);
                    const double Sb = S.get(i, j, kb);
                    const double Ca = C.get(i, j, ka);
                    const double Cb = C.get(i, j, kb);
                    const double Aa = A.get(i, j, ka);
                    const double Ab = A.get(i, j, kb);
                    const double chi_f = 0.5 * (chi_C(p, Aa) + chi_C(p, Ab));
                    const double C_f = 0.5 * (Ca + Cb);
                    const double dSdz = (Sb - Sa) * inv_dz;
                    return chi_f * C_f * dSdz;
                };

                const double Fx_p = flux_x(i, i + 1);
                const double Fx_m = flux_x(i - 1, i);
                const double Fy_p = flux_y(j, j + 1);
                const double Fy_m = flux_y(j - 1, j);
                const double Fz_p = flux_z(k, k + 1);
                const double Fz_m = flux_z(k - 1, k);

                const double divF = (Fx_p - Fx_m) * inv_dx + (Fy_p - Fy_m) * inv_dy + (Fz_p - Fz_m) * inv_dz;
                out.set(i, j, k, -divF);
            }
        }
    }

    return out;
}

static inline std::size_t idx3(const Grid3D& g, int i, int j, int k) {
    return (static_cast<std::size_t>(k) * static_cast<std::size_t>(g.ny) + static_cast<std::size_t>(j)) *
        static_cast<std::size_t>(g.nx) +
        static_cast<std::size_t>(i);
}

static inline double sample_clamped(const Grid3D& g, const std::vector<double>& buf, int i, int j, int k) {
    i = clamp_int(i, 0, g.nx - 1);
    j = clamp_int(j, 0, g.ny - 1);
    k = clamp_int(k, 0, g.nz - 1);
    return buf[idx3(g, i, j, k)];
}

static void implicit_diffusion_solve(Field& u, const Field& rhs, double D, double dt, int max_iters, double tol) {
    const auto& g = u.g;
    if (D <= 0.0 || dt <= 0.0) {
        u.v = rhs.v;
        return;
    }

    const double ax = dt * D / (g.dx * g.dx);
    const double ay = dt * D / (g.dy * g.dy);
    const double az = dt * D / (g.dz * g.dz);
    const double denom = 1.0 + 2.0 * (ax + ay + az);

    std::vector<double> cur = rhs.v;
    std::vector<double> nxt = cur;

    for (int iter = 0; iter < max_iters; ++iter) {
        double max_delta = 0.0;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) reduction(max : max_delta)
#endif
        for (int k = 0; k < g.nz; ++k) {
            for (int j = 0; j < g.ny; ++j) {
                for (int i = 0; i < g.nx; ++i) {
                    const std::size_t id = idx3(g, i, j, k);

                    const double ue = sample_clamped(g, cur, i + 1, j, k);
                    const double uw = sample_clamped(g, cur, i - 1, j, k);
                    const double un = sample_clamped(g, cur, i, j + 1, k);
                    const double us = sample_clamped(g, cur, i, j - 1, k);
                    const double ut = sample_clamped(g, cur, i, j, k + 1);
                    const double ub = sample_clamped(g, cur, i, j, k - 1);

                    const double b = rhs.v[id];
                    const double new_u = (b + ax * (ue + uw) + ay * (un + us) + az * (ut + ub)) / denom;
                    nxt[id] = new_u;

                    const double delta = std::abs(new_u - cur[id]);
                    if (delta > max_delta) max_delta = delta;
                }
            }
        }

        cur.swap(nxt);
        if (max_delta < tol) break;
    }

    u.v.swap(cur);
}

static void init_state(State& s, double tumor_radius, double tumor_value, double cart_layer, double cart_radius,
    double cart_value) {
    const auto& g = s.C.g;
    const double cx = 0.5 * g.Lx;
    const double cy = 0.5 * g.Ly;
    const double cz = 0.5 * g.Lz;

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int k = 0; k < g.nz; ++k) {
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                const double x = i * g.dx;
                const double y = j * g.dy;
                const double z = k * g.dz;

                const double dx = x - cx;
                const double dy = y - cy;
                const double dz = z - cz;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

                s.TA.set(i, j, k, (r <= tumor_radius) ? tumor_value : 0.0);
                s.TB.set(i, j, k, 0.0);
                s.S.set(i, j, k, 0.0);
                s.A.set(i, j, k, 0.0);
                s.H.set(i, j, k, 0.0);
                s.E.set(i, j, k, 0.0);

                const bool in_layer = (x <= cart_layer);
                const double ryz = std::sqrt((y - cy) * (y - cy) + (z - cz) * (z - cz));
                const bool in_disk = (ryz <= cart_radius);
                s.C.set(i, j, k, (in_layer && in_disk) ? cart_value : 0.0);
            }
        }
    }
}

static void compute_explicit_rhs(const State& s, const Params& p, Deriv& out) {
    const auto& g = s.C.g;
    Field chemo = compute_chemotaxis_divergence(g, p, s.C, s.S, s.A);

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int k = 0; k < g.nz; ++k) {
        for (int j = 0; j < g.ny; ++j) {
            for (int i = 0; i < g.nx; ++i) {
                const double C = s.C.get(i, j, k);
                const double E = s.E.get(i, j, k);
                const double TA = s.TA.get(i, j, k);
                const double TB = s.TB.get(i, j, k);
                const double S = s.S.get(i, j, k);
                const double A = s.A.get(i, j, k);
                const double H = s.H.get(i, j, k);

                const double ke = k_e(p, A, H);
                const double muab = mu_AB(p, C);
                const double muba = mu_BA(p, C);
                const double fH = f_H(p, H);

                const double pr_eff = p.p_r / (1.0 + A);

                const double growth_C = p.rho_C * (C * TA) / (p.K_C + C + 1e-30);
                const double kill_C = p.delta_CT * C * TA;

                const double dCdt = growth_C - kill_C - ke * C + pr_eff * E - p.d_C * C + chemo.get(i, j, k);
                const double dEdt = ke * C - pr_eff * E - p.d_E * E;

                const double N = TA + TB;
                const double logistic = p.r_T * (1.0 - N / p.K_T);

                const double dTAdt = logistic * TA - p.gamma * C * TA - muab * TA + muba * TB - p.d_TA * TA;
                const double dTBdt = logistic * TB + muab * TA - muba * TB - p.d_TB * TB;

                const double dSdt = p.sigma_S * TA - p.lambda_S * S;
                const double dAdt = p.sigma_A * (TA + TB) + p.sigma_M * fH - p.lambda_A * A;
                const double dHdt = p.eta * (TA + TB) - p.lambda_H * H;

                out.dC.set(i, j, k, dCdt);
                out.dE.set(i, j, k, dEdt);
                out.dTA.set(i, j, k, dTAdt);
                out.dTB.set(i, j, k, dTBdt);
                out.dS.set(i, j, k, dSdt);
                out.dA.set(i, j, k, dAdt);
                out.dH.set(i, j, k, dHdt);

                (void)S;
            }
        }
    }
}

static void state_add_scaled(State& s, const Deriv& k, double a) {
    const std::size_t n = s.C.v.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const std::size_t id = static_cast<std::size_t>(i);
        s.C.v[id] += a * k.dC.v[id];
        s.E.v[id] += a * k.dE.v[id];
        s.TA.v[id] += a * k.dTA.v[id];
        s.TB.v[id] += a * k.dTB.v[id];
        s.S.v[id] += a * k.dS.v[id];
        s.A.v[id] += a * k.dA.v[id];
        s.H.v[id] += a * k.dH.v[id];
    }
}

static State state_copy_add_scaled(const State& base, const Deriv& k, double a) {
    State out(base.C.g);
    out.C.v = base.C.v;
    out.E.v = base.E.v;
    out.TA.v = base.TA.v;
    out.TB.v = base.TB.v;
    out.S.v = base.S.v;
    out.A.v = base.A.v;
    out.H.v = base.H.v;
    state_add_scaled(out, k, a);
    return out;
}

static void explicit_rk2(State& s, const Params& p, double dt, int substeps) {
    if (substeps < 1) substeps = 1;
    const double h = dt / static_cast<double>(substeps);
    const auto& g = s.C.g;

    Deriv k1(g);
    Deriv k2(g);

    for (int m = 0; m < substeps; ++m) {
        compute_explicit_rhs(s, p, k1);
        State tmp = state_copy_add_scaled(s, k1, h);
        tmp.clamp_nonneg();
        compute_explicit_rhs(tmp, p, k2);

        state_add_scaled(s, k1, 0.5 * h);
        state_add_scaled(s, k2, 0.5 * h);
        s.clamp_nonneg();
    }
}

static void diffusion_implicit(State& s, const Params& p, double dt, int diff_iters, double diff_tol) {
    const auto& g = s.C.g;

    Field rhsC(g), rhsE(g), rhsTA(g), rhsTB(g), rhsS(g), rhsA(g), rhsH(g);
    rhsC.v = s.C.v;
    rhsE.v = s.E.v;
    rhsTA.v = s.TA.v;
    rhsTB.v = s.TB.v;
    rhsS.v = s.S.v;
    rhsA.v = s.A.v;
    rhsH.v = s.H.v;

    implicit_diffusion_solve(s.C, rhsC, p.D_C, dt, diff_iters, diff_tol);
    implicit_diffusion_solve(s.E, rhsE, p.D_E, dt, diff_iters, diff_tol);
    implicit_diffusion_solve(s.TA, rhsTA, p.D_T, dt, diff_iters, diff_tol);
    implicit_diffusion_solve(s.TB, rhsTB, p.D_T, dt, diff_iters, diff_tol);
    implicit_diffusion_solve(s.S, rhsS, p.D_S, dt, diff_iters, diff_tol);
    implicit_diffusion_solve(s.A, rhsA, p.D_A, dt, diff_iters, diff_tol);
    implicit_diffusion_solve(s.H, rhsH, p.D_H, dt, diff_iters, diff_tol);

    s.clamp_nonneg();
}

static void step_strang(State& s, const Params& p, double dt, int exp_substeps, int diff_iters, double diff_tol) {
    explicit_rk2(s, p, 0.5 * dt, exp_substeps);
    diffusion_implicit(s, p, dt, diff_iters, diff_tol);
    explicit_rk2(s, p, 0.5 * dt, exp_substeps);
}

static void write_csv(const std::string& path, const State& s, double t, int precision) {
    const auto& g = s.C.g;
    std::ofstream out(path.c_str(), std::ios::out | std::ios::trunc);
    if (!out) throw std::runtime_error("Failed to open output file: " + path);

    out << std::setprecision(precision);
    out << "t,i,j,k,x,y,z,C,E,TA,TB,S,A,H\n";

    for (int k = 0; k < g.nz; ++k) {
        const double z = k * g.dz;
        for (int j = 0; j < g.ny; ++j) {
            const double y = j * g.dy;
            for (int i = 0; i < g.nx; ++i) {
                const double x = i * g.dx;
                out << t << ',' << i << ',' << j << ',' << k << ',';
                out << x << ',' << y << ',' << z << ',';
                out << s.C.get(i, j, k) << ',' << s.E.get(i, j, k) << ',' << s.TA.get(i, j, k) << ','
                    << s.TB.get(i, j, k) << ',' << s.S.get(i, j, k) << ',' << s.A.get(i, j, k) << ','
                    << s.H.get(i, j, k) << '\n';
            }
        }
    }
}

static void print_usage() {
    std::cout
        << "Usage: sim [--key=value ...]\n"
        << "Threads: --threads=0 (0=default OpenMP)\n"
        << "Grid: --nx=40 --ny=40 --nz=40 --Lx=1 --Ly=1 --Lz=1\n"
        << "Time: --dt=0.01 --T=30 --output_every=1\n"
        << "Scheme: --exp_substeps=1 --diff_iters=60 --diff_tol=1e-8\n"
        << "Init: --tumor_radius=0.2 --tumor_value=2.4e7 --cart_layer=0.05 --cart_radius=0.2 --cart_value=1e7\n"
        << "Output: --out_dir=output --csv_precision=16\n";
}

int main(int argc, char** argv) {
    try {
        const auto kv = parse_kv_args(argc, argv);
        if (kv.count("help") || kv.count("h")) {
            print_usage();
            return 0;
        }

        int threads = parse_num<int>(kv, "threads", 0);
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#endif

        Grid3D g;
        g.nx = parse_num<int>(kv, "nx", g.nx);
        g.ny = parse_num<int>(kv, "ny", g.ny);
        g.nz = parse_num<int>(kv, "nz", g.nz);
        g.Lx = parse_num<double>(kv, "Lx", g.Lx);
        g.Ly = parse_num<double>(kv, "Ly", g.Ly);
        g.Lz = parse_num<double>(kv, "Lz", g.Lz);
        g.finalize();

        double dt = parse_num<double>(kv, "dt", 0.01);
        double T = parse_num<double>(kv, "T", 30.0);
        int output_every = parse_num<int>(kv, "output_every", 1);

        int exp_substeps = parse_num<int>(kv, "exp_substeps", 1);
        int diff_iters = parse_num<int>(kv, "diff_iters", 60);
        double diff_tol = parse_num<double>(kv, "diff_tol", 1e-8);

        int csv_precision = parse_num<int>(kv, "csv_precision", 16);
        std::string out_dir = parse_str(kv, "out_dir", "output");

        if (!(dt > 0.0) || !(T >= 0.0)) throw std::runtime_error("dt must be > 0 and T must be >= 0");
        if (output_every < 1) output_every = 1;

        if (!ensure_dir(out_dir)) throw std::runtime_error("Failed to create output dir: " + out_dir);

        Params p;
        p.gamma = p.delta_CT;
        p.gamma = parse_num<double>(kv, "gamma", p.gamma);

        double tumor_radius = parse_num<double>(kv, "tumor_radius", 0.2);
        double tumor_value = parse_num<double>(kv, "tumor_value", 0.1 * p.K_T);
        double cart_layer = parse_num<double>(kv, "cart_layer", 0.05);
        double cart_radius = parse_num<double>(kv, "cart_radius", 0.2);
        double cart_value = parse_num<double>(kv, "cart_value", 1e7);

        State s(g);
        init_state(s, tumor_radius, tumor_value, cart_layer, cart_radius, cart_value);

        const int steps = static_cast<int>(std::ceil(T / dt));

        auto make_name = [&](int n) -> std::string {
            std::ostringstream oss;
            oss << "step_" << std::setw(6) << std::setfill('0') << n << ".csv";
            return join_path(out_dir, oss.str());
        };

        write_csv(make_name(0), s, 0.0, csv_precision);

        for (int n = 1; n <= steps; ++n) {
            step_strang(s, p, dt, exp_substeps, diff_iters, diff_tol);

            if (n % output_every == 0 || n == steps) {
                const double t = n * dt;
                write_csv(make_name(n), s, t, csv_precision);
                std::cout << "Saved " << make_name(n) << " at t=" << t << "\n";
            }
        }

        std::cout << "Done.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "Use --help=1\n";
        return 1;
    }
}