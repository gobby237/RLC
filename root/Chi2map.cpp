// ============================================================
//  chi2map.C  –  Mappa del chi² in funzione di due parametri
//  del fit multiparametrico V_L(t) per circuito RLC
//
//  *** COME SCEGLIERE I DUE PARAMETRI ***
//  Modifica PARAM_X e PARAM_Y nella SEZIONE 1:
//    0  →  V_L0    (V)
//    1  →  omega0  (rad/s)
//    2  →  delta   (rad/s)
//    3  →  Voff    (V)
//    4  →  t0      (s)
// ============================================================

#include <TMath.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TH2D.h>
#include <TGraph.h>
#include <TAxis.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <TLine.h>
#include <TPad.h>
#include <TMarker.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TColor.h>
#include <TGaxis.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <fstream>

// ============================================================
//  SEZIONE 1: SCEGLI QUI I DUE PARAMETRI DA VISUALIZZARE
// ============================================================

const int PARAM_X = 1;   // asse X  →  omega0  (rad/s)
const int PARAM_Y = 2;   // asse Y  →  delta   (rad/s)

// Risoluzione della griglia
const int N_GRID = 100;

// Ampiezza della finestra attorno al best-fit (in unità di sigma)
// 2 = zoom stretto, 3 = medio, 5 = largo
const double NSIGMA = 2.0;

// Range colorbar: mostra fino a chi2_min + questo valore
const double CHI2_DISPLAY_DELTA = 12.0;

// ============================================================
//  SEZIONE 2: PARAMETRI DEL FIT
// ============================================================

const double T_FIT_MIN = 0.0;
const double T_FIT_MAX = 46.78e-6;
const double SIGMA_V   = 0.02;

const double VL0_MIN   = 0.5,    VL0_MAX   = 5.0;
const double OM0_MIN   = 6.0e5,  OM0_MAX   = 1.1e6;
const double DELTA_MIN = 1.0e4,  DELTA_MAX = 2.0e5;
const double VOFF_MIN  = -0.3,   VOFF_MAX  =  0.3;
const double T0_MIN    = -2.0e-6, T0_MAX   =  2.0e-6;

const int    N_OM0   = 5;
const int    N_DELTA = 4;
const double VL0_INIT  = 2.3;
const double VOFF_INIT = 0.0;
const double T0_INIT   = 0.0;

// ============================================================
//  SEZIONE 3: FUNZIONE DI FIT
// ============================================================

double fitFunc_map(double *x, double *par) {
    const double t      = x[0];
    const double VL0    = par[0];
    const double omega0 = par[1];
    const double delta  = par[2];
    const double Voff   = par[3];
    const double t0     = par[4];
    const double dt     = t - t0;
    const double od2    = omega0*omega0 - delta*delta;
    if (od2 <= 0.0) return Voff;
    const double omega_d = TMath::Sqrt(od2);
    return VL0 * TMath::Exp(-delta * dt)
               * (TMath::Cos(omega_d * dt) - (delta / omega_d) * TMath::Sin(omega_d * dt))
           + Voff;
}

// ============================================================
//  SEZIONE 4: CALCOLO CHI²
// ============================================================

double computeChi2(const std::vector<double>& t_s,
                   const std::vector<double>& V_V,
                   double sigma, double pars[5]) {
    double chi2 = 0.0;
    const int N = (int)t_s.size();
    for (int i = 0; i < N; i++) {
        double xi = t_s[i];
        double r  = (V_V[i] - fitFunc_map(&xi, pars)) / sigma;
        chi2 += r * r;
    }
    return chi2;
}

// ============================================================
//  SEZIONE 5: PALETTE TENUE (giallo → arancio → viola)
// ============================================================

void setMutedPalette() {
    const int NRGBs = 5;
    double stops[NRGBs] = { 0.00, 0.25, 0.50, 0.75, 1.00 };
    double red[NRGBs]   = { 0.98, 0.94, 0.82, 0.55, 0.24 };
    double green[NRGBs] = { 0.95, 0.78, 0.55, 0.25, 0.03 };
    double blue[NRGBs]  = { 0.72, 0.44, 0.38, 0.50, 0.49 };
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, 256);
    gStyle->SetNumberContours(256);
}

// ============================================================
//  SEZIONE 6: FUNZIONE PRINCIPALE
// ============================================================

void Chi2map() {

    setMutedPalette();
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetPadGridX(kFALSE);
    gStyle->SetPadGridY(kFALSE);
    gStyle->SetFrameBorderMode(0);
    gStyle->SetCanvasBorderMode(0);
    // Forza notazione scientifica sugli assi con offset (×10^n visibile)
    TGaxis::SetMaxDigits(3);

    // ----------------------------------------------------------
    // 6.1  Lettura dati
    // ----------------------------------------------------------
    std::vector<double> t_raw, V_raw;
    std::ifstream infile("dati.txt");
    if (!infile.is_open()) { std::cerr << "ERRORE: impossibile aprire dati.txt\n"; return; }
    double t_val, v_val;
    while (infile >> t_val >> v_val) { t_raw.push_back(t_val); V_raw.push_back(v_val); }
    infile.close();

    const int N = (int)t_raw.size();
    if (N == 0) { std::cerr << "ERRORE: nessun dato letto.\n"; return; }

    std::vector<double> t_s(N), V_V(N);
    for (int i = 0; i < N; i++) {
        t_s[i] = t_raw[i] * 1e-12;
        V_V[i] = V_raw[i] * 1e-6;
    }

    // ----------------------------------------------------------
    // 6.2  Best-fit multi-start
    // ----------------------------------------------------------
    TGraphErrors *gr = new TGraphErrors(N, t_s.data(), V_V.data(), nullptr, nullptr);
    for (int i = 0; i < N; i++) gr->SetPointError(i, 0, SIGMA_V);

    TF1 *f = new TF1("f_vL", fitFunc_map, T_FIT_MIN, T_FIT_MAX, 5);
    f->SetParNames("V_{L0}", "#omega_{0}", "#delta", "V_{off}", "t_{0}");
    f->SetParLimits(0, VL0_MIN, VL0_MAX);
    f->SetParLimits(1, OM0_MIN, OM0_MAX);
    f->SetParLimits(2, DELTA_MIN, DELTA_MAX);
    f->SetParLimits(3, VOFF_MIN, VOFF_MAX);
    f->SetParLimits(4, T0_MIN, T0_MAX);

    double best_chi2 = std::numeric_limits<double>::max();
    double best_pars[5], best_errs[5];
    int best_ndf = 0;

    gErrorIgnoreLevel = kWarning;
    for (int i = 0; i < N_OM0; i++) {
        double om0_try = OM0_MIN + (OM0_MAX - OM0_MIN) * (i + 0.5) / N_OM0;
        for (int j = 0; j < N_DELTA; j++) {
            double delta_try = DELTA_MIN + (DELTA_MAX - DELTA_MIN) * (j + 0.5) / N_DELTA;
            if (om0_try <= delta_try) continue;
            f->SetParameters(VL0_INIT, om0_try, delta_try, VOFF_INIT, T0_INIT);
            auto res = gr->Fit(f, "SRQNE", "", T_FIT_MIN, T_FIT_MAX);
            bool converged = (res.Get() && res->IsValid() && res->Status() == 0);
            double chi2_try = f->GetChisquare();
            if (converged && chi2_try < best_chi2) {
                best_chi2 = chi2_try;
                best_ndf  = f->GetNDF();
                for (int k = 0; k < 5; k++) {
                    best_pars[k] = f->GetParameter(k);
                    best_errs[k] = f->GetParError(k);
                }
            }
        }
    }
    gErrorIgnoreLevel = kPrint;

    if (best_chi2 >= std::numeric_limits<double>::max()) {
        std::cerr << "ERRORE: nessun fit convergito.\n"; return;
    }

    f->SetParameters(best_pars[0], best_pars[1], best_pars[2], best_pars[3], best_pars[4]);
    gr->Fit(f, "SRE", "", T_FIT_MIN, T_FIT_MAX);
    for (int k = 0; k < 5; k++) {
        best_pars[k] = f->GetParameter(k);
        best_errs[k] = f->GetParError(k);
    }
    best_chi2 = f->GetChisquare();
    best_ndf  = f->GetNDF();

    // ----------------------------------------------------------
    // 6.3  Nomi parametri
    // ----------------------------------------------------------
    const char* par_names[5]       = {"V_{L0}", "#omega_{0}", "#delta", "V_{off}", "t_{0}"};
    const char* par_units[5]       = {"V", "rad/s", "rad/s", "V", "s"};
    const char* par_names_plain[5] = {"VL0", "omega0", "delta", "Voff", "t0"};

    // ----------------------------------------------------------
    // 6.4  Range griglia
    // ----------------------------------------------------------
    const double cx = best_pars[PARAM_X];
    const double cy = best_pars[PARAM_Y];
    const double sx = best_errs[PARAM_X];
    const double sy = best_errs[PARAM_Y];

    const double xlo = cx - NSIGMA * sx;
    const double xhi = cx + NSIGMA * sx;
    const double ylo = cy - NSIGMA * sy;
    const double yhi = cy + NSIGMA * sy;

    printf("\n========================================\n");
    printf("  MAPPA CHI2 — %s (X)  vs  %s (Y)\n",
           par_names_plain[PARAM_X], par_names_plain[PARAM_Y]);
    printf("========================================\n");
    printf("  Best-fit  %s = %.6g  +-  %.3g  %s\n", par_names_plain[PARAM_X], cx, sx, par_units[PARAM_X]);
    printf("  Best-fit  %s = %.6g  +-  %.3g  %s\n", par_names_plain[PARAM_Y], cy, sy, par_units[PARAM_Y]);
    printf("  chi2_min  = %.2f / %d = %.3f\n", best_chi2, best_ndf, best_chi2/best_ndf);
    printf("  Griglia:  %d x %d  (finestra +/- %.0f sigma)\n\n", N_GRID, N_GRID, NSIGMA);

    // ----------------------------------------------------------
    // 6.5  Calcolo mappa chi²
    // ----------------------------------------------------------
    TH2D *h = new TH2D("h_chi2", "", N_GRID, xlo, xhi, N_GRID, ylo, yhi);

    double chi2_map_min =  std::numeric_limits<double>::max();
    double chi2_map_max = -std::numeric_limits<double>::max();
    double pars_scan[5];
    for (int k = 0; k < 5; k++) pars_scan[k] = best_pars[k];

    for (int ix = 1; ix <= N_GRID; ix++) {
        pars_scan[PARAM_X] = h->GetXaxis()->GetBinCenter(ix);
        for (int iy = 1; iy <= N_GRID; iy++) {
            pars_scan[PARAM_Y] = h->GetYaxis()->GetBinCenter(iy);
            double v = computeChi2(t_s, V_V, SIGMA_V, pars_scan);
            h->SetBinContent(ix, iy, v);
            chi2_map_min = std::min(chi2_map_min, v);
            chi2_map_max = std::max(chi2_map_max, v);
        }
    }

    const double chi2_display_max = chi2_map_min + CHI2_DISPLAY_DELTA;
    h->SetMinimum(chi2_map_min);
    h->SetMaximum(chi2_display_max);

    printf("  chi2 mappa: min=%.2f  max=%.2f\n\n", chi2_map_min, chi2_map_max);

    // ----------------------------------------------------------
    // 6.6  Contorno chi²_min + 1  (bisection lungo 1440 raggi)
    // ----------------------------------------------------------
    const double chi2_lev = chi2_map_min + 1.0;
    std::vector<double> xc, yc;
    const int N_ANGLE = 1440;

    for (int ia = 0; ia < N_ANGLE; ia++) {
        double angle  = 2.0 * TMath::Pi() * ia / N_ANGLE;
        double dxd    = TMath::Cos(angle);
        double dyd    = TMath::Sin(angle);
        double r_lo   = 0.0;
        double r_hi   = NSIGMA * TMath::Sqrt(sx*sx*dxd*dxd + sy*sy*dyd*dyd) * 1.5;

        pars_scan[PARAM_X] = cx + r_hi * dxd;
        pars_scan[PARAM_Y] = cy + r_hi * dyd;
        if (computeChi2(t_s, V_V, SIGMA_V, pars_scan) < chi2_lev) continue;

        for (int iter = 0; iter < 60; iter++) {
            double r_mid = 0.5 * (r_lo + r_hi);
            pars_scan[PARAM_X] = cx + r_mid * dxd;
            pars_scan[PARAM_Y] = cy + r_mid * dyd;
            if (computeChi2(t_s, V_V, SIGMA_V, pars_scan) < chi2_lev) r_lo = r_mid;
            else                                                        r_hi = r_mid;
            if ((r_hi - r_lo) < 1e-12 * (TMath::Abs(cx) + TMath::Abs(cy) + 1e-30)) break;
        }
        double rb = 0.5 * (r_lo + r_hi);
        xc.push_back(cx + rb * dxd);
        yc.push_back(cy + rb * dyd);
    }
    if (!xc.empty()) { xc.push_back(xc[0]); yc.push_back(yc[0]); }

    double xc_min = *std::min_element(xc.begin(), xc.end());
    double xc_max = *std::max_element(xc.begin(), xc.end());
    double yc_min = *std::min_element(yc.begin(), yc.end());
    double yc_max = *std::max_element(yc.begin(), yc.end());

    printf("  Contorno chi2_min+1:\n");
    printf("    %s: [%.6g, %.6g]  sigma=%.3g %s\n",
           par_names_plain[PARAM_X], xc_min, xc_max, 0.5*(xc_max-xc_min), par_units[PARAM_X]);
    printf("    %s: [%.6g, %.6g]  sigma=%.3g %s\n\n",
           par_names_plain[PARAM_Y], yc_min, yc_max, 0.5*(yc_max-yc_min), par_units[PARAM_Y]);

    TGraph *gr_contour = new TGraph((int)xc.size(), xc.data(), yc.data());
    gr_contour->SetLineColor(kBlack);
    gr_contour->SetLineWidth(3);
    gr_contour->SetLineStyle(2);

    // ----------------------------------------------------------
    // 6.7  Calcolo proiezioni
    //      (PRIMA del canvas, cosi' i range sono noti per allineare gli assi)
    // ----------------------------------------------------------

    // --- Proiezione X: chi2(x, cy) per x in [xlo, xhi] ---
    const int NX = N_GRID * 5;
    std::vector<double> xvec(NX), chi2x(NX);
    for (int k = 0; k < NX; k++) {
        xvec[k] = xlo + (xhi - xlo) * k / (NX - 1);
        for (int p = 0; p < 5; p++) pars_scan[p] = best_pars[p];
        pars_scan[PARAM_X] = xvec[k];
        pars_scan[PARAM_Y] = cy;
        chi2x[k] = computeChi2(t_s, V_V, SIGMA_V, pars_scan);
    }
    double chi2x_min = *std::min_element(chi2x.begin(), chi2x.end());
    double chi2x_max = *std::max_element(chi2x.begin(), chi2x.end());
    // Range Y di projX: deve contenere chi2_min+1 e avere margine
    double yX_lo = chi2x_min - 0.12 * (chi2x_max - chi2x_min);
    double yX_hi = chi2x_max + 0.25 * (chi2x_max - chi2x_min);
    // Assicura che il livello chi2_min+1 stia dentro
    yX_hi = TMath::Max(yX_hi, chi2_map_min + 1.0 + 0.15*(chi2x_max - chi2x_min));

    // --- Proiezione Y: chi2(cx, y) per y in [ylo, yhi] ---
    const int NY = N_GRID * 5;
    std::vector<double> yvec(NY), chi2y(NY);
    for (int k = 0; k < NY; k++) {
        yvec[k] = ylo + (yhi - ylo) * k / (NY - 1);
        for (int p = 0; p < 5; p++) pars_scan[p] = best_pars[p];
        pars_scan[PARAM_X] = cx;
        pars_scan[PARAM_Y] = yvec[k];
        chi2y[k] = computeChi2(t_s, V_V, SIGMA_V, pars_scan);
    }
    double chi2y_min = *std::min_element(chi2y.begin(), chi2y.end());
    double chi2y_max = *std::max_element(chi2y.begin(), chi2y.end());
    // Range X di projY: deve contenere chi2_min+1 e avere margine
    double xY_lo = chi2y_min - 0.12 * (chi2y_max - chi2y_min);
    double xY_hi = chi2y_max + 0.25 * (chi2y_max - chi2y_min);
    xY_hi = TMath::Max(xY_hi, chi2_map_min + 1.0 + 0.15*(chi2y_max - chi2y_min));

    // ----------------------------------------------------------
    // 6.8  Canvas — layout con 4 pad
    //      pad_map   : mappa 2D (centro)
    //      pad_projX : proiezione X (sotto la mappa, stessa larghezza)
    //      pad_projY : proiezione Y (a sinistra della mappa, stessa altezza)
    //      pad_cbar  : colorbar (a destra, un po' piu' bassa della mappa)
    // ----------------------------------------------------------
    TCanvas *c = new TCanvas("c_chi2map",
        Form("chi2 map: %s vs %s", par_names_plain[PARAM_X], par_names_plain[PARAM_Y]),
        1150, 980);

    // Frazioni NDC
    const double mL    = 0.03;
    const double mR    = 0.02;
    const double mT    = 0.08;
    const double mB    = 0.04;
    const double lfrac = 0.22;
    const double bfrac = 0.25;
    const double cfrac = 0.08;   // larghezza colorbar (ridotta)
    // La colorbar e' piu' bassa della mappa: parte da map_y1 + 10% e finisce a map_y2 - 10%
    const double cbar_vmargin = 0.07;

    double map_x1   = mL + lfrac,        map_x2   = 1.0 - mR - cfrac - 0.01;
    double map_y1   = mB + bfrac,        map_y2   = 1.0 - mT;
    double projX_x1 = map_x1,            projX_x2 = map_x2;
    double projX_y1 = mB,                projX_y2 = mB + bfrac;
    double projY_x1 = mL,                projY_x2 = mL + lfrac;
    double projY_y1 = map_y1,            projY_y2 = map_y2;
    double cbar_x1  = map_x2 + 0.01,    cbar_x2  = map_x2 + 0.01 + cfrac;
    // Colorbar piu' bassa della mappa
    double cbar_y1  = map_y1 + cbar_vmargin * (map_y2 - map_y1);
    double cbar_y2  = map_y2 - cbar_vmargin * (map_y2 - map_y1);

    TPad *pad_map   = new TPad("pad_map",   "", map_x1,   map_y1,   map_x2,   map_y2);
    TPad *pad_projX = new TPad("pad_projX", "", projX_x1, projX_y1, projX_x2, projX_y2);
    TPad *pad_projY = new TPad("pad_projY", "", projY_x1, projY_y1, projY_x2, projY_y2);
    TPad *pad_cbar  = new TPad("pad_cbar",  "", cbar_x1,  cbar_y1,  cbar_x2,  cbar_y2);

    pad_map->SetLeftMargin(0.01);   pad_map->SetRightMargin(0.01);
    pad_map->SetTopMargin(0.10);    pad_map->SetBottomMargin(0.01);

    pad_projX->SetLeftMargin(0.01); pad_projX->SetRightMargin(0.01);
    pad_projX->SetTopMargin(0.01);  pad_projX->SetBottomMargin(0.38);

    pad_projY->SetLeftMargin(0.5); pad_projY->SetRightMargin(0.01);
    pad_projY->SetTopMargin(0.10);  pad_projY->SetBottomMargin(0.01);

    pad_cbar->SetLeftMargin(0.35);  pad_cbar->SetRightMargin(0.08);
    pad_cbar->SetTopMargin(0.02);   pad_cbar->SetBottomMargin(0.02);

    pad_map->Draw();
    pad_projX->Draw();
    pad_projY->Draw();
    pad_cbar->Draw();

    // Stile linee tratteggiate
    const int dash_col = kGray + 1;
    const int dash_sty = 2;
    const int dash_wid = 2;

    // Font uniforme per le etichette dei due parametri
    // (calibrato rispetto alle dimensioni fisiche dei pad)
    // projX: pad alto bfrac*(980) ~ 245 px, testo ad asse X
    // projY: pad largo lfrac*(1150) ~ 253 px, testo ad asse Y
    // → usiamo la stessa dimensione fisica, convertita in NDC per pad
    const double PARAM_TITLE_SIZE_X  = 0.130;   // projX: TitleSize asse X
    const double PARAM_LABEL_SIZE_X  = 0.105;   // projX: LabelSize asse X
    const double PARAM_TITLE_SIZE_Y  = 0.130;   // projY: TitleSize asse Y (pad piu' largo)
    const double PARAM_LABEL_SIZE_Y  = 0.105;   // projY: LabelSize asse Y

    // ============================================================
    //  MAPPA 2D
    // ============================================================
    pad_map->cd();

    h->GetXaxis()->SetLabelSize(0.0);
    h->GetYaxis()->SetLabelSize(0.0);
    h->GetXaxis()->SetTitleSize(0.0);
    h->GetYaxis()->SetTitleSize(0.0);
    h->SetContour(256);
    h->Draw("COL");

    gr_contour->Draw("L SAME");

    auto mkLine = [&](double x1, double y1, double x2, double y2) -> TLine* {
        TLine *l = new TLine(x1, y1, x2, y2);
        l->SetLineStyle(dash_sty);
        l->SetLineColor(dash_col);
        l->SetLineWidth(dash_wid);
        return l;
    };

    // Tangenti al contorno: linee complete da bordo a bordo
    mkLine(xc_min, ylo, xc_min, yhi)->Draw("SAME");
    mkLine(xc_max, ylo, xc_max, yhi)->Draw("SAME");
    mkLine(xlo, yc_min, xhi, yc_min)->Draw("SAME");
    mkLine(xlo, yc_max, xhi, yc_max)->Draw("SAME");

    // Linee centrali (minimo)
    TLine *lcx = new TLine(cx, ylo, cx, yhi);
    TLine *lcy = new TLine(xlo, cy, xhi, cy);
    lcx->SetLineStyle(3); lcx->SetLineColor(kGray); lcx->SetLineWidth(1); lcx->Draw("SAME");
    lcy->SetLineStyle(3); lcy->SetLineColor(kGray); lcy->SetLineWidth(1); lcy->Draw("SAME");

    TMarker *mmin = new TMarker(cx, cy, 5);
    mmin->SetMarkerColor(kBlack); mmin->SetMarkerSize(2.2); mmin->Draw("SAME");

    TLatex ltex;
    ltex.SetTextAlign(22); ltex.SetTextSize(0.045);
    ltex.SetTextColor(dash_col); ltex.SetTextFont(42);
    ltex.DrawLatex(cx, cy - 0.40*sy, Form("%.0f", chi2_map_min));

    double xl = cx + (xc_max - cx) * 1.10;
    double yl = cy + (yc_max - cy) * 1.10;
    if (xl > xhi - 0.05*(xhi-xlo)) xl = xhi - 0.08*(xhi-xlo);
    if (yl > yhi - 0.05*(yhi-ylo)) yl = yhi - 0.08*(yhi-ylo);
    ltex.SetTextSize(0.042); ltex.SetTextAlign(12);
    ltex.DrawLatex(xl, yl, Form("%.0f", chi2_map_min + 1.0));

    TLatex title_map;
    title_map.SetNDC(); title_map.SetTextAlign(22);
    title_map.SetTextSize(0.085); title_map.SetTextFont(62);
    title_map.DrawLatex(0.50, 0.955,
        Form("#chi^{2}(%s, %s)", par_names[PARAM_X], par_names[PARAM_Y]));

    pad_map->Update();

    // ============================================================
    //  PROIEZIONE X (sotto la mappa)
    //  Le linee verticali xc_min, xc_max ATTRAVERSANO il pad
    //  dall'alto in basso (yX_lo → yX_hi)
    // ============================================================
    pad_projX->cd();

    TGraph *gr_projX = new TGraph(NX, xvec.data(), chi2x.data());
    gr_projX->SetLineColor(kBlue+1);
    gr_projX->SetLineWidth(3);
    gr_projX->SetTitle("");
    gr_projX->GetXaxis()->SetLimits(xlo, xhi);
    gr_projX->GetYaxis()->SetRangeUser(yX_lo, yX_hi);
    gr_projX->GetXaxis()->SetTitle(Form("%s [%s]", par_names[PARAM_X], par_units[PARAM_X]));
    gr_projX->GetXaxis()->SetTitleSize(PARAM_TITLE_SIZE_X);
    gr_projX->GetXaxis()->SetLabelSize(PARAM_LABEL_SIZE_X);
    gr_projX->GetXaxis()->SetTitleOffset(0.85);
    gr_projX->GetYaxis()->SetTitleSize(0.0);
    gr_projX->GetYaxis()->SetLabelSize(0.0);
    gr_projX->Draw("AL");
    TGaxis::SetExponentOffset(0.5, 0.2, "x");

    // Livello chi2+1 (linea rossa orizzontale)
    TLine *lX_lev = new TLine(xlo, chi2_map_min + 1.0, xhi, chi2_map_min + 1.0);
    lX_lev->SetLineStyle(2); lX_lev->SetLineColor(kRed+1); lX_lev->SetLineWidth(2);
    lX_lev->Draw("SAME");

    // Linee verticali tratteggiate: xc_min e xc_max attraversano TUTTO il range Y
    TLine *lXv_l = new TLine(xc_min, yX_lo, xc_min, yX_hi);
    TLine *lXv_r = new TLine(xc_max, yX_lo, xc_max, yX_hi);
    for (auto l : {lXv_l, lXv_r}) {
        l->SetLineStyle(dash_sty); l->SetLineColor(dash_col);
        l->SetLineWidth(dash_wid); l->Draw("SAME");
    }

    // Etichette ±sigma
    TLatex ltxX;
    ltxX.SetTextSize(0.110); ltxX.SetTextColor(kRed+1); ltxX.SetTextAlign(22);
    double ylab_X = chi2x_min + 0.30*(chi2x_max - chi2x_min);
    ltxX.DrawLatex(xc_min, ylab_X, Form("-%.3g", sx));
    ltxX.DrawLatex(xc_max, ylab_X, Form("+%.3g", sx));

    TLatex ltxX2;
    ltxX2.SetTextSize(0.095); ltxX2.SetTextColor(kGray+2); ltxX2.SetTextAlign(22);
    ltxX2.DrawLatex(cx, chi2x_min + 0.88*(chi2x_max - chi2x_min), Form("%.6g", cx));

    pad_projX->Update();

    // ============================================================
    //  PROIEZIONE Y (a sinistra della mappa)
    //  Le linee orizzontali yc_min, yc_max ATTRAVERSANO il pad
    //  da sinistra a destra (xY_lo → xY_hi)
    // ============================================================
    pad_projY->cd();

    TGraph *gr_projY = new TGraph(NY, chi2y.data(), yvec.data());
    gr_projY->SetLineColor(kBlue+1);
    gr_projY->SetLineWidth(3);
    gr_projY->SetTitle("");
    gr_projY->GetYaxis()->SetLimits(ylo, yhi);
    gr_projY->GetXaxis()->SetRangeUser(xY_lo, xY_hi);
    gr_projY->GetYaxis()->SetTitle(Form("%s [%s]", par_names[PARAM_Y], par_units[PARAM_Y]));
    gr_projY->GetYaxis()->SetTitleSize(PARAM_TITLE_SIZE_Y);
    gr_projY->GetYaxis()->SetLabelSize(PARAM_LABEL_SIZE_Y);
    gr_projY->GetYaxis()->SetTitleOffset(1.60);
    gr_projY->GetXaxis()->SetTitleSize(0.0);
    gr_projY->GetXaxis()->SetLabelSize(0.0);
    gr_projY->Draw("AL");

    // Livello chi2+1
    TLine *lY_lev = new TLine(chi2_map_min + 1.0, ylo, chi2_map_min + 1.0, yhi);
    lY_lev->SetLineStyle(2); lY_lev->SetLineColor(kRed+1); lY_lev->SetLineWidth(2);
    lY_lev->Draw("SAME");

    // Linee orizzontali tratteggiate: yc_min e yc_max attraversano TUTTO il range X
    TLine *lYh_b = new TLine(xY_lo, yc_min, xY_hi, yc_min);
    TLine *lYh_t = new TLine(xY_lo, yc_max, xY_hi, yc_max);
    for (auto l : {lYh_b, lYh_t}) {
        l->SetLineStyle(dash_sty); l->SetLineColor(dash_col);
        l->SetLineWidth(dash_wid); l->Draw("SAME");
    }

    // Etichette ±sigma
    TLatex ltyY;
    ltyY.SetTextSize(0.110);
    ltyY.SetTextColor(kRed+1);
    ltyY.SetTextAlign(12);
    double xlab_Y = chi2y_min + 0.08*(chi2y_max - chi2y_min);
    ltyY.DrawLatex(xlab_Y, yc_min, Form("-%.3g", sy));
    ltyY.DrawLatex(xlab_Y, yc_max, Form("+%.3g", sy));

    TLatex ltyY2;
    ltyY2.SetTextSize(0.095);
    ltyY2.SetTextColor(kGray+2);
    ltyY2.SetTextAlign(12);
    ltyY2.DrawLatex(chi2y_min + 0.50*(chi2y_max - chi2y_min), cy, Form("%.6g", cy));

    pad_projY->Update();

    // ============================================================
    //  COLORBAR — piu' corta della mappa, font grande
    // ============================================================
    pad_cbar->cd();

    TH2D *h_cbar = new TH2D("h_cbar", "", 1, 0, 1, 256,
                             chi2_map_min, chi2_display_max);
    for (int iy = 1; iy <= 256; iy++)
        h_cbar->SetBinContent(1, iy,
            chi2_map_min + (chi2_display_max - chi2_map_min) * (iy - 0.5) / 256.0);
    h_cbar->SetMinimum(chi2_map_min);
    h_cbar->SetMaximum(chi2_display_max);

    h_cbar->GetYaxis()->SetTitle("#chi^{2}");
    h_cbar->GetYaxis()->SetTitleSize(0.145);
    h_cbar->GetYaxis()->SetLabelSize(0.140);
    h_cbar->GetYaxis()->SetTitleOffset(0.42);
    h_cbar->GetXaxis()->SetLabelSize(0.0);
    h_cbar->GetXaxis()->SetTitleSize(0.0);
    h_cbar->Draw("COLZ");

    TLine *lcbar_lev = new TLine(0.0, chi2_map_min + 1.0, 1.0, chi2_map_min + 1.0);
    lcbar_lev->SetLineStyle(2); lcbar_lev->SetLineColor(kBlack); lcbar_lev->SetLineWidth(2);
    lcbar_lev->Draw("SAME");

    pad_cbar->Update();
    c->Update();

    printf("  Mappa completata.\n");
    printf("  Per cambiare parametri: modifica PARAM_X e PARAM_Y in cima al file.\n");
    printf("  0=VL0  1=omega0  2=delta  3=Voff  4=t0\n\n");
}