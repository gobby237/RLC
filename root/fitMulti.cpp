// ============================================================
//  rlc_fit.C  –  Fit multiparametrico V_L(t) per circuito RLC
//  Legge i dati da "dati.txt" (tre colonne: tempo_ps  tensione_uV  errore_uV)
//  Il fit è pesato con gli errori individuali di ogni punto.
// ============================================================

#include <TMath.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TAxis.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <TLine.h>
#include <TPad.h>
#include <TString.h>
#include <TFitResult.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>

// ============================================================
//  SEZIONE 1: PARAMETRI — modifica solo qui
// ============================================================

// --- Range temporale del fit ---
const double T_FIT_MIN = 0.0;       // s
const double T_FIT_MAX = 46.78e-6;  // s

// --- Limiti fisici dei parametri ---
const double VL0_MIN   = 0.5,    VL0_MAX   = 5.0;       // V
const double OM0_MIN   = 6.0e5,  OM0_MAX   = 1.1e6;     // rad/s
const double DELTA_MIN = 1.0e4,  DELTA_MAX = 2.0e5;     // rad/s
const double VOFF_MIN  = -0.3,   VOFF_MAX  =  0.3;      // V
const double T0_MIN    = -2.0e-6, T0_MAX   =  2.0e-6;   // s

// --- Griglia multi-start ---
const int    N_OM0   = 5;
const int    N_DELTA = 4;
const double VL0_INIT  = 2.3;
const double VOFF_INIT = 0.0;
const double T0_INIT   = 0.0;

// --- Box risultati fit (coordinate NDC del pad superiore) ---
const double BOX_x1 = 0.13, BOX_y1 = 0.45, BOX_x2 = 0.55, BOX_y2 = 0.93;
const double BOX_textsize = 0.033;

// --- Scala asse Y dei residui (in unità di sigma, residui normalizzati) ---
const double RES_YMAX = 3.5;   // unità di sigma

// ============================================================
//  SEZIONE 2: FUNZIONE DI FIT
//  par[0]=VL0, par[1]=omega0, par[2]=delta, par[3]=Voff, par[4]=t0
// ============================================================

double fitFunc(double *x, double *par) {
    const double t      = x[0];
    const double VL0    = par[0];
    const double omega0 = par[1];
    const double delta  = par[2];
    const double Voff   = par[3];
    const double t0     = par[4];

    const double dt  = t - t0;
    const double od2 = omega0*omega0 - delta*delta;
    if (od2 <= 0.0) return Voff;

    const double omega_d = TMath::Sqrt(od2);

    return VL0 * TMath::Exp(-delta * dt)
               * (TMath::Cos(omega_d * dt) - (delta / omega_d) * TMath::Sin(omega_d * dt))
           + Voff;
}

// ============================================================
//  SEZIONE 3: FUNZIONI DI FORMATTAZIONE
// ============================================================

double roundToSigFigs(double x, int n = 1) {
    if (x == 0.0) return 0.0;
    const double ax    = TMath::Abs(x);
    const double exp10 = TMath::Floor(TMath::Log10(ax));
    const double scale = TMath::Power(10.0, exp10 - n + 1);
    return TMath::Nint(x / scale) * scale;
}

int decimalsFromError(double err) {
    if (err <= 0.0) return 0;
    const double exp10 = TMath::Floor(TMath::Log10(err));
    return (exp10 < 0) ? (int)(-exp10) : 0;
}

TString fmtFixed(double val, double err, const char* unit) {
    if (err <= 0.0) return Form("%.3g %s", val, unit);
    const double err1 = roundToSigFigs(err, 1);
    const int    dec  = decimalsFromError(err1);
    const double val1 = TMath::Nint(val * TMath::Power(10.0, dec)) / TMath::Power(10.0, dec);
    return Form("%.*f #pm %.*f %s", dec, val1, dec, err1, unit);
}

TString fmtSci(double val, double err, const char* unit) {
    if (val == 0.0 || err <= 0.0) return Form("%.3e %s", val, unit);
    const double aval  = TMath::Abs(val);
    const int    expv  = (int)TMath::Floor(TMath::Log10(aval));
    const double scale = TMath::Power(10.0, expv);
    const double vm    = val / scale;
    const double em    = err / scale;
    const double em1   = roundToSigFigs(em, 1);
    const int    dec   = decimalsFromError(em1);
    const double vm1   = TMath::Nint(vm * TMath::Power(10.0, dec)) / TMath::Power(10.0, dec);
    return Form("(%.*f #pm %.*f) #times 10^{%d} %s", dec, vm1, dec, em1, expv, unit);
}

// ============================================================
//  SEZIONE 4: FUNZIONE PRINCIPALE
// ============================================================

void fitMulti() {

    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetPadGridX(kFALSE);
    gStyle->SetPadGridY(kFALSE);
    gStyle->SetFrameBorderMode(0);
    gStyle->SetCanvasBorderMode(0);

    // ----------------------------------------------------------
    // 4.1  Lettura dati da file "dati.txt"
    //      Il file deve contenere TRE colonne:
    //        tempo [ps]   tensione [µV]   errore_tensione [µV]
    //
    //      Righe che iniziano con '#' sono trattate come commenti.
    // ----------------------------------------------------------
    std::vector<double> t_raw, V_raw, sV_raw;
    std::ifstream infile("dati.txt");
    if (!infile.is_open()) {
        std::cerr << "ERRORE: impossibile aprire il file dati.txt\n";
        return;
    }

    std::string line;
    int line_num = 0;
    int skipped  = 0;
    while (std::getline(infile, line)) {
        line_num++;
        // Ignora righe vuote e commenti
        if (line.empty() || line[0] == '#') continue;

        std::istringstream iss(line);
        double t_val, v_val, sv_val;
        if (!(iss >> t_val >> v_val >> sv_val)) {
            std::cerr << "ATTENZIONE: riga " << line_num
                      << " non ha tre colonne valide, saltata.\n";
            skipped++;
            continue;
        }
        if (sv_val <= 0.0) {
            std::cerr << "ATTENZIONE: riga " << line_num
                      << " ha errore <= 0 (" << sv_val << " µV), saltata.\n";
            skipped++;
            continue;
        }
        t_raw.push_back(t_val);
        V_raw.push_back(v_val);
        sV_raw.push_back(sv_val);
    }
    infile.close();

    if (t_raw.empty()) {
        std::cerr << "ERRORE: nessun dato valido letto da dati.txt\n";
        return;
    }

    printf("\n  Letti %d punti da dati.txt", (int)t_raw.size());
    if (skipped > 0) printf("  (%d righe saltate)", skipped);
    printf("\n");

    // ----------------------------------------------------------
    // 4.2  Conversione delle unità
    //      t_raw  in ps  →  s     (* 1e-12)
    //      V_raw  in µV  →  V     (* 1e-6)
    //      sV_raw in µV  →  V     (* 1e-6)
    // ----------------------------------------------------------
    const int N = (int)t_raw.size();
    std::vector<double> t_s(N), V_V(N), sV(N);

    for (int i = 0; i < N; i++) {
        t_s[i] = t_raw[i]  * 1e-12;
        V_V[i] = V_raw[i]  * 1e-6;
        sV[i]  = sV_raw[i] * 1e-6;   // errore individuale, in Volt
    }

    // Stampa statistiche degli errori
    double sV_min = *std::min_element(sV.begin(), sV.end());
    double sV_max = *std::max_element(sV.begin(), sV.end());
    double sV_sum = 0; for (auto v : sV) sV_sum += v;
    printf("  Errori sigma_V: min=%.4g V  max=%.4g V  media=%.4g V\n\n",
           sV_min, sV_max, sV_sum / N);

    // ----------------------------------------------------------
    // 4.3  TGraphErrors con errori individuali
    // ----------------------------------------------------------
    TGraphErrors *gr = new TGraphErrors(N, t_s.data(), V_V.data(), nullptr, sV.data());
    gr->SetMarkerStyle(20);
    gr->SetMarkerColor(kBlack);
    gr->SetMarkerSize(0.3);
    gr->SetLineColor(kBlack);

    // ----------------------------------------------------------
    // 4.4  Definizione TF1
    // ----------------------------------------------------------
    TF1 *f = new TF1("f_vL", fitFunc, T_FIT_MIN, T_FIT_MAX, 5);
    f->SetParNames("V_{L0}", "#omega_{0}", "#delta", "V_{off}", "t_{0}");
    f->SetParLimits(0, VL0_MIN,   VL0_MAX);
    f->SetParLimits(1, OM0_MIN,   OM0_MAX);
    f->SetParLimits(2, DELTA_MIN, DELTA_MAX);
    f->SetParLimits(3, VOFF_MIN,  VOFF_MAX);
    f->SetParLimits(4, T0_MIN,    T0_MAX);
    f->SetLineColor(kRed);
    f->SetLineWidth(2);

    // ----------------------------------------------------------
    // 4.5  Fit multi-start sulla griglia (omega0, delta)
    //      Il fit è automaticamente pesato perché gr ha errori
    //      individuali: ROOT minimizza sum[ (V_i - f(t_i))^2 / sV_i^2 ]
    // ----------------------------------------------------------
    double best_chi2 = std::numeric_limits<double>::max();
    std::vector<double> best_pars(5), best_errs(5);
    int best_ndf = 0;

    gErrorIgnoreLevel = kWarning;

    for (int i = 0; i < N_OM0; i++) {
        double om0_try = OM0_MIN + (OM0_MAX - OM0_MIN) * (i + 0.5) / N_OM0;

        for (int j = 0; j < N_DELTA; j++) {
            double delta_try = DELTA_MIN + (DELTA_MAX - DELTA_MIN) * (j + 0.5) / N_DELTA;
            if (om0_try <= delta_try) continue;

            f->SetParameters(VL0_INIT, om0_try, delta_try, VOFF_INIT, T0_INIT);
            f->SetLineStyle(0);

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
        std::cerr << "\nATTENZIONE: nessun fit e' convergito! Prova ad allargare i limiti.\n";
        return;
    }

    // ----------------------------------------------------------
    // 4.6  Fit finale con i migliori parametri trovati
    // ----------------------------------------------------------
    f->SetParameters(best_pars[0], best_pars[1], best_pars[2], best_pars[3], best_pars[4]);
    f->SetLineStyle(1);

    TVirtualFitter::SetDefaultFitter("Minuit");
    auto fitRes = gr->Fit(f, "SRE", "", T_FIT_MIN, T_FIT_MAX);

    // ----------------------------------------------------------
    // 4.7  Estrazione risultati
    // ----------------------------------------------------------
    const double VL0_fit    = f->GetParameter(0);  const double sVL0    = f->GetParError(0);
    const double omega0_fit = f->GetParameter(1);  const double somega0 = f->GetParError(1);
    const double delta_fit  = f->GetParameter(2);  const double sdelta  = f->GetParError(2);
    const double Voff_fit   = f->GetParameter(3);  const double sVoff   = f->GetParError(3);
    const double t0_fit     = f->GetParameter(4);  const double st0     = f->GetParError(4);

    const double chi2   = f->GetChisquare();
    const int    ndf    = f->GetNDF();
    const double prob   = TMath::Prob(chi2, ndf);
    const double chi2_r = chi2 / (double)ndf;

    // Grandezze derivate
    const double od2     = omega0_fit*omega0_fit - delta_fit*delta_fit;
    const double omega_d = (od2 > 0) ? TMath::Sqrt(od2) : 0.0;
    const double T_d     = (omega_d > 0) ? 2.0 * TMath::Pi() / omega_d : 0.0;
    const double Q       = (delta_fit > 0) ? omega0_fit / (2.0 * delta_fit) : 0.0;

    // L e R con C dal METRIX
    const double C  = 4.76e-9;
    const double sC = 0.02e-9;

    const double L = 1.0 / (omega0_fit * omega0_fit * C);
    const double sL_over_L = TMath::Sqrt(TMath::Power(2.0 * somega0 / omega0_fit, 2)
                                       + TMath::Power(sC / C, 2));
    const double sL = L * sL_over_L;

    const double R  = 2.0 * L * delta_fit;
    const double sR = R * TMath::Sqrt(sL_over_L * sL_over_L
                                    + TMath::Power(sdelta / delta_fit, 2));

    double somega_d = 0.0;
    if (omega_d > 0)
        somega_d = TMath::Sqrt(TMath::Power(omega0_fit/omega_d * somega0, 2)
                             + TMath::Power(delta_fit /omega_d * sdelta,  2));

    // ----------------------------------------------------------
    // 4.8  Formattazione stringhe
    // ----------------------------------------------------------
    TString s_VL0    = fmtFixed(VL0_fit,    sVL0,     "V");
    TString s_omega0 = fmtSci  (omega0_fit, somega0,  "rad/s");
    TString s_delta  = fmtSci  (delta_fit,  sdelta,   "rad/s");
    TString s_Voff   = fmtFixed(Voff_fit,   sVoff,    "V");
    TString s_omegad = fmtSci  (omega_d,    somega_d, "rad/s");
    TString s_t0     = fmtSci  (t0_fit,     st0,      "s");
    TString sL_str   = fmtSci  (L,          sL,       "H");
    TString sR_str   = fmtSci  (R,          sR,       "#Omega");

    // ----------------------------------------------------------
    // 4.9  Stampa a terminale
    // ----------------------------------------------------------
    printf("\n========================================\n");
    printf("  RISULTATI FIT MULTIPARAMETRICO V_L(t)\n");
    printf("  (fit pesato con errori individuali)\n");
    printf("========================================\n");
    printf("Tentativi griglia: %d x %d = %d\n", N_OM0, N_DELTA, N_OM0 * N_DELTA);
    printf("chi2 best (griglia) = %.2f\n\n", best_chi2);
    printf("Fit finale:\n");
    printf("  chi2/ndf = %.1f / %d = %.3f\n", chi2, ndf, chi2_r);
    printf("  Prob     = %.6f\n\n", prob);
    printf("  V_L0    = %s\n",  s_VL0.Data());
    printf("  omega0  = %s\n",  s_omega0.Data());
    printf("  delta   = %s\n",  s_delta.Data());
    printf("  Voff    = %s\n",  s_Voff.Data());
    printf("  t0      = %s\n",  s_t0.Data());
    printf("  omega_d = %s\n",  s_omegad.Data());
    printf("  T_d     = %.4e s\n", T_d);
    printf("  Q       = %.3f\n\n", Q);
    printf("  L = %s\n",   sL_str.Data());
    printf("  R = %s\n\n", sR_str.Data());

    // ----------------------------------------------------------
    // 4.10  Residui veri (non normalizzati)  V_i - f(t_i)
    // ----------------------------------------------------------
    std::vector<double> res_raw(N);
    for (int i = 0; i < N; i++)
        res_raw[i] = V_V[i] - f->Eval(t_s[i]);

    // Crea un TGraph semplice (senza barre di errore) per i residui
    TGraph *gr_res = new TGraph(N, t_s.data(), res_raw.data());
    gr_res->SetMarkerStyle(20);
    gr_res->SetMarkerColor(kBlack);
    gr_res->SetMarkerSize(0.3);
    gr_res->SetLineColor(kBlack);
    // ----------------------------------------------------------
    // 4.11  Canvas e pad
    // ----------------------------------------------------------
    TCanvas *c = new TCanvas("c_fit", "RLC - Fit V_{L}(t)", 1000, 750);

    TPad *pad_top = new TPad("pad_top", "", 0.0, 0.30, 1.0, 1.0);
    TPad *pad_bot = new TPad("pad_bot", "", 0.0, 0.00, 1.0, 0.30);

    pad_top->SetBottomMargin(0.02);  pad_top->SetLeftMargin(0.11);
    pad_top->SetRightMargin(0.12);   pad_top->SetTopMargin(0.10);
    pad_bot->SetTopMargin(0.02);     pad_bot->SetBottomMargin(0.35);
    pad_bot->SetLeftMargin(0.11);    pad_bot->SetRightMargin(0.12);

    pad_top->Draw();
    pad_bot->Draw();

    // --- Pad superiore: dati + curva di fit ---
    pad_top->cd();

    gr->SetTitle("Fit V_{L}(t) in scarica");
    gr->GetXaxis()->SetLimits(T_FIT_MIN, T_FIT_MAX);
    gr->GetYaxis()->SetRangeUser(-2.0, 3.0);
    gr->GetXaxis()->SetTitleSize(0.0);
    gr->GetYaxis()->SetTitleSize(0.052);
    gr->GetXaxis()->SetLabelSize(0.0);
    gr->GetYaxis()->SetLabelSize(0.045);
    gr->GetYaxis()->SetTitle("V_{L} [V]");
    gr->GetYaxis()->SetTitleOffset(0.90);
    gr->Draw("AP");
    f->Draw("SAME");

    TLine *lz = new TLine(T_FIT_MIN, 0.0, T_FIT_MAX, 0.0);
    lz->SetLineStyle(2);  lz->SetLineColor(kGray+1);
    lz->Draw("SAME");

    // Box risultati
    TPaveText *pt = new TPaveText(BOX_x1, BOX_y1, BOX_x2, BOX_y2, "NDC");
    pt->SetFillColor(kWhite);  pt->SetBorderSize(1);
    pt->SetTextAlign(12);      pt->SetTextSize(BOX_textsize);
    pt->SetMargin(0.03);
    pt->AddText("V_{L}(t) = V_{L0} e^{-#delta(t-t_{0})}[cos(#omega_{d}(t-t_{0})) - (#delta/#omega_{d})sin(#omega_{d}(t-t_{0}))] + V_{off}");
    pt->AddText(Form("#chi^{2}/ndf = %.1f / %d = %.2f", chi2, ndf, chi2_r));
    pt->AddText(Form("V_{L0} = %s",     s_VL0.Data()));
    pt->AddText(Form("#omega_{0} = %s", s_omega0.Data()));
    pt->AddText(Form("#delta = %s",     s_delta.Data()));
    pt->AddText(Form("V_{off} = %s",    s_Voff.Data()));
    pt->AddText(Form("t_{0} = %s",      s_t0.Data()));
    pt->Draw();

    TLegend *leg = new TLegend(0.80, 0.80, 1.0, 0.91);
    leg->SetBorderSize(1);  leg->SetFillColor(kWhite);  leg->SetTextSize(0.036);
    leg->AddEntry(gr, "Dati sperimentali", "p");
    leg->AddEntry(f,  "Fit pesato",        "l");
    leg->Draw();

    // --- Pad inferiore: residui veri (non normalizzati) ---
    pad_bot->cd();

    // Calcola automaticamente il range Y con un margine del 20%
    double res_min = *std::min_element(res_raw.begin(), res_raw.end());
    double res_max = *std::max_element(res_raw.begin(), res_raw.end());
    double margin = 0.2 * (res_max - res_min);
    if (margin == 0) margin = 0.1;  // caso tutti residui uguali

    gr_res->SetTitle("");
    gr_res->GetXaxis()->SetLimits(T_FIT_MIN, T_FIT_MAX);
    gr_res->GetYaxis()->SetRangeUser(res_min - margin, res_max + margin);
    gr_res->GetXaxis()->SetTitle("Tempi [s]");
    gr_res->GetYaxis()->SetTitle("Residui [V]");
    gr_res->GetXaxis()->SetTitleSize(0.12);
    gr_res->GetYaxis()->SetTitleSize(0.10);
    gr_res->GetXaxis()->SetLabelSize(0.09);
    gr_res->GetYaxis()->SetLabelSize(0.085);
    gr_res->GetXaxis()->SetTitleOffset(0.95);
    gr_res->GetYaxis()->SetTitleOffset(0.50);
    gr_res->GetYaxis()->SetNdivisions(505);
    gr_res->Draw("AP");   // "A" per assi, "P" per punti

    // Linea orizzontale a zero (solo riferimento)
    TLine *lz2 = new TLine(T_FIT_MIN, 0.0, T_FIT_MAX, 0.0);
    lz2->SetLineStyle(2);
    lz2->SetLineColor(kRed);
    lz2->SetLineWidth(1);
    lz2->Draw("SAME");

    c->Update();
}