// ============================================================
//  RLC – Stima del periodo T tramite regressione lineare
//  su 4 dataset: t+, t-, t_max, t_min  vs  n
//
//  Modello: t(n) = T * n + t0
//  Pendenza T = periodo dell'oscillazione
//
//  4 dataset separati (come nella slide del prof):
//    t+   : tempi zeri crescenti   (n = 2, 3, 4, 5)
//    t-   : tempi zeri decrescenti (n = 1.5, 2.5, 3.5, 4.5, 5.5)
//    t_max: tempi massimi = t+ - T/4  (n = 2, 3, 4, 5)
//    t_min: tempi minimi  = t- + T/4  (n = 1.5, 2.5, 3.5, 4.5)
//
//  Tutti i tempi in microsecondi (us).
//  Incertezza sui tempi: sigma_t = 0.04 us (risoluzione cursore).
//
//  Risultato finale: media pesata di T dai 4 fit.
//  Inset: zoom sui valori di T con barre di errore (come nella slide).
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
#include <TLatex.h>
#include <iostream>
#include <vector>
#include <cmath>

// ============================================================

void FitStimaPeriodo() {

    // --------------------------------------------------------
    // 0. Stile
    // --------------------------------------------------------
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetPadGridX(kFALSE);
    gStyle->SetPadGridY(kFALSE);
    gStyle->SetFrameBorderMode(0);
    gStyle->SetCanvasBorderMode(0);

    // ============================================================
    //  PARAMETRI GRAFICI – modifica qui
    // ============================================================
    // Box risultati (coordinate NDC del pad principale)
    const double BOX_x1 = 0.13, BOX_y1 = 0.60, BOX_x2 = 0.49, BOX_y2 = 0.87;
    const double BOX_textsize = 0.030;

    // Inset (zoom su T): coordinate NDC nel pad principale
    const double INSET_x1 = 0.58, INSET_y1 = 0.35, INSET_x2 = 0.95, INSET_y2 = 0.92;
    // ============================================================

    // --------------------------------------------------------
    // 1. Dati (tempi in microsecondi, misurati con cursore oscilloscopio)
    // --------------------------------------------------------
    const double sigma_t = 0.04;  // us – risoluzione cursore

    // t+ : zeri crescenti
    std::vector<double> n_plus  = {2.0, 3.0, 4.0, 5.0};
    std::vector<double> t_plus  = {5.6, 13.2, 20.8, 28.32};

    // t- : zeri decrescenti
    std::vector<double> n_minus = {1.5, 2.5, 3.5, 4.5, 5.5};
    std::vector<double> t_minus = {1.8, 9.4, 17.0, 24.6, 32.2};

    // t_max : stima dai dati – massimo cade T/4 prima dello zero crescente
    // t_max[i] = t_plus[i] - T_approx/4   (T_approx = 7.6 us)
    std::vector<double> n_max = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> t_max = {0.16, 7.4, 15.06, 22.68, 30.32};
    // t_min : stima dai dati – minimo cade T/4 dopo lo zero decrescente
    // t_min[i] = t_minus[i] + T_approx/4  (primi 4 punti)
    std::vector<double> n_min = {1.5, 2.5, 3.5, 4.5};
    std::vector<double> t_min = {3.64, 11.30, 18.84, 26.46, 34.16}; 

    // Errori (stessa sigma_t su tutti, propagazione nulla sulla stima T/4)
    int Np = (int)n_plus.size(),  Nm = (int)n_minus.size();
    int NM = (int)n_max.size(),   Nn = (int)n_min.size();
    std::vector<double> s_plus(Np, sigma_t), s_minus(Nm, sigma_t);
    std::vector<double> s_max(NM, sigma_t),  s_min(Nn, sigma_t);

    // --------------------------------------------------------
    // 2. TGraphErrors per i 4 dataset
    // --------------------------------------------------------
    // Colori coerenti con la slide: t+ rosso, t- blu, t_max arancio, t_min verde
    auto makeGraph = [](const std::vector<double>& n, const std::vector<double>& t,
                        const std::vector<double>& s,
                        Color_t col, Style_t mstyle, double msize) -> TGraphErrors* {
        TGraphErrors *g = new TGraphErrors((int)n.size(),
                                           n.data(), t.data(),
                                           nullptr, s.data());
        g->SetMarkerStyle(mstyle);
        g->SetMarkerColor(col);
        g->SetLineColor(col);
        g->SetMarkerSize(msize);
        return g;
    };

    TGraphErrors *gr_plus  = makeGraph(n_plus,  t_plus,  s_plus,  kRed,      22, 1.2);
    TGraphErrors *gr_minus = makeGraph(n_minus, t_minus, s_minus, kBlue,     23, 1.2);
    TGraphErrors *gr_max   = makeGraph(n_max,   t_max,   s_max,   kBlue, 24, 1.2);
    TGraphErrors *gr_min   = makeGraph(n_min,   t_min,   s_min,   kRed,  24, 1.2);

    // --------------------------------------------------------
    // 3. Fit lineari: t = T*n + t0   (pol1)
    // --------------------------------------------------------
    double xfit_min = 1.0, xfit_max = 6.5;

    auto doFit = [&](TGraphErrors *g, Color_t col, int lstyle)
                    -> std::pair<TF1*, TFitResultPtr> {
        TF1 *f = new TF1(Form("f_%p", (void*)g), "pol1", xfit_min, xfit_max);
        f->SetLineColor(col);
        f->SetLineStyle(lstyle);
        f->SetLineWidth(1);
        auto res = g->Fit(f, "SQR");
        return {f, res};
    };

    auto [f_plus,  r_plus]  = doFit(gr_plus,  kRed,      2);
    auto [f_minus, r_minus] = doFit(gr_minus, kBlue,     2);
    auto [f_max,   r_max]   = doFit(gr_max,   kOrange+1, 2);
    auto [f_min,   r_min]   = doFit(gr_min,   kGreen+2,  2);

    // Estrai T (pendenza = p1) e t0 (intercetta = p0) con errori
    double T_p = f_plus ->GetParameter(1), sT_p = f_plus ->GetParError(1);
    double T_m = f_minus->GetParameter(1), sT_m = f_minus->GetParError(1);
    double T_M = f_max  ->GetParameter(1), sT_M = f_max  ->GetParError(1);
    double T_n = f_min  ->GetParameter(1), sT_n = f_min  ->GetParError(1);

    double chi2_p = f_plus ->GetChisquare(); int ndf_p = f_plus ->GetNDF();
    double chi2_m = f_minus->GetChisquare(); int ndf_m = f_minus->GetNDF();
    double chi2_M = f_max  ->GetChisquare(); int ndf_M = f_max  ->GetNDF();
    double chi2_n = f_min  ->GetChisquare(); int ndf_n = f_min  ->GetNDF();

    // --------------------------------------------------------
    // 4. Media pesata di T dai 4 dataset
    // --------------------------------------------------------
    // Gruppo 1: t+ + t-   (zeri)
    double w_p = 1.0/(sT_p*sT_p), w_m = 1.0/(sT_m*sT_m);
    double T_zeros   = (w_p*T_p + w_m*T_m) / (w_p + w_m);
    double sT_zeros  = 1.0 / TMath::Sqrt(w_p + w_m);

    // Gruppo 2: t_max + t_min   (estremi)
    double w_M = 1.0/(sT_M*sT_M), w_n = 1.0/(sT_n*sT_n);
    double T_extrema  = (w_M*T_M + w_n*T_n) / (w_M + w_n);
    double sT_extrema = 1.0 / TMath::Sqrt(w_M + w_n);

    // Media pesata globale
    double W1 = 1.0/(sT_zeros*sT_zeros), W2 = 1.0/(sT_extrema*sT_extrema);
    double T_tot  = (W1*T_zeros + W2*T_extrema) / (W1 + W2);
    double sT_tot = 1.0 / TMath::Sqrt(W1 + W2);

    // omega_d = 2*pi / T   (T in us -> converti in s)
    double T_s    = T_tot   * 1e-6;
    double sT_s   = sT_tot  * 1e-6;
    double omega_d   = 2.0 * TMath::Pi() / T_s;
    double s_omega_d = omega_d * sT_s / T_s;   // = 2pi * sT / T^2

    // --------------------------------------------------------
    // 5. Stampa risultati
    // --------------------------------------------------------
    std::cout << "\n========================================" << std::endl;
    std::cout << "  STIMA DEL PERIODO T" << std::endl;
    std::cout << "========================================" << std::endl;
    printf("  [t+]    T = %.4f +/- %.4f us,  chi2/ndf = %.3f/%d\n", T_p, sT_p, chi2_p, ndf_p);
    printf("  [t-]    T = %.4f +/- %.4f us,  chi2/ndf = %.3f/%d\n", T_m, sT_m, chi2_m, ndf_m);
    printf("  [t_max] T = %.4f +/- %.4f us,  chi2/ndf = %.3f/%d\n", T_M, sT_M, chi2_M, ndf_M);
    printf("  [t_min] T = %.4f +/- %.4f us,  chi2/ndf = %.3f/%d\n", T_n, sT_n, chi2_n, ndf_n);
    printf("\n  Media pesata t+ + t-   : T = %.4f +/- %.4f us\n", T_zeros,   sT_zeros);
    printf("  Media pesata t_max+t_min: T = %.4f +/- %.4f us\n", T_extrema, sT_extrema);
    printf("\n  MEDIA PESATA GLOBALE: T = %.4f +/- %.4f us\n",    T_tot,     sT_tot);
    printf("  Omega_d = 2pi/T = %.4e +/- %.4e rad/s\n", omega_d, s_omega_d);
    printf("  Omega_d / (2pi) = %.4f +/- %.4f kHz\n",
           omega_d/(2.0*TMath::Pi())*1e-3, s_omega_d/(2.0*TMath::Pi())*1e-3);

    // Compatibilita' tra i due gruppi
    double diff   = TMath::Abs(T_zeros - T_extrema);
    double s_diff = TMath::Sqrt(sT_zeros*sT_zeros + sT_extrema*sT_extrema);
    printf("\n  Compatibilita' zeri vs estremi: |DT| = %.4f us, sigma = %.4f us, pull = %.2f\n",
           diff, s_diff, diff/s_diff);
    if (diff/s_diff < 2.0)
        std::cout << "  --> COMPATIBILI (pull < 2)" << std::endl;
    else
        std::cout << "  --> NON compatibili (pull >= 2)" << std::endl;

    // --------------------------------------------------------
    // 6. Canvas principale: t vs n con i 4 dataset
    // --------------------------------------------------------
    TCanvas *c = new TCanvas("c_period", "RLC - Stima periodo T", 900, 680);
    c->SetLeftMargin(0.11);
    c->SetRightMargin(0.04);
    c->SetBottomMargin(0.12);
    c->SetTopMargin(0.05);

    // Frame vuoto per definire gli assi
    double xmin_plot = 1.0, xmax_plot = 6.5;
    double ymin_plot = -5.0, ymax_plot = 40.0;  // us

    gr_plus->SetTitle("Stima Periodo T da regressione lineare su t vs n");
    gr_plus->GetXaxis()->SetLimits(xmin_plot, xmax_plot);
    gr_plus->GetYaxis()->SetRangeUser(ymin_plot, ymax_plot);
    gr_plus->GetXaxis()->SetTitle("n");
    gr_plus->GetYaxis()->SetTitle("tempi  (#mus)");
    gr_plus->GetXaxis()->SetTitleSize(0.050);
    gr_plus->GetYaxis()->SetTitleSize(0.050);
    gr_plus->GetXaxis()->SetLabelSize(0.042);
    gr_plus->GetYaxis()->SetLabelSize(0.042);
    gr_plus->GetXaxis()->SetTitleOffset(1.0);
    gr_plus->GetYaxis()->SetTitleOffset(0.95);

    gr_plus->Draw("AP");
    gr_minus->Draw("P SAME");
    gr_max->Draw("P SAME");
    gr_min->Draw("P SAME");
    f_plus->Draw("SAME");
    f_minus->Draw("SAME");
    f_max->Draw("SAME");
    f_min->Draw("SAME");

    // Legenda
    TLegend *leg = new TLegend(0.12, 0.68, 0.42, 0.92);
    leg->SetBorderSize(1);
    leg->SetFillColor(kWhite);
    leg->SetTextSize(0.036);
    leg->AddEntry(gr_plus,  "t^{+}  (zeri crescenti)",   "p");
    leg->AddEntry(gr_minus, "t^{-}  (zeri decrescenti)", "p");
    leg->AddEntry(gr_max,   "t^{max}  (massimi)",        "p");
    leg->AddEntry(gr_min,   "t^{min}  (minimi)",         "p");
    leg->Draw();

    // Box risultati
    TPaveText *pt = new TPaveText(BOX_x1, BOX_y1, BOX_x2, BOX_y2, "NDC");
    pt->SetFillColor(kWhite);
    pt->SetBorderSize(1);
    pt->SetTextAlign(12);
    pt->SetTextSize(BOX_textsize);
    pt->SetTextColor(kBlack);
    pt->AddText(Form("t^{+} + t^{-}  #rightarrow  T = (%.2f #pm %.2f) #mus",
                     T_zeros, sT_zeros));
    pt->AddText(Form("t^{max} + t^{min}  #rightarrow  T = (%.2f #pm %.2f) #mus",
                     T_extrema, sT_extrema));
    pt->AddText(Form("Media pesata  #rightarrow  T = (%.2f #pm %.2f) #mus",
                     T_tot, sT_tot));
    pt->AddText(Form("#Omega = 2#pi/T = (%.1f #pm %.1f) kHz",
                     omega_d/(2.0*TMath::Pi())*1e-3,
                     s_omega_d/(2.0*TMath::Pi())*1e-3));
    pt->Draw();

    
    c->cd();  // torna al canvas principale

    c->Update();
}