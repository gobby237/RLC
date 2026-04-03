// ============================================================
//  RLC Oscillazioni Smorzate – Analisi ai capi dell'induttore
//  Fit lineare di ln|Vn| vs n per massimi (+) e minimi (-)
//  Stima di delta e Q = omega_d / (2*delta)
//  Con pannelli dei residui
// ============================================================

#include <TMath.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TAxis.h>
#include <TLegend.h>
#include <TPaveText.h>
#include <TStyle.h>
#include <TLatex.h>
#include <TLine.h>
#include <TPad.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

void rlc_analysis() {

    // ============================================================
    //  PARAMETRO DA MODIFICARE: offset del ground in milliVolt
    // ============================================================
    double offset_mV = 20.0;   // <--- MODIFICA QUI (in milliVolt)
    double offset_V  = offset_mV * 1e-3;

    // ============================================================
    //  DIMENSIONI E POSIZIONE DEI BOX CHI2 (coordinate NDC 0->1)
    // ============================================================

    // -- Canvas 1: box MASSIMI (blu, sinistra) --
    double bB_x1 = 0.13, bB_y1 = 0.72, bB_x2 = 0.42, bB_y2 = 0.93;
    double bB_textsize = 0.032;

    // -- Canvas 1: box MINIMI (rosso, destra) --
    double bR_x1 = 0.51, bR_y1 = 0.72, bR_x2 = 0.80, bR_y2 = 0.93;
    double bR_textsize = 0.032;

    // -- Canvas 2: box FIT COMBINATO (nero) --
    double bC_x1 = 0.13, bC_y1 = 0.72, bC_x2 = 0.42, bC_y2 = 0.93;
    double bC_textsize = 0.032;

    // --------------------------------------------------------
    // 0. Stile
    // --------------------------------------------------------
    gStyle->SetOptStat(0);
    gStyle->SetOptFit(0);
    gStyle->SetPadGridX(kFALSE);
    gStyle->SetPadGridY(kFALSE);
    gStyle->SetFrameBorderMode(0);
    gStyle->SetCanvasBorderMode(0);

    // --------------------------------------------------------
    // 1. Dati sperimentali (valori misurati grezzi)
    // --------------------------------------------------------

    // --- Massimi (n+, V+) ---
    std::vector<double> n_plus   = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> V_plus   = {2.26, 1.06, 0.48, 0.22, 0.10};
    std::vector<double> sV_plus  = {0.04, 0.03, 0.02, 0.02, 0.02};

    // --- Minimi (n-, |V-| gia' in valore assoluto come misurati) ---
    std::vector<double> n_minus  = {1.5, 2.5, 3.5, 4.5, 5.5};
    std::vector<double> V_minus  = {1.56, 0.66, 0.28, 0.10, 0.04};
    std::vector<double> sV_minus = {0.03, 0.02, 0.02, 0.02, 0.02};

    // --------------------------------------------------------
    // 1b. Applica correzione offset ground
    // --------------------------------------------------------
    int Np = (int)n_plus.size();
    int Nm = (int)n_minus.size();

    for (int i = 0; i < Np; i++)
        V_plus[i]  = V_plus[i]  - offset_V;
    for (int i = 0; i < Nm; i++)
        V_minus[i] = V_minus[i] + offset_V;

    if (TMath::Abs(offset_mV) > 1e-9) {
        printf("\n  Offset ground applicato: %.2f mV\n", offset_mV);
        printf("  Valori corretti:\n");
        for (int i = 0; i < Np; i++)
            printf("    V+[%d] = %.4f V\n", i+1, V_plus[i]);
        for (int i = 0; i < Nm; i++)
            printf("    |V-[%.1f]| = %.4f V\n", n_minus[i], V_minus[i]);
    }

    // --- Tempi (microsecondi) ---
    std::vector<double> t_plus  = {5.6, 13.2, 20.8, 28.32};
    std::vector<double> t_minus = {1.8, 9.4, 17.0, 24.6, 32.2};
    double sigma_t = 0.04;

    // --------------------------------------------------------
    // 2. Stima del periodo T
    // --------------------------------------------------------
    std::vector<double> T2_pos, sT2_pos, T2_neg, sT2_neg;
    for (int i = 0; i < 4; i++) {
        T2_pos.push_back(t_minus[i+1] - t_plus[i]);
        sT2_pos.push_back(TMath::Sqrt(2.0) * sigma_t);
        T2_neg.push_back(t_plus[i] - t_minus[i]);
        sT2_neg.push_back(TMath::Sqrt(2.0) * sigma_t);
    }

    auto weightedMean = [](const std::vector<double>& v, const std::vector<double>& sv,
                           double &mean, double &sigma_mean) {
        double sw = 0, swx = 0;
        for (size_t i = 0; i < v.size(); i++) {
            double w = 1.0 / (sv[i]*sv[i]);
            sw += w; swx += w * v[i];
        }
        mean       = swx / sw;
        sigma_mean = 1.0 / TMath::Sqrt(sw);
    };

    double T2pos_mean, sT2pos_mean, T2neg_mean, sT2neg_mean;
    weightedMean(T2_pos, sT2_pos, T2pos_mean, sT2pos_mean);
    weightedMean(T2_neg, sT2_neg, T2neg_mean, sT2neg_mean);

    std::cout << "\n========================================" << std::endl;
    std::cout << "  ANALISI DEL PERIODO T" << std::endl;
    std::cout << "========================================" << std::endl;
    for (size_t i = 0; i < T2_pos.size(); i++)
        printf("  T/2+[%zu] = %.4f +/- %.4f us\n", i+1, T2_pos[i], sT2_pos[i]);
    printf("  Media pesata T/2+ = %.4f +/- %.4f us\n", T2pos_mean, sT2pos_mean);
    for (size_t i = 0; i < T2_neg.size(); i++)
        printf("  T/2-[%zu] = %.4f +/- %.4f us\n", i+1, T2_neg[i], sT2_neg[i]);
    printf("  Media pesata T/2- = %.4f +/- %.4f us\n", T2neg_mean, sT2neg_mean);

    double diff_T2    = TMath::Abs(T2pos_mean - T2neg_mean);
    double sigma_diff = TMath::Sqrt(sT2pos_mean*sT2pos_mean + sT2neg_mean*sT2neg_mean);
    double pull_T     = diff_T2 / sigma_diff;
    printf("  |T/2+ - T/2-| = %.4f us,  sigma_diff = %.4f us,  pull = %.2f\n",
           diff_T2, sigma_diff, pull_T);
    if (pull_T < 2.0)
        std::cout << "  --> COMPATIBILI (pull < 2): nessun offset di ground rilevato." << std::endl;
    else
        std::cout << "  --> NON compatibili (pull >= 2): possibile offset di ground!" << std::endl;

    double T_est  = (t_minus.back() - t_minus[0]) / 4.0;
    double sT_est = TMath::Sqrt(2.0) * sigma_t / 4.0;
    printf("\n  Stima T = (t-_last - t-_first)/4 = %.4f +/- %.4f us\n", T_est, sT_est);
    double T_s  = T_est  * 1e-6;
    double sT_s = sT_est * 1e-6;

    // --------------------------------------------------------
    // 3. ln|Vn| e propagazione errori
    // --------------------------------------------------------
    std::vector<double> lnVp(Np), slnVp(Np), lnVm(Nm), slnVm(Nm);
    for (int i = 0; i < Np; i++) {
        lnVp[i]  = TMath::Log(V_plus[i]);
        slnVp[i] = sV_plus[i] / V_plus[i];
    }
    for (int i = 0; i < Nm; i++) {
        lnVm[i]  = TMath::Log(V_minus[i]);
        slnVm[i] = sV_minus[i] / V_minus[i];
    }

    // --------------------------------------------------------
    // 4. TGraphErrors e fit lineari
    // --------------------------------------------------------
    TGraphErrors *gr_plus  = new TGraphErrors(Np, n_plus.data(),  lnVp.data(), nullptr, slnVp.data());
    TGraphErrors *gr_minus = new TGraphErrors(Nm, n_minus.data(), lnVm.data(), nullptr, slnVm.data());

    gr_plus->SetMarkerStyle(24); gr_plus->SetMarkerColor(kBlue); gr_plus->SetLineColor(kBlue); gr_plus->SetMarkerSize(0.9);
    gr_minus->SetMarkerStyle(20); gr_minus->SetMarkerColor(kRed); gr_minus->SetLineColor(kRed); gr_minus->SetMarkerSize(0.9);

    TF1 *f_plus  = new TF1("f_plus",  "pol1", 0.5, 6.5);
    TF1 *f_minus = new TF1("f_minus", "pol1", 0.5, 6.5);
    f_plus ->SetLineColor(kBlue); f_plus ->SetLineStyle(2); f_plus ->SetLineWidth(1);
    f_minus->SetLineColor(kRed);  f_minus->SetLineStyle(2); f_minus->SetLineWidth(1);

    gr_plus ->Fit(f_plus,  "SQR");
    gr_minus->Fit(f_minus, "SQR");

    double p0p = f_plus->GetParameter(0),  s_p0p = f_plus->GetParError(0);
    double p1p = f_plus->GetParameter(1),  s_p1p = f_plus->GetParError(1);
    double chi2p = f_plus->GetChisquare(), ndfp  = f_plus->GetNDF();

    double p0m = f_minus->GetParameter(0), s_p0m = f_minus->GetParError(0);
    double p1m = f_minus->GetParameter(1), s_p1m = f_minus->GetParError(1);
    double chi2m = f_minus->GetChisquare(), ndfm = f_minus->GetNDF();

    // --------------------------------------------------------
    // 5. Stima delta (=1/tau), omega_d (=Omega), omega_0 e Q
    //
    //  Dalla slide del prof:
    //    delta = 1/tau  -->  nostro delta dal fit: delta = -p1 / T
    //    Omega = omega_d = 2*pi / T   (frequenza oscillazione smorzata)
    //    omega_0 = sqrt(Omega^2 + delta^2)   (frequenza propria)
    //    Q = omega_0 / (2 * delta)
    //
    //  Propagazione errori omega_0:
    //    s_omega0 = sqrt( (Omega/omega_0)^2 * s_Omega^2
    //                   + (delta/omega_0)^2 * s_delta^2 )
    //
    //  Propagazione errori Q:
    //    s_Q = Q * sqrt( (s_omega0/omega_0)^2 + (s_delta/delta)^2 )
    // --------------------------------------------------------

    auto calcDelta = [&](double m, double sm, double T, double sT,
                         double &delta, double &s_delta) {
        // m = -delta * T  =>  delta = -m / T
        delta   = -m / T;
        s_delta = TMath::Sqrt(TMath::Power(sm/T, 2) +
                              TMath::Power(m*sT/(T*T), 2));
    };

    // omega_d = Omega = 2*pi / T
    double omega_d   = 2.0 * TMath::Pi() / T_s;
    double s_omega_d = 2.0 * TMath::Pi() * sT_s / (T_s * T_s);

    auto calcOmega0 = [&](double delta, double s_delta,
                          double &omega0, double &s_omega0) {
        // omega_0 = sqrt(omega_d^2 + delta^2)
        omega0   = TMath::Sqrt(omega_d*omega_d + delta*delta);
        // s_omega0 = sqrt( (omega_d/omega0)^2 * s_omega_d^2
        //                + (delta/omega0)^2    * s_delta^2  )
        s_omega0 = TMath::Sqrt(
            TMath::Power((omega_d / omega0) * s_omega_d, 2) +
            TMath::Power((delta   / omega0) * s_delta,   2) );
    };

    auto calcQ = [&](double omega0, double s_omega0,
                     double delta,  double s_delta,
                     double &Q, double &sQ) {
        // Q = omega_0 / (2 * delta)
        Q  = omega0 / (2.0 * delta);
        // s_Q = Q * sqrt( (s_omega0/omega0)^2 + (s_delta/delta)^2 )
        sQ = Q * TMath::Sqrt(TMath::Power(s_omega0 / omega0, 2) +
                             TMath::Power(s_delta  / delta,  2));
    };

    double delta_p, s_delta_p, delta_m, s_delta_m;
    calcDelta(p1p, s_p1p, T_s, sT_s, delta_p, s_delta_p);
    calcDelta(p1m, s_p1m, T_s, sT_s, delta_m, s_delta_m);

    double omega0_p, s_omega0_p, omega0_m, s_omega0_m;
    calcOmega0(delta_p, s_delta_p, omega0_p, s_omega0_p);
    calcOmega0(delta_m, s_delta_m, omega0_m, s_omega0_m);

    double Q_p, sQ_p, Q_m, sQ_m;
    calcQ(omega0_p, s_omega0_p, delta_p, s_delta_p, Q_p, sQ_p);
    calcQ(omega0_m, s_omega0_m, delta_m, s_delta_m, Q_m, sQ_m);

    // --------------------------------------------------------
    // 5b. Test compatibilita' e fit combinato
    // --------------------------------------------------------
    double diff_p1  = TMath::Abs(p1p - p1m);
    double sigma_p1 = TMath::Sqrt(s_p1p*s_p1p + s_p1m*s_p1m);
    double pull_p1  = diff_p1 / sigma_p1;
    bool compatible = (pull_p1 < 2.0);

    std::vector<double> n_all, lnV_all, slnV_all;
    for (int i = 0; i < Np; i++) { n_all.push_back(n_plus[i]);  lnV_all.push_back(lnVp[i]);  slnV_all.push_back(slnVp[i]); }
    for (int i = 0; i < Nm; i++) { n_all.push_back(n_minus[i]); lnV_all.push_back(lnVm[i]);  slnV_all.push_back(slnVm[i]); }
    int Ntot = (int)n_all.size();

    TGraphErrors *gr_all = new TGraphErrors(Ntot, n_all.data(), lnV_all.data(), nullptr, slnV_all.data());
    TF1 *f_all = new TF1("f_all", "pol1", 0.5, 7.0);
    f_all->SetLineColor(kBlack); f_all->SetLineStyle(2); f_all->SetLineWidth(1);

    gr_all->SetMarkerStyle(20);
    gr_all->SetMarkerColor(kBlack);
    gr_all->SetLineColor(kBlack);
    gr_all->SetMarkerSize(0.9);

    gr_all->Fit(f_all, "SQR");

    double p0c = f_all->GetParameter(0), s_p0c = f_all->GetParError(0);
    double p1c = f_all->GetParameter(1), s_p1c = f_all->GetParError(1);
    double chi2c = f_all->GetChisquare(), ndfc = f_all->GetNDF();

    double delta_c, s_delta_c, omega0_c, s_omega0_c, Q_c, sQ_c;
    calcDelta(p1c, s_p1c, T_s, sT_s, delta_c, s_delta_c);
    calcOmega0(delta_c, s_delta_c, omega0_c, s_omega0_c);
    calcQ(omega0_c, s_omega0_c, delta_c, s_delta_c, Q_c, sQ_c);

    // --------------------------------------------------------
    // 5c. Residui
    // --------------------------------------------------------
    std::vector<double> res_plus_sep(Np), res_minus_sep(Nm);
    std::vector<double> res_all_comb(Ntot);

    for (int i = 0; i < Np; i++)
        res_plus_sep[i] = lnVp[i] - f_plus->Eval(n_plus[i]);
    for (int i = 0; i < Nm; i++)
        res_minus_sep[i] = lnVm[i] - f_minus->Eval(n_minus[i]);
    for (int i = 0; i < Ntot; i++)
        res_all_comb[i] = lnV_all[i] - f_all->Eval(n_all[i]);

    TGraphErrors *gr_res_plus_sep  = new TGraphErrors(Np,   n_plus.data(),  res_plus_sep.data(),  nullptr, slnVp.data());
    TGraphErrors *gr_res_minus_sep = new TGraphErrors(Nm,   n_minus.data(), res_minus_sep.data(), nullptr, slnVm.data());
    TGraphErrors *gr_res_all_comb  = new TGraphErrors(Ntot, n_all.data(),   res_all_comb.data(),  nullptr, slnV_all.data());

    gr_res_plus_sep->SetMarkerStyle(24); gr_res_plus_sep->SetMarkerColor(kBlue);  gr_res_plus_sep->SetLineColor(kBlue);  gr_res_plus_sep->SetMarkerSize(0.9);
    gr_res_minus_sep->SetMarkerStyle(20); gr_res_minus_sep->SetMarkerColor(kRed);   gr_res_minus_sep->SetLineColor(kRed);   gr_res_minus_sep->SetMarkerSize(0.9);
    gr_res_all_comb->SetMarkerStyle(20); gr_res_all_comb->SetMarkerColor(kBlack); gr_res_all_comb->SetLineColor(kBlack); gr_res_all_comb->SetMarkerSize(0.9);

    double maxAbsRes_sep = 0.0;
    for (int i = 0; i < Np;   i++) maxAbsRes_sep  = std::max(maxAbsRes_sep,  std::abs(res_plus_sep[i]));
    for (int i = 0; i < Nm;   i++) maxAbsRes_sep  = std::max(maxAbsRes_sep,  std::abs(res_minus_sep[i]));
    double resY_sep  = 2 * std::max(0.15, 1.35 * maxAbsRes_sep);

    double maxAbsRes_comb = 0.0;
    for (int i = 0; i < Ntot; i++) maxAbsRes_comb = std::max(maxAbsRes_comb, std::abs(res_all_comb[i]));
    double resY_comb = 2 * std::max(0.15, 1.35 * maxAbsRes_comb);

    // --------------------------------------------------------
    // 6. Stampa risultati
    // --------------------------------------------------------
    std::cout << "\n========================================" << std::endl;
    std::cout << "  RISULTATI FIT" << std::endl;
    std::cout << "========================================" << std::endl;

    printf("  [MASSIMI +]  chi2/ndf = %.4f/%d = %.4f,  Prob = %.4f\n",
           chi2p,(int)ndfp,chi2p/ndfp,TMath::Prob(chi2p,ndfp));
    printf("  p0 = %.4f +/- %.4f\n  p1 = %.4f +/- %.4f\n", p0p,s_p0p,p1p,s_p1p);
    printf("  Omega (=omega_d) = %.4e +/- %.4e  rad/s\n", omega_d,  s_omega_d);
    printf("  delta (=1/tau)   = %.4e +/- %.4e  rad/s\n", delta_p,  s_delta_p);
    printf("  omega_0          = %.4e +/- %.4e  rad/s\n", omega0_p, s_omega0_p);
    printf("  Q+               = %.4f +/- %.4f\n\n",      Q_p,      sQ_p);

    printf("  [MINIMI -]   chi2/ndf = %.4f/%d = %.4f,  Prob = %.4f\n",
           chi2m,(int)ndfm,chi2m/ndfm,TMath::Prob(chi2m,ndfm));
    printf("  p0 = %.4f +/- %.4f\n  p1 = %.4f +/- %.4f\n", p0m,s_p0m,p1m,s_p1m);
    printf("  Omega (=omega_d) = %.4e +/- %.4e  rad/s\n", omega_d,  s_omega_d);
    printf("  delta (=1/tau)   = %.4e +/- %.4e  rad/s\n", delta_m,  s_delta_m);
    printf("  omega_0          = %.4e +/- %.4e  rad/s\n", omega0_m, s_omega0_m);
    printf("  Q-               = %.4f +/- %.4f\n\n",      Q_m,      sQ_m);

    printf("  TEST p1+/p1-: pull = %.2f  --> %s\n", pull_p1,
           compatible ? "COMPATIBILI" : "NON compatibili");

    printf("\n  [FIT COMBINATO] chi2/ndf = %.4f/%d = %.4f,  Prob = %.4f\n",
           chi2c,(int)ndfc,chi2c/ndfc,TMath::Prob(chi2c,ndfc));
    printf("  p0 = %.4f +/- %.4f\n  p1 = %.4f +/- %.4f\n", p0c,s_p0c,p1c,s_p1c);
    printf("  Omega (=omega_d) = %.4e +/- %.4e  rad/s\n", omega_d,  s_omega_d);
    printf("  delta (=1/tau)   = %.4e +/- %.4e  rad/s\n", delta_c,  s_delta_c);
    printf("  omega_0          = %.4e +/- %.4e  rad/s\n", omega0_c, s_omega0_c);
    printf("  Q_comb           = %.4f +/- %.4f\n\n",      Q_c,      sQ_c);

    // --------------------------------------------------------
    // 7. Canvas 1 - fit separati + residui
    // --------------------------------------------------------
    double xmin = 0.5, xmax = 6.0;
    double ymin = -4.5, ymax = 2.0;

    TCanvas *c1 = new TCanvas("c1",
        Form("RLC - ln|Vn| vs n (separati, offset=%.1f mV)", offset_mV), 850, 900);

    TPad *pad1_top = new TPad("pad1_top", "pad1_top", 0.0, 0.38, 1.0, 1.0);
    TPad *pad1_mid = new TPad("pad1_mid", "pad1_mid", 0.0, 0.19, 1.0, 0.38);
    TPad *pad1_bot = new TPad("pad1_bot", "pad1_bot", 0.0, 0.00, 1.0, 0.19);

    pad1_top->SetBottomMargin(0.02); pad1_top->SetLeftMargin(0.12);
    pad1_top->SetRightMargin(0.05);  pad1_top->SetTopMargin(0.05);
    pad1_mid->SetTopMargin(0.02);    pad1_mid->SetBottomMargin(0.02);
    pad1_mid->SetLeftMargin(0.12);   pad1_mid->SetRightMargin(0.05);
    pad1_bot->SetTopMargin(0.02);    pad1_bot->SetBottomMargin(0.28);
    pad1_bot->SetLeftMargin(0.12);   pad1_bot->SetRightMargin(0.05);

    pad1_top->Draw(); pad1_mid->Draw(); pad1_bot->Draw();

    pad1_top->cd();
    gr_plus->SetTitle("");
    gr_plus->GetXaxis()->SetLimits(xmin, xmax);
    gr_plus->GetYaxis()->SetRangeUser(ymin, ymax);
    gr_plus->GetXaxis()->SetTitle("n");
    gr_plus->GetYaxis()->SetTitle("ln(V_{n})");
    gr_plus->GetXaxis()->SetTitleSize(0.050); gr_plus->GetYaxis()->SetTitleSize(0.050);
    gr_plus->GetXaxis()->SetLabelSize(0.0);   gr_plus->GetYaxis()->SetLabelSize(0.040);
    gr_plus->GetXaxis()->SetTitleOffset(0.95); gr_plus->GetYaxis()->SetTitleOffset(1.10);
    gr_plus->Draw("AP");
    gr_minus->Draw("P SAME");
    f_plus->Draw("SAME");
    f_minus->Draw("SAME");

    TPaveText *ptB = new TPaveText(bB_x1, bB_y1, bB_x2, bB_y2, "NDC");
    ptB->SetFillColor(kWhite); ptB->SetBorderSize(1);
    ptB->SetTextAlign(12); ptB->SetTextSize(bB_textsize); ptB->SetTextColor(kBlue);
    ptB->AddText(Form("#chi^{2} / ndf      %.4f / %d", chi2p, (int)ndfp));
    ptB->AddText(Form("Prob              %.4f", TMath::Prob(chi2p, ndfp)));
    ptB->AddText(Form("p0      %.4f #pm %.5f", p0p, s_p0p));
    ptB->AddText(Form("p1      %.4f #pm %.5f", p1p, s_p1p));
    ptB->Draw();

    TPaveText *ptR = new TPaveText(bR_x1, bR_y1, bR_x2, bR_y2, "NDC");
    ptR->SetFillColor(kWhite); ptR->SetBorderSize(1);
    ptR->SetTextAlign(12); ptR->SetTextSize(bR_textsize); ptR->SetTextColor(kRed);
    ptR->AddText(Form("#chi^{2} / ndf      %.4f / %d", chi2m, (int)ndfm));
    ptR->AddText(Form("Prob              %.4f", TMath::Prob(chi2m, ndfm)));
    ptR->AddText(Form("p0      %.4f #pm %.5f", p0m, s_p0m));
    ptR->AddText(Form("p1      %.4f #pm %.5f", p1m, s_p1m));
    ptR->Draw();

    pad1_mid->cd();
    gr_res_plus_sep->SetTitle("");
    gr_res_plus_sep->GetXaxis()->SetLimits(xmin, xmax);
    gr_res_plus_sep->GetYaxis()->SetRangeUser(-resY_sep, resY_sep);
    gr_res_plus_sep->GetXaxis()->SetTitle("n");
    gr_res_plus_sep->GetYaxis()->SetTitle("Residui massimi");
    gr_res_plus_sep->GetXaxis()->SetTitleSize(0.0);   gr_res_plus_sep->GetYaxis()->SetTitleSize(0.11);
    gr_res_plus_sep->GetXaxis()->SetLabelSize(0.0);   gr_res_plus_sep->GetYaxis()->SetLabelSize(0.09);
    gr_res_plus_sep->GetYaxis()->SetTitleOffset(0.45); gr_res_plus_sep->GetYaxis()->SetNdivisions(505);
    gr_res_plus_sep->Draw("AP");
    TLine *line0_1 = new TLine(xmin, 0.0, xmax, 0.0);
    line0_1->SetLineStyle(2); line0_1->SetLineColor(kBlack); line0_1->Draw("SAME");

    pad1_bot->cd();
    gr_res_minus_sep->SetTitle("");
    gr_res_minus_sep->GetXaxis()->SetLimits(xmin, xmax);
    gr_res_minus_sep->GetYaxis()->SetRangeUser(-resY_sep, resY_sep);
    gr_res_minus_sep->GetXaxis()->SetTitle("n");
    gr_res_minus_sep->GetYaxis()->SetTitle("Residui minimi");
    gr_res_minus_sep->GetXaxis()->SetTitleSize(0.13);  gr_res_minus_sep->GetYaxis()->SetTitleSize(0.11);
    gr_res_minus_sep->GetXaxis()->SetLabelSize(0.10);  gr_res_minus_sep->GetYaxis()->SetLabelSize(0.09);
    gr_res_minus_sep->GetXaxis()->SetTitleOffset(0.95); gr_res_minus_sep->GetYaxis()->SetTitleOffset(0.45);
    gr_res_minus_sep->GetYaxis()->SetNdivisions(505);
    gr_res_minus_sep->Draw("AP");
    TLine *line0_2 = new TLine(xmin, 0.0, xmax, 0.0);
    line0_2->SetLineStyle(2); line0_2->SetLineColor(kBlack); line0_2->Draw("SAME");

    c1->Update();

    // --------------------------------------------------------
    // 8. Canvas 2 - fit combinato + residui
    // --------------------------------------------------------
    TCanvas *c2 = new TCanvas("c2",
        Form("RLC - ln|Vn| vs n (combinato, offset=%.1f mV)", offset_mV), 850, 760);

    TPad *pad2_top = new TPad("pad2_top", "pad2_top", 0.0, 0.28, 1.0, 1.0);
    TPad *pad2_bot = new TPad("pad2_bot", "pad2_bot", 0.0, 0.00, 1.0, 0.28);

    pad2_top->SetBottomMargin(0.02); pad2_top->SetLeftMargin(0.12);
    pad2_top->SetRightMargin(0.05);  pad2_top->SetTopMargin(0.05);
    pad2_bot->SetTopMargin(0.02);    pad2_bot->SetBottomMargin(0.25);
    pad2_bot->SetLeftMargin(0.12);   pad2_bot->SetRightMargin(0.05);

    pad2_top->Draw(); pad2_bot->Draw();

    pad2_top->cd();
    gr_all->SetTitle("");
    gr_all->GetXaxis()->SetLimits(xmin, xmax);
    gr_all->GetYaxis()->SetRangeUser(ymin, ymax);
    gr_all->GetXaxis()->SetTitle("n");
    gr_all->GetYaxis()->SetTitle("ln(V_{n})");
    gr_all->GetXaxis()->SetTitleSize(0.050); gr_all->GetYaxis()->SetTitleSize(0.050);
    gr_all->GetXaxis()->SetLabelSize(0.0);   gr_all->GetYaxis()->SetLabelSize(0.040);
    gr_all->GetXaxis()->SetTitleOffset(0.95); gr_all->GetYaxis()->SetTitleOffset(1.10);
    gr_all->Draw("AP");
    f_all->Draw("SAME");

    TPaveText *ptC = new TPaveText(bC_x1, bC_y1, bC_x2, bC_y2, "NDC");
    ptC->SetFillColor(kWhite); ptC->SetBorderSize(1);
    ptC->SetTextAlign(12); ptC->SetTextSize(bC_textsize); ptC->SetTextColor(kBlack);
    ptC->AddText(Form("#chi^{2} / ndf      %.4f / %d", chi2c, (int)ndfc));
    ptC->AddText(Form("Prob              %.4f", TMath::Prob(chi2c, ndfc)));
    ptC->AddText(Form("p0      %.4f #pm %.5f", p0c, s_p0c));
    ptC->AddText(Form("p1      %.4f #pm %.5f", p1c, s_p1c));
    ptC->Draw();

    pad2_bot->cd();
    gr_res_all_comb->SetTitle("");
    gr_res_all_comb->GetXaxis()->SetLimits(xmin, xmax);
    gr_res_all_comb->GetYaxis()->SetRangeUser(-resY_comb, resY_comb);
    gr_res_all_comb->GetXaxis()->SetTitle("n");
    gr_res_all_comb->GetYaxis()->SetTitle("Residui");
    gr_res_all_comb->GetXaxis()->SetTitleSize(0.12);  gr_res_all_comb->GetYaxis()->SetTitleSize(0.10);
    gr_res_all_comb->GetXaxis()->SetLabelSize(0.09);  gr_res_all_comb->GetYaxis()->SetLabelSize(0.085);
    gr_res_all_comb->GetXaxis()->SetTitleOffset(0.95); gr_res_all_comb->GetYaxis()->SetTitleOffset(0.50);
    gr_res_all_comb->GetYaxis()->SetNdivisions(505);
    gr_res_all_comb->Draw("AP");
    TLine *line0_3 = new TLine(xmin, 0.0, xmax, 0.0);
    line0_3->SetLineStyle(2); line0_3->SetLineColor(kBlack); line0_3->Draw("SAME");

    c2->Update();
}