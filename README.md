# Autoencoder-Asset-Pricing-Models

🧐 [Report](https://www.richardsong.space/autoencoder-asset-pricing-models)
## Set Up

```bash
# generate preprocessed data and download portfolio returns
python data_prepare.py

# train models (ALL Together)
python main.py --Model 'FF PCA IPCA CA0 CA1 CA2 CA3' --K '1 2 3 4 5 6'

# train models (selected models and K)
python main.py --Model 'IPCA CA3' --K '5 6'

# analyze models (calculate R^2)
python analysis.py

# analyze characteristics' importance
python main.py --Model 'IPCA CA0 CA1 CA2 CA3' --K '5' --omit_char 'absacc acc age agr bm bm_ia cashdebt cashpr cfp cfp_ia chatoia chcsho chempia chinv chpmia convind currat depr divi divo dy egr ep gma grcapx grltnoa herf hire invest lev lgr mve_ia operprof orgcap pchcapx_ia pchcurrat pchdepr pchgm_pchsale pchquick pchsale_pchinvt pchsale_pchrect pchsale_pchxsga pchsaleinv pctacc ps quick rd rd_mve rd_sale realestate roic salecash saleinv salerec secured securedind sgr sin sp tang tb aeavol cash chtx cinvest ear ms nincr roaq roavol roeq rsup stdacc stdcf baspread beta betasq chmom dolvol idiovol ill indmom maxret mom12m mom1m mom36m mom6m mvel1 pricedelay retvol std_dolvol std_turn turn zerotrade'
```
## Results
### Total R^2

|(%)| K=1  | K=2  |  K=3 |  K=4 |  K=5 |  K=6 |
|---|---|---|---|---|---|---|
| FF |  8.53 | 15.76 | 19.86 | 20.32 | 31.40 | 36.16 |
|PCA | 40.61 | 53.00 | 59.13 | 62.46 | 64.68 | 67.20 |
|IPCA| 66.96 | 77.11 | 81.98 | 85.68 | 86.99 | 88.64 |
|CA0 | 56.78 | 67.61 | 70.76 | 69.14 | 69.21 | 71.11 |
|CA1 | 55.18 | 70.26 | 68.58 | 66.64 | 68.42 | 70.53 |
|CA2 | 59.67 | 66.27 | 66.09 | 70.70 | 68.61 | 67.68 |
|CA3 | 49.28 | 57.65 | 57.17 | 58.24 | 56.64 | 50.85 |
<img src="/imgs/total_R2.png" width=100%>


### Predict R^2
|(%)| K=1  | K=2  |  K=3 |  K=4 |  K=5 |  K=6 |
|---|---|---|---|---|---|---|
| FF | -0.59 | -1.10 | -0.72 | -0.46 | -0.38 | -0.45 |
|PCA |  0.30 |  0.60 |  0.69 |  0.75 |  0.80 |  0.82 |
|IPCA|  0.15 |  0.53 | -0.34 | -0.17 | -0.36 | -0.07 |
|CA0 | -1.23 | -2.45 | -3.54 | -1.68 | -3.35 | -1.36 |
|CA1 | -3.39 | -6.12 | -3.02 | -1.67 | -2.88 | -1.85 |
|CA2 | -1.87 | -0.31 | -1.19 | -1.14 | -1.13 | -1.94 |
|CA3 | -0.04 | -1.00 | -2.24 | -1.25 | -0.64 | -1.50 |
<img src="/imgs/pred_R2.png" width=100%>

### Risk Premia v.s. Mispricing
<table>
<tr>
<td><img src="imgs/alpha/FF_5_inference_alpha_plot.png" border=0></td>
<td><img src="imgs/alpha/PCA_5_inference_alpha_plot.png" border=0></td>
</tr>
<tr>
<td><img src="imgs/alpha/IPCA_5_inference_alpha_plot.png" border=0></td>
<td><img src="imgs/alpha/CA1_5_inference_alpha_plot.png" border=0></td>
</tr>
<tr>
<td><img src="imgs/alpha/CA2_5_inference_alpha_plot.png" border=0></td>
<td><img src="imgs/alpha/CA3_5_inference_alpha_plot.png" border=0></td>
</tr>
</table>
