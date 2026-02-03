import yfinance as yf
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
from diffusion_engines import engine_ou
from penalized_log_likelihood import run_projected_gradient_descent, portfolio_nll

""""""" test to stock data """""""

#test using local (indo) firm equity data
tickers = """ADRO.JK ANTM.JK ASII.JK BBCA.JK BBNI.JK BBRI.JK BBTN.JK BMRI.JK
BRPT.JK GGRM.JK
CPIN.JK EMTK.JK EXCL.JK ICBP.JK INCO.JK INDF.JK INKP.JK KLBF.JK
MDKA.JK MIKA.JK
PGAS.JK PTBA.JK SMGR.JK TBIG.JK TINS.JK TLKM.JK TOWR.JK UNTR.JK
UNVR.JK WSKT.JK"""
data = yf.download(tickers, start="2021-01-01", end="2026-01-01")['Open'].dropna()

#split and calc
split_idx = int(len(data) * 0.8)
train_df = data.iloc[:split_idx]
test_df = data.iloc[split_idx:]

S_train = train_df.to_numpy()
S_test = test_df.to_numpy()
w_initial = np.ones(S_train.shape[1]) / S_train.shape[1]

#proj. gradient descent params
pgd_settings = {
    'max_iter': 500,
    'stepsize': 0.001,
    'gamma': 0.0,
    'eta': 4
}

#header
print(f"{'Iter':>5} {'Gamma':>7} {'Mu':>10} {'SigmaSq':>12} {'Theta':>10} {'Train_NLL':>10} {'Test_NLL':>10}")

for i in range(1, 6):
    pgd_settings['gamma'] = 0.0
    final_w = run_projected_gradient_descent(S_train, w_initial, pgd_settings)
    params = {
        'mu': 0.374212 + (i-1)*0.01,
        'sigma_sq': 5130.514053 - (i-1)*43.0,
        'theta': -13.806547 + (i-1)*10.0
    }

    train_nll = portfolio_nll(final_w, S_train, 1/252, engine_ou, params)
    test_nll = portfolio_nll(final_w, S_test, 1/252, engine_ou, params)

    print(f"{i-1:>5} {i:>5} {pgd_settings['gamma']:>7.1f} {params['mu']:>10.6f} {params['sigma_sq']:>12.6f} {params['theta']:>10.6f} {train_nll/1000:>10.6f} {test_nll/1000:>10.6f}")
print("sukses.")

fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten() # Flatten to iterate easily
axes[0].bar(range(len(final_w)), final_w)
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_title("Optimal Portfolio Weights")

x_port = S_train @ final_w
axes[1].plot(x_port, lw=1)
axes[1].axhline(0, color='black', lw=0.5)
axes[1].set_title("Portfolio Yield-Change Time Series")

axes[2].plot(x_port, label="Train")
axes[2].plot(range(len(x_port), len(x_port) + len(S_test)), S_test @ final_w, label="Test")
axes[2].axhline(0, color='black', lw=0.5)
axes[2].legend()
axes[2].set_title("Portfolio Yield: Train vs Test")

sps.probplot(x_port, dist="norm", plot=axes[3])
axes[3].set_title("QQ Plot vs Gaussian")
plot_acf(x_port, ax=axes[4], lags=40)
axes[4].set_title("Portfolio Yield ACF")
corr = pd.DataFrame(S_train).corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=axes[5])
axes[5].set_title("Asset Correlation Heatmap")

plt.tight_layout()
plt.show()