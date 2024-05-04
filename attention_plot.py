import yfinance as yf


df = yf.download('^GSPC', start="2000-01-01", end="2022-09-01", interval="1d")

mod_kns = sm.tsa.MarkovRegression(ex_ret.dropna(), k_regimes=2, trend='n', switching_variance=True)
res_kns = mod_kns.fit()
res_kns.summary()