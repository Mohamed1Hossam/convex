"""visualizations.py
Functions to create and save plots used in Results/plots/.
"""
import matplotlib.pyplot as plt
import pandas as pd


def plot_price_trends(price_df, out_path):
    plt.figure(figsize=(10,6))
    # Expect wide-format price_df where columns are tickers
    price_df.plot(legend=False)
    plt.title('S&P Price Trends (placeholder)')
    plt.savefig(out_path)
    plt.close()

def plot_risk_return(portfolios, out_path, title='Risk vs Return'):
    # portfolios: list of dicts with 'risk' and 'return'
    risks = [p['risk'] for p in portfolios]
    rets = [p['return'] for p in portfolios]
    plt.figure(figsize=(7,5))
    plt.scatter(risks, rets)
    plt.xlabel('Risk (std)')
    plt.ylabel('Return')
    plt.title(title)
    plt.savefig(out_path)
    plt.close()

if __name__ == '__main__':
    print('visualizations: plot_price_trends, plot_risk_return')
