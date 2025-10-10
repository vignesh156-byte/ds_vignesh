import pandas as pd

# Load the correct sentiment file
sent = pd.read_csv("fear_greed_index.csv")

# Load the trader data file
trader = pd.read_csv("historical_data.csv", low_memory=False)

print("\n--- Sentiment Data ---")
print(sent.shape)
print(sent.head())

print("\n--- Trader Data ---")
print(trader.shape)
print(trader.head())

# Standardize column names
sent.columns = [c.strip().lower().replace(" ", "_") for c in sent.columns]
trader.columns = [c.strip().lower().replace(" ", "_") for c in trader.columns]

print(sent.columns)
print(trader.columns)

# --- Convert timestamps ---
sent['date'] = pd.to_datetime(sent['timestamp'], unit='s').dt.normalize()
trader['trade_date'] = pd.to_datetime(trader['timestamp'] / 1000, unit='s').dt.normalize()

print(sent[['timestamp', 'date']].head())
print(trader[['timestamp', 'trade_date']].head())
def simplify_sentiment(x):
    if "Fear" in x:
        return "Fear"
    elif "Greed" in x:
        return "Greed"
    else:
        return "Neutral"

sent['sentiment'] = sent['classification'].apply(simplify_sentiment)
print(sent['sentiment'].value_counts())
# Convert to numeric safely
for col in ['execution_price', 'size_tokens', 'size_usd', 'closed_pnl', 'fee']:
    trader[col] = pd.to_numeric(trader[col], errors='coerce')

# Drop rows with missing important fields
trader = trader.dropna(subset=['execution_price', 'size_usd', 'trade_date'])

# Optional: total USD value traded
trader['trade_value'] = trader['execution_price'] * trader['size_tokens']
daily = trader.groupby('trade_date').agg(
    num_trades=('execution_price', 'count'),
    total_volume_usd=('size_usd', 'sum'),
    avg_price=('execution_price', 'mean'),
    avg_pnl=('closed_pnl', 'mean'),
    avg_leverage=('size_usd', 'mean')
).reset_index()

print(daily.head())
merged = pd.merge(
    daily,
    sent[['date', 'sentiment']],
    left_on='trade_date',
    right_on='date',
    how='left'
)

print(merged.head())
print(merged['sentiment'].value_counts(dropna=False))
merged.to_csv("csv_files/merged_trader_sentiment.csv", index=False)
print("✅ Cleaned and merged dataset saved to csv_files/merged_trader_sentiment.csv")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

merged = pd.read_csv("csv_files/merged_trader_sentiment.csv")

print(merged.info())
print(merged.describe())
print(merged['sentiment'].value_counts())

sns.countplot(x='sentiment', data=merged)
plt.title("Sentiment Distribution (Fear vs Greed)")
plt.savefig('outputs/sentiment_distribution.png', bbox_inches='tight', dpi=150)
plt.show()
sns.boxplot(x='sentiment', y='total_volume_usd', data=merged)
plt.title("Trading Volume by Sentiment")
plt.yscale('log')   # if volumes vary a lot
plt.savefig('outputs/volume_by_sentiment.png', bbox_inches='tight', dpi=150)
plt.show()
sns.boxplot(x='sentiment', y='avg_pnl', data=merged)
plt.title("Average Profit/Loss by Sentiment")
plt.savefig('outputs/pnl_by_sentiment.png', bbox_inches='tight', dpi=150)
plt.show()
plt.figure(figsize=(12,5))
sns.lineplot(x='trade_date', y='total_volume_usd', data=merged, label='Daily Volume')
sns.lineplot(x='trade_date', y='avg_pnl', data=merged, label='Avg PnL')
plt.legend()
plt.title("Daily Trading Trends")
plt.savefig('outputs/daily_trends.png', bbox_inches='tight', dpi=150)
plt.show()
corr_cols = ['num_trades', 'total_volume_usd', 'avg_price', 'avg_pnl', 'avg_leverage']
corr = merged[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig('outputs/correlation_matrix.png', bbox_inches='tight', dpi=150)
plt.show()
summary = merged.groupby('sentiment')[['num_trades', 'total_volume_usd', 'avg_pnl']].mean()
summary.to_csv("csv_files/sentiment_summary.csv")
print("Summary saved → csv_files/sentiment_summary.csv")
print(summary)
from scipy import stats

fear = merged[merged['sentiment']=='Fear']['avg_pnl'].dropna()
greed = merged[merged['sentiment']=='Greed']['avg_pnl'].dropna()

t_stat, p_val = stats.ttest_ind(fear, greed, equal_var=False)
print(f"T-test avg PnL Fear vs Greed: t={t_stat:.3f}, p={p_val:.3f}")
plt.close()





