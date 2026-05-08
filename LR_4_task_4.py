import numpy as np
from sklearn import covariance, cluster
import yfinance as yf

# 1. Словник компаній (замість JSON-файлу)
company_symbols_map = {
    "TOT": "Total",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "AAPL": "Apple",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "GS": "Goldman Sachs",
    "JPM": "JPMorgan Chase",
    "GE": "General Electric",
    "F": "Ford",
    "GM": "General Motors",
    "INTC": "Intel",
    "HPQ": "HP",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "ADBE": "Adobe",
    "YHOO": "Yahoo",
    "EBAY": "eBay",
    "AIG": "AIG",
    "AXP": "American Express",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "C": "Citigroup"
}

# Отримуємо символи та назви компаній
symbols = list(company_symbols_map.keys())
names = np.array(list(company_symbols_map.values()))

# 2. Завантаження даних з Yahoo Finance
start_date = "2023-01-01"
end_date = "2024-01-01"

print(f"Завантаження даних для {len(symbols)} компаній...")

data = yf.download(
    symbols,
    start=start_date,
    end=end_date,
    progress=False
)

# Перевірка
if data.empty:
    print("Не вдалося отримати дані.")
    exit()

# 3. Отримання цін відкриття та закриття
quotes_open = data["Open"].T
quotes_close = data["Close"].T

# 4. Обчислення денних змін
variation = (quotes_close - quotes_open).values

# Видаляємо компанії з пропущеними значеннями
mask = ~np.isnan(variation).any(axis=1)
variation = variation[mask]
names = names[mask]

# 5. Нормалізація
X = variation.copy()
X /= X.std(axis=1)[:, np.newaxis]

# 6. Побудова Graphical Lasso
print("Навчання моделі...")
edge_model = covariance.GraphicalLassoCV()
edge_model.fit(X.T)

# 7. Кластеризація
print("Кластеризація...")
_, labels = cluster.affinity_propagation(
    edge_model.covariance_,
    random_state=0
)

num_labels = labels.max()

# 8. Вивід результатів
print("\n=== Результати кластеризації ===")

for i in range(num_labels + 1):
    cluster_names = names[labels == i]

    if len(cluster_names) > 0:
        print(f"Група {i+1}:")
        for company in cluster_names:
            print(f" - {company}")
        print()