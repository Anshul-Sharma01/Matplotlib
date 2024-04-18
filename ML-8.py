import pandas as pd
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates

plt.style.use('seaborn')

data = pd.read_csv('data4.csv')

data["Data"]=pd.to_datetime(data["Data"])
data.sort_values("Date",inplace=True)

price_date = data['Date']
price_close = data['Close']

plt.title('Bitcoin Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')

plt.plot_date(price_date,price_close,linestyle="solid")

plt.gcf().autofmt_xdate()
plt.tight_layout()

plt.show()