'''
Author: Benjamin Bradford

MIT License

Copyright (c) 2017 backtest-rookies.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import backtrader as bt

import datetime
import backtrader.indicators as btind
import backtrader.filters as btfilters
#from bbdelta import bbDelta
import bbdelta

class firstStrategy(bt.Strategy):

    def __init__(self):

        #        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=21)

        # Inputs:
        #    option_type = "p" or "c"
        #    fs          = price of underlying
        #    x           = strike
        #    t           = time to expiration
        #    v           = implied volatility
        #    r           = risk free rate
        #    q           = dividend payment
        #    b           = cost of carry

        kwargs = dict(
            optstring=soptstring,
            riskfreerate=0.0142,
            dividends=0.0143, )
            #greek=1, )
        #          print "testimpvol - imp vol is: ", bbImpVol(**kwargs)

        print(len(data0.array))
        print(len(data1.array))
        print "data1 arry len: ", len(data1.array)
        # print "data_filled: ", len(data_filled.array)

        #        print (type(bbImpVol(**kwargs)))
        bbdelta._gamma(*self.datas, **kwargs)

    def next(self):
        pass

        #print(len(self.ivline.array))

# TICKER-YYYYMMDD-EXCHANGE-CURRENCY-STRIKE-RIGHT # OPT
soptstring = 'AAPL-20180119-SMART-USD-170-PUT'                   # e.g. 'AAPL-20180119-SMART-USD-170-CALL'
sunderlying = 'AAPL-STK-SMART-USD'

# Variable for our starting cash
startcash = 10000

# Create an instance of cerebro`
cerebro = bt.Cerebro()

# Add our strategy
cerebro.addstrategy(firstStrategy)

# Get Apple data from Yahoo Finance.
'''
data0 = bt.feeds.YahooFinanceData(
     dataname='AAPL',
     fromdate=datetime.date(2017, 9, 1),
     todate=datetime.date(2017, 12, 8),
     buffered=True
)


'''
# Get underlying (e.g. AAPL) data from IB.
stockkwargs = dict(
    dataname=sunderlying,
    historical=True,
    fromdate=datetime.date(2017,11,12),  # get data from
    todate=datetime.date(2017,12,12),
    timeframe=bt.TimeFrame.Minutes,
    compression=1,
    sessionstart=datetime.time(8, 30, 0),
    sessionend=datetime.time(15, 15, 0),
)

ibstore = bt.stores.IBStore(host='127.0.0.1', port=4001, clientId=31)
data0 = ibstore.getdata(**stockkwargs)

# Get Option data from IB.
stockkwargs = dict(
    dataname=soptstring,
    historical=True,
    fromdate=datetime.date(2017,11,12),  # get data from
    todate=datetime.date(2017, 12, 12),
    timeframe=bt.TimeFrame.Minutes,
    compression=1,
    sessionstart=datetime.time(8, 30, 0),
    sessionend=datetime.time(15, 15, 0),
)

ibstore = bt.stores.IBStore(host='127.0.0.1', port=4001, clientId=30)
data1 = ibstore.getdata(**stockkwargs)


# Add the data to Cerebro

data0.resample(timeframe=bt.TimeFrame.Minutes, compression=5)
#data1.resample(timeframe=bt.TimeFrame.Minutes, compression=5)
# Some issues with SessionFiller:
# data_filled = data1.clone(filters=[btfilters.SessionFiller], timeframe=bt.TimeFrame.Minutes, compression=60)
# print "data_filled: ", len(data_filled.array)

cerebro.adddata(data0)
cerebro.adddata(data1)
# cerebro.adddata(data_filled)

# Set our desired cash start
cerebro.broker.setcash(startcash)

# Run over everything
cerebro.run()        # preload=False)  #runonce=False)

# print "cerebro run completed"
# print("data0 arry len: ",len(data0.array))
# print("data1 arry len: ", len(data1.array))
# print(len(data_filled.array))

# Get final portfolio Value
portvalue = cerebro.broker.getvalue()
pnl = portvalue - startcash

# Print out the final result
print('Final Portfolio Value: ${}'.format(portvalue))
print('P/L: ${}'.format(pnl))

# Finally plot the end results
cerebro.plot(style='candlestick')
