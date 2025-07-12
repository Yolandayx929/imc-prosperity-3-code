import jsonpickle
import numpy as np
import math
import pandas as pd
from typing import List
from datamodel import Order, OrderDepth, TradingState

class Trader:
    def __init__(self):
        self.squid_prices = []      # 历史价格数据
        self.pnl_history = []       # 盈亏记录
        self.band_history = []      # 动态布林带系数历史
        self.trade_log = []         # 交易日志
        self.last_trade_price = None

    def squid_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        orders = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        # 取得订单簿最佳报价
        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2.0
        self.squid_prices.append(mid_price)
        if len(self.squid_prices) > 100:
            self.squid_prices = self.squid_prices[-100:]
        if len(self.squid_prices) < 30:
            return orders

        # 计算动态布林带系数
        band_coef = self.best_band_coef()
        self.band_history.append(band_coef)
        
        # 计算均值和标准差，确定布林带上轨、下轨
        sma_val = self.sma(self.squid_prices, 20)
        std_val = np.std(self.squid_prices[-20:])
        upper_band = sma_val + band_coef * std_val
        lower_band = sma_val - band_coef * std_val

        # 计算其他技术指标
        slope_val = self.slope(self.squid_prices, 10)
        zscore_val = self.zscore(self.squid_prices, 14)
        bias_val = self.bias(self.squid_prices, 10)
        mtm_std_val = self.mtm_std(self.squid_prices, 10)

        # 从订单簿中获取成交量
        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = -order_depth.sell_orders[best_ask]

        # 自适应交易量（根据近期盈亏调整）
        recent_pnl = sum(self.pnl_history[-20:]) if len(self.pnl_history) >= 20 else 0
        if recent_pnl < -3000:
            max_trade_size = 5
        elif recent_pnl > 2000:
            max_trade_size = 25
        else:
            max_trade_size = 10

        # ---- Modified Risk Filter 1: 波动率过滤 ----
        # 计算 vol_ratio = std / sma; 原阈值为 0.05，现调整为 0.08
        vol_ratio = std_val / sma_val if sma_val != 0 else 0
        if vol_ratio > 0.08:
            self.trade_log.append("High volatility condition met; skipping active trades")
            grid_spread = 3
            passive_qty = 5
            if position < position_limit:
                orders.append(Order("SQUID_INK", best_bid - grid_spread, passive_qty))
            if position > -position_limit:
                orders.append(Order("SQUID_INK", best_ask + grid_spread, -passive_qty))
            return orders

        # ---- Modified Risk Filter 2: 亏损过滤 ----
        # 原 threshold -5000，现调整为 -8000，以便减少主动交易跳过频率
        if recent_pnl < -8000:
            self.trade_log.append("Recent pnl too negative; skipping active trades")
            grid_spread = 3
            passive_qty = 5
            if position < position_limit:
                orders.append(Order("SQUID_INK", best_bid - grid_spread, passive_qty))
            if position > -position_limit:
                orders.append(Order("SQUID_INK", best_ask + grid_spread, -passive_qty))
            return orders

        # ---- 主动交易信号（基于 V形反转与 zscore） ----
        # BUY 信号：若价格低于下轨且短期斜率为正
        if mid_price < lower_band and slope_val > 0:
            # Modified：将 zscore 阈值从 -1.5 调整为 -1.2
            if self.detect_v_shape(self.squid_prices, threshold=0.2) or zscore_val < -1.2:
                quantity = min(min(position_limit - position, ask_volume), max_trade_size)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_ask, quantity))
                    # pnl 计算：买入时成本 best_ask，记录盈亏供后续参考
                    pnl_update = (self.last_trade_price if self.last_trade_price is not None else mid_price) - best_ask
                    self.pnl_history.append(pnl_update)
                    self.last_trade_price = best_ask
                    self.trade_log.append(f"BUY at {best_ask}, qty {quantity}, mid {mid_price}")
        # SELL 信号：若价格高于上轨且短期斜率为负
        elif mid_price > upper_band and slope_val < 0:
            if zscore_val > 1.2:
                quantity = min(min(position + position_limit, bid_volume), max_trade_size)
                if quantity > 0:
                    orders.append(Order("SQUID_INK", best_bid, -quantity))
                    pnl_update = best_bid - (self.last_trade_price if self.last_trade_price is not None else mid_price)
                    self.pnl_history.append(pnl_update)
                    self.last_trade_price = best_bid
                    self.trade_log.append(f"SELL at {best_bid}, qty {quantity}, mid {mid_price}")

        # ---- 被动网格订单（补充流动性） ----
        grid_spread = 3
        passive_qty = 5
        if position < position_limit:
            orders.append(Order("SQUID_INK", best_bid - grid_spread, passive_qty))
        if position > -position_limit:
            orders.append(Order("SQUID_INK", best_ask + grid_spread, -passive_qty))

        return orders

    # ---------------- 辅助技术指标 ----------------

    def zscore(self, prices: List[float], window: int = 14) -> float:
        if len(prices) < window:
            return 0.0
        mean = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        return (prices[-1] - mean) / std if std > 0 else 0.0

    def sma(self, prices: List[float], window: int = 10) -> float:
        if len(prices) < window:
            return prices[-1]
        return np.mean(prices[-window:])

    def slope(self, prices: List[float], window: int = 10) -> float:
        if len(prices) < window:
            return 0.0
        return np.polyfit(range(window), prices[-window:], 1)[0]

    def bias(self, prices: List[float], window: int = 10) -> float:
        prices_series = pd.Series(prices)
        ma = prices_series.rolling(window=window, min_periods=1).mean()
        bias = prices_series / ma - 1
        alpha = bias.rolling(window=window, min_periods=1).mean()
        return alpha.iloc[-1]

    def mtm_std(self, prices: List[float], window: int = 10) -> float:
        prices_series = pd.Series(prices)
        mtm = prices_series.pct_change(window)
        mtm_mean = mtm.rolling(window=window, min_periods=1).mean()
        mtm_std = mtm.rolling(window=window, min_periods=1).std()
        alpha = mtm_mean * mtm_std
        return alpha.iloc[-1]

    def detect_v_shape(self, prices: List[float], threshold: float = 0.2) -> bool:
        if len(prices) < 7:
            return False
        pre3 = prices[-7:-4]
        mid = prices[-4:-2]
        post2 = prices[-2:]
        return np.mean(mid) < np.mean(pre3) * (1 - threshold) and np.mean(post2) > np.mean(mid) * (1 + threshold)

    def best_band_coef(self) -> float:
        coefs = [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5]
        best_coef = 1.5
        best_score = -float('inf')
        if len(self.squid_prices) < 30:
            return best_coef
        for c in coefs:
            pnl = 0
            for i in range(30, len(self.squid_prices)):
                window = self.squid_prices[i-20:i]
                mid = self.squid_prices[i]
                sma_val = np.mean(window)
                std = np.std(window)
                upper, lower = sma_val + c * std, sma_val - c * std
                if mid < lower:
                    pnl += window[-1] - mid
                elif mid > upper:
                    pnl += mid - window[-1]
            if pnl > best_score:
                best_score = pnl
                best_coef = c
        return best_coef

    # ---------------- 主入口 ----------------

    def run(self, state: TradingState):
        result = {}
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_order_depth = state.order_depths["SQUID_INK"]
            result["SQUID_INK"] = self.squid_orders(squid_order_depth, squid_position, 100)
        trader_data = jsonpickle.encode({
            "squid_prices": self.squid_prices,
            "pnl_history": self.pnl_history,
            "band_history": self.band_history,
            "trade_log": self.trade_log
        })
        return result, 1, trader_data


#Profit summary:
#Round 1 day -2: 8,858
#Round 1 day -1: 208
#Round 1 day 0: 796
#Total profit: 9,862
