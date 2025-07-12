from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import pandas as pd
import math

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Product:
    RESIN = "RESIN"
    KELP = "KELP"
    SQUIDINK = "SQUIDINK"


PARAMS = {
    Product.RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 25,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "soft_position_limit": 25,
    },
    Product.SQUIDINK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.25,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "soft_position_limit": 25,
    },
}

class Trader:
    def __init__(self, params=None):
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = [] 
        self.squid_prices = []
        self.pnl_history = []
        self.band_history = []  # 🆕 用于滑窗回测band
        self.trade_log = []  # 🆕 新增：交易记录
        self.last_trade_price = None
        if params is None:
            params = PARAMS
        self.params = params
        # Updated position limits for each product are now 50
        self.LIMIT = {Product.RESIN: 50, Product.KELP: 50, Product.SQUIDINK: 50}
        self.recent_prices = {}
    
    # --- 技术指标函数 ---

    def zscore(self, prices: List[float], window: int = 14) -> float:
        if len(prices) < window:
            return 0.0
        mean = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        return (prices[-1] - mean) / std if std > 0 else 0

    def sma(self, prices: List[float], window: int = 10) -> float:
        if len(prices) < window:
            return prices[-1]
        return np.mean(prices[-window:])

    def slope(self, prices: List[float], window: int = 10) -> float:
        if len(prices) < window:
            return 0.0
        return np.polyfit(range(window), prices[-window:], 1)[0]

    def bias(self, prices: List[float], window: int = 10) -> float:
        # Convert the list to a pandas Series
        prices_series = pd.Series(prices)

        # Calculate the rolling mean
        ma = prices_series.rolling(window=window, min_periods=1).mean()

        # Calculate the bias (current price / moving average - 1)
        bias = prices_series / ma - 1

        # Calculate the rolling average of bias over the window
        alpha = bias.rolling(window=window, min_periods=1).mean()

        # Return the final bias value (use the latest value in the series)
        return alpha.iloc[-1]

    def mtm_std(self, prices: List[float], window: int = 10) -> float:
        # Convert the list to a pandas Series
        prices_series = pd.Series(prices)

        # Calculate the momentum (percentage change over 'window' periods)
        mtm = prices_series.pct_change(window)

        # Calculate the rolling mean and standard deviation of momentum
        mtm_mean = mtm.rolling(window=window, min_periods=1).mean()
        mtm_std = mtm.rolling(window=window, min_periods=1).std()

        # Calculate the alpha as the product of mean momentum and standard deviation
        alpha = mtm_mean * mtm_std

        # Return the final alpha value (use the latest value in the series)
        return alpha.iloc[-1]

    def detect_v_shape(self, prices: List[float], threshold: float = 0.2) -> bool:
        if len(prices) < 7:
            return False
        pre3 = prices[-7:-4]
        mid = prices[-4:-2]
        post3 = prices[-2:]
        return np.mean(mid) < np.mean(pre3) * (1 - threshold) and np.mean(post3) > np.mean(mid) * (1 + threshold)

    # --- 动态band优化（滑窗回测） ---
    def best_band_coef(self) -> float:
        coefs = [1.0, 1.2, 1.5, 1.8, 2.0]
        best_coef = 1.5
        best_score = -float('inf')
        for c in coefs:
            pnl = 0
            for i in range(30, len(self.squid_prices)):
                window = self.squid_prices[i-20:i]
                mid = self.squid_prices[i]
                sma_val = np.mean(window)
                std = np.std(window)
                upper, lower = sma_val + c*std, sma_val - c*std
                if mid < lower:
                    pnl += window[-1] - mid
                elif mid > upper:
                    pnl += mid - window[-1]
            if pnl > best_score:
                best_score = pnl
                best_coef = c
        return best_coef
    
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume
    
    def clear_position_order2(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)
        # fair_for_ask = fair_for_bid = fair

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                # clear_quantity = position_after_take
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                # clear_quantity = abs(position_after_take)
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int,
                              position_limit: int, product: str, buy_order_volume: int,
                              sell_order_volume: int, fair_value: float) -> List[int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
            sell_order_volume += abs(sent_quantity)

        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
            buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def squid_orders(self, order_depth: OrderDepth, position: int, position_limit: int) -> List[Order]:
        orders = []
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders)
        best_ask = min(order_depth.sell_orders)
        mid_price = (best_bid + best_ask) / 2
        self.squid_prices.append(mid_price)
        if len(self.squid_prices) > 100:
            self.squid_prices = self.squid_prices[-100:]

        if len(self.squid_prices) < 30:
            return orders

        band_coef = self.best_band_coef()
        self.band_history.append(band_coef)

        sma_val = self.sma(self.squid_prices, 20)
        std_val = np.std(self.squid_prices[-20:])
        upper_band = sma_val + band_coef * std_val
        lower_band = sma_val - band_coef * std_val
        #slope_val = self.slope(self.squid_prices, 10)
        bias_val = self.bias(self.squid_prices, 10)
        mtm_std_val = self.mtm_std(self.squid_prices, 10)


        bid_volume = order_depth.buy_orders[best_bid]
        ask_volume = -order_depth.sell_orders[best_ask]

        # 🧠 PnL adaptive trade size
        recent_pnl = sum(self.pnl_history[-20:]) if len(self.pnl_history) >= 20 else 0
        if recent_pnl < -3000:
            max_trade_size = 5
        elif recent_pnl > 2000:
            max_trade_size = 20
        else:
            max_trade_size = 10

        # ✅ pattern: V-shape bottom bounce
        if mtm_std_val < 0 and position < position_limit and bias_val > 0:
            # Only trade if the market is volatile (negative momentum) and there is room to add to the position
            # Also, ensure the market bias is positive (uptrend or positive outlook)
            if self.detect_v_shape(self.squid_prices):  # Additional pattern detection check
                quantity = min(position_limit - position, ask_volume,
                               max_trade_size)  # Trade only within position limits
                if quantity > 0:  # Only place an order if there is an actual quantity to trade
                    orders.append(Order("SQUID_INK", best_ask, quantity))  # Place buy order at the best ask price
                    self.pnl_history.append((self.last_trade_price or mid_price) - best_ask)  # Record PnL
                    self.last_trade_price = best_ask  # Update last trade price to the current best ask

        elif mtm_std_val > 0 and position > -position_limit and bias_val < 0:
            # Only trade if the market is volatile (positive momentum) and there is room to reduce the position
            # Also, ensure the market bias is negative (downtrend or negative outlook)
            quantity = min(position + position_limit, bid_volume, max_trade_size)  # Trade only within position limits
            if quantity > 0:  # Only place an order if there is an actual quantity to trade
                orders.append(Order("SQUID_INK", best_bid, -quantity))  # Place sell order at the best bid price
                self.pnl_history.append(best_bid - (self.last_trade_price or mid_price))  # Record PnL
                self.last_trade_price = best_bid  # Update last trade price to the current best bid

        return orders
    
    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("KELP_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["KELP_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("KELP_last_price", None) != None:
                last_price = traderObject["KELP_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["KELP_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int): # type: ignore
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order3(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def kelp_orders(self, order_depth: OrderDepth, position: int, position_limit: int, window_size: int = 10, threshold: float = 1.5) -> List[Order]:
    
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # 【新增代码】动态计算当前中间价
        if order_depth.sell_orders and order_depth.buy_orders:
            current_mid = (min(order_depth.sell_orders.keys()) + max(order_depth.buy_orders.keys())) / 2
        else:
            current_mid = fair_value  # 如果订单簿数据不足，使用传入的静态值

        # 将当前中间价添加到 kelp_prices 历史记录中
        self.kelp_prices.append(current_mid)
        # 限制历史记录长度（例如保留最近10个数据）
        if len(self.kelp_prices) > 10:
            self.kelp_prices = self.kelp_prices[-10:]

        # 如果有足够的历史数据，计算移动平均和标准差
        if len(self.kelp_prices) >= 5:
            mean_price = np.mean(self.kelp_prices)
            std_price = np.std(self.kelp_prices)
            # 使用移动平均作为动态fair_value
            dynamic_fair_value = mean_price
            # （可选）可以根据标准差来设定交易阈值，例如：
            # threshold = 1.0 * std_price  
        else:
            dynamic_fair_value = fair_value
        # 将动态计算得到的 fair_value 替换原来的值
        fair_value = dynamic_fair_value

        # 继续原有逻辑：
        baaf = min([price for price in order_depth.sell_orders.keys() if price > fair_value + 1])
        bbbf = max([price for price in order_depth.buy_orders.keys() if price < fair_value - 1])

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)  # 最大购买量
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)  # 最大可卖数量
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -1 * quantity))
                    sell_order_volume += quantity
        
        buy_order_volume, sell_order_volume = self.clear_position_order(orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value, 1)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", bbbf + 1, buy_quantity))  # 补充买单

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", baaf - 1, -sell_quantity))  # 补充卖单

        return orders

    def clear_position_order3(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int, product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float, width: int) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = round(fair_value)
        fair_for_bid = math.floor(fair_value)
        fair_for_ask = math.ceil(fair_value)

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
    
        return buy_order_volume, sell_order_volume
    
    def resin_fair_value(self, order_depth: OrderDepth, method = "mid_price", min_vol = 0) -> float:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if len([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol]) ==0 or len([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol]) ==0:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2
                return mid_price
            else:   
                best_ask = min([price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max([price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = (best_ask + best_bid) / 2
            return mid_price

    def resin_orders(self, order_depth: OrderDepth, timespan:int, width: float, resin_take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []

        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:    

            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else best_ask
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else best_bid
            
            mmmid_price = (mm_ask + mm_bid) / 2    
            self.resin_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
            self.resin_vwap.append({"vol": volume, "vwap": vwap})
            
            if len(self.resin_vwap) > timespan:
                self.resin_vwap.pop(0)
            
            if len(self.resin_prices) > timespan:
                self.resin_prices.pop(0)
        
            fair_value = sum([x["vwap"]*x['vol'] for x in self.resin_vwap]) / sum([x['vol'] for x in self.resin_vwap])
            
            fair_value = mmmid_price

            if best_ask <= fair_value - resin_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + resin_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("RAINFOREST_RESIN", best_bid, -1 * quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order2(orders, order_depth, position, position_limit, "RAINFOREST_RESIN", buy_order_volume, sell_order_volume, fair_value, 2)
            
            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if len(aaf) > 0 else fair_value + 2
            bbbf = max(bbf) if len(bbf) > 0 else fair_value - 2
           
            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))  # Buy order

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))  # Sell order

        return orders

    def run(self, state: TradingState):

        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # ----------------------------
        # RAINFOREST_RESIN 参数设置
        resin_position_limit = 50         # 持仓上限
        resin_width = 2                   # （暂未深入使用，可根据策略进一步调整）
        # 对于 Resin，我们依旧采用原有的 resin_orders 策略（利用 VWAP 和中间价计算）
    
        # ----------------------------
        # KELP 参数设置
        kelp_position_limit = 50          # Kelp 的持仓上限
        kelp_window_size = 10             # 用于动态计算移动平均的窗口大小（历史数据个数）
        kelp_threshold = 1.5              # 标准差倍数，只有当当前中间价偏离均值超过此倍数时才下单

        KELP_make_width = 3.5
        KELP_take_width = 1
        KELP_position_limit = 50
        KELP_timemspan = 10


        # ----------------------------
        # 处理 RAINFOREST_RESIN
        if "RAINFOREST_RESIN" in state.order_depths:
            resin_order_depth = state.order_depths["RAINFOREST_RESIN"]
            resin_orders = self.resin_orders(
                resin_order_depth,
                kelp_window_size,                  # 此处传入的 timespan（历史窗口）暂用 kelp_window_size 数值
                resin_width,                       # 固定宽度参数
                kelp_threshold,                    # 此处使用与 Kelp 相同的参数（或根据情况独立设置）
                state.position.get("RAINFOREST_RESIN", 0),
                resin_position_limit
            )
            result["RAINFOREST_RESIN"] = resin_orders

        # ----------------------------
        # KELP - Momentum/Trend-Following Strategy
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )
        
        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_order_depth = state.order_depths["SQUID_INK"]
            result["SQUID_INK"] = self.squid_orders(squid_order_depth, squid_position, 50)

        traderData = jsonpickle.encode({
            "resin_prices": self.resin_prices,
            "resin_vwap": self.resin_vwap,
            "kelp_prices": self.kelp_prices,
            "squid_prices": self.squid_prices,
            "pnl_history": self.pnl_history,
            "band_history": self.band_history,
            "trade_log": self.trade_log
        })


        # 设定转换参数，本例中设置为1（具体看交易需求）
        conversions = 1

        logger.flush(state, result, conversions, traderData)
        # 返回结果字典、转换参数和 traderData（状态数据字符串）
        return result, conversions, traderData
