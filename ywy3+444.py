from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple, Optional, Any
import jsonpickle
import numpy as np
import pandas as pd
import math
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from collections import defaultdict
from typing import Dict, List

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
    PICNIC_BASKET1 = 'PICNIC_BASKET1'
    PICNIC_BASKET2 = 'PICNIC_BASKET2'
    CROISSANTS = 'CROISSANTS'
    JAMS = 'JAMS'
    DJEMBES = 'DJEMBES'

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
    Product.CROISSANTS: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 100,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
        "soft_position_limit": 200,
    },
    Product.DJEMBES: {
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
    Product.SPREAD: {
        "default_spread_mean": 28.52,
        "default_spread_std": 76.07966,
        "spread_std_window": 75,
        "zscore_threshold": 9,
        "target_position": 41,
    },
    "update_interval": 1000,  # 更新固定均值的周期

}

BASKET_WEIGHTS = {
    Product.PICNIC_BASKET2: 1,
    Product.CROISSANTS: 2,
    Product.JAMS: 1,
    Product.DJEMBES: 1,}

    #########################################
    # 归一化函数：合并同产品、价格差异在一定容差内的订单
def normalize_orders(orders: Dict[str, List[Order]], tolerance: float = 0.0005) -> Dict[str, List[Order]]:
    grouped = defaultdict(list)
    for prod, order_list in orders.items():
        order_list.sort(key=lambda o: o.price)
        merged = []
        for order in order_list:
            if not merged:
                merged.append(order)
            else:
                last = merged[-1]
                if abs(order.price - last.price) / last.price < tolerance:
                    merged[-1] = Order(prod, last.price, last.quantity + order.quantity)
                else:
                    merged.append(order)
        grouped[prod] = [o for o in merged if o.quantity != 0]
    
    return dict(grouped)
    #########################################

class Trader:
    def __init__(self, params=None):
        self.resin_prices = []
        self.resin_vwap = []
        self.kelp_prices = [] 
        self.squid_prices = []
        self.pnl_history = []
        self.band_history = []
        self.trade_log = []
        self.last_trade_price = None
        self.params = params if params else PARAMS
        self.LIMIT = {
            Product.RESIN: 50, 
            Product.KELP: 50,
            Product.SQUIDINK: 50,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60
        }
        self.recent_prices = {}

    # --- Technical Indicators ---
    def zscore(self, prices: List[float], window: int = 14) -> float:
        if len(prices) < window:
            return 0.0
        mean = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        return (prices[-1] - mean) / std if std > 0 else 0

    def sma(self, prices: List[float], window: int = 10) -> float:
        return np.mean(prices[-window:]) if len(prices) >= window else prices[-1] if prices else 0

    def slope(self, prices: List[float], window: int = 10) -> float:
        if len(prices) < window:
            return 0.0
        x = np.arange(window)
        y = np.array(prices[-window:])
        return np.polyfit(x, y, 1)[0]

    def bias(self, prices: List[float], window: int = 10) -> float:
        if not prices:
            return 0.0
        prices_series = pd.Series(prices)
        ma = prices_series.rolling(window=window, min_periods=1).mean()
        bias = (prices_series / ma - 1).rolling(window=window, min_periods=1).mean()
        return bias.iloc[-1] if not bias.empty else 0.0

    def detect_v_shape(self, prices: List[float], threshold: float = 0.2) -> bool:
        if len(prices) < 7:
            return False
        pre3 = prices[-7:-4]
        mid = prices[-4:-2]
        post3 = prices[-2:]
        return (np.mean(mid) < np.mean(pre3) * (1 - threshold)) and (np.mean(post3) > np.mean(mid) * (1 + threshold))


    # --- Order Management ---
    def take_best_orders(self, product: str, fair_value: float, take_width: float,
                        orders: List[Order], order_depth: OrderDepth, position: int,
                        buy_order_volume: int, sell_order_volume: int,
                        prevent_adverse: bool = False, adverse_volume: int = 0) -> Tuple[int, int]:
        position_limit = self.LIMIT.get(product, 0)
        
        # Process buy orders
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            
            if (not prevent_adverse or abs(best_ask_amount) <= adverse_volume) and best_ask <= fair_value - take_width:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    # Update order depth (simulated execution)
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        # Process sell orders
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            
            if (not prevent_adverse or abs(best_bid_amount) <= adverse_volume) and best_bid >= fair_value + take_width:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(product, best_bid, -quantity))
                    sell_order_volume += quantity
                    # Update order depth (simulated execution)
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int,
                   position: int, buy_order_volume: int, sell_order_volume: int) -> Tuple[int, int]:
        position_limit = self.LIMIT.get(product, 0)
        
        # Post buy order
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))

        # Post sell order
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))

        return buy_order_volume, sell_order_volume

    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, 
                           position: int, position_limit: int, product: str,
                           buy_order_volume: int, sell_order_volume: int,
                           fair_value: float, width: int) -> Tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = math.floor(fair_value - width)
        fair_for_ask = math.ceil(fair_value + width)

        # Clear long position
        if position_after_take > 0 and fair_for_ask in order_depth.buy_orders:
            clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
            sent_quantity = min(position_limit + position - sell_order_volume, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -sent_quantity))
                sell_order_volume += sent_quantity

        # Clear short position
        if position_after_take < 0 and fair_for_bid in order_depth.sell_orders:
            clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
            sent_quantity = min(position_limit - position - buy_order_volume, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, sent_quantity))
                buy_order_volume += sent_quantity

        return buy_order_volume, sell_order_volume
        
    def clear_position_order2(
        self,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        position_limit: int,
        product: str,
        buy_order_volume: int,
        sell_order_volume: int,
        fair_value: float,
        width: int
    ) -> Tuple[int, int]:
        """
        Improved version of clear_position_order with better error handling and position management
        
        Args:
            orders: List to accumulate new orders
            order_depth: Current market order depth
            position: Current position
            position_limit: Maximum allowed position
            product: Product being traded
            buy_order_volume: Current pending buy volume
            sell_order_volume: Current pending sell volume
            fair_value: Calculated fair value
            width: Price width around fair value to clear positions
            
        Returns:
            Updated buy_order_volume and sell_order_volume
        """
        # Calculate net position after pending orders
        position_after_take = position + buy_order_volume - sell_order_volume
        
        # Calculate clearing price levels
        try:
            fair_for_bid = math.floor(fair_value - width)
            fair_for_ask = math.ceil(fair_value + width)
        except Exception as e:
            logger.print(f"Error calculating clearing prices for {product}: {str(e)}")
            return buy_order_volume, sell_order_volume

        # Calculate remaining buy/sell capacity
        remaining_buy = max(0, position_limit - (position + buy_order_volume))
        remaining_sell = max(0, position_limit + (position - sell_order_volume))

        # Clear long positions if we're net long
        if position_after_take > 0:
            try:
                # Aggregate all buy orders at or above our ask price
                clear_quantity = sum(
                    vol for price, vol in order_depth.buy_orders.items()
                    if price >= fair_for_ask
                )
                clear_quantity = min(clear_quantity, position_after_take)
                
                # Only clear what we have capacity to sell
                sent_quantity = min(remaining_sell, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -sent_quantity))
                    sell_order_volume += sent_quantity
            except Exception as e:
                logger.print(f"Error clearing long position for {product}: {str(e)}")

        # Clear short positions if we're net short
        elif position_after_take < 0:
            try:
                # Aggregate all sell orders at or below our bid price
                clear_quantity = sum(
                    abs(vol) for price, vol in order_depth.sell_orders.items()
                    if price <= fair_for_bid
                )
                clear_quantity = min(clear_quantity, abs(position_after_take))
                
                # Only clear what we have capacity to buy
                sent_quantity = min(remaining_buy, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, sent_quantity))
                    buy_order_volume += sent_quantity
            except Exception as e:
                logger.print(f"Error clearing short position for {product}: {str(e)}")

        return buy_order_volume, sell_order_volume
        
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

    # --- Fair Value Calculation ---
    def general_fair_value(self, product: str, order_depth: OrderDepth, traderObject: dict) -> Optional[float]:
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        try:
            # Get best prices with volume filtering
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            
            # Filter out small orders
            min_volume = self.params.get(product, {}).get("adverse_volume", 10)
            valid_asks = [p for p, vol in order_depth.sell_orders.items() if abs(vol) >= min_volume]
            valid_bids = [p for p, vol in order_depth.buy_orders.items() if abs(vol) >= min_volume]
            
            mm_ask = min(valid_asks) if valid_asks else best_ask
            mm_bid = max(valid_bids) if valid_bids else best_bid
            
            # Calculate mid price
            current_mid = (mm_ask + mm_bid) / 2
            
            # Mean reversion adjustment
            last_price_key = f"{product}_last_price"
            if last_price_key in traderObject:
                last_price = traderObject[last_price_key]
                price_change = (current_mid - last_price) / (last_price + 1e-6)  # Avoid division by zero
                reversion_beta = self.params.get(product, {}).get("reversion_beta", -0.2)
                fair_value = current_mid * (1 + price_change * reversion_beta)
            else:
                fair_value = current_mid
                
            # Update last price
            traderObject[last_price_key] = current_mid
            return fair_value
            
        except Exception as e:
            logger.print(f"Error calculating fair value for {product}: {str(e)}")
            return None

    # --- Order Generation ---
    def generate_product_orders(self, product: str, order_depth: OrderDepth,
                            position: int, position_limit: int,
                                traderObject: dict) -> List[Order]:
        orders = []
        if product not in self.params:
            return orders

        params = self.params[product]
        fair_value = self.general_fair_value(product, order_depth, traderObject)
        if fair_value is None:
            return orders

        # Step 1: Take liquidity orders
        take_orders, buy_vol, sell_vol = [], 0, 0
        if params.get("take_width", 0) > 0:
            buy_vol, sell_vol = self.take_best_orders(
                product, fair_value, params["take_width"],
                take_orders, order_depth, position,
                buy_vol, sell_vol,
                params.get("prevent_adverse", False),
                params.get("adverse_volume", 0)
            )
        orders += take_orders

        # Step 2: Clear position orders
        clear_orders = []
        if params.get("clear_width", 0) > 0:
            buy_vol, sell_vol = self.clear_position_order(
                clear_orders, order_depth,
                position, position_limit, product,
                buy_vol, sell_vol,
                fair_value, params["clear_width"]
            )
        orders += clear_orders

        # Step 3: Market making orders
        make_orders = []
        if (params.get("disregard_edge", 0) > 0 or
            params.get("default_edge", 0) > 0):
            # If you have a method like `self.market_make` or similar, use that instead
            buy_vol, sell_vol = self.market_make(
                product, make_orders,
                fair_value - params["default_edge"], fair_value + params["default_edge"],
                position, buy_vol, sell_vol
            )
        orders += make_orders

        return orders
        
    def get_swmid(self, order_depth: OrderDepth) -> float:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_vol = abs(order_depth.buy_orders[best_bid])
            best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (best_bid_vol + best_ask_vol)
        
        # 构造 synthetic basket：利用 PICNIC_BASKET2, CROISSANTS, JAMS, DJEMBES 的报价及权重
    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth]) -> OrderDepth:
            basket2 = BASKET_WEIGHTS[Product.PICNIC_BASKET2]
            croissants = BASKET_WEIGHTS[Product.CROISSANTS]
            jams = BASKET_WEIGHTS[Product.JAMS]
            djembes = BASKET_WEIGHTS[Product.DJEMBES]
            synthetic_od = OrderDepth()
            
            cb_bid = max(order_depths[Product.CROISSANTS].buy_orders.keys()) if order_depths[Product.CROISSANTS].buy_orders else 0
            cb_ask = min(order_depths[Product.CROISSANTS].sell_orders.keys()) if order_depths[Product.CROISSANTS].sell_orders else float("inf")
            ja_bid = max(order_depths[Product.JAMS].buy_orders.keys()) if order_depths[Product.JAMS].buy_orders else 0
            ja_ask = min(order_depths[Product.JAMS].sell_orders.keys()) if order_depths[Product.JAMS].sell_orders else float("inf")
            dj_bid = max(order_depths[Product.DJEMBES].buy_orders.keys()) if order_depths[Product.DJEMBES].buy_orders else 0
            dj_ask = min(order_depths[Product.DJEMBES].sell_orders.keys()) if order_depths[Product.DJEMBES].sell_orders else float("inf")
            b2_bid = min(order_depths[Product.PICNIC_BASKET2].buy_orders.keys()) if order_depths[Product.PICNIC_BASKET2].buy_orders else 0
            b2_ask = min(order_depths[Product.PICNIC_BASKET2].sell_orders.keys()) if order_depths[Product.PICNIC_BASKET2].sell_orders else float("inf")
            
            implied_bid = cb_bid * croissants + ja_bid * jams + dj_bid * djembes + b2_bid * basket2
            implied_ask = cb_ask * croissants + ja_ask * jams + dj_ask * djembes + b2_ask * basket2
            
            if implied_bid > 0:
                q_cb = order_depths[Product.CROISSANTS].buy_orders.get(cb_bid, 0) // croissants
                q_ja = order_depths[Product.JAMS].buy_orders.get(ja_bid, 0) // jams
                q_dj = order_depths[Product.DJEMBES].buy_orders.get(dj_bid, 0) // djembes
                q_b2 = order_depths[Product.PICNIC_BASKET2].buy_orders.get(b2_bid, 0) // basket2
                implied_bid_volume = min(q_cb, q_ja, q_dj, q_b2)
                synthetic_od.buy_orders[implied_bid] = implied_bid_volume
            if implied_ask < float("inf"):
                q_cb = -order_depths[Product.CROISSANTS].sell_orders.get(cb_ask, 0) // croissants
                q_ja = -order_depths[Product.JAMS].sell_orders.get(ja_ask, 0) // jams
                q_dj = -order_depths[Product.DJEMBES].sell_orders.get(dj_ask, 0) // djembes
                q_b2 = -order_depths[Product.PICNIC_BASKET2].sell_orders.get(b2_ask, 0) // basket2
                implied_ask_volume = min(q_cb, q_ja, q_dj, q_b2)
                synthetic_od.sell_orders[implied_ask] = -implied_ask_volume
            
        return synthetic_od

    def convert_synthetic_basket_orders(self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]) -> Dict[str, List[Order]]:
            component_orders = {
                Product.CROISSANTS: [],
                Product.JAMS: [],
                Product.DJEMBES: [],
                Product.PICNIC_BASKET2: [],
            }
            synthetic_od = self.get_synthetic_basket_order_depth(order_depths)
            best_bid = max(synthetic_od.buy_orders.keys()) if synthetic_od.buy_orders else 0
            best_ask = min(synthetic_od.sell_orders.keys()) if synthetic_od.sell_orders else float("inf")
            for order in synthetic_orders:
                price = order.price
                qty = order.quantity
                if qty > 0 and price >= best_ask:
                    c_price = min(order_depths[Product.CROISSANTS].sell_orders.keys())
                    j_price = min(order_depths[Product.JAMS].sell_orders.keys())
                    dj_price = min(order_depths[Product.DJEMBES].sell_orders.keys())
                    b2_price = min(order_depths[Product.PICNIC_BASKET2].sell_orders.keys())
                elif qty < 0 and price <= best_bid:
                    c_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                    j_price = max(order_depths[Product.JAMS].buy_orders.keys())
                    dj_price = max(order_depths[Product.DJEMBES].buy_orders.keys())
                    b2_price = max(order_depths[Product.PICNIC_BASKET2].buy_orders.keys())
                else:
                    continue
                component_orders[Product.CROISSANTS].append(Order(Product.CROISSANTS, c_price, qty * BASKET_WEIGHTS[Product.CROISSANTS]))
                component_orders[Product.JAMS].append(Order(Product.JAMS, j_price, qty * BASKET_WEIGHTS[Product.JAMS]))
                component_orders[Product.DJEMBES].append(Order(Product.DJEMBES, dj_price, qty * BASKET_WEIGHTS[Product.DJEMBES]))
                component_orders[Product.PICNIC_BASKET2].append(Order(Product.PICNIC_BASKET2, b2_price, qty * BASKET_WEIGHTS[Product.PICNIC_BASKET2]))
        return component_orders

    def execute_spread_orders(self, target_position: int, basket_position: int, order_depths: Dict[str, OrderDepth]):
        if target_position == basket_position:
            return None
        target_qty = abs(target_position - basket_position)
        basket_od = order_depths[Product.PICNIC_BASKET1]
        synthetic_od = self.get_synthetic_basket_order_depth(order_depths)
        if target_position > basket_position:
            b_ask = min(basket_od.sell_orders.keys())
            b_ask_vol = abs(basket_od.sell_orders[b_ask])
            s_bid = max(synthetic_od.buy_orders.keys())
            s_bid_vol = abs(synthetic_od.buy_orders[s_bid])
            order_vol = min(b_ask_vol, s_bid_vol, target_qty)
            basket_orders = [Order(Product.PICNIC_BASKET1, b_ask, order_vol)]
            synthetic_orders = [Order(Product.SYNTHETIC, s_bid, -order_vol)]
            agg_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            agg_orders[Product.PICNIC_BASKET1] = basket_orders
            return agg_orders
        else:
            b_bid = max(basket_od.buy_orders.keys())
            b_bid_vol = abs(basket_od.buy_orders[b_bid])
            s_ask = min(synthetic_od.sell_orders.keys())
            s_ask_vol = abs(synthetic_od.sell_orders[s_ask])
            order_vol = min(b_bid_vol, s_ask_vol, target_qty)
            basket_orders = [Order(Product.PICNIC_BASKET1, b_bid, -order_vol)]
            synthetic_orders = [Order(Product.SYNTHETIC, s_ask, order_vol)]
            agg_orders = self.convert_synthetic_basket_orders(synthetic_orders, order_depths)
            agg_orders[Product.PICNIC_BASKET1] = basket_orders
            return agg_orders

    # 在 spread_orders 中增加固定均值更新逻辑：当数据充足时更新，
    # 否则保持用默认值
    def spread_orders(self, order_depths: Dict[str, OrderDepth], product: Product, basket_position: int, spread_data: Dict[str, Any], state: TradingState):
        if Product.PICNIC_BASKET1 not in order_depths:
            return None
        basket_od = order_depths[Product.PICNIC_BASKET1]
        synthetic_od = self.get_synthetic_basket_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_od)
        synthetic_swmid = self.get_swmid(synthetic_od)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)
        
        # 如果数据不足时，直接采用默认的 fixed_mean，不做更新
        window = self.params[Product.SPREAD]["spread_std_window"]
        if len(spread_data["spread_history"]) < window:
            fixed_mean = self.params[Product.SPREAD]["default_spread_mean"]
        else:
            # 每隔 update_interval 更新一次固定均值
            update_interval = self.params.get("update_interval", 1000)
            if "last_mean_update" not in spread_data:
                spread_data["last_mean_update"] = state.timestamp
                fixed_mean = self.params[Product.SPREAD]["default_spread_mean"]
            elif state.timestamp - spread_data["last_mean_update"] >= update_interval:
                new_fixed_mean = statistics.median(spread_data["spread_history"])
                fixed_mean = new_fixed_mean
                spread_data["last_mean_update"] = state.timestamp
            else:
                fixed_mean = spread_data.get("fixed_mean", self.params[Product.SPREAD]["default_spread_mean"])
            # 存储固定均值，便于下次调用
            spread_data["fixed_mean"] = fixed_mean

        # 保持历史数据长度
        if len(spread_data["spread_history"]) > window:
            spread_data["spread_history"].pop(0)
        
        # 计算鲁棒标准差（MAD）
        median_spread = statistics.median(spread_data["spread_history"])
        mad = statistics.median([abs(x - median_spread) for x in spread_data["spread_history"]])
        robust_std = 1.4826 * mad if mad > 0 else 1e-6
        
        zscore = (spread - fixed_mean) / robust_std
        
        if zscore >= self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(-self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        if zscore <= -self.params[Product.SPREAD]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD]["target_position"]:
                return self.execute_spread_orders(self.params[Product.SPREAD]["target_position"], basket_position, order_depths)
        spread_data["prev_zscore"] = zscore
        return None

    # 对冲篮子1未对冲仓位的组合对冲
    def hedge_basket1_combo(self, state: TradingState) -> Dict[str, List[Order]]:
        orders = {}
        unhedged = state.position.get(Product.PICNIC_BASKET1, 0)
        if unhedged == 0:
            return orders
        direction = -1 if unhedged > 0 else 1
        combo = {
            Product.PICNIC_BASKET2: 1,
            Product.CROISSANTS: 2,
            Product.JAMS: 1,
            Product.DJEMBES: 1,
        }
        for prod, multiplier in combo.items():
            if prod not in state.order_depths:
                continue
            od = state.order_depths[prod]
            if not od.buy_orders or not od.sell_orders:
                continue
            if direction == -1:
                price = min(od.sell_orders.keys())
                avail = abs(od.sell_orders.get(price, 0))
            else:
                price = max(od.buy_orders.keys())
                avail = abs(od.buy_orders.get(price, 0))
            desired = multiplier * abs(unhedged)
            exec_qty = min(desired, avail)
            if exec_qty > 0:
                orders.setdefault(prod, []).append(Order(prod, price, direction * exec_qty))
        return orders

    def aggregate_orders(self, order_dicts: List[Dict[str, List[Order]]]) -> Dict[str, List[Order]]:
        combined = defaultdict(list)
        for od in order_dicts:
            for prod, ol in od.items():
                combined[prod].extend(ol)
        return normalize_orders(combined, tolerance=0.0005)


    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}
        result = {}
        
        # Process each product
        for product in state.order_depths:
            position = state.position.get(product, 0)
            position_limit = self.LIMIT.get(product, 0)
            
            if product == "RAINFOREST_RESIN":
                # Special handling for resin
                result[product] = self.resin_orders(
                    state.order_depths[product],
                    width=2,
                    timespan = 10,
                    resin_take_width=1,
                    position=position,
                    position_limit=position_limit
                )
            else:
                # General handling for other products
                result[product] = self.generate_product_orders(
                    product,
                    state.order_depths[product],
                    position,
                    position_limit,
                    traderObject
                )
        
        # Serialize trader data
        traderData = jsonpickle.encode({
            **traderObject,
            "resin_prices": self.resin_prices[-100:],
            "kelp_prices": self.kelp_prices[-100:],
            "pnl_history": self.pnl_history[-100:],
            "band_history": self.band_history[-100:],
            "trade_log": self.trade_log[-100:]
        })
        
        return result, 1, traderData
