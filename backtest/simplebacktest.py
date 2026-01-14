# -*- coding: utf-8 -*-
import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ====================== 基础指标工具（本地版 metrics） ====================== #

DEFAULT_TRADING_DAYS = 250


def _to_series(x, name: str = "value") -> pd.Series:
    """把输入转为 float64 的 Series。"""
    if isinstance(x, pd.Series):
        return x.astype("float64")
    if isinstance(x, (pd.Index, np.ndarray, list, tuple)):
        return pd.Series(x, dtype="float64", name=name)
    return pd.Series([x], dtype="float64", name=name)


def returns_from_nav(nav) -> pd.Series:
    """净值序列 -> 简单收益序列 r_t = nav_t / nav_{t-1} - 1。"""
    s = _to_series(nav, name="nav").replace([np.inf, -np.inf], np.nan)
    r = s.pct_change()
    return r.dropna()


def annualized_return(
    returns,
    periods_per_year: float = DEFAULT_TRADING_DAYS,
    geometric: bool = True,
) -> float:
    """年化收益率（默认几何口径）。"""
    r = _to_series(returns, name="ret").dropna()
    n = len(r)
    if n == 0:
        return np.nan

    ppy = periods_per_year

    if geometric:
        x = 1.0 + r.values
        if np.all(x > 0.0):
            a = np.log1p(r).mean()   # mean(log(1+r))
            ann = np.expm1(a * ppy)  # exp(a*ppy)-1
        else:
            gross = np.prod(x)
            if gross <= 0.0:
                return np.nan if gross < 0.0 else -1.0
            ann = gross ** (ppy / n) - 1.0
    else:
        ann = float(r.mean() * ppy)

    return float(ann)


def annualized_volatility(
    returns,
    periods_per_year: float = DEFAULT_TRADING_DAYS,
    ddof: int = 0,
) -> float:
    """年化波动率。"""
    r = _to_series(returns, name="ret").dropna()
    if len(r) <= ddof:
        return np.nan
    vol = float(r.std(ddof=ddof))
    return vol * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns,
    rf_annual: float = 0.0,
    periods_per_year: float = DEFAULT_TRADING_DAYS,
    geometric: bool = True,
    ddof: int = 0,
) -> float:
    """Sharpe = (年化收益 - 年化无风险) / 年化波动。"""
    ann_ret = annualized_return(returns, periods_per_year=periods_per_year, geometric=geometric)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year, ddof=ddof)
    if np.isnan(ann_ret) or np.isnan(ann_vol) or ann_vol == 0.0:
        return np.nan
    return (ann_ret - rf_annual) / ann_vol


@dataclass(frozen=True)
class DrawdownStats:
    """最大回撤统计。"""
    max_drawdown: float
    peak_index: Optional[Union[pd.Timestamp, int]]
    trough_index: Optional[Union[pd.Timestamp, int]]
    duration: int


def max_drawdown(nav_or_returns) -> DrawdownStats:
    """最大回撤（自动识别传入的是净值还是收益）。"""
    s = _to_series(nav_or_returns)
    # 粗判一下像不像净值
    if s.min() >= 0 and (abs(s.iloc[0] - 1.0) < 1e-6 or s.iloc[0] > 0.5):
        nav = s
    else:
        r = s.dropna()
        if r.empty:
            return DrawdownStats(np.nan, None, None, 0)
        nav = (1.0 + r).cumprod()

    nav = nav.replace([np.inf, -np.inf], np.nan).dropna()
    if len(nav) == 0:
        return DrawdownStats(np.nan, None, None, 0)

    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    trough_loc = drawdown.idxmin()
    peak_loc = (nav.loc[:trough_loc]).idxmax()

    mdd = float(drawdown.min())
    try:
        duration = int(nav.index.get_loc(trough_loc) - nav.index.get_loc(peak_loc))
    except Exception:
        duration = 0

    return DrawdownStats(mdd, peak_loc, trough_loc, duration)


# ====================== 简单回测类（含 benchmark） ====================== #


class SimpleBackTest:
    """
    简单回测利用 ret_o2c 和 ret_c2o 两列计算策略收益时间序列。
    同时内置基于 ret_df 的全市场等权 benchmark：

        benchmark_pct = ret_df.groupby('date').apply(
            lambda x: ((x['ret_c2o'] + 1) * (x['ret_o2c'] + 1) - 1).mean()
        )

    在 __init__ 中完成：
      - 过滤日期
      - 计算 benchmark 日收益 + 净值
      - 用 pd.merge 对齐 score_df 和 ret_df，形成 merged_df
    """

    def __init__(self, score_df: pd.DataFrame, ret_df: pd.DataFrame, backtest_config: dict):

        self.raw_score_df = score_df.copy()
        self.raw_ret_df = ret_df.copy()

        self.start_date = backtest_config['start_date']
        self.end_date = backtest_config['end_date']

        # ========= 1. 先过滤 ret_df & score_df 到回测区间 ========= #
        ret_df = ret_df.copy()
        ret_df['date'] = pd.to_datetime(ret_df['date'])
        ret_df = ret_df[(ret_df['date'] >= self.start_date) & (ret_df['date'] <= self.end_date)]

        score_df = score_df.copy()
        score_df['date'] = pd.to_datetime(score_df['date'])
        score_df = score_df[(score_df['date'] >= self.start_date) & (score_df['date'] <= self.end_date)]

        # ========= 2. 基于 ret_df 计算全市场 benchmark 日收益 & 净值 ========= #
        # 等权市场组合：每天所有股票的 (1+ret_c2o)*(1+ret_o2c)-1 的截面均值
        daily_bench_ret = ret_df.groupby('date').apply(
            lambda x: ((x['ret_c2o'] + 1.0) * (x['ret_o2c'] + 1.0) - 1.0).mean()
        )
        daily_bench_ret = daily_bench_ret.sort_index()
        self.benchmark_ret_series = daily_bench_ret.astype('float64')
        self.benchmark_nav_series = (1.0 + self.benchmark_ret_series).cumprod()

        # ========= 3. 用 pd.merge 对齐 score_df 到 ret_df（以 ret_df 为主） ========= #
        score_cols = ['date', 'code', 'score']
        score_cols = [c for c in score_cols if c in score_df.columns]

        merged_df = pd.merge(
            ret_df,
            score_df[score_cols],
            on=['date', 'code'],
            how='left',
            sort=False,
        )

        self.merged_df = merged_df
        self.date_list = merged_df['date'].sort_values().unique()

        # ============= 一些配置 ============= #
        self.stock_percent_range = backtest_config.get('stock_percent_range', None)
        if self.stock_percent_range is not None:
            start, end = self.stock_percent_range
            if not (0 <= start < end <= 1):
                raise ValueError("stock_percent_range 必须是 [start, end] 且 0 <= start < end <= 1")

        self.max_holding_days = backtest_config['max_holding_days']
        self.stock_number = backtest_config['stock_number']
        self.buy_cost = backtest_config['buy_cost']
        self.sell_cost = backtest_config['sell_cost']
        self.initial_cash = backtest_config['cash']
        self.cash = backtest_config['cash']
        self.total_value = backtest_config['cash']

        self.trading_style = backtest_config['trading_style']
        self.drop_st = backtest_config['drop_st']

        self.limit_up_buy_constraint = backtest_config.get('limit_up_buy_constraint', True)
        self.limit_dn_buy_constraint = backtest_config.get('limit_dn_buy_constraint', True)
        self.limit_dn_sell_constraint = backtest_config.get('limit_dn_sell_constraint', True)
        self.limit_up_sell_constraint = backtest_config.get('limit_up_sell_constraint', True)
        self.sell_dn_as_soon_as_possible = backtest_config.get('sell_dn_as_soon_as_possible', True)
        self.sell_up_as_soon_as_possible = backtest_config.get('sell_up_as_soon_as_possible', True)

        self.equal_weight = backtest_config.get('equal_weight', True)
        self.max_weight = backtest_config.get('max_weight', 1)
        self.weight_method = backtest_config.get('weight_method', None)
        self.tau = backtest_config.get('tau', 1)
        self.p = backtest_config.get('p', 1)

        self.stock_daily_account = pd.DataFrame(columns=['date', 'cash', 'market_value', 'total_value'])
        self.stock_daily_account = self.stock_daily_account.set_index('date')
        self.stock_position = pd.DataFrame(columns=['code', 'market_value', 'holding_days', 'ret'])
        self.stock_daily_position = pd.DataFrame(columns=['date', 'code', 'market_value', 'holding_days'])

        # 预先按日期排序好（score 降序，code 升序）
        self.score_df_groupby_date = self.merged_df.groupby('date').apply(
            lambda x: x.sort_values(['score', 'code'], ascending=[False, True])
        )

    # 日期备注：2022-01-04 reto2c代表当天的，retc2o代表前一天到2022-01-04开盘价的收益

    def update_stock_valuation(self):
        self.stock_position = self.stock_position[['code', 'market_value', 'holding_days']]
        if self.when == 'open':
            ret_df = self.score_df_groupby_date.loc[self.date][['code', 'ret_c2o']]
            self.stock_position = pd.merge(self.stock_position, ret_df, on='code', how='left')
            self.stock_position = self.stock_position.rename(columns={'ret_c2o': 'ret'})
        elif self.when == 'close':
            ret_df = self.score_df_groupby_date.loc[self.date][['code', 'ret_o2c']]
            self.stock_position = pd.merge(self.stock_position, ret_df, on='code', how='left')
            self.stock_position = self.stock_position.rename(columns={'ret_o2c': 'ret'})

        # 停牌/退市用 0 收益
        self.stock_position['ret'] = self.stock_position['ret'].fillna(0)
        self.stock_position['market_value'] = self.stock_position['market_value'] * (1 + self.stock_position['ret'])

        market_value = float(self.stock_position['market_value'].sum())
        self.total_value = market_value + self.cash

    def buy(self):
        score_current_date = self.score_df_groupby_date.loc[self.date].dropna(subset=['score'])

        # 剔除停牌
        mask = score_current_date['is_suspended'] == True
        # 剔除 ST
        if self.drop_st:
            mask |= (score_current_date['is_st'] == True)
        # 剔除开盘涨停
        if self.limit_up_buy_constraint:
            mask |= (score_current_date[f'{self.when}_type'] == 1)
        # 剔除开盘跌停
        if self.limit_dn_buy_constraint:
            mask |= (score_current_date[f'{self.when}_type'] == -1)

        score_current_date = score_current_date[~mask]

        # 支持按比例选股
        if self.stock_percent_range is not None:
            total_stocks = len(score_current_date)
            if total_stocks == 0:
                return
            start_percent, end_percent = self.stock_percent_range
            start_idx = int(total_stocks * start_percent)
            end_idx = int(total_stocks * end_percent)
            start_idx = min(start_idx, total_stocks - 1)
            end_idx = max(end_idx, start_idx + 1)
            to_buy = score_current_date.iloc[start_idx:end_idx].copy()
        else:
            # 否则按固定数目
            to_buy = score_current_date.head(self.stock_number).copy()

        if to_buy.empty:
            return

        total_trade_value = self.total_value / self.max_holding_days

        if self.equal_weight:
            # 等权
            unit_value = min(self.cash, total_trade_value) / len(to_buy)
            to_buy['market_value'] = unit_value * (1 - self.buy_cost)
            self.cash -= unit_value * len(to_buy)
        else:
            # 非等权（score→weight）
            if self.weight_method == 'softmax':
                to_buy['weight'] = np.exp((to_buy['score'] - to_buy['score'].max()) * self.tau)
                to_buy['weight'] = to_buy['weight'] / to_buy['weight'].sum()
            elif self.weight_method == 'rank':
                to_buy['weight'] = to_buy['score'].rank(ascending=False, method="first")
                to_buy['weight'] = to_buy['weight'] / to_buy['weight'].sum()
            elif self.weight_method == 'power':
                to_buy['weight'] = (to_buy['score'] - to_buy['score'].min()) ** self.p
                to_buy['weight'] = to_buy['weight'] / to_buy['weight'].sum()
            elif self.weight_method == 'custom':
                if 'weight' not in to_buy.columns:
                    raise ValueError("weight_method='custom' 时需要提供 'weight' 列")
            else:
                raise ValueError(f"未知的 weight_method: {self.weight_method}")

            # 限最大权重
            to_buy['weight'] = np.minimum(to_buy['weight'], self.max_weight)
            to_buy['weight'] = to_buy['weight'] / to_buy['weight'].sum()

            invest_value = min(self.cash, total_trade_value)
            to_buy['market_value'] = invest_value * to_buy['weight'] * (1 - self.buy_cost)
            self.cash -= invest_value

        to_buy['holding_days'] = 0
        to_buy['ret'] = np.nan
        to_buy = to_buy[['code', 'market_value', 'holding_days', 'ret']]

        self.stock_position = pd.concat([self.stock_position, to_buy], ignore_index=True)
        self.stock_position = self.stock_position.reset_index(drop=True)

    def sell(self):
        to_sell = self.stock_position[self.stock_position['holding_days'] >= self.max_holding_days]
        to_sell = to_sell.reset_index()
        if to_sell.empty:
            return

        to_sell = pd.merge(to_sell, self.score_df_groupby_date.loc[self.date], on='code', how='left')
        to_sell.set_index('index', inplace=True)

        mask = to_sell['is_suspended'] == True
        if self.limit_up_sell_constraint:
            mask |= (to_sell[f'{self.when}_type'] == 1)
        if self.limit_dn_sell_constraint:
            mask |= (to_sell[f'{self.when}_type'] == -1)

        to_sell = to_sell[~mask]

        if not to_sell.empty:
            self.cash += float(to_sell['market_value'].sum()) * (1 - self.sell_cost)
            self.stock_position = self.stock_position.drop(to_sell.index)

    def process_market_open(self):
        self.when = 'open'
        self.update_stock_valuation()

        if self.sell_up_as_soon_as_possible:
            self.sell()

        if self.sell_dn_as_soon_as_possible:
            self.sell()

        if self.trading_style == 'o2c':
            self.buy()
        elif self.trading_style == 'c2o':
            self.sell()
        elif self.trading_style == 'o2o':
            self.sell()
            self.buy()

    def process_market_close(self):
        self.when = 'close'
        self.update_stock_valuation()
        if not self.stock_position.empty:
            self.stock_position['holding_days'] += 1

        if self.trading_style == 'o2c':
            self.sell()
        elif self.trading_style == 'c2o':
            self.buy()
        elif self.trading_style == 'c2c':
            self.sell()
            self.buy()

    def update_account(self):
        self.stock_daily_account.loc[self.date] = [
            self.cash,
            self.total_value - self.cash,
            self.total_value,
        ]
        self.stock_daily_account.index = pd.to_datetime(self.stock_daily_account.index)

        stock_position = self.stock_position.copy()[['code', 'market_value', 'holding_days']]
        stock_position['date'] = self.date
        self.stock_daily_position = pd.concat([self.stock_daily_position, stock_position], ignore_index=True)

    def run(self):
        for date in self.date_list:
            self.date = date
            print(date)
            self.process_market_open()
            self.process_market_close()
            self.update_account()

    # ---------------- 对外取序列 ---------------- #

    def get_nv_series(self) -> pd.Series:
        """策略净值序列（以 initial_cash 归一到 1 开始）。"""
        return self.stock_daily_account['total_value'] / self.initial_cash

    def get_ret_series(self) -> pd.Series:
        """策略简单收益序列。"""
        first_ret = self.stock_daily_account['total_value'].iloc[0] / self.initial_cash - 1
        return self.stock_daily_account['total_value'].pct_change().fillna(first_ret)

    def get_benchmark_nav_series(self) -> pd.Series:
        """返回回测区间内的 benchmark 净值序列。"""
        return self.benchmark_nav_series.loc[self.date_list]

    # ---------------- 单策略 + 基准绩效评估 ---------------- #

    @staticmethod
    def calculate_single_strategy_metrics(
        nav_series: pd.Series,
        benchmark_nav_series: Optional[pd.Series] = None,
        rf_annual: float = 0.0,
        periods_per_year: int = DEFAULT_TRADING_DAYS,
    ) -> pd.DataFrame:
        """
        计算单策略（+ 可选基准）的绩效：
        - final_nav, annual_return, annual_volatility, sharpe_ratio, max_drawdown
        - 若给基准，则额外给：excess_return, alpha_return, beta_return, alpha, beta
        """
        nav_series = nav_series.dropna()
        if nav_series.empty:
            return pd.DataFrame()

        nav_norm = nav_series / nav_series.iloc[0]
        returns = returns_from_nav(nav_norm)

        strategy_metrics = {
            "final_nav": float(nav_series.iloc[-1]),
            "annual_return": annualized_return(returns, periods_per_year, geometric=True),
            "annual_volatility": annualized_volatility(returns, periods_per_year),
            "sharpe_ratio": sharpe_ratio(returns, rf_annual, periods_per_year),
            "max_drawdown": max_drawdown(nav_norm).max_drawdown,
        }

        benchmark_metrics = {}
        if benchmark_nav_series is not None:
            bench_nav_aligned = benchmark_nav_series.loc[nav_series.index].dropna()
            if not bench_nav_aligned.empty:
                bench_nav_norm = bench_nav_aligned / bench_nav_aligned.iloc[0]
                bench_returns = returns_from_nav(bench_nav_norm)

                bench_ann_ret = annualized_return(bench_returns, periods_per_year, geometric=True)
                strategy_ann_ret = strategy_metrics["annual_return"]

                benchmark_metrics = {
                    "final_nav": float(bench_nav_aligned.iloc[-1]),
                    "annual_return": bench_ann_ret,
                    "annual_volatility": annualized_volatility(bench_returns, periods_per_year),
                    "sharpe_ratio": sharpe_ratio(bench_returns, rf_annual, periods_per_year),
                    "max_drawdown": max_drawdown(bench_nav_norm).max_drawdown,
                    "excess_return": 0.0,
                    "alpha_return": 0.0,
                    "beta_return": bench_ann_ret,
                    "alpha": 0.0,
                    "beta": 1.0,
                }

                # 手写 CAPM，算 alpha / beta / excess / alpha_return
                df = pd.concat(
                    [returns.rename("strategy"), bench_returns.rename("benchmark")],
                    axis=1,
                ).dropna()

                if len(df) >= 2:
                    s = df["strategy"]
                    b = df["benchmark"]

                    var_b = float(np.var(b.values, ddof=0))
                    cov_sb = float(np.cov(b.values, s.values, ddof=0)[0, 1]) if var_b > 0 else np.nan
                    if var_b > 0 and not np.isnan(cov_sb):
                        beta = cov_sb / var_b
                        alpha_per_period = float(s.mean() - beta * b.mean())
                        alpha_annual = (1.0 + alpha_per_period) ** periods_per_year - 1.0
                        beta_return = beta * bench_ann_ret

                        strategy_metrics.update({
                            "excess_return": strategy_ann_ret - bench_ann_ret,
                            "alpha_return": strategy_ann_ret - beta_return,
                            "beta_return": beta_return,
                            "alpha": alpha_annual,
                            "beta": beta,
                        })

        metrics_data = {
            "strategy": strategy_metrics,
            "benchmark": benchmark_metrics,
        }
        return pd.DataFrame(metrics_data)


    def get_metrics(self):
        nav = self.get_nv_series()
        bench_nav = self.get_benchmark_nav_series()
        metrics_df = self.calculate_single_strategy_metrics(nav, benchmark_nav_series=bench_nav)
        print(metrics_df)
        return metrics_df


if __name__ == "__main__":
    score_df = pd.read_parquet(r'E:\models_output\quanthw_202509\backtest\score_df_example_2022-01-04_2023-12-29.parquet')
    ret_df = pd.read_parquet(r'E:\models_output\quanthw_202509\backtest\ret_df_2022-01-04_2023-12-29.parquet')

    backtest_config = {
        'start_date': '2022-01-04',
        'end_date': '2023-12-29',
        'buy_cost': 0.0002,
        'sell_cost': 0.0012,
        'stock_number': 10,
        'max_holding_days': 2,
        'cash': 10000000,
        'trading_style': "o2c",
        'drop_st': True,
        'limit_up_buy_constraint': True,
        'limit_dn_buy_constraint': False,
        'limit_up_sell_constraint': False,
        'limit_dn_sell_constraint': True,
        'sell_dn_as_soon_as_possible': False,
        'sell_up_as_soon_as_possible': False,
        'equal_weight': True,
    }

    bt = SimpleBackTest(score_df, ret_df, backtest_config)
    bt.run()

    nav = bt.get_nv_series() ##净值
    metrics_df = bt.get_metrics() ##评估指标
