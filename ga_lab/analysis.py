from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

from .database import Database


def visualize_results(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates and saves all visualizations for the top strategies."""
    if df.empty:
        return

    output_dir.mkdir(exist_ok=True)
    print(f"\n--- Generating Visualizations in {output_dir} ---")

    # 1. Fitness Distribution
    plt.figure(figsize=(12, 7))
    sns.histplot(df["fitness"], kde=True, bins=20, color="skyblue")
    plt.title("Distribution of Fitness Scores", fontsize=16)
    plt.xlabel("Fitness")
    plt.ylabel("Frequency")
    fitness_plot_path = output_dir / "fitness_distribution.png"
    plt.savefig(fitness_plot_path)
    plt.close()
    print(f"Saved fitness distribution plot to {fitness_plot_path}")

    # 2. Parameter Correlation Heatmap
    numeric_cols = df.select_dtypes(include=["number"]).columns
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(24, 20))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
    plt.title("Correlation Heatmap of Strategy Parameters", fontsize=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    heatmap_plot_path = output_dir / "parameter_heatmap.png"
    plt.savefig(heatmap_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter heatmap to {heatmap_plot_path}")

    # 3. Key Parameter Distributions
    dist_cols = [c for c in df.columns if c.startswith(('params.', 'thresholds.')) and len(df[c].unique()) > 1][:9]
    if dist_cols:
        num_plots = len(dist_cols)
        num_cols = 3
        num_rows = (num_plots + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows), constrained_layout=True)
        axes = axes.flatten()
        fig.suptitle('Distributions of Key Strategy Parameters', fontsize=22)
        for i, col in enumerate(dist_cols):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(col, fontsize=14)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
        for i in range(num_plots, len(axes)):
            fig.delaxes(axes[i])
        param_dist_path = output_dir / "parameter_distributions.png"
        plt.savefig(param_dist_path)
        plt.close()
        print(f"Saved parameter distributions plot to {param_dist_path}")

    # 4. 2D Performance Scatter Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="win_rate", y="profit_factor", size="trades", hue="fitness", palette="viridis", sizes=(50, 500), alpha=0.7)
    plt.title("Strategy Performance: Win Rate vs. Profit Factor")
    plt.xlabel("Win Rate")
    plt.ylabel("Profit Factor")
    plt.legend(title="Fitness")
    plt.grid(True)
    scatter_plot_path = output_dir / "performance_scatterplot.png"
    plt.savefig(scatter_plot_path)
    plt.close()
    print(f"Saved performance scatter plot to {scatter_plot_path}")

    # 5. 3D Scatter Plot of Fitness, Win Rate, and Profit Factor
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['win_rate'], df['profit_factor'], df['fitness'], c=df['fitness'], cmap='viridis', s=df['trades']/df['trades'].max()*200+50, alpha=0.8)
    ax.set_xlabel('Win Rate')
    ax.set_ylabel('Profit Factor')
    ax.set_zlabel('Fitness')
    ax.set_title('3D View: Fitness, Win Rate, and Profit Factor')
    plt.colorbar(sc, label='Fitness Score')
    threed_scatter_path = output_dir / "3d_scatter_plot.png"
    plt.savefig(threed_scatter_path)
    plt.close()
    print(f"Saved 3D scatter plot to {threed_scatter_path}")


def generate_html_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates a static HTML report with embedded plots and data."""
    if df.empty:
        return

    top_strategy = df.iloc[0]

    def embed_image(path):
        try:
            with path.open("rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except FileNotFoundError:
            return None

    fitness_plot_b64 = embed_image(output_dir / "fitness_distribution.png")
    heatmap_plot_b64 = embed_image(output_dir / "parameter_heatmap.png")
    param_dist_b64 = embed_image(output_dir / "parameter_distributions.png")
    scatter_plot_b64 = embed_image(output_dir / "performance_scatterplot.png")
    threed_scatter_b64 = embed_image(output_dir / "3d_scatter_plot.png")

    def series_to_html_table(series, title):
        table = f'<h3>{title}</h3><table class="param-table">'
        for idx, val in series.items():
            name = idx.split('.')[-1].replace('_', ' ').title()
            val_str = f'{val:.4f}' if isinstance(val, float) else str(val)
            table += f'<tr><td><strong>{name}</strong></td><td>{val_str}</td></tr>'
        return table + '</table>'

    params_html = series_to_html_table(top_strategy.filter(like="params."), "Indicator Parameters")
    thresholds_html = series_to_html_table(top_strategy.filter(like="thresholds."), "Trading Thresholds")
    weights_html = series_to_html_table(top_strategy.filter(like="weights."), "Indicator Weights")

    def generate_formula_html(strategy):
        weights = strategy.filter(like='weights.')
        thresholds = strategy.filter(like='thresholds.')
        buy_conditions = [f"({w:.2f} * {name.split('.')[-1]})" for name, w in weights.items()]
        buy_formula = " + ".join(buy_conditions)

        buy_threshold = thresholds.get('thresholds.buy', 'N/A')
        sell_threshold = thresholds.get('thresholds.sell', 'N/A')

        buy_threshold_str = f"{buy_threshold:.2f}" if isinstance(buy_threshold, (int, float)) else buy_threshold
        sell_threshold_str = f"{sell_threshold:.2f}" if isinstance(sell_threshold, (int, float)) else sell_threshold

        html = f"""<h3>Strategy Logic</h3>
        <div class='formula'>
            <p><strong>Buy Signal:</strong></p>
            <code>({buy_formula}) > {buy_threshold_str}</code>
            <p><strong>Sell Signal:</strong></p>
            <code>({buy_formula}) < {sell_threshold_str}</code>
        </div>"""
        return html

    formula_html = generate_formula_html(top_strategy)
    strategies_table_html = df.to_html(index=False, classes='data-table')

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GA Strategy Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #3a3a3a; background-color: #f4f7f6; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: auto; background: #ffffff; padding: 40px; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.05); }}
        h1, h2, h3 {{ color: #2c3e50; }}
        h1 {{ font-size: 3em; text-align: center; margin-bottom: 20px; border-bottom: 4px solid #4a90e2; padding-bottom: 20px; }}
        h2 {{ font-size: 2.4em; margin-top: 60px; border-bottom: 2px solid #e8e8e8; padding-bottom: 15px; }}
        h3 {{ font-size: 1.6em; margin-top: 30px; }}
        .section {{ margin-bottom: 60px; }}
        .narrative {{ background-color: #f8f9fa; border-left: 5px solid #4a90e2; padding: 20px; margin: 25px 0; border-radius: 8px; font-size: 1.1em; }}
        .narrative p {{ margin: 0; }}
        .formula {{ background-color: #2d3748; color: #f7fafc; padding: 20px; border-radius: 8px; margin-top: 15px; font-family: 'SF Mono', 'Courier New', Courier, monospace; }}
        .formula code {{ color: #a7c7e7; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }}
        th, td {{ padding: 16px; text-align: left; border: 1px solid #e1e1e1; }}
        .data-table th {{ background-color: #4a90e2; color: white; font-size: 1.1em; }}
        .data-table tr:nth-child(even) {{ background-color: #fbfcfd; }}
        .param-table td:first-child {{ font-weight: bold; width: 60%; }}
        img {{ max-width: 100%; height: auto; border-radius: 12px; margin-top: 20px; box-shadow: 0 6px 18px rgba(0,0,0,0.1); display: block; margin-left: auto; margin-right: auto; }}
        .plot-container img {{ width: 90%; }}
        .top-strategy-layout {{ display: flex; gap: 40px; align-items: flex-start; }}
        .top-strategy-details {{ flex: 1; }}
        .top-strategy-params {{ flex: 1; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Genetic Algorithm Strategy Analysis</h1>
        <p style="text-align:center; font-style: italic; color: #777;">Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="section top-strategy-layout">
            <div class="top-strategy-details">
                <h2>Top Performing Strategy</h2>
                <h3>Performance Overview</h3>
                <table class="param-table">
                    <tr><td><strong>Fitness Score</strong></td><td>{top_strategy['fitness']:.2f}</td></tr>
                    <tr><td><strong>Win Rate</strong></td><td>{top_strategy['win_rate']:.2%}</td></tr>
                    <tr><td><strong>Profit Factor</strong></td><td>{top_strategy['profit_factor']:.2f}</td></tr>
                    <tr><td><strong>Total Trades</strong></td><td>{top_strategy['trades']}</td></tr>
                </table>
                {formula_html}
            </div>
            <div class="top-strategy-params">
                <h3>Parameters & Weights</h3>
                {params_html}
                {thresholds_html}
                {weights_html}
            </div>
        </div>

        <div class="section plot-container">
            <h2>Strategy Parameter Correlation Heatmap</h2>
            <div class="narrative">
                <p><strong>What it is:</strong> This heatmap reveals the relationships between different strategy parameters and their impact on overall fitness. The 'fitness' row is the most critical for analysis.
                <br><strong>How to use it:</strong> Look for strong colors in the 'fitness' row. Bright red indicates a strong positive correlation (higher parameter value = higher fitness), while bright blue indicates a strong negative correlation. This helps you identify the most influential parameters for building a successful strategy.</p>
            </div>
            {f'<img src="data:image/png;base64,{heatmap_plot_b64}" alt="Parameter Heatmap">' if heatmap_plot_b64 else ''}
        </div>

        <div class="section plot-container">
            <h2>Fitness Distribution</h2>
            <div class="narrative">
                <p><strong>What it is:</strong> This histogram shows the distribution of fitness scores across all the top strategies discovered by the algorithm.
                <br><strong>How to use it:</strong> A distribution skewed to the right is a great sign, indicating the algorithm consistently found high-performing solutions. A wide distribution suggests a diverse set of successful strategies was explored, while a narrow peak might mean only one type of strategy was dominant.</p>
            </div>
            {f'<img src="data:image/png;base64,{fitness_plot_b64}" alt="Fitness Distribution">' if fitness_plot_b64 else ''}
        </div>

        <div class="section plot-container">
            <h2>Strategy Performance: Win Rate vs. Profit Factor</h2>
            <div class="narrative">
                <p><strong>What it is:</strong> This scatter plot maps each strategy based on its win rate and profit factor. The size of each bubble represents the number of trades, and the color indicates its fitness score.
                <br><strong>How to use it:</strong> The ideal strategy is in the top-right corner (high win rate, high profit factor). This plot helps you visually identify trade-offs. Some strategies might have a high win rate but low profit, while others might be the opposite. The color helps confirm if these strategies also have high fitness scores.</p>
            </div>
            {f'<img src="data:image/png;base64,{scatter_plot_b64}" alt="Performance Scatter Plot">' if scatter_plot_b64 else ''}
        </div>

        <div class="section plot-container">
            <h2>3D View: Fitness, Win Rate, and Profit Factor</h2>
            <div class="narrative">
                <p><strong>What it is:</strong> This 3D plot provides a multi-dimensional view of strategy performance, plotting fitness, win rate, and profit factor together. The size of each point still represents the number of trades.
                <br><strong>How to use it:</strong> This view helps uncover more complex relationships. Look for clusters of points in the upper regions of the cube, as they represent the most desirable strategies across all three key metrics. It's a powerful way to see if the highest-fitness strategies are also the ones with the best win rates and profit factors.</p>
            </div>
            {f'<img src="data:image/png;base64,{threed_scatter_b64}" alt="3D Scatter Plot">' if threed_scatter_b64 else ''}
        </div>

        <div class="section plot-container">
            <h2>Distributions of Key Strategy Parameters</h2>
            <div class="narrative">
                <p><strong>What it is:</strong> These histograms show how the values for the most important parameters are distributed across the top-performing strategies.
                <br><strong>How to use it:</strong> Look for clear peaks or trends. If a parameter's distribution is tightly clustered around a specific value, it suggests that this value is critical for success. A flat or uniform distribution might mean the parameter is less influential. This can provide strong hints for manually fine-tuning strategies.</p>
            </div>
            {f'<img src="data:image/png;base64,{param_dist_b64}" alt="Parameter Distributions">' if param_dist_b64 else ''}
        </div>

        <div class="section">
            <h2>Top {len(df)} Strategies Data</h2>
            {strategies_table_html}
        </div>
    </div>
</body>
</html>"""

    report_path = output_dir / "strategy_report.html"
    with report_path.open("w") as f:
        f.write(html_content)

    print(f"\n--- HTML Report Generation ---")
    print(f"Saved HTML report to {report_path}")


def view_results(db: Database, top_n: int = 20, visualize: bool = False, report: bool = False) -> None:
    """Fetches and displays top N strategies from the database."""
    print(f"--- Top {top_n} Strategies ---")
    strategies_df = db.get_all_strategies(limit=top_n)

    if strategies_df.empty:
        print("No strategies found in the database.")
        return

    expanded_df = strategies_df.copy()
    for col_name in ["params", "thresholds", "weights"]:
        if col_name in expanded_df.columns:
            json_data = expanded_df[col_name].apply(json.loads)
            normalized_df = pd.json_normalize(json_data).add_prefix(f"{col_name}.")
            expanded_df = expanded_df.drop(col_name, axis=1).join(normalized_df)

    pd.set_option("display.max_rows", top_n)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 250)

    print(expanded_df)

    output_dir = Path("analysis_plots")
    if visualize or report:
        output_dir.mkdir(exist_ok=True)
        if visualize or report:
            visualize_results(expanded_df, output_dir)
        
        if report:
            generate_html_report(expanded_df, output_dir)
