import pandas as pd
from pathlib import Path
import argparse

# --------------------------------
# 1. Load cleaned wildfire dataset
# --------------------------------

def load_data(file_path=None):
    if file_path:
        path = Path(file_path)
    else:
        candidates = [
            Path("data/cleaned_data.csv"),
            Path("cleaned_data.csv"),
            Path("data/cleaned.csv"),
            Path("clean_data.csv")
        ]

        path = next(
            (p for p in candidates if p.exists()),
            None
        )

        if path is None:
            raise FileNotFoundError(
                "Cleaned CSV not found. "
                "Please provide its file path."
            )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    print("Dataset loaded:", path)
    print("Total records:", len(df))

    return df


# --------------------------------
# 2. Identify dataset columns
# --------------------------------

def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def analyze_data(df):

    results = []

    date_col = find_column(
        df, ["acq_date", "date", "fire_date", "datetime"]
    )

    year_col = find_column(df, ["year"])
    month_col = find_column(df, ["month"])

    region_col = find_column(
        df, ["region", "state", "province", "country"]
    )

    brightness_col = find_column(
        df, ["brightness", "bright_ti4", "bright_ti5"]
    )

    frp_col = find_column(
        df, ["frp", "fire_radiative_power"]
    )

    confidence_col = find_column(
        df, ["confidence", "conf"]
    )

    daynight_col = find_column(
        df, ["daynight", "day_night"]
    )

    # Extract year and month from date
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce")

        if not year_col:
            df["year"] = dates.dt.year
            year_col = "year"

        if not month_col:
            df["month"] = dates.dt.month
            month_col = "month"

    # --------------------------------
    # Helper function for summaries
    # --------------------------------

    def add_summary(column, analysis_name):
        if column is None:
            return

        grouped = df.groupby(column, dropna=False)

        for category, group in grouped:
            results.append({
                "analysis": analysis_name,
                "category": str(category),
                "fire_count": len(group),
                "average_brightness": (
                    group[brightness_col].mean()
                    if brightness_col else None
                ),
                "average_frp": (
                    group[frp_col].mean()
                    if frp_col else None
                )
            })

    # --------------------------------
    # 3. Year-wise analysis
    # --------------------------------

    add_summary(year_col, "Year-wise Analysis")

    # --------------------------------
    # 4. Month-wise analysis
    # --------------------------------

    add_summary(month_col, "Month-wise Analysis")

    # --------------------------------
    # 5. Region-wise analysis
    # --------------------------------

    add_summary(region_col, "Region-wise Analysis")

    # --------------------------------
    # 6. Fire intensity analysis
    # --------------------------------

    for column, name in [
        (brightness_col, "Brightness Analysis"),
        (frp_col, "FRP Analysis")
    ]:
        if column:
            results.append({
                "analysis": name,
                "category": "Overall",
                "fire_count": df[column].count(),
                "average_brightness": (
                    df[brightness_col].mean()
                    if brightness_col else None
                ),
                "average_frp": (
                    df[frp_col].mean()
                    if frp_col else None
                )
            })

    # --------------------------------
    # 7. Confidence level analysis
    # --------------------------------

    add_summary(confidence_col, "Confidence Analysis")

    # --------------------------------
    # 8. Day vs Night analysis
    # --------------------------------

    add_summary(daynight_col, "Day vs Night Analysis")

    return pd.DataFrame(results)


# --------------------------------
# 9. Identify important patterns
# --------------------------------

def generate_report(df, results):

    lines = [
        "# Wildfire Data Analysis",
        "",
        "## Dataset Overview",
        "",
        f"- Total wildfire records: {len(df)}",
        f"- Total columns: {len(df.columns)}",
        ""
    ]

    if "year" in df.columns:
        yearly = df.groupby("year").size()

        lines += [
            "## Year-wise Analysis",
            "",
            yearly.to_string(),
            ""
        ]

        if not yearly.empty:
            highest = yearly.idxmax()
            lowest = yearly.idxmin()

            lines += [
                f"- Highest recorded year: {highest} "
                f"({yearly.max()} records)",
                f"- Lowest recorded year: {lowest} "
                f"({yearly.min()} records)",
                ""
            ]

    if "month" in df.columns:
        monthly = df.groupby("month").size()

        lines += [
            "## Month-wise Analysis",
            "",
            monthly.to_string(),
            ""
        ]

    if results.empty:
        lines.append("No grouped analysis could be generated.")
    else:
        lines += [
            "## Analysis Results",
            "",
            results.to_markdown(index=False),
            ""
        ]

    lines += [
        "## Important Observations",
        "",
        "The generated CSV contains the grouped analysis "
        "for available dataset columns.",
        "",
        "Interpret year-wise, month-wise, regional, "
        "intensity, confidence, and day/night patterns "
        "using the results above.",
        "",
        "Note: Missing dataset columns are skipped. "
        "No unavailable values are assumed."
    ]

    Path("doc").mkdir(exist_ok=True)

    with open("doc/data_analysis.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --------------------------------
# 10. Main execution
# --------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Wildfire Data Analysis"
    )

    parser.add_argument(
        "--input",
        help="Path to cleaned wildfire CSV file"
    )

    args = parser.parse_args()

    df = load_data(args.input)

    results = analyze_data(df)

    results.to_csv(
        "analysis_results.csv",
        index=False
    )

    generate_report(df, results)

    print("\nAnalysis completed successfully!")
    print("Generated files:")
    print("1. analysis_results.csv")
    print("2. doc/data_analysis.md")


if __name__ == "__main__":
    main()
