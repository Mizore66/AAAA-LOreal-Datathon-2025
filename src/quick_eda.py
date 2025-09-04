from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / 'data' / 'processed'
INTERIM = ROOT / 'data' / 'interim'
INTERIM.mkdir(parents=True, exist_ok=True)


def main():
    lines = ['# Quick EDA Summary', '']
    for p in sorted(PROC.glob('*.parquet')):
        try:
            df = pd.read_parquet(p)
            lines.append(f"## {p.name}")
            lines.append(f"- Shape: {df.shape}")
            head = df.head(3)
            cols = ", ".join(map(str, head.columns[:20]))
            lines.append(f"- Columns: {cols}{'...' if head.shape[1] > 20 else ''}")
            lines.append("- Head:\n\n" + head.to_markdown(index=False))
            lines.append("")
        except Exception as e:
            lines.append(f"## {p.name}")
            lines.append(f"- ERROR reading parquet: {e}")
    out = INTERIM / 'eda_quick_summary.md'
    with open(out, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
